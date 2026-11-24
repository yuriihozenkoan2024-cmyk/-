// MAVKA offline test bench (no JUCE needed)
//   mavka_test list
//   mavka_test render <in.wav> <out.wav> <preset> [--bpm 95] [--sr] [--set id=value ...]
//   mavka_test renderall <in.wav> <outdir> [--bpm 95]
//   mavka_test latency | alias | stress | cpu | switch | duck | crush | flange | melodic | reverbsafe | tuning
#include "MavkaEngine.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#if defined(__SSE__) || defined(__x86_64__)
 #include <xmmintrin.h>
 static void noDenormals() { _mm_setcsr (_mm_getcsr() | 0x8040); }
#else
 static void noDenormals() {}
#endif

using namespace mavka;

//==============================================================================
struct Audio { int sr = 48000; std::vector<std::vector<float>> ch; size_t frames() const { return ch.empty() ? 0 : ch[0].size(); } };

static bool readWav (const std::string& path, Audio& a)
{
    FILE* f = std::fopen (path.c_str(), "rb");
    if (! f) return false;
    std::vector<unsigned char> d;
    unsigned char buf[65536];
    size_t r;
    while ((r = std::fread (buf, 1, sizeof (buf), f)) > 0) d.insert (d.end(), buf, buf + r);
    std::fclose (f);
    auto u16 = [&] (size_t o) { return (unsigned) d[o] | ((unsigned) d[o + 1] << 8); };
    auto u32 = [&] (size_t o) { return u16 (o) | (u16 (o + 2) << 16); };
    if (d.size() < 12 || std::string ((char*) &d[0], 4) != "RIFF") return false;
    size_t pos = 12; int fmt = 0, nch = 0, bits = 0;
    while (pos + 8 <= d.size())
    {
        std::string id ((char*) &d[pos], 4);
        const size_t sz = u32 (pos + 4);
        const size_t body = pos + 8;
        if (id == "fmt ")
        {
            fmt = (int) u16 (body); nch = (int) u16 (body + 2); a.sr = (int) u32 (body + 4); bits = (int) u16 (body + 14);
            if (fmt == 0xFFFE) fmt = (int) u16 (body + 24);
        }
        else if (id == "data")
        {
            const int bps = bits / 8;
            const size_t n = std::min (sz, d.size() - body) / (size_t) (bps * nch);
            a.ch.assign ((size_t) nch, std::vector<float> (n));
            for (size_t i = 0; i < n; ++i)
                for (int c = 0; c < nch; ++c)
                {
                    const size_t o = body + (i * (size_t) nch + (size_t) c) * (size_t) bps;
                    float v = 0.0f;
                    if (fmt == 3 && bits == 32) { uint32_t x = u32 (o); std::memcpy (&v, &x, 4); }
                    else if (bits == 16) v = (float) (int16_t) u16 (o) / 32768.0f;
                    else if (bits == 24) { int32_t x = (int32_t) ((d[o] << 8) | (d[o + 1] << 16) | (d[o + 2] << 24)); v = (float) (x >> 8) / 8388608.0f; }
                    else if (bits == 32) v = (float) (int32_t) u32 (o) / 2147483648.0f;
                    a.ch[(size_t) c][i] = v;
                }
            return true;
        }
        pos = body + sz + (sz & 1);
    }
    return false;
}

static bool writeWav (const std::string& path, const Audio& a)
{
    FILE* f = std::fopen (path.c_str(), "wb");
    if (! f) return false;
    const int nch = (int) a.ch.size();
    const uint32_t n = (uint32_t) a.frames();
    const uint32_t dataBytes = n * (uint32_t) nch * 3;
    auto w32 = [&] (uint32_t v) { std::fwrite (&v, 4, 1, f); };
    auto w16 = [&] (uint16_t v) { std::fwrite (&v, 2, 1, f); };
    std::fwrite ("RIFF", 1, 4, f); w32 (36 + dataBytes); std::fwrite ("WAVEfmt ", 1, 8, f);
    w32 (16); w16 (1); w16 ((uint16_t) nch); w32 ((uint32_t) a.sr); w32 ((uint32_t) a.sr * (uint32_t) nch * 3); w16 ((uint16_t) (nch * 3)); w16 (24);
    std::fwrite ("data", 1, 4, f); w32 (dataBytes);
    for (uint32_t i = 0; i < n; ++i)
        for (int c = 0; c < nch; ++c)
        {
            const float v = clampf (a.ch[(size_t) c][i], -1.0f, 1.0f);
            const int32_t x = (int32_t) std::lrint (v * 8388607.0f);
            const unsigned char b[3] = { (unsigned char) (x & 255), (unsigned char) ((x >> 8) & 255), (unsigned char) ((x >> 16) & 255) };
            std::fwrite (b, 1, 3, f);
        }
    std::fclose (f);
    return true;
}

//==============================================================================
// Runs the engine over a whole buffer with host-like blocks and a playing transport.
static void runEngine (Engine& e, std::vector<std::vector<float>>& io, double sr, double bpm, int block = 512,
                       const std::function<void (size_t)>& perBlock = {})
{
    const size_t n = io[0].size();
    const int nch = (int) io.size();
    for (size_t pos = 0; pos < n; pos += (size_t) block)
    {
        const int len = (int) std::min ((size_t) block, n - pos);
        if (perBlock) perBlock (pos);
        e.setTransport (true, bpm, (double) pos / sr * bpm / 60.0);
        float* ptr[2] = { io[0].data() + pos, io[(size_t) std::min (1, nch - 1)].data() + pos };
        e.process (ptr, nch, len);
    }
}

static void linearParams (float* v)
{
    defaultValues (v);
    const char* zeros[] = { "drive", "grit", "melody", "motion", "ink", "slant", "ligature", "glass", "fold", "doppler",
                            "bias", "ghostDown", "ghostUp", "consist", "focus", "smooth", "preTilt", "tone", "warp",
                            "echo", "forest", "twin", "crush", "flange", "clip", "clipForce", "prism" };
    for (auto z : zeros) v[findParam (z)] = 0.0f;
    v[P_LAMP] = 1.0f; v[P_MIX] = 1.0f; v[P_LOWCUT] = 20.0f; v[P_HIGHCUT] = 20000.0f;
    v[P_SING] = 1.0f; v[P_DIRT] = 1.0f;   // the plain distortion (the overtone filters are idle)
}

// the melodic distortion switched to "plain": exactly what an ordinary distortion would add
static void plainDistortion (float* v)
{
    v[P_SING] = 1.0f; v[P_DIRT] = 1.0f; v[P_FORCE] = 0.0f; v[P_PRISM] = 0.0f; v[P_REVSAFE] = 0.0f;
    for (int h = 0; h < 11; ++h) v[P_OT2 + h] = 0.0f;
}


//==============================================================================
static int testLatency()
{
    const double sr = 48000.0;
    const size_t n = (size_t) (sr * 1.5);
    std::vector<float> in (n);
    for (size_t i = 0; i < n; ++i)
    {
        const double t = (double) i / sr;
        in[i] = (float) (0.001 * (std::sin (2 * kPiD * 100 * t) + std::sin (2 * kPiD * 1000 * t) + std::sin (2 * kPiD * 6000 * t)));
    }
    std::vector<std::vector<float>> outs;
    int lat = 0;
    for (int os = 1; os <= 5; ++os)
    {
        Engine e; e.prepare (sr, 1);
        float v[P_COUNT]; linearParams (v); v[P_OS] = (float) os; e.setParams (v);
        std::vector<std::vector<float>> io { in };
        runEngine (e, io, sr, 95.0);
        lat = e.getLatency();
        outs.push_back (io[0]);
    }
    // reported latency already includes the lead (lamp) pre-delay
    const int expected = lat;
    std::printf ("reported latency (all factors) = %d samples\n", lat);
    const size_t a = (size_t) (0.5 * sr), b = n - 2000;
    int fails = 0;
    for (int k = 0; k < 5; ++k)
    {
        // best integer lag against the input
        double best = -1e30; int bestLag = 0;
        for (int lag = expected - 20; lag <= expected + 20; ++lag)
        {
            double s = 0; for (size_t i = a; i < b; ++i) s += (double) outs[(size_t) k][i] * in[i - (size_t) lag];
            if (s > best) { best = s; bestLag = lag; }
        }
        double diff = 0, ref = 0;
        for (size_t i = a; i < b; ++i) { const double d = outs[(size_t) k][i] - outs[1][i]; diff += d * d; ref += (double) outs[1][i] * outs[1][i]; }
        const double rel = 10 * std::log10 (diff / ref + 1e-30);
        std::printf ("  %2dx: best lag %d, difference vs 2x: %.1f dB\n", 1 << k, bestLag, k == 1 ? -999.0 : rel);
        if (std::abs (bestLag - expected) > 1) ++fails;
        if (k >= 2 && rel > -60.0) ++fails;
    }
    std::printf (fails ? "LATENCY: FAIL\n" : "LATENCY: OK\n");
    return fails;
}

static void fftMag (const std::vector<double>& x, std::vector<double>& mag)
{
    const size_t n = x.size();
    std::vector<std::complex<double>> a (n);
    for (size_t i = 0; i < n; ++i) a[i] = x[i];
    for (size_t i = 1, j = 0; i < n; ++i)
    {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap (a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1)
    {
        const double ang = -2 * kPiD / (double) len;
        const std::complex<double> wl (std::cos (ang), std::sin (ang));
        for (size_t i = 0; i < n; i += len)
        {
            std::complex<double> w (1);
            for (size_t k = 0; k < len / 2; ++k) { auto u = a[i + k], t = a[i + k + len / 2] * w; a[i + k] = u + t; a[i + k + len / 2] = u - t; w *= wl; }
        }
    }
    mag.resize (n / 2);
    for (size_t i = 0; i < n / 2; ++i) mag[i] = std::norm (a[i]);
}

static int testAlias()
{
    const double sr = 48000.0, f0 = 3150.0;
    const size_t N = 65536;
    std::printf ("aliasing test: %.0f Hz sine, heavy settings (drive 100%%, grit 80%%, fold 60%%) at %.0f Hz\n", f0, sr);
    double res[7] = {};
    for (int os = 0; os <= 5; ++os)
    {
        Engine e; e.prepare (sr, 1);
        float v[P_COUNT]; linearParams (v);
        v[P_DRIVE] = 1.0f; v[P_GRIT] = 0.8f; v[P_FOLD] = 0.6f; v[P_GLASS] = 0.5f; v[P_CONSIST] = 1.0f; v[P_OS] = (float) os;
        e.setParams (v);
        const size_t n = N + 24000;
        std::vector<std::vector<float>> io (1, std::vector<float> (n));
        for (size_t i = 0; i < n; ++i) io[0][i] = (float) (0.25 * std::sin (2 * kPiD * f0 * (double) i / sr));
        runEngine (e, io, sr, 95.0);
        std::vector<double> x (N);
        for (size_t i = 0; i < N; ++i)
        {
            const double w = 0.35875 - 0.48829 * std::cos (2 * kPiD * i / (N - 1)) + 0.14128 * std::cos (4 * kPiD * i / (N - 1)) - 0.01168 * std::cos (6 * kPiD * i / (N - 1));
            x[i] = io[0][i + 24000] * w;
        }
        std::vector<double> mag; fftMag (x, mag);
        double harm = 0, alias = 0;
        const double binHz = sr / (double) N;
        for (size_t k = 3; k < mag.size(); ++k)
        {
            const double f = (double) k * binHz;
            if (f > 20000.0) break;
            const double h = f / f0, dist = std::fabs (h - std::round (h)) * f0 / binHz;
            if (std::round (h) >= 1 && dist <= 6) harm += mag[k]; else alias += mag[k];
        }
        res[os] = 10 * std::log10 (alias / harm + 1e-30);
        std::printf ("  %-5s (running %2dx): non-harmonic energy %.1f dB below the harmonics\n", kOsChoices[os], e.getActiveFactor(), -res[os]);
    }
    const bool ok = res[0] < -60.0 && res[4] < -60.0 && res[5] < res[3];
    std::printf (ok ? "ALIAS: OK\n" : "ALIAS: CHECK\n");
    return ok ? 0 : 1;
}

static int testStress()
{
    std::mt19937 rng (1234);
    std::uniform_real_distribution<float> U (0.0f, 1.0f);
    const double rates[] = { 44100.0, 48000.0, 96000.0, 192000.0 };
    int fails = 0;
    for (double sr : rates)
        for (int nch = 1; nch <= 2; ++nch)
        {
            Engine e; e.prepare (sr, nch);
            const size_t n = (size_t) (sr * 12.0);
            std::vector<std::vector<float>> io ((size_t) nch, std::vector<float> (n));
            for (size_t i = 0; i < n; ++i)
            {
                const double t = (double) i / sr;
                const int seg = (int) (t * 4.0) % 8;
                float s = 0.0f;
                switch (seg)
                {
                    case 0: s = 0.0f; break;
                    case 1: s = (U (rng) * 2.0f - 1.0f); break;
                    case 2: s = (float) std::sin (2 * kPiD * (80 + 6000 * std::fmod (t, 1.0)) * t); break;
                    case 3: s = (std::fmod (t * 110.0, 1.0) < 0.5 ? 1.0f : -1.0f); break;
                    case 4: s = (i % 4800 == 0) ? 1.0f : 0.0f; break;
                    case 5: s = 1.0e-30f; break;
                    case 6: s = 0.3f * (float) std::sin (2 * kPiD * 140 * t) * (U (rng) > 0.5f ? 1.0f : 0.2f); break;
                    default: s = 1.0f; break;   // DC
                }
                for (int c = 0; c < nch; ++c) io[(size_t) c][i] = s * (c ? 0.8f : 1.0f);
            }
            float v[P_COUNT];
            const ParamDef* d = paramDefs();
            runEngine (e, io, sr, 140.0, 333, [&] (size_t pos)
            {
                if (pos % (size_t) (sr * 0.25) < 333)
                    for (int p = 0; p < P_COUNT; ++p)
                    {
                        float x = d[p].minV + U (rng) * (d[p].maxV - d[p].minV);
                        if (d[p].type != PType::Float) x = std::round (x);
                        v[p] = x;
                    }
                v[P_OUTPUT] = 0.0f;
                e.setParams (v);
            });
            float peak = 0.0f; bool finite = true;
            for (auto& c : io) for (float x : c) { if (! std::isfinite (x)) finite = false; peak = std::max (peak, std::fabs (x)); }
            std::printf ("  %6.0f Hz, %d ch: finite=%s peak=%.2f\n", sr, nch, finite ? "yes" : "NO", peak);
            if (! finite || peak > 12.0f) ++fails;
        }
    std::printf (fails ? "STRESS: FAIL\n" : "STRESS: OK\n");
    return fails;
}

static int testCpu()
{
    const double sr = 48000.0;
    const size_t n = (size_t) (sr * 20.0);
    std::mt19937 rng (7);
    std::normal_distribution<float> G (0.0f, 0.1f);
    std::vector<float> src (n);
    for (size_t i = 0; i < n; ++i) src[i] = 0.3f * (float) std::sin (2 * kPiD * 150 * (double) i / sr) + G (rng);
    const char* names[] = { "Mavka (Auto)", "Torn Tape (Auto)", "Mavka forced 16x" };
    for (int t = 0; t < 3; ++t)
    {
        Engine e; e.prepare (sr, 2);
        float v[P_COUNT]; presetValues (t == 1 ? 7 : 0, v);
        if (t == 2) v[P_OS] = 5.0f;
        e.setParams (v);
        std::vector<std::vector<float>> io { src, src };
        const auto t0 = std::chrono::steady_clock::now();
        runEngine (e, io, sr, 95.0, 256);
        const double secs = std::chrono::duration<double> (std::chrono::steady_clock::now() - t0).count();
        std::printf ("  %-18s running %2dx: %.2f%% of one core (stereo, 48 kHz)\n", names[t], e.getActiveFactor(), 100.0 * secs / 20.0);
    }
    return 0;
}

static int testSwitch()
{
    const double sr = 48000.0;
    const size_t n = (size_t) (sr * 3.0);
    Engine e; e.prepare (sr, 1);
    float v[P_COUNT]; presetValues (0, v); v[P_MOTION] = 0.0f; v[P_INK] = 0.0f; v[P_SLANT] = 0.0f; v[P_LIGATURE] = 0.0f; v[P_MELODY] = 0.0f;
    v[P_OS] = 2.0f;  // 2x
    e.setParams (v);
    std::vector<std::vector<float>> io (1, std::vector<float> (n));
    for (size_t i = 0; i < n; ++i) io[0][i] = 0.2f * (float) std::sin (2 * kPiD * 440 * (double) i / sr);
    runEngine (e, io, sr, 95.0, 256, [&] (size_t pos)
    {
        if (pos >= (size_t) sr && std::fabs (v[P_OS] - 4.0f) > 0.5f) { v[P_OS] = 4.0f; e.setParams (v); }   // -> 8x at t = 1 s
    });
    auto maxD2 = [&] (size_t a, size_t b)
    {
        double m = 0; for (size_t i = a + 2; i < b; ++i) m = std::max (m, (double) std::fabs (io[0][i] - 2 * io[0][i - 1] + io[0][i - 2])); return m;
    };
    const double steady = maxD2 ((size_t) (0.5 * sr), (size_t) (0.95 * sr));
    const double around = maxD2 ((size_t) (0.98 * sr), (size_t) (1.2 * sr));
    std::printf ("  factor switch 2x -> 8x: max 2nd difference steady %.5f, during switch %.5f (now running %dx)\n", steady, around, e.getActiveFactor());
    const bool ok = around < steady * 1.5;
    std::printf (ok ? "SWITCH: OK (no click)\n" : "SWITCH: CHECK\n");
    return ok ? 0 : 1;
}


static int testDuck()
{
    const float fs = 48000.0f;
    const size_t n = (size_t) (fs * 4.0f);
    std::mt19937 rng (11);
    std::normal_distribution<float> G (0.0f, 0.1f);
    std::vector<float> noise (n), key (n), outQuiet (n), outSung (n);
    // FX = an "echo" of the voice in the same band (louder than voice - 6 dB) + broadband noise
    for (size_t i = 0; i < n; ++i)
    {
        key[i] = 0.25f * (float) std::sin (2 * kPiD * 1000.0 * (double) i / fs);
        noise[i] = G (rng) + 0.4f * (float) std::sin (2 * kPiD * 1010.0 * (double) i / fs + 1.0);
    }
    int fails = 0;
    {   // silent voice: the filterbank must give back the exact input
        DuckKey k; DuckFx d; k.prepare (fs); d.prepare (fs);
        double err = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            k.process (0.0f, 0.0f);
            float L = noise[i], R = noise[i];
            d.process (k, L, R, 0.5f, 0.125f);
            err = std::max (err, (double) std::fabs (L - noise[i]));
        }
        std::printf ("  no voice: max deviation from the input %.2e\n", err);
        if (err > 1.0e-4) ++fails;
    }
    {   // a voice at 1 kHz: the FX must step aside around 1 kHz only
        DuckKey k; DuckFx d; k.prepare (fs); d.prepare (fs);
        for (size_t i = 0; i < n; ++i)
        {
            k.process (key[i], 1.0f);
            float L = noise[i], R = noise[i];
            d.process (k, L, R, 0.5f, 0.125f);
            outSung[i] = L;
        }
        auto bandDb = [&] (const std::vector<float>& x, double f0, double f1)
        {
            const size_t N = 65536; std::vector<double> w (N);
            for (size_t i = 0; i < N; ++i) w[i] = x[n - N + i] * (0.5 - 0.5 * std::cos (2 * kPiD * i / (N - 1)));
            std::vector<double> mag; fftMag (w, mag);
            double e = 0; for (size_t b = (size_t) (f0 / fs * N); b < (size_t) (f1 / fs * N); ++b) e += mag[b];
            return 10 * std::log10 (e + 1e-30);
        };
        const double d1k = bandDb (outSung, 850, 1180) - bandDb (noise, 850, 1180);
        const double d200 = bandDb (outSung, 150, 250) - bandDb (noise, 150, 250);
        const double d6k = bandDb (outSung, 5000, 7000) - bandDb (noise, 5000, 7000);
        std::printf ("  voice at 1 kHz, echo at 1 kHz + noise: FX change around 1 kHz %.1f dB, at 200 Hz %.1f dB, at 6 kHz %.1f dB\n", d1k, d200, d6k);
        if (d1k > -8.0 || std::fabs (d200) > 1.5 || std::fabs (d6k) > 1.5) ++fails;
    }
    std::printf (fails ? "DUCK: FAIL\n" : "DUCK: OK\n");
    return fails;
}


static int duckStats (const std::string& path, double bpm)
{
    Audio a;
    if (! readWav (path, a)) return 1;
    if (a.ch.size() == 1) a.ch.push_back (a.ch[0]);
    float v[P_COUNT]; presetValues (0, v); v[P_ECHO] = 0.6f; v[P_FOREST] = 0.4f;
    Engine e; e.prepare (a.sr, 2); e.setParams (v);
    double sum[16] = {}; double minG[16]; for (double& m : minG) m = 1.0; int active = 0, total = 0;
    const int block = 512;
    for (size_t pos = 0; pos + block < a.frames(); pos += block)
    {
        e.setTransport (true, bpm, (double) pos / a.sr * bpm / 60.0);
        float* ptr[2] = { a.ch[0].data() + pos, a.ch[1].data() + pos };
        e.process (ptr, 2, block);
        ++total;
        if (e.getRelativeDb() > -12.0f)   // the voice is singing
        {
            ++active;
            for (int b = 0; b < 16; ++b) { sum[b] += e.getDuckGain (b); minG[b] = std::min (minG[b], (double) e.getDuckGain (b)); }
        }
    }
    std::printf ("blocks with voice: %d of %d\nband  centre   avg cut while singing   deepest cut\n", active, total);
    for (int b = 0; b < 16; ++b)
        std::printf ("%4d %7.0f Hz   %6.1f dB              %6.1f dB\n", b, DuckBands::centreHz (b),
                     20 * std::log10 (sum[b] / std::max (1, active) + 1e-9), 20 * std::log10 (minG[b] + 1e-9));
    return 0;
}


static int testCrush()
{
    const double sr = 48000.0, f0 = 196.0;   // G3
    const size_t N = 65536;
    std::printf ("crusher test: voice-like tone at %.0f Hz, Crush 80%%\n", f0);
    double res[3] = {};
    const char* names[3] = { "Harmonic", "Digital", "off" };
    for (int mode = 0; mode < 3; ++mode)
    {
        Engine e; e.prepare (sr, 1);
        float v[P_COUNT]; linearParams (v);
        v[P_CRUSH] = mode == 2 ? 0.0f : 0.8f; v[P_CRUSHMODE] = (float) std::min (mode, 1); v[P_OS] = 4.0f;
        e.setParams (v);
        const size_t n = N + 48000;
        std::vector<std::vector<float>> io (1, std::vector<float> (n));
        for (size_t i = 0; i < n; ++i)
        {
            double x = 0; for (int h = 1; h <= 8; ++h) x += std::sin (2 * kPiD * f0 * h * (double) i / sr + 0.3 * h) / h;
            io[0][i] = (float) (0.12 * x);
        }
        runEngine (e, io, sr, 95.0);
        std::vector<double> x (N);
        for (size_t i = 0; i < N; ++i)
            x[i] = io[0][i + 48000] * (0.35875 - 0.48829 * std::cos (2 * kPiD * i / (N - 1)) + 0.14128 * std::cos (4 * kPiD * i / (N - 1)) - 0.01168 * std::cos (6 * kPiD * i / (N - 1)));
        std::vector<double> mag; fftMag (x, mag);
        double harm = 0, other = 0;
        const double binHz = sr / (double) N;
        for (size_t k = 30; k < mag.size(); ++k)
        {
            const double f = (double) k * binHz;
            if (f > 20000.0) break;
            const double h = f / f0, dist = std::fabs (h - std::round (h)) * f0 / binHz;
            if (std::round (h) >= 1 && dist <= 8) harm += mag[k]; else other += mag[k];
        }
        res[mode] = 10 * std::log10 (other / harm + 1e-30);
        std::printf ("  %-9s non-harmonic energy %.1f dB below the harmonics\n", names[mode], -res[mode]);
    }
    const bool ok = res[0] < res[1] - 15.0;
    std::printf (ok ? "CRUSH: OK (harmonic crush stays in tune)\n" : "CRUSH: CHECK\n");
    return ok ? 0 : 1;
}


// A steady tone through the engine. Levels (dB re the fundamental) at r * f0 for r = 0.5, 1, 1.5 .. 12.5 (index 2r),
// per channel and for the mono sum (index 2).
struct ToneLevels { double lv[3][26] {}; };

// vibCents > 0: a sung vibrato (5.5 Hz) - the levels are then summed over the band the overtone sweeps
static ToneLevels measureTone (const float* v, int nch, double f0, double amp, double sr = 48000.0, double vibCents = 0.0)
{
    Engine e; e.prepare (sr, nch); e.setParams (v);
    const size_t N = 65536, n = N + 48000;
    std::vector<std::vector<float>> io ((size_t) nch, std::vector<float> (n));
    double ph = 0.0, jit = 0.0, jitT = 0.0;
    std::mt19937 rng (11); std::normal_distribution<double> J (0.0, 1.0);
    for (size_t i = 0; i < n; ++i)
    {
        // a sung voice: vibrato, plus a little period-to-period jitter (0.3%)
        if (vibCents > 0.0 && (i % (size_t) (sr / f0)) == 0) jitT = 0.003 * J (rng);
        jit += (jitT - jit) * 0.02;
        const double f = f0 * std::pow (2.0, vibCents / 1200.0 * std::sin (2 * kPiD * 5.5 * (double) i / sr)) * (1.0 + jit);
        ph += 2 * kPiD * f / sr;
        const float x = (float) (amp * std::sin (ph));
        for (auto& c : io) c[i] = x;
    }
    runEngine (e, io, sr, 95.0);
    ToneLevels t;
    const double binHz = sr / (double) N;
    for (int c = 0; c < 3; ++c)
    {
        if (c == 1 && nch < 2) continue;
        std::vector<double> x (N);
        for (size_t i = 0; i < N; ++i)
        {
            const double s = c == 2 ? 0.5 * (io[0][i + 48000] + io[(size_t) (nch - 1)][i + 48000]) : io[(size_t) c][i + 48000];
            x[i] = s * (0.35875 - 0.48829 * std::cos (2 * kPiD * i / (N - 1)) + 0.14128 * std::cos (4 * kPiD * i / (N - 1))
                        - 0.01168 * std::cos (6 * kPiD * i / (N - 1)));
        }
        std::vector<double> mag; fftMag (x, mag);
        double ref = 0;
        for (int r2 = 1; r2 <= 25; ++r2)
        {
            const double fc = 0.5 * r2 * f0;
            const int b = (int) std::lround (fc / binHz);
            double m = 0;
            if (vibCents > 0.0)
            {   // energy over the swept band
                const int w = 4 + (int) std::ceil (fc * (std::pow (2.0, vibCents / 1200.0) - 1.0) / binHz);
                for (int q = b - w; q <= b + w; ++q) m += mag[(size_t) q];
            }
            else
                for (int q = b - 4; q <= b + 4; ++q) m = std::max (m, mag[(size_t) q]);
            t.lv[c][r2] = 10 * std::log10 (m + 1e-30);   // mag is power
            if (r2 == 2) ref = t.lv[c][r2];
        }
        for (int r2 = 1; r2 <= 25; ++r2) t.lv[c][r2] -= ref;
    }
    return t;
}

static int testMelodic()
{
    // A sung A3, the song in A minor (A C E). Every distortion adds C# (5th overtone: the major third that clashes
    // with minor), G (7th) and B (9th). With Dirt at 0 MAVKA must take those out - across the WHOLE core
    // (crusher, lamp, glass, fold, clip) - and keep / sing the E (3rd, 6th, 12th) and the octaves.
    const double f0 = 220.0;
    const char* names[13] = { "", "A", "A", "E", "A", "C#", "E", "G-", "A", "B", "C#", "D#-", "E" };
    const bool fits[13]  = { false, true, true, true, true, false, true, false, true, false, false, false, true };
    bool allOk = true;
    auto drop = [] (const ToneLevels& p, const ToneLevels& m, int h) { return p.lv[0][2 * h] - m.lv[0][2 * h]; };

    // 1. the clip alone, hard pressed, at 4x (the usual), 1x and 16x: the filters step at different rates;
    //    then a sung vibrato (+-40 cents) at 4x
    for (float osChoice : { 3.0f, 1.0f, 5.0f, 13.0f })
    {
        const double vib = osChoice > 10.0f ? 40.0 : 0.0;
        if (osChoice > 10.0f) osChoice = 3.0f;
        float v[P_COUNT]; linearParams (v);
        v[P_CLIP] = 0.9f; v[P_GRIT] = 0.6f; v[P_SHOULDER] = 1.0f - 0.36f; v[P_DRIVE] = 0.3f;
        v[P_KEY] = 9.0f; v[P_CHORD] = 3.0f; v[P_OS] = osChoice;
        const ToneLevels plain = measureTone (v, 1, f0, 0.3, 48000.0, vib);
        v[P_SING] = 1.2f; v[P_DIRT] = 0.0f;
        const ToneLevels mel = measureTone (v, 1, f0, 0.3, 48000.0, vib);
        if (osChoice > 2.5f && osChoice < 3.5f && vib <= 0.0)
        {
            std::printf ("MELODIC test 1: A3 into a hard clip, song in A minor (A C E)\n  overtone  note  in chord   plain clip   Sing 120%% Dirt 0\n");
            for (int h = 2; h <= 12; ++h)
                std::printf ("  %5d     %-4s  %-8s  %7.1f dB   %7.1f dB\n", h, names[h], fits[h] ? "yes" : "no",
                             plain.lv[0][2 * h], mel.lv[0][2 * h]);
        }
        const double d5 = drop (plain, mel, 5), d7 = drop (plain, mel, 7), d9 = drop (plain, mel, 9), k3 = -drop (plain, mel, 3);
        std::printf ("  %2dx%s: C# %.1f dB lower, G %.1f dB lower, B %.1f dB lower, E %+.1f dB\n", 1 << ((int) osChoice - 1),
                     vib > 0.0 ? " with a 40-cent vibrato" : "", d5, d7, d9, k3);
        allOk = allOk && d5 > (vib > 0.0 ? 15.0 : 30.0) && d7 > (vib > 0.0 ? 12.0 : 30.0) && k3 > 0.5;
    }

    // 2. the whole core, factory presets (Auto oversampling): Dirt 0 against the plain distortion
    std::printf ("MELODIC test 2: the whole core, A3 in A minor, Dirt 0 against the plain distortion\n");
    std::printf ("  preset          C#(5)    G(7)     B(9)     C#(10)   E(3) kept   half-way products C#5 / G5 / B5\n");
    for (const char* name : { "Mavka", "Likho", "Pressure", "Torn Tape", "Glass Harp" })
    {
        int pc = 0; const Preset* pl = factoryPresets (pc);
        int idx = 0; for (int i = 0; i < pc; ++i) if (std::string (pl[i].name) == name) idx = i;
        float v[P_COUNT]; presetValues (idx, v);
        v[P_KEY] = 9.0f; v[P_CHORD] = 3.0f; v[P_PRISM] = 0.0f;
        v[P_ECHO] = 0.0f; v[P_FOREST] = 0.0f; v[P_TWIN] = 0.0f; v[P_LIGATURE] = 0.0f; v[P_MELODY] = 0.0f; v[P_FLANGE] = 0.0f;
        float vp[P_COUNT]; std::copy (v, v + P_COUNT, vp); plainDistortion (vp);
        const ToneLevels plain = measureTone (vp, 1, f0, 0.2);
        v[P_DIRT] = 0.0f; v[P_FORCE] = 0.0f;
        const ToneLevels mel = measureTone (v, 1, f0, 0.2);
        std::printf ("  %-13s %6.1f   %6.1f   %6.1f   %6.1f   %+6.1f      %6.1f %6.1f %6.1f (plain %5.1f %5.1f %5.1f)\n", name,
                     drop (plain, mel, 5), drop (plain, mel, 7), drop (plain, mel, 9), drop (plain, mel, 10), -drop (plain, mel, 3),
                     mel.lv[0][5], mel.lv[0][7], mel.lv[0][9], plain.lv[0][5], plain.lv[0][7], plain.lv[0][9]);
        allOk = allOk && drop (plain, mel, 5) > 12.0;
    }

    // 3. loud AND clean: everything slammed, Dirt 0, Force 0
    {
        float v[P_COUNT]; linearParams (v);
        v[P_DRIVE] = 1.0f; v[P_GRIT] = 1.0f; v[P_SHOULDER] = 0.0f; v[P_CLIP] = 1.0f; v[P_FOLD] = 0.6f; v[P_GLASS] = 0.6f;
        v[P_KEY] = 9.0f; v[P_CHORD] = 3.0f; v[P_OS] = 0.0f;
        const ToneLevels plain = measureTone (v, 1, f0, 0.3);
        v[P_SING] = 1.0f; v[P_DIRT] = 0.0f;
        const ToneLevels mel = measureTone (v, 1, f0, 0.3);
        std::printf ("MELODIC test 3: loud and clean (drive, grit, clip, fold all up): C# %.1f dB lower, G %.1f, B %.1f, E %+.1f\n",
                     drop (plain, mel, 5), drop (plain, mel, 7), drop (plain, mel, 9), -drop (plain, mel, 3));
        allOk = allOk && drop (plain, mel, 5) > 12.0;
    }

    // 4. the overtone map: one overtone dragged up, another one down
    {
        float v[P_COUNT]; linearParams (v);
        v[P_CLIP] = 0.9f; v[P_GRIT] = 0.6f; v[P_SHOULDER] = 0.64f; v[P_DRIVE] = 0.3f; v[P_KEY] = 9.0f; v[P_CHORD] = 3.0f; v[P_OS] = 3.0f;
        const ToneLevels plain = measureTone (v, 1, f0, 0.3);
        v[P_OT5] = 12.0f; v[P_OT3] = -30.0f;
        const ToneLevels map = measureTone (v, 1, f0, 0.3);
        const double up5 = -drop (plain, map, 5), down3 = drop (plain, map, 3), d7 = drop (plain, map, 7);
        std::printf ("MELODIC test 4: overtone map  5th +12 dB -> %+.1f dB, 3rd -30 dB -> %+.1f dB, untouched 7th %+.1f dB\n", up5, -down3, -d7);
        allOk = allOk && up5 > 10.0 && down3 > 24.0 && std::fabs (d7) < 1.5;
    }

    // 5. Prism: the chord tones refract between the channels, the mono sum stays
    {
        float v[P_COUNT]; linearParams (v);
        v[P_CLIP] = 0.9f; v[P_GRIT] = 0.6f; v[P_SHOULDER] = 0.64f; v[P_DRIVE] = 0.3f; v[P_KEY] = 9.0f; v[P_CHORD] = 3.0f; v[P_OS] = 3.0f;
        v[P_SING] = 1.6f; v[P_DIRT] = 1.0f;
        const ToneLevels flat = measureTone (v, 2, f0, 0.3);
        v[P_PRISM] = 1.0f;
        const ToneLevels pr = measureTone (v, 2, f0, 0.3);
        double worstMono = 0, bestSpread = 0;
        for (int h : { 2, 3, 4, 6, 8, 12 })
        {
            worstMono = std::max (worstMono, std::fabs (pr.lv[2][2 * h] - flat.lv[2][2 * h]));
            bestSpread = std::max (bestSpread, std::fabs (pr.lv[0][2 * h] - pr.lv[1][2 * h]));
        }
        std::printf ("MELODIC test 5: Prism 100%%: chord tones up to %.1f dB apart between L and R, mono sum within %.2f dB\n",
                     bestSpread, worstMono);
        allOk = allOk && bestSpread > 6.0 && worstMono < 1.5;
    }

    std::printf (allOk ? "MELODIC: OK (the whole core only adds notes of the chord; dirt is a choice)\n" : "MELODIC: CHECK\n");
    return allOk ? 0 : 1;
}

// diagnostics: a steady A3 through a preset (+ --set overrides), levels at every half overtone
static int toneSpectrum (int preset, const std::vector<std::string>& sets)
{
    float v[P_COUNT]; presetValues (preset, v);
    v[P_KEY] = 9.0f; v[P_CHORD] = 3.0f;
    for (const auto& kv : sets)
    {
        const size_t eq = kv.find ('=');
        const int p = eq == std::string::npos ? -1 : findParam (kv.substr (0, eq).c_str());
        if (p >= 0) v[p] = (float) std::atof (kv.substr (eq + 1).c_str());
    }
    const ToneLevels t = measureTone (v, 1, 220.0, 0.2);
    const char* nm[26] = { "", "A2", "A3", "E4", "A4", "C#5", "E5", "G5", "A5", "B5", "C#6", "D#6", "E6", "F#6", "G6-",
                           "G#6", "A6", "A#6", "B6", "C7", "C#7", "D7", "D#7-", "D#7", "E7", "F7" };
    for (int r2 = 1; r2 <= 25; ++r2)
        std::printf ("  %4.1f x f0  %-5s %7.1f dB\n", 0.5 * r2, nm[r2], t.lv[0][r2]);
    return 0;
}

static int testReverbSafe()
{
    // words of an A3 (a sung "aa" with an attack), Dirt 100%: Reverb safe must keep the dirt to the attacks
    // and send no rumble / narrower lows on
    const double sr = 48000.0, f0 = 220.0;
    const size_t word = (size_t) (0.9 * sr), gap = (size_t) (0.35 * sr), words = 6, n = words * (word + gap);
    std::vector<float> src (n, 0.0f);
    std::mt19937 rng (3); std::normal_distribution<float> G (0.0f, 1.0f);
    for (size_t w = 0; w < words; ++w)
        for (size_t i = 0; i < word; ++i)
        {
            const double t = (double) i / sr, env = std::min (1.0, t / 0.01) * std::min (1.0, (double) (word - i) / (0.02 * sr));
            // a voice without a 5th overtone of its own: every C# in the output is what the distortion added
            double x = 0; for (int h : { 1, 2, 3, 4, 6 }) x += std::sin (2 * kPiD * f0 * h * t + 0.4 * h) / (h * h);
            src[w * (word + gap) + i] = (float) (0.25 * env * x + 0.002 * G (rng));
        }
    for (size_t i = 0; i < n; ++i)   // and some rumble under it all (the reverb must not get it)
        src[i] += (float) (0.01 * std::sin (2 * kPiD * 50.0 * (double) i / sr));
    double lowDb[2] = {}, sideDb[2] = {}, dirtSus[2] = {}, dirtAtk[2] = {};
    for (int pass = 0; pass < 2; ++pass)
    {
        Engine e; e.prepare (sr, 2);
        float v[P_COUNT]; defaultValues (v);
        v[P_KEY] = 9.0f; v[P_CHORD] = 3.0f; v[P_MIX] = 1.0f; v[P_DIRT] = 1.0f; v[P_FORCE] = 0.0f; v[P_DRIVE] = 0.6f;
        v[P_GHOSTDOWN] = 0.4f; v[P_REVSAFE] = pass == 1 ? 1.0f : 0.0f; v[P_ECHO] = 0.0f; v[P_FOREST] = 0.0f;
        v[P_MELODY] = 0.0f; v[P_FLANGE] = 0.0f; v[P_TWIN] = 0.0f;   // the core alone (the strings have overtones of their own)
        e.setParams (v);
        std::vector<std::vector<float>> io (2, src);
        runEngine (e, io, sr, 95.0);
        const size_t lat = (size_t) e.getLatency();
        double side = 0, mid = 0;
        std::complex<double> hum = 0, fund = 0;
        for (size_t i = lat + (size_t) sr; i < n; ++i)
        {
            const double m = 0.5 * (io[0][i] + io[1][i]), sd = 0.5 * (io[0][i] - io[1][i]);
            side += sd * sd; mid += m * m;
            hum += m * std::exp (std::complex<double> (0, -2 * kPiD * 50.0 * (double) i / sr));
            fund += m * std::exp (std::complex<double> (0, -2 * kPiD * f0 * (double) i / sr));
        }
        lowDb[pass] = 20 * std::log10 (std::abs (hum) / std::abs (fund));
        sideDb[pass] = 10 * std::log10 (side / mid);
        // C# (5th overtone) in the attack (first 120 ms) and in the body (400 .. 740 ms) of the last 4 words
        auto csharp = [&] (size_t start, size_t len)
        {
            std::vector<double> x (8192, 0.0);
            for (size_t i = 0; i < len && i < 8192; ++i)
                x[i] = 0.5 * (io[0][start + i] + io[1][start + i]) * (0.5 - 0.5 * std::cos (2 * kPiD * i / (len - 1)));
            std::vector<double> mag; fftMag (x, mag);
            const double binHz = sr / 8192.0;
            auto pk = [&] (double f) { const int b = (int) std::lround (f / binHz); double m = 0; for (int q = b - 2; q <= b + 2; ++q) m = std::max (m, mag[(size_t) q]); return m; };
            return 10 * std::log10 (pk (5 * f0) / pk (f0));
        };
        for (size_t w = 2; w < words; ++w)
        {
            const size_t s0 = w * (word + gap) + lat;
            dirtAtk[pass] += csharp (s0, (size_t) (0.12 * sr)) / 4.0;
            dirtSus[pass] += csharp (s0 + (size_t) (0.40 * sr), (size_t) (0.34 * sr)) / 4.0;
        }
    }
    std::printf ("REVERB SAFE: 50 Hz rumble %.1f -> %.1f dB re the note, side %.1f -> %.1f dB re mid\n", lowDb[0], lowDb[1], sideDb[0], sideDb[1]);
    std::printf ("  dirt (C#) in the attack %.1f -> %.1f dB, in the body of the word %.1f -> %.1f dB re the note\n",
                 dirtAtk[0], dirtAtk[1], dirtSus[0], dirtSus[1]);
    const bool ok = lowDb[1] < lowDb[0] - 8.0 && sideDb[1] < sideDb[0] - 2.0 && dirtSus[1] < dirtSus[0] - 10.0
                    && dirtAtk[1] > dirtAtk[0] - 6.0;
    std::printf (ok ? "REVERBSAFE: OK (dirt only on the attacks, clean lows and sides)\n" : "REVERBSAFE: CHECK\n");
    return ok ? 0 : 1;
}

static int testTuning()
{
    // a beat tuned to A = 447 Hz (+27.3 cents): with Detune set to it, everything that plays in key moves along
    const double sr = 48000.0, cents = a4ToDetune (447.0f), k = std::pow (2.0, cents / 1200.0);
    bool ok = true;

    // 1. the Tuned flanger rings on the beat's C3 harmonics, not on the 440 Hz ones: where are its comb peaks?
    auto peakCents = [&] (float detune)
    {
        Engine e; e.prepare (sr, 1);
        float v[P_COUNT]; linearParams (v);
        v[P_FLANGE] = 1.0f; v[P_FLANGEMODE] = 0.0f; v[P_FLANGEFB] = 0.85f; v[P_KEY] = 0.0f; v[P_CHORD] = 3.0f;
        v[P_REGISTER] = 1.0f; v[P_OS] = 2.0f; v[P_DETUNE] = detune;
        e.setParams (v);
        const size_t n = (size_t) (sr * 1.2);
        std::mt19937 rng (5); std::normal_distribution<float> G (0.0f, 0.05f);
        std::vector<std::vector<float>> io (1, std::vector<float> (n));
        for (auto& x : io[0]) x = G (rng);
        runEngine (e, io, sr, 120.0);   // beat 1 (0 .. 0.5 s at 120 bpm) rings on the root
        const size_t N = 32768, start = (size_t) (0.02 * sr) + (size_t) e.getLatency();
        std::vector<double> w (N, 0.0);
        const size_t len = (size_t) (0.45 * sr);
        for (size_t i = 0; i < len; ++i) w[i] = io[0][start + i] * (0.5 - 0.5 * std::cos (2 * kPiD * i / (len - 1)));
        std::vector<double> mag; fftMag (w, mag);
        double acc = 0; int cnt = 0;
        for (int h = 6; h <= 14; ++h)   // where each comb peak sits, against the 440 Hz C3 harmonic (power-weighted)
        {
            const double f = 130.8128 * h;
            const int lo = (int) (f * 0.97 / sr * N), hi = (int) (f * 1.03 / sr * N);
            double sw = 0, sf = 0;
            for (int q = lo; q <= hi; ++q) { const double m = mag[(size_t) q] * mag[(size_t) q]; sw += m; sf += m * q; }
            acc += 1200.0 * std::log2 ((sf / sw) * sr / N / f); ++cnt;
        }
        return acc / cnt;
    };
    const double at440 = peakCents (0.0f), at447 = peakCents ((float) cents);
    std::printf ("TUNING: Detune %+.1f cents (A = 447 Hz): the Tuned flanger's peaks move from %+.1f to %+.1f cents\n",
                 cents, at440, at447);
    ok = ok && std::fabs ((at447 - at440) - cents) < 8.0;

    // 2. a sample tuned 45 cents sharp: with Detune left at 440 the melodic distortion mistakes its chord tones for
    //    dirt; with Detune set to the sample, it hears them right again
    const double sharp = 45.0, ks = std::pow (2.0, sharp / 1200.0);
    double drop[2] = {}, keepE[2] = {};
    for (int pass = 0; pass < 2; ++pass)
    {
        float v[P_COUNT]; linearParams (v);
        v[P_CLIP] = 0.9f; v[P_GRIT] = 0.6f; v[P_SHOULDER] = 0.64f; v[P_DRIVE] = 0.3f; v[P_KEY] = 9.0f; v[P_CHORD] = 3.0f; v[P_OS] = 3.0f;
        v[P_DETUNE] = pass == 1 ? (float) sharp : 0.0f;
        const ToneLevels plain = measureTone (v, 1, 220.0 * ks, 0.3);
        v[P_SING] = 1.2f; v[P_DIRT] = 0.0f;
        const ToneLevels mel = measureTone (v, 1, 220.0 * ks, 0.3);
        drop[pass] = plain.lv[0][10] - mel.lv[0][10];
        keepE[pass] = mel.lv[0][6] - plain.lv[0][6];
    }
    std::printf ("  A3 sung 45 cents sharp: Detune at 440 -> E (a chord tone) %+.1f dB, C# %.1f dB lower; "
                 "Detune +45 -> E %+.1f dB, C# %.1f dB lower\n", keepE[0], drop[0], keepE[1], drop[1]);
    ok = ok && keepE[0] < -6.0 && keepE[1] > 0.5 && drop[1] > 30.0;
    std::printf (ok ? "TUNING: OK (harp, flanger and the melodic distortion follow the beat's tuning)\n" : "TUNING: CHECK\n");
    return ok ? 0 : 1;
}

static int testFlange()
{
    // Tuned mode: white noise through the flanger must ring on the chord tone of the current beat
    const double sr = 48000.0, bpm = 120.0;
    Engine e; e.prepare (sr, 1);
    float v[P_COUNT]; linearParams (v);
    v[P_FLANGE] = 1.0f; v[P_FLANGEMODE] = 0.0f; v[P_FLANGEFB] = 0.85f; v[P_KEY] = 0.0f; v[P_CHORD] = 3.0f;   // C minor
    v[P_REGISTER] = 1.0f; v[P_OS] = 2.0f;
    e.setParams (v);
    const size_t n = (size_t) (sr * 2.0);
    std::mt19937 rng (5); std::normal_distribution<float> G (0.0f, 0.05f);
    std::vector<std::vector<float>> io (1, std::vector<float> (n));
    for (auto& x : io[0]) x = G (rng);
    runEngine (e, io, sr, bpm);
    auto contrast = [&] (size_t start, double f0)
    {
        const size_t N = 16384; std::vector<double> w (N);
        for (size_t i = 0; i < N; ++i) w[i] = io[0][start + i] * (0.5 - 0.5 * std::cos (2 * kPiD * i / (N - 1)));
        std::vector<double> mag; fftMag (w, mag);
        double on = 0, off = 0;
        for (int h = 1; h <= 8; ++h)
        {
            const double f = f0 * h, fo = f0 * (h + 0.5);
            const size_t b = (size_t) std::lrint (f / sr * N), bo = (size_t) std::lrint (fo / sr * N);
            double m1 = 0, m2 = 0;
            for (int d = -2; d <= 2; ++d) { m1 = std::max (m1, mag[b + (size_t) d]); m2 = std::max (m2, mag[bo + (size_t) d]); }
            on += m1; off += m2;
        }
        return 10 * std::log10 (on / off);
    };
    // the delay line output is aligned to the plugin latency; beat 1 = root C3, beat 2 = fifth G3 (arp 0,2,1,3)
    const size_t lat = (size_t) e.getLatency();
    const double cRoot = contrast ((size_t) (0.08 * sr) + lat, 130.81), cFifthOnBeat2 = contrast ((size_t) (0.58 * sr) + lat, 196.0);
    const double cWrong = contrast ((size_t) (0.08 * sr) + lat, 130.81 * std::pow (2.0, 0.5 / 12.0));   // a quarter tone off
    std::printf ("  tuned flanger, beat 1: peaks on C3 harmonics +%.1f dB over the gaps\n", cRoot);
    std::printf ("  tuned flanger, beat 2: peaks on G3 harmonics +%.1f dB\n", cFifthOnBeat2);
    std::printf ("  control, beat 1, a quarter tone off the chord tone: %+.1f dB\n", cWrong);
    const bool ok = cRoot > 8.0 && cFifthOnBeat2 > 8.0 && cRoot - cWrong > 6.0;
    std::printf (ok ? "FLANGE: OK (the flanger plays the chord)\n" : "FLANGE: CHECK\n");
    return ok ? 0 : 1;
}

//==============================================================================
// THE DOUBLES: fine controls. Renders aligned to the plugin's latency (output i = what came in at i).
static std::vector<std::vector<float>> renderAligned (const float* v, const std::vector<std::vector<float>>& in, double sr, double bpm,
                                                      const std::function<void (Engine&, size_t)>& perBlock = {})
{
    Engine e; e.prepare (sr, (int) in.size()); e.setParams (v);
    std::vector<std::vector<float>> io = in;
    const size_t lat0 = (size_t) e.getLatency();
    for (auto& c : io) c.resize (c.size() + lat0, 0.0f);
    runEngine (e, io, sr, bpm, 512, [&] (size_t pos) { if (perBlock) perBlock (e, pos); });
    const size_t lat = (size_t) e.getLatency();
    std::vector<std::vector<float>> out (in.size(), std::vector<float> (in[0].size(), 0.0f));
    for (size_t c = 0; c < in.size(); ++c)
        for (size_t i = 0; i < in[0].size() && i + lat < io[c].size(); ++i)
            out[c][i] = io[c][i + lat];
    return out;
}

static void doublesParams (float* v, bool glass)
{
    linearParams (v);
    v[P_LAMP] = 0.0f; v[P_GLASS] = glass ? 1.0f : 0.0f; v[P_FOLD] = glass ? 0.0f : 1.0f;
    v[P_ENSEMBLE] = 0.0f; v[P_PULSE] = 0.0f; v[P_ORBIT] = 0.0f; v[P_VIBRATO] = 0.0f; v[P_WIDTH] = 0.0f;
    v[P_AUTOGAIN] = 0.0f;
}

// where the burst comes out (ms after it went in): the peak of the cross-correlation with the burst
static std::vector<double> burstEchoes (const std::vector<float>& y, const std::vector<float>& burst, size_t at, double sr, int count, double minGapMs)
{
    std::vector<double> corr (y.size(), 0.0);
    for (size_t i = at; i + burst.size() < y.size(); ++i)
    {
        double a = 0.0;
        for (size_t k = 0; k < burst.size(); ++k) a += (double) y[i + k] * burst[k];
        corr[i] = std::fabs (a);
    }
    std::vector<double> res;   // pairs: delay ms, peak
    std::vector<bool> taken (y.size(), false);
    for (int c = 0; c < count; ++c)
    {
        size_t best = 0; double bv = 0.0;
        for (size_t i = at; i < corr.size(); ++i)
            if (! taken[i] && corr[i] > bv) { bv = corr[i]; best = i; }
        if (bv <= 0.0) break;
        res.push_back ((double) (best - at) / sr * 1000.0);
        res.push_back (bv);
        const size_t g = (size_t) (minGapMs * 0.001 * sr);
        for (size_t i = best > g ? best - g : 0; i < std::min (corr.size(), best + g); ++i) taken[i] = true;
    }
    return res;
}

static double bandPower (const std::vector<float>& x, size_t start, double sr, double f1, double f2)
{
    const size_t N = 65536;
    std::vector<double> w (N);
    for (size_t i = 0; i < N; ++i) w[i] = (start + i < x.size() ? x[start + i] : 0.0f) * (0.5 - 0.5 * std::cos (2 * kPiD * i / (N - 1)));
    std::vector<double> mag; fftMag (w, mag);
    double p = 0.0;
    for (size_t k = (size_t) (f1 / sr * N); k <= (size_t) (f2 / sr * N) && k < mag.size(); ++k) p += mag[k];
    return p + 1.0e-30;
}

static double pitchSpanCents (const std::vector<float>& y, size_t from, size_t to, double sr, double f0)
{
    // instantaneous pitch from zero crossings, averaged over 8 periods
    std::vector<double> zc;
    for (size_t i = from + 1; i < to; ++i)
        if (y[i - 1] < 0.0f && y[i] >= 0.0f)
            zc.push_back ((double) (i - 1) + (double) (-y[i - 1]) / (double) (y[i] - y[i - 1]));
    double lo = 1e9, hi = -1e9;
    for (size_t k = 8; k < zc.size(); ++k)
    {
        const double f = 8.0 * sr / (zc[k] - zc[k - 8]);
        const double c = 1200.0 * std::log2 (f / f0);
        lo = std::min (lo, c); hi = std::max (hi, c);
    }
    return hi - lo;
}

static int testDoubles()
{
    const double sr = 48000.0, bpm = 120.0;
    bool ok = true;
    // a short 1 kHz burst, alone in the silence
    std::vector<float> burst ((size_t) (0.004 * sr));
    for (size_t i = 0; i < burst.size(); ++i)
        burst[i] = (float) (0.25 * std::sin (2 * kPiD * 1000.0 * i / sr) * (0.5 - 0.5 * std::cos (2 * kPiD * i / (burst.size() - 1))));
    const size_t at = (size_t) (0.25 * sr);
    std::vector<std::vector<float>> in (1, std::vector<float> ((size_t) (1.2 * sr), 0.0f));
    std::copy (burst.begin(), burst.end(), in[0].begin() + (long) at);

    auto timeOf = [&] (bool glass, float timeMs, float sync, float repeats, int count)
    {
        float v[P_COUNT]; doublesParams (v, glass);
        v[glass ? P_GTIME : P_FTIME] = timeMs; v[glass ? P_GSYNC : P_FSYNC] = sync; v[glass ? P_GREPEATS : P_FREPEATS] = repeats;
        auto out = renderAligned (v, in, sr, bpm);
        return burstEchoes (out[0], burst, at, sr, count, 20.0);
    };
    {
        const auto g = timeOf (true, 80.0f, 0.0f, 0.0f, 1), f = timeOf (false, 50.0f, 0.0f, 0.0f, 1);
        const auto gs = timeOf (true, 23.0f, 4.0f, 0.0f, 1);   // 1/16 at 120 bpm = 125 ms
        std::printf ("  Glass Time 80 ms: the double comes %.2f ms after the voice\n", g[0]);
        std::printf ("  Fold Time 50 ms: %.2f ms\n", f[0]);
        std::printf ("  Glass Sync 1/16 at 120 bpm (125 ms): %.2f ms\n", gs[0]);
        ok = ok && std::fabs (g[0] - 80.0) < 1.0 && std::fabs (f[0] - 50.0) < 1.0 && std::fabs (gs[0] - 125.0) < 1.0;
    }
    {
        const auto r = timeOf (true, 60.0f, 0.0f, 0.6f, 3);
        std::vector<std::pair<double, double>> e;
        for (size_t i = 0; i + 1 < r.size(); i += 2) e.push_back ({ r[i], r[i + 1] });
        std::sort (e.begin(), e.end());
        std::printf ("  Glass Repeats 60%%, 60 ms: repeats at");
        for (auto& p : e) std::printf (" %.1f", p.first);
        std::printf (" ms, each %.2f / %.2f of the one before\n", e.size() > 1 ? e[1].second / e[0].second : 0.0,
                     e.size() > 2 ? e[2].second / e[1].second : 0.0);
        ok = ok && e.size() == 3 && std::fabs (e[1].first - 120.0) < 1.5 && std::fabs (e[2].first - 180.0) < 2.0
                && e[1].second / e[0].second > 0.45 && e[1].second / e[0].second < 0.75
                && e[2].second / e[1].second > 0.45 && e[2].second / e[1].second < 0.75;
    }
    {
        // ATTACK / OPEN: syllables with hard onsets; the double's first 25 ms against its sustain
        std::vector<std::vector<float>> syl (1, std::vector<float> ((size_t) (2.0 * sr), 0.0f));
        for (int s = 0; s < 3; ++s)
            for (size_t i = 0; i < (size_t) (0.25 * sr); ++i)
            {
                const double t = (double) i / sr;
                double x = 0.0;
                for (int h = 1; h <= 8; ++h) x += std::sin (2 * kPiD * 220.0 * h * t) / h;
                syl[0][(size_t) ((0.2 + 0.55 * s) * sr) + i] = (float) (0.2 * x * std::min (1.0, t / 0.002) * std::min (1.0, (0.25 - t) / 0.01));
            }
        auto ratio = [&] (float attack)
        {
            float v[P_COUNT]; doublesParams (v, true);
            v[P_GTIME] = 30.0f; v[P_GATTACK] = attack; v[P_GOPEN] = 120.0f;
            auto out = renderAligned (v, syl, sr, bpm);
            double early = 0.0, late = 0.0;
            for (int s = 1; s < 3; ++s)
            {
                const size_t on = (size_t) ((0.2 + 0.55 * s + 0.030) * sr);
                for (size_t i = 0; i < (size_t) (0.025 * sr); ++i) early += out[0][on + i] * out[0][on + i];
                for (size_t i = (size_t) (0.18 * sr); i < (size_t) (0.205 * sr); ++i) late += out[0][on + i] * out[0][on + i];
            }
            return 10.0 * std::log10 ((early + 1e-30) / (late + 1e-30));
        };
        const double r0 = ratio (0.0f), rSkip = ratio (-1.0f), rHit = ratio (1.0f);
        std::printf ("  Glass Attack: first 25 ms vs the sustain: %+.1f dB normal, %+.1f dB at -100%% (skips the attack), "
                     "%+.1f dB at +100%% (grabs it)\n", r0, rSkip, rHit);
        ok = ok && rSkip < r0 - 8.0 && rHit > r0 + 4.0;
    }
    {
        // TONE: noise through the double, the top against the bottom
        std::vector<std::vector<float>> nz (1, std::vector<float> ((size_t) (2.0 * sr)));
        std::mt19937 rng (11); std::normal_distribution<float> G (0.0f, 0.05f);
        for (auto& x : nz[0]) x = G (rng);
        auto tilt = [&] (float tone)
        {
            float v[P_COUNT]; doublesParams (v, true);
            v[P_GTONE] = tone;
            auto out = renderAligned (v, nz, sr, bpm);
            return 10.0 * std::log10 (bandPower (out[0], (size_t) (0.4 * sr), sr, 4000.0, 8000.0)
                                      / bandPower (out[0], (size_t) (0.4 * sr), sr, 200.0, 400.0));
        };
        const double t0 = tilt (0.0f), tDark = tilt (-1.0f), tBright = tilt (1.0f);
        std::printf ("  Glass Tone: 4-8 kHz vs 200-400 Hz: %+.1f dB flat, %+.1f dB dark, %+.1f dB bright\n", t0, tDark, tBright);
        ok = ok && tDark < t0 - 10.0 && tBright > t0 + 10.0;
    }
    {
        // FLOAT: a steady tone floats around its pitch
        std::vector<std::vector<float>> sn (1, std::vector<float> ((size_t) (2.5 * sr)));
        for (size_t i = 0; i < sn[0].size(); ++i) sn[0][i] = (float) (0.2 * std::sin (2 * kPiD * 440.0 * i / sr));
        auto span = [&] (float cents)
        {
            float v[P_COUNT]; doublesParams (v, true);
            v[P_GFLOAT] = cents; v[P_GRATE] = 1.0f;
            auto out = renderAligned (v, sn, sr, bpm);
            return pitchSpanCents (out[0], (size_t) (0.5 * sr), (size_t) (2.4 * sr), sr, 440.0);
        };
        const double s0 = span (0.0f), s30 = span (30.0f);
        std::printf ("  Glass Float 30 ct at 1 Hz: the pitch swings %.1f ct peak to peak (%.1f ct without float)\n", s30, s0);
        ok = ok && s0 < 6.0 && s30 > 40.0 && s30 < 80.0;
    }
    {
        // DRIVE: the double's own drive
        std::vector<std::vector<float>> sn (1, std::vector<float> ((size_t) (1.5 * sr)));
        for (size_t i = 0; i < sn[0].size(); ++i) sn[0][i] = (float) (0.3 * std::sin (2 * kPiD * 220.0 * i / sr));
        auto thd = [&] (float db)
        {
            float v[P_COUNT]; doublesParams (v, true);
            v[P_DRIVE] = 0.15f; v[P_GDRIVE] = db;
            auto out = renderAligned (v, sn, sr, bpm);
            const double f = bandPower (out[0], (size_t) (0.3 * sr), sr, 200.0, 240.0);
            return 10.0 * std::log10 ((bandPower (out[0], (size_t) (0.3 * sr), sr, 400.0, 2400.0) + 1e-30) / f);
        };
        const double d0 = thd (0.0f), d12 = thd (12.0f), dm12 = thd (-12.0f), d6 = thd (6.0f), dm6 = thd (-6.0f);
        std::printf ("  Glass Drive: harmonics vs fundamental %+.1f dB at -12 dB, %+.1f at -6, %+.1f dB at 0, %+.1f at +6, %+.1f dB at +12 dB\n", dm12, dm6, d0, d6, d12);
        ok = ok && d12 > d0 + 8.0 && dm12 < d0 - 8.0;
    }
    {
        // FOLD THROW: Fold gets its own doppler
        std::vector<std::vector<float>> sn (1, std::vector<float> ((size_t) (1.5 * sr)));
        for (size_t i = 0; i < sn[0].size(); ++i)
            sn[0][i] = (float) (0.3 * std::sin (2 * kPiD * 220.0 * i / sr) * (0.6 + 0.4 * std::sin (2 * kPiD * 3.0 * i / sr)));
        float v[P_COUNT]; doublesParams (v, false); v[P_DRIVE] = 0.3f;
        const auto a = renderAligned (v, sn, sr, bpm);
        v[P_FTHROW] = 1.0f;
        const auto b = renderAligned (v, sn, sr, bpm);
        double d = 0.0, p = 0.0;
        for (size_t i = (size_t) (0.3 * sr); i < a[0].size(); ++i) { d += (a[0][i] - b[0][i]) * (a[0][i] - b[0][i]); p += a[0][i] * a[0][i]; }
        const double rel = 10.0 * std::log10 ((d + 1e-30) / (p + 1e-30));
        bool finite = true;
        for (float x : b[0]) finite = finite && std::isfinite (x);
        std::printf ("  Fold Throw 100%%: the fold double moves (difference %+.1f dB re the double)\n", rel);
        ok = ok && finite && rel > -30.0;
    }
    std::printf (ok ? "DOUBLES: OK (time, sync, repeats, attack, tone, float, drive, throw)\n" : "DOUBLES: CHECK\n");
    return ok ? 0 : 1;
}

//==============================================================================
// IMAGER: the width of the wet sound in four bands
static int testImager()
{
    const double sr = 48000.0, bpm = 120.0;
    bool ok = true;
    const size_t n = (size_t) (3.0 * sr), at = (size_t) (1.0 * sr);
    std::mt19937 rng (21); std::normal_distribution<float> G (0.0f, 0.05f);
    std::vector<float> n1 (n), n2 (n);
    for (auto& x : n1) x = G (rng);
    for (auto& x : n2) x = G (rng);
    std::vector<std::vector<float>> wide (2, std::vector<float> (n)), mono (2, n1), apart (2);
    for (size_t i = 0; i < n; ++i) { wide[0][i] = n1[i]; wide[1][i] = 0.6f * n1[i] + 0.8f * n2[i]; }
    apart[0] = n1; apart[1] = n2;
    auto ms = [] (const std::vector<std::vector<float>>& y, std::vector<float>& m, std::vector<float>& s)
    {
        m.resize (y[0].size()); s.resize (y[0].size());
        for (size_t i = 0; i < m.size(); ++i) { m[i] = 0.5f * (y[0][i] + y[1][i]); s[i] = 0.5f * (y[0][i] - y[1][i]); }
    };
    auto db = [] (double a, double b) { return 10.0 * std::log10 (a / b); };
    const double bands[4][2] = { { 30.0, 70.0 }, { 380.0, 560.0 }, { 2200.0, 3200.0 }, { 10000.0, 14000.0 } };

    float base[P_COUNT]; linearParams (base); base[P_AUTOGAIN] = 0.0f;
    {
        // widths -100 / +50 / -50 / +100 %: the side follows, the mid (the mono sum) does not move
        const auto off = renderAligned (base, wide, sr, bpm);
        float v[P_COUNT]; std::copy (base, base + P_COUNT, v);
        v[P_IW1] = -1.0f; v[P_IW2] = 0.5f; v[P_IW3] = -0.5f; v[P_IW4] = 1.0f;
        const auto on = renderAligned (v, wide, sr, bpm);
        std::vector<float> m0, s0, m1, s1; ms (off, m0, s0); ms (on, m1, s1);
        // what the ideal (analog) Linkwitz-Riley tree of the same settings gives, averaged over the band like the noise is
        const double xs[3] = { 180.0, 1200.0, 5500.0 }, gs[4] = { 0.0, 1.5, 0.5, 2.0 };
        auto ideal = [&] (double f1, double f2)
        {
            using Cx = std::complex<double>;
            auto bw = [] (double f, double fc, bool hp) { const Cx q (0.0, f / fc); const Cx d = q * q + std::sqrt (2.0) * q + 1.0; return hp ? q * q / d : 1.0 / d; };
            auto lr = [&] (double f, double fc, bool hp) { const Cx b = bw (f, fc, hp); return b * b; };
            double acc = 0.0;
            for (int i = 0; i < 64; ++i)
            {
                const double f = f1 + (f2 - f1) * (i + 0.5) / 64.0;
                const Cx b1 = lr (f, xs[0], false), r1 = lr (f, xs[0], true);
                const Cx b2 = r1 * lr (f, xs[1], false), r2 = r1 * lr (f, xs[1], true);
                const Cx b3 = r2 * lr (f, xs[2], false), b4 = r2 * lr (f, xs[2], true);
                const Cx ap1 = lr (f, xs[1], false) + lr (f, xs[1], true), ap2 = lr (f, xs[2], false) + lr (f, xs[2], true);
                acc += std::norm ((ap1 * gs[0] * b1 + gs[1] * b2) * ap2 + gs[2] * b3 + gs[3] * b4);
            }
            return 10.0 * std::log10 (acc / 64.0 + 1e-30);
        };
        std::printf ("  widths -100 / +50 / -50 / +100 %%:\n");
        for (int b = 0; b < 4; ++b)
        {
            const double dm = db (bandPower (m1, at, sr, bands[b][0], bands[b][1]), bandPower (m0, at, sr, bands[b][0], bands[b][1]));
            const double ds = db (bandPower (s1, at, sr, bands[b][0], bands[b][1]), bandPower (s0, at, sr, bands[b][0], bands[b][1]));
            const double want = ideal (bands[b][0], bands[b][1]);
            std::printf ("    %5.0f-%5.0f Hz: side %+6.1f dB (ideal Linkwitz-Riley %+6.1f), mono sum %+.2f dB\n", bands[b][0], bands[b][1],
                         ds, want, dm);
            ok = ok && std::fabs (dm) < 0.25 && (b == 0 ? ds < -25.0 : std::fabs (ds - want) < 0.75);
        }
    }
    {
        // SPREAD on a mono source: width appears in its band only, the mono sum stays
        const auto off = renderAligned (base, mono, sr, bpm);
        float v[P_COUNT]; std::copy (base, base + P_COUNT, v);
        v[P_IS4] = 1.0f;
        const auto on = renderAligned (v, mono, sr, bpm);
        std::vector<float> m0, s0, m1, s1; ms (off, m0, s0); ms (on, m1, s1);
        const double sHigh = db (bandPower (s1, at, sr, 10000.0, 14000.0), bandPower (m1, at, sr, 10000.0, 14000.0));
        const double sLow = db (bandPower (s1, at, sr, 30.0, 70.0), bandPower (m1, at, sr, 30.0, 70.0));
        double worstM = 0.0;
        for (auto& b : bands)
            worstM = std::max (worstM, std::fabs (db (bandPower (m1, at, sr, b[0], b[1]), bandPower (m0, at, sr, b[0], b[1]))));
        std::printf ("  Spread 100%% (high band) on a mono voice: side %+.1f dB re mid up high, %+.1f dB in the lows; "
                     "mono sum moves %.2f dB at most\n", sHigh, sLow, worstM);
        ok = ok && sHigh > -5.0 && sHigh < -1.5 && sLow < -30.0 && worstM < 0.25;
    }
    {
        // RECOVER: a band narrowed to mono gets its side back in the middle
        const auto off = renderAligned (base, apart, sr, bpm);
        float v[P_COUNT]; std::copy (base, base + P_COUNT, v);
        v[P_IW2] = -1.0f;
        const auto narrow = renderAligned (v, apart, sr, bpm);
        v[P_IRECOVER] = 1.0f;
        const auto rec = renderAligned (v, apart, sr, bpm);
        std::vector<float> m0, s0, m1, s1, m2, s2; ms (off, m0, s0); ms (narrow, m1, s1); ms (rec, m2, s2);
        const double side = db (bandPower (s2, at, sr, 380.0, 560.0), bandPower (s0, at, sr, 380.0, 560.0));
        const double body = db (bandPower (m2, at, sr, 380.0, 560.0), bandPower (m0, at, sr, 380.0, 560.0));
        const double bodyN = db (bandPower (m1, at, sr, 380.0, 560.0), bandPower (m0, at, sr, 380.0, 560.0));
        const double top = db (bandPower (m2, at, sr, 10000.0, 14000.0), bandPower (m0, at, sr, 10000.0, 14000.0));
        std::printf ("  Recover 100%%, low mids at mono: side %+.1f dB, mid %+.2f dB (%+.2f dB without Recover), up high %+.2f dB\n",
                     side, body, bodyN, top);
        ok = ok && side < -20.0 && body > 1.0 && body < 2.5 && std::fabs (bodyN) < 0.25 && std::fabs (top) < 0.25;
    }
    {
        // the dry gets the same all-pass: at 50 % mix the imager must not comb the dry against the wet
        float v0[P_COUNT]; std::copy (base, base + P_COUNT, v0); v0[P_MIX] = 0.5f;
        const auto off = renderAligned (v0, wide, sr, bpm);
        float v[P_COUNT]; std::copy (v0, v0 + P_COUNT, v); v[P_IW4] = 0.001f;
        const auto on = renderAligned (v, wide, sr, bpm);
        double worst = 0.0;
        const double around[3][2] = { { 150.0, 210.0 }, { 1000.0, 1400.0 }, { 4700.0, 6300.0 } };
        for (auto& b : around)
            for (int c = 0; c < 2; ++c)
                worst = std::max (worst, std::fabs (db (bandPower (on[(size_t) c], at, sr, b[0], b[1]), bandPower (off[(size_t) c], at, sr, b[0], b[1]))));
        std::printf ("  50%% mix, imager engaged at neutral: the crossovers move the sound by %.2f dB at most (no comb)\n", worst);
        ok = ok && worst < 0.3;
    }
    {
        // back to neutral: after its second of grace and the fade, the imager is gone without a trace
        const auto off = renderAligned (base, wide, sr, bpm);
        float v[P_COUNT]; std::copy (base, base + P_COUNT, v);
        v[P_IW1] = -1.0f; v[P_IW4] = 0.8f; v[P_IS3] = 0.5f; v[P_IRECOVER] = 0.7f;
        const auto back = renderAligned (v, wide, sr, bpm, [&] (Engine& e, size_t pos)
        {
            if (pos >= (size_t) (0.6 * sr) && pos < (size_t) (0.6 * sr) + 512) { e.setParams (base); }
        });
        double d = 0.0;
        for (size_t c = 0; c < 2; ++c)
            for (size_t i = (size_t) (2.0 * sr); i < n; ++i)
                d = std::max (d, (double) std::fabs (back[c][i] - off[c][i]));
        // a mono track: nothing to widen, nothing changes
        std::vector<std::vector<float>> one (1, n1);
        const auto mOff = renderAligned (base, one, sr, bpm);
        const auto mOn = renderAligned (v, one, sr, bpm);
        double dm = 0.0;
        for (size_t i = 0; i < n; ++i) dm = std::max (dm, (double) std::fabs (mOn[0][i] - mOff[0][i]));
        std::printf ("  back to neutral: %.3g max difference after 1.4 s; a mono track: %.3g\n", d, dm);
        ok = ok && d == 0.0 && dm == 0.0;
    }
    std::printf (ok ? "IMAGER: OK (widths, spread, recover, the mono sum never moves)\n" : "IMAGER: CHECK\n");
    return ok ? 0 : 1;
}

//==============================================================================
// LIGATURE: the ping-pong echo between the words, finely
static double shortBand (const std::vector<float>& x, size_t start, size_t N, double sr, double f1, double f2)
{
    std::vector<double> w (N);
    for (size_t i = 0; i < N; ++i) w[i] = (start + i < x.size() ? x[start + i] : 0.0f) * (0.5 - 0.5 * std::cos (2 * kPiD * i / (N - 1)));
    std::vector<double> mag; fftMag (w, mag);
    double p = 0.0;
    for (size_t k = (size_t) (f1 / sr * N); k <= (size_t) (f2 / sr * N) && k < mag.size(); ++k) p += mag[k];
    return p + 1.0e-30;
}

static int testLigature()
{
    const double sr = 48000.0, bpm = 120.0;
    bool ok = true;
    auto ligParams = [] (float* v) { linearParams (v); v[P_LIGATURE] = 0.5f; v[P_AUTOGAIN] = 0.0f; };
    auto db = [] (double a, double b) { return 10.0 * std::log10 (a / b); };

    // a short 1 kHz burst alone in the silence: where the left and the right echo come back
    std::vector<float> burst ((size_t) (0.004 * sr));
    for (size_t i = 0; i < burst.size(); ++i)
        burst[i] = (float) (0.25 * std::sin (2 * kPiD * 1000.0 * i / sr) * (0.5 - 0.5 * std::cos (2 * kPiD * i / (burst.size() - 1))));
    const size_t at = (size_t) (0.2 * sr);
    std::vector<std::vector<float>> in (2, std::vector<float> ((size_t) (1.6 * sr), 0.0f));
    for (auto& c : in) std::copy (burst.begin(), burst.end(), c.begin() + (long) at);
    auto firstEcho = [&] (const std::vector<float>& y)
    {
        const size_t from = at + (size_t) (0.08 * sr);
        const auto e = burstEchoes (y, burst, from, sr, 1, 20.0);
        return e.empty() ? -1.0 : e[0] + 80.0;
    };
    {
        float v[P_COUNT]; ligParams (v);
        auto a = renderAligned (v, in, sr, bpm);
        v[P_LLEFT] = 0.0f; v[P_LLEFTMS] = 180.0f; v[P_LRIGHT] = 0.0f; v[P_LRIGHTMS] = 270.0f;
        auto b = renderAligned (v, in, sr, bpm);
        v[P_LLEFT] = 9.0f; v[P_LRIGHT] = 3.0f;   // 1/4 and 1/16
        auto c = renderAligned (v, in, sr, bpm);
        const double aL = firstEcho (a[0]), aR = firstEcho (a[1]), bL = firstEcho (b[0]), bR = firstEcho (b[1]);
        const double cL = firstEcho (c[0]), cR = firstEcho (c[1]);
        std::printf ("  Time: default %.1f / %.1f ms (1/8 / dotted 1/8 at 120 bpm = 250 / 375), Free 180 / 270 ms: %.1f / %.1f, "
                     "1/4 / 1/16: %.1f / %.1f ms\n", aL, aR, bL, bR, cL, cR);
        ok = ok && std::fabs (aL - 250.0) < 1.5 && std::fabs (aR - 375.0) < 1.5 && std::fabs (bL - 180.0) < 1.5
                && std::fabs (bR - 270.0) < 1.5 && std::fabs (cL - 500.0) < 1.5 && std::fabs (cR - 125.0) < 1.5;
    }
    {
        // REPEATS: the second left echo (it came back through the right side) against the first
        auto ratio = [&] (float rep)
        {
            float v[P_COUNT]; ligParams (v); v[P_DUCK] = 2.0f; v[P_LREPEATS] = rep;
            auto y = renderAligned (v, in, sr, bpm);
            const auto e = burstEchoes (y[0], burst, at + (size_t) (0.08 * sr), sr, 2, 60.0);
            if (e.size() < 4) return 0.0;
            return e[0] < e[2] ? e[3] / e[1] : e[1] / e[3];
        };
        const double r0 = ratio (0.0f), rUp = ratio (0.8f), rDn = ratio (-0.8f);
        std::printf ("  Repeats: each repeat %.2f of the one before at 0 (the Ligature knob's own), %.2f at +80%%, %.2f at -80%%\n",
                     r0, rUp, rDn);
        ok = ok && r0 > 0.3 && r0 < 0.5 && rUp > 0.7 && rUp < 0.95 && rDn < 0.12;
    }
    {
        // TONE and DRIVE of the repeats (the first echo is the voice itself; the second has been through the loop)
        std::vector<std::vector<float>> nz (2, std::vector<float> ((size_t) (1.6 * sr), 0.0f));
        std::mt19937 rng (7); std::normal_distribution<float> G (0.0f, 0.1f);
        for (size_t i = 0; i < (size_t) (0.03 * sr); ++i) { const float x = G (rng); nz[0][at + i] = x; nz[1][at + i] = x; }
        auto tilt = [&] (float tone)
        {
            float v[P_COUNT]; ligParams (v); v[P_DUCK] = 2.0f; v[P_LTONE] = tone;
            auto y = renderAligned (v, nz, sr, bpm);
            const size_t e1 = at + (size_t) (0.245 * sr), e2 = at + (size_t) (0.620 * sr);
            return db (shortBand (y[0], e2, 4096, sr, 4000.0, 8000.0), shortBand (y[0], e2, 4096, sr, 200.0, 400.0))
                 - db (shortBand (y[0], e1, 4096, sr, 4000.0, 8000.0), shortBand (y[0], e1, 4096, sr, 200.0, 400.0));
        };
        const double t0 = tilt (0.0f), tD = tilt (-1.0f), tB = tilt (1.0f);
        std::printf ("  Tone: a repeat's top vs its bottom, against the first echo: %+.1f dB neutral, %+.1f dark, %+.1f bright\n", t0, tD, tB);
        ok = ok && tD < t0 - 6.0 && tB > t0 + 6.0;

        std::vector<std::vector<float>> sn (2, std::vector<float> ((size_t) (1.6 * sr), 0.0f));
        for (size_t i = 0; i < (size_t) (0.06 * sr); ++i)
        {
            const float x = (float) (0.5 * std::sin (2 * kPiD * 1000.0 * i / sr) * std::min (1.0, std::min (i, (size_t) (0.06 * sr) - i) / (0.005 * sr)));
            sn[0][at + i] = x; sn[1][at + i] = x;
        }
        auto thd = [&] (float d)
        {
            float v[P_COUNT]; ligParams (v); v[P_DUCK] = 2.0f; v[P_LDRIVE] = d;
            auto y = renderAligned (v, sn, sr, bpm);
            const size_t e2 = at + (size_t) (0.625 * sr);
            return db (shortBand (y[0], e2, 2048, sr, 2500.0, 6500.0), shortBand (y[0], e2, 2048, sr, 900.0, 1100.0));
        };
        const double d0 = thd (0.0f), dHi = thd (18.0f), dLo = thd (-12.0f);
        std::printf ("  Drive: harmonics of a repeat %+.1f dB at -12 dB, %+.1f at 0, %+.1f at +18 dB\n", dLo, d0, dHi);
        ok = ok && dHi > d0 + 6.0 && dLo < d0 + 0.5;   // below 0 dB the repeats are as clean as the voice itself
    }
    {
        // FLOAT: the echo of a held note floats around its pitch
        std::vector<std::vector<float>> tn (2, std::vector<float> ((size_t) (1.2 * sr), 0.0f));
        for (size_t i = 0; i < (size_t) (0.5 * sr); ++i) { const float x = (float) (0.2 * std::sin (2 * kPiD * 440.0 * i / sr)); tn[0][i] = x; tn[1][i] = x; }
        auto span = [&] (float cents)
        {
            float v[P_COUNT]; ligParams (v); v[P_DUCK] = 2.0f; v[P_LFLOAT] = cents; v[P_LRATE] = 8.0f;
            auto y = renderAligned (v, tn, sr, bpm);
            return pitchSpanCents (y[0], (size_t) (0.56 * sr), (size_t) (0.74 * sr), sr, 440.0);
        };
        const double s0 = span (0.0f), s30 = span (30.0f);
        std::printf ("  Float 30 ct at 8 Hz: the echo's pitch swings %.1f ct (%.1f ct without)\n", s30, s0);
        ok = ok && s0 < 5.0 && s30 > 35.0 && s30 < 80.0;
    }
    {
        // ATTACK: what of each word goes into the echo (the duck off, to see only the door)
        std::vector<std::vector<float>> w (2, std::vector<float> ((size_t) (2.6 * sr), 0.0f));
        for (int k = 0; k < 3; ++k)
            for (size_t i = 0; i < (size_t) (0.2 * sr); ++i)
            {
                const double t = (double) i / sr;
                double x = 0.0;
                for (int h = 1; h <= 8; ++h) x += std::sin (2 * kPiD * 220.0 * h * t) / h;
                const float y = (float) (0.2 * x * std::min (1.0, t / 0.002) * std::min (1.0, (0.2 - t) / 0.01));
                w[0][(size_t) ((0.2 + 0.8 * k) * sr) + i] = y; w[1][(size_t) ((0.2 + 0.8 * k) * sr) + i] = y;
            }
        auto early = [&] (float att)
        {
            float v[P_COUNT]; ligParams (v); v[P_DUCK] = 2.0f; v[P_LATTACK] = att; v[P_LOPEN] = 80.0f;
            auto y = renderAligned (v, w, sr, bpm);
            double e = 0.0, l = 0.0;
            for (int k = 1; k < 3; ++k)
            {
                const size_t on = (size_t) ((0.2 + 0.8 * k + 0.25) * sr);
                for (size_t i = 0; i < (size_t) (0.025 * sr); ++i) e += y[0][on + i] * y[0][on + i];
                for (size_t i = (size_t) (0.15 * sr); i < (size_t) (0.175 * sr); ++i) l += y[0][on + i] * y[0][on + i];
            }
            return 10.0 * std::log10 ((e + 1e-30) / (l + 1e-30));
        };
        const double a0 = early (0.0f), aS = early (-1.0f), aG = early (1.0f);
        std::printf ("  Attack: the echo's first 25 ms vs its body: %+.1f dB natural, %+.1f Skip 100%%, %+.1f Grab 100%%\n", a0, aS, aG);
        ok = ok && aS < a0 - 8.0 && aG > a0 + 4.0;
    }
    {
        // DUCK / SOFT / RETURN (classic duck): one note, then another; the echo of the first plays under the second
        std::vector<std::vector<float>> two (2, std::vector<float> ((size_t) (2.4 * sr), 0.0f));
        for (size_t i = 0; i < (size_t) (2.0 * sr); ++i)
        {
            const double t = (double) i / sr, f = t < 1.0 ? 300.0 : 700.0;
            const float x = (float) (0.2 * std::sin (2 * kPiD * f * t));
            two[0][i] = x; two[1][i] = x;
        }
        auto under = [&] (float duckDb, float soft, float ret, double from, double f1, double f2)
        {
            float v[P_COUNT]; ligParams (v); v[P_DUCK] = 1.0f; v[P_LDUCK] = duckDb; v[P_LSOFT] = soft; v[P_LRETURN] = ret;
            auto y = renderAligned (v, two, sr, bpm);
            return 10.0 * std::log10 (shortBand (y[0], (size_t) (from * sr), 4096, sr, f1, f2));
        };
        // the echo of the 300 Hz note under the 700 Hz one (1.06 - 1.14 s)
        const double u0 = under (0.0f, 4.0f, 140.0f, 1.06, 270.0, 330.0), uShallow = under (-24.0f, 4.0f, 140.0f, 1.06, 270.0, 330.0),
                     uDeep = under (12.0f, 4.0f, 140.0f, 1.06, 270.0, 330.0);
        std::printf ("  Duck: the echo under the voice %+.1f dB at 0, %+.1f at -24 dB (barely ducks), %+.1f at +12 dB (deeper)\n",
                     u0 - u0, uShallow - u0, uDeep - u0);
        ok = ok && uShallow > u0 + 12.0 && uDeep < u0 - 6.0;
        // RETURN: after the voice stops (2.0 s), how fast the echo of its last note comes back
        const double rFast = under (0.0f, 4.0f, 20.0f, 2.12, 670.0, 730.0), rSlow = under (0.0f, 4.0f, 1500.0f, 2.12, 670.0, 730.0);
        // SOFT: a word, a pause, and the next word comes in while the echo (a quarter note) of the first still plays;
        // a soft duck lets the echo fade under the new word instead of vanishing at once
        std::vector<std::vector<float>> gap (2, std::vector<float> ((size_t) (1.6 * sr), 0.0f));
        for (size_t i = 0; i < (size_t) (1.2 * sr); ++i)
        {
            const double t = (double) i / sr;
            if (t >= 0.4 && t < 0.6) continue;
            const float x = (float) (0.2 * std::sin (2 * kPiD * (t < 0.4 ? 300.0 : 700.0) * t));
            gap[0][i] = x; gap[1][i] = x;
        }
        auto echoAtOnset = [&] (float soft)
        {
            float v[P_COUNT]; ligParams (v); v[P_DUCK] = 1.0f; v[P_LSOFT] = soft; v[P_LLEFT] = 9.0f;   // 1/4: the echo plays 0.5 - 0.9 s
            auto y = renderAligned (v, gap, sr, bpm);
            return 10.0 * std::log10 (shortBand (y[0], (size_t) (0.62 * sr), 4096, sr, 270.0, 330.0));
        };
        const double sHard = echoAtOnset (4.0f), sSoft = echoAtOnset (200.0f);
        std::printf ("  Soft 200 ms: the echo stays %+.1f dB louder as the voice comes in; Return 20 ms: the echo is back %+.1f dB "
                     "sooner than at 1500 ms\n", sSoft - sHard, rFast - rSlow);
        ok = ok && sSoft > sHard + 6.0 && rFast > rSlow + 8.0;
    }
    {
        // the IMAGER aimed at the Ligature: the echo goes mono, the doubles keep their width
        std::vector<std::vector<float>> w (2, std::vector<float> ((size_t) (3.0 * sr), 0.0f));
        for (int k = 0; k < 3; ++k)
            for (size_t i = 0; i < (size_t) (0.3 * sr); ++i)
            {
                const double t = (double) i / sr;
                double x = 0.0;
                for (int h = 1; h <= 10; ++h) x += std::sin (2 * kPiD * 196.0 * h * t) / h;
                const float y = (float) (0.15 * x * std::min (1.0, t / 0.005) * std::min (1.0, (0.3 - t) / 0.02));
                w[0][(size_t) ((0.2 + 0.9 * k) * sr) + i] = y; w[1][(size_t) ((0.2 + 0.9 * k) * sr) + i] = y;
            }
        float v[P_COUNT]; ligParams (v); v[P_LIGATURE] = 0.6f; v[P_GLASS] = 0.8f; v[P_WIDTH] = 0.8f; v[P_DUCK] = 2.0f;
        const auto off = renderAligned (v, w, sr, bpm);
        for (int b = 0; b < 4; ++b) v[P_LIW1 + b] = -1.0f;
        const auto lig = renderAligned (v, w, sr, bpm);
        for (int b = 0; b < 4; ++b) { v[P_LIW1 + b] = 0.0f; v[P_IW1 + b] = -1.0f; }
        const auto all = renderAligned (v, w, sr, bpm);
        auto sideIn = [&] (const std::vector<std::vector<float>>& y, double t0, double t1)
        {
            double s = 0.0;
            for (size_t i = (size_t) (t0 * sr); i < (size_t) (t1 * sr); ++i) { const double d = 0.5 * (y[0][i] - y[1][i]); s += d * d; }
            return s + 1e-30;
        };
        // the first word before any echo (0.25 - 0.42 s: the voice and its doubles) and a pause (1.55 - 1.95 s: only the echo)
        const double wordLig = db (sideIn (lig, 0.25, 0.42), sideIn (off, 0.25, 0.42)), pauseLig = db (sideIn (lig, 1.55, 1.95), sideIn (off, 1.55, 1.95));
        const double wordAll = db (sideIn (all, 0.25, 0.42), sideIn (off, 0.25, 0.42));
        const double pauseAll = db (sideIn (all, 1.55, 1.95), sideIn (off, 1.55, 1.95));
        std::printf ("  The Ligature's imager, every band at mono: the echo's side %+.1f dB, the word's side (the doubles) %+.2f dB;\n"
                     "  the doubles' imager at mono: the word's side %+.1f dB, the echo's side %+.1f dB (it narrows only what echoes "
                     "the doubles)\n", pauseLig, wordLig, wordAll, pauseAll);
        ok = ok && pauseLig < -30.0 && std::fabs (wordLig) < 0.5 && wordAll < -30.0;
    }
    std::printf (ok ? "LIGATURE: OK (times, repeats, tone, drive, float, attack, duck, soft, return, its own width)\n" : "LIGATURE: CHECK\n");
    return ok ? 0 : 1;
}

//==============================================================================
static int findPreset (const std::string& s)
{
    int count = 0; const Preset* p = factoryPresets (count);
    for (int i = 0; i < count; ++i) if (s == p[i].name) return i;
    return std::atoi (s.c_str());
}

static int render (const std::string& inPath, const std::string& outPath, int preset, double bpm, const std::vector<std::string>& sets)
{
    Audio a;
    if (! readWav (inPath, a)) { std::fprintf (stderr, "cannot read %s\n", inPath.c_str()); return 1; }
    if (a.ch.size() == 1) a.ch.push_back (a.ch[0]);
    float v[P_COUNT]; presetValues (preset, v);
    for (auto& s : sets)
    {
        const auto eq = s.find ('=');
        const int p = findParam (s.substr (0, eq).c_str());
        if (p >= 0) v[p] = (float) std::atof (s.substr (eq + 1).c_str());
    }
    Engine e; e.prepare (a.sr, 2); e.setNonRealtime (true); e.setParams (v);
    const int lat = e.getLatency();
    for (auto& c : a.ch) c.insert (c.end(), (size_t) lat + (size_t) a.sr * 2, 0.0f);   // room for the tails
    runEngine (e, a.ch, a.sr, bpm, 512);
    for (auto& c : a.ch) c.erase (c.begin(), c.begin() + lat);                       // latency compensation
    writeWav (outPath, a);
    float peak = 0; for (auto& c : a.ch) for (float x : c) peak = std::max (peak, std::fabs (x));
    int count = 0; const Preset* pl = factoryPresets (count);
    std::printf ("rendered %-14s -> %s (peak %.2f dBFS, ran at %dx)\n", pl[preset].name, outPath.c_str(), 20 * std::log10 (peak + 1e-9), e.getActiveFactor());
    return 0;
}

// POLISH check on a real vocal: at several Mix / Drive points - how much of what MAVKA adds is dirt (non-chord
// overtones), how dense the result is against the dry voice, how wide, how much top
static int polish (const std::string& inPath, int preset, double bpm, const std::vector<std::string>& sets)
{
    Audio a;
    if (! readWav (inPath, a)) { std::fprintf (stderr, "cannot read %s\n", inPath.c_str()); return 1; }
    if (a.ch.size() == 1) a.ch.push_back (a.ch[0]);
    const size_t n = a.frames();
    std::vector<char> sungMask;   // the frames where the DRY voice sings (the same frames for every render)
    auto stats = [&] (const std::vector<std::vector<float>>& io, double& crest, double& dynStd, double& sideDb, double& hfDb)
    {
        double peak = 0, sum = 0, side = 0, mid = 0, hf = 0;
        Biquad hp; hp.highpass (7000.0, 0.707, a.sr);
        std::vector<double> frames;
        const size_t fl = (size_t) (0.03 * a.sr);
        double fe = 0; size_t fc = 0;
        for (size_t i = 0; i < n; ++i)
        {
            const double m = 0.5 * (io[0][i] + io[1][i]), sd = 0.5 * (io[0][i] - io[1][i]);
            peak = std::max (peak, std::fabs (m)); sum += m * m; mid += m * m; side += sd * sd;
            const double h = hp.process ((float) m); hf += h * h;
            fe += m * m;
            if (++fc == fl) { frames.push_back (10 * std::log10 (fe / (double) fl + 1e-12)); fe = 0; fc = 0; }
        }
        crest = 20 * std::log10 (peak / std::sqrt (sum / (double) n));
        sideDb = 10 * std::log10 (side / mid + 1e-12);
        hfDb = 10 * std::log10 (hf / mid + 1e-12);
        if (sungMask.empty())
        {
            double top = -200; for (double f : frames) top = std::max (top, f);
            for (double f : frames) sungMask.push_back (f > top - 25.0 ? 1 : 0);
        }
        std::vector<double> sung;
        for (size_t i = 0; i < frames.size() && i < sungMask.size(); ++i) if (sungMask[i]) sung.push_back (frames[i]);
        std::sort (sung.begin(), sung.end());
        // how far the quiet sung moments sit under the loud ones (10th .. 90th percentile)
        dynStd = sung.size() > 10 ? sung[sung.size() * 9 / 10] - sung[sung.size() / 10] : 0.0;
    };
    double dC, dD, dS, dH;
    stats (a.ch, dC, dD, dS, dH);
    std::printf ("dry voice: crest %.1f dB, syllable spread %.2f dB, side %.1f dB, top (>7k) %.1f dB\n", dC, dD, dS, dH);
    std::printf ("  mix  drive | dirt share  chord tones | crest   spread   top     side (re mid)\n");
    const float mixes[] = { 0.15f, 0.25f, 0.45f, 1.0f, 0.25f, 0.25f, 0.25f };
    const float drives[] = { -1.0f, -1.0f, -1.0f, -1.0f, 0.10f, 0.60f, -1.0f };
    for (int pass = 0; pass < 7; ++pass)
    {
        const bool plain = pass == 6;
        const int q = pass;
        float v[P_COUNT]; presetValues (preset, v);
        for (auto& st : sets)
        {
            const auto eq = st.find ('=');
            const int p = findParam (st.substr (0, eq).c_str());
            if (p >= 0) v[p] = (float) std::atof (st.substr (eq + 1).c_str());
        }
        v[P_MIX] = mixes[q];
        if (drives[q] >= 0.0f) v[P_DRIVE] = drives[q];
        if (plain) plainDistortion (v);
        Engine e; e.prepare (a.sr, 2); e.setNonRealtime (true); e.setParams (v); e.setMetering (true);
        auto io = a.ch;
        for (auto& c : io) c.insert (c.end(), (size_t) e.getLatency(), 0.0f);
        double dirtAcc = 0, melAcc = 0; int blocks = 0;
        runEngine (e, io, a.sr, bpm, 512, [&] (size_t)
        {
            Engine::OvertoneView ov; e.getOvertones (ov);
            if (ov.conf < 0.6f) return;
            double tot = 0, dirt = 0, mel = 0;
            for (int h = 0; h < InnerControls::kBands; ++h)
            {
                if (h >= InnerControls::kHarm && ! ov.halves) break;
                const double pw = std::pow (10.0, 0.1 * ov.level[h]) * ov.gain[h] * ov.gain[h];
                tot += pw; dirt += pw * (1.0 - ov.fit[h]); mel += pw * ov.fit[h];
            }
            if (tot <= 0) return;
            dirtAcc += dirt / tot; melAcc += 10 * std::log10 (mel + 1e-12); ++blocks;
        });
        for (auto& c : io) c.erase (c.begin(), c.begin() + e.getLatency());
        double c, d, sd, h;
        stats (io, c, d, sd, h);
        std::printf ("  %3.0f%%  %3.0f%%  | %6.1f%%    %7.1f dB  | %+5.1f   %+5.2f   %+5.1f   %6.1f dB  %s\n", 100 * v[P_MIX], 100 * v[P_DRIVE],
                     blocks ? 100.0 * dirtAcc / blocks : 0.0, blocks ? melAcc / blocks : -120.0, c - dC, d - dD, h - dH, sd,
                     plain ? "(plain distortion)" : "");
    }
    return 0;
}

int main (int argc, char** argv)
{
    noDenormals();
    if (argc < 2) { std::printf ("usage: see header\n"); return 1; }
    const std::string cmd = argv[1];
    if (cmd == "list") { int c = 0; const Preset* p = factoryPresets (c); for (int i = 0; i < c; ++i) std::printf ("%d %s\n", i, p[i].name); return 0; }
    if (cmd == "latency") return testLatency();
    if (cmd == "alias")   return testAlias();
    if (cmd == "stress")  return testStress();
    if (cmd == "cpu")     return testCpu();
    if (cmd == "switch")  return testSwitch();
    if (cmd == "duck")    return testDuck();
    if (cmd == "crush")   return testCrush();
    if (cmd == "flange")  return testFlange();
    if (cmd == "melodic" || cmd == "clipkey") return testMelodic();
    if (cmd == "reverbsafe") return testReverbSafe();
    if (cmd == "tuning") return testTuning();
    if (cmd == "doubles") return testDoubles();
    if (cmd == "imager") return testImager();
    if (cmd == "ligature") return testLigature();
    double bpm = 95.0; std::vector<std::string> sets;
    std::vector<std::string> pos;
    for (int i = 2; i < argc; ++i)
    {
        const std::string s = argv[i];
        if (s == "--bpm" && i + 1 < argc) bpm = std::atof (argv[++i]);
        else if (s == "--set" && i + 1 < argc) sets.push_back (argv[++i]);
        else pos.push_back (s);
    }
    if (cmd == "duckstats" && pos.size() >= 1) return duckStats (pos[0], bpm);
    if (cmd == "tone" && pos.size() >= 1) return toneSpectrum (findPreset (pos[0]), sets);
    if (cmd == "polish" && pos.size() >= 2) return polish (pos[0], findPreset (pos[1]), bpm, sets);
    if (cmd == "render" && pos.size() >= 3) return render (pos[0], pos[1], findPreset (pos[2]), bpm, sets);
    if (cmd == "renderall" && pos.size() >= 2)
    {
        int c = 0; const Preset* p = factoryPresets (c);
        for (int i = 0; i < c; ++i)
        {
            std::string name = p[i].name; for (auto& ch : name) if (ch == ' ') ch = '_';
            char buf[512]; std::snprintf (buf, sizeof (buf), "%s/%02d_%s.wav", pos[1].c_str(), i + 1, name.c_str());
            render (pos[0], buf, i, bpm, sets);
        }
        return 0;
    }
    std::printf ("unknown command\n");
    return 1;
}
