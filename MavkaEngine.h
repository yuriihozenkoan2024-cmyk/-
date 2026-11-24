// MAVKA - vocal "sauce" distortion engine (no JUCE dependency)
//
// Signal flow (per channel):
//   in -> pre EQ -> consistency (level-independent drive) -> warp (phase) -> ghosts (sub/fifth + octave)
//      -> [oversampled core: Lamp | Doppler Glass | Fold voices -> clip]  -> DC -> smooth (anti-harsh)
//      -> restore dynamics * vowel focus * ink -> + Harp (strings in key) -> tone -> mix with delayed dry
//
// Handwriting ("Ink"): every word starts clean and the distortion spreads over it (Ink > 0),
// or hits hard and lifts off (Press, Ink < 0).
// Chord clock: all modulators run at rates that form a chord (4:5:6, 10:12:15, ...) locked to the host bars.
#pragma once

#include "MavkaAura.h"
#include "MavkaDSP.h"
#include "MavkaParams.h"

#include <atomic>
#include <cstring>

namespace mavka
{

inline float smoothCoef (float ms, float fs) noexcept { return 1.0f - std::exp (-1.0f / (0.001f * ms * fs)); }
inline bool  differs (float a, float b) noexcept { return std::memcmp (&a, &b, sizeof (float)) != 0; }

//==============================================================================
struct InnerControls
{
    float dLamp, dGlass, dDop, dFold;       // ms
    float kLamp, cLamp, tcLamp;             // lamp drive, bias offset, tanh(bias)
    float kGlass, dopSens;                  // glass drive, doppler modulator sensitivity
    float kFold, sFold, ssFold;             // fold drive, symmetry offset, sin(symmetry)
    float gLamp, gGlass, gFold;             // voice gains (level comp, pan, pulse)
    float clipIn, clipH, clipOut;           // final clipper
    float crushRate, crushStep, crushMix;   // crusher: hold rate (Hz), quantiser step, blend
    float crushSync;                        // 1 = a new period of the voice starts here: restart the step grid
    float clipPos;                          // positive ceiling (lower on vowels = asymmetric, warm)
    // THE DOUBLES, finely: each double hears its own line - the voice plus its own repeats, through its Tone
    float fbG, fbF;                         // Repeats: loop gain (< 1)
    float toneG, toneF;                     // Tone: -1 dark .. 0 untouched .. +1 bright (what the double hears)
    float dDopF;                            // Fold's own Doppler throw (ms)
    // MELODIC DISTORTION: everything the core adds (crusher, the three machines, the clip) is measured against
    // the clean voice and split by the voice's overtones; each overtone the distortion adds is then kept, raised
    // or taken back (Sing for the notes of the song's chord, Dirt for the rest, plus the overtone map)
    static constexpr int kHarm = 11;        // overtones 2 .. 12 (the overtone map)
    static constexpr int kHalf = 7;         // half overtones 1.5 .. 7.5: what the sub-octave ghost adds through the core
    static constexpr int kBands = kHarm + kHalf;
    float hOn;                              // 0 = identity: the overtone filters are idle
    float hHalf;                            // 1 = the half-overtone filters run too (Ghost Down is on)
    float hF0;                              // the voice's pitch (Hz) the overtone filters are tuned to
    // per band, how much of what the core added there is taken back (in phase / in quadrature): the bands overlap,
    // so these come from the wanted per-overtone amounts through the leakage compensation (inkey::Comp)
    float hW[kBands], hQ[kBands];
};

namespace inkey
{
    constexpr int kHarm = InnerControls::kHarm, kBands = InnerControls::kBands;
    static const float kMult[kBands]  = { 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
                                          1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f };
    static const float kSemis[kBands] = { 12.0f, 19.01955f, 24.0f, 27.86314f, 31.01955f, 33.68826f, 36.0f,
                                          38.03910f, 39.86314f, 41.51318f, 43.01955f,
                                          7.01955f, 15.86314f, 21.68826f, 26.03910f, 29.51318f, 32.40528f, 34.88269f };
    // SVF damping per band: an overtone band is a quarter of the voice's pitch wide, a half-overtone band an eighth
    // (they sit between the overtones)
    static const float kR[kBands] = { 0.125f / 2.0f, 0.125f / 3.0f, 0.125f / 4.0f, 0.125f / 5.0f, 0.125f / 6.0f,
                                      0.125f / 7.0f, 0.125f / 8.0f, 0.125f / 9.0f, 0.125f / 10.0f, 0.125f / 11.0f,
                                      0.125f / 12.0f,
                                      0.0625f / 1.5f, 0.0625f / 2.5f, 0.0625f / 3.5f, 0.0625f / 4.5f, 0.0625f / 5.5f,
                                      0.0625f / 6.5f, 0.0625f / 7.5f };
    inline bool isOctave (int h) noexcept { return h == 0 || h == 2 || h == 6; }   // 2, 4, 8: the sung note itself
    // POLISH bloom: with the knobs turned down the lowest chord tones (warmth: 2nd, 3rd, 4th, the sub's fifth)
    // come forward and the top ones (fizz) step back
    static const float kBloom[kBands] = { 1.0f, 0.8f, 0.6f, 0.35f, 0.3f, 0.1f, 0.0f, -0.25f, -0.3f, -0.35f, -0.4f,
                                          0.8f, 0.4f, 0.2f, 0.0f, -0.2f, -0.3f, -0.35f };
    constexpr float kTopHz = 9000.0f;       // overtones above this are left alone

    // Leakage compensation. Every band also passes a little of its neighbours, so taking back 100% of three
    // clashing overtones in a row would take back too much of each (the skirts add up). The bands all scale with
    // the voice's pitch, so how much band k passes at overtone j is a constant: solve once for the weights
    // (in phase on the band-pass, in quadrature on the low-pass) that give exactly the wanted amount AT every
    // overtone. m[i][k]: realisation weight i (0..n-1 in phase, n..2n-1 quadrature) per wanted weight k.
    struct Comp
    {
        int n = 0;
        float m[2 * kBands][kBands] {};

        void design (int numBands) noexcept
        {
            n = numBands;
            const int N = 2 * n;
            double a[2 * kBands][2 * kBands + kBands] {};
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < n; ++k)
                {
                    // band k (centre kMult[k] f0, damping kR[k]) at overtone j
                    const double x = (double) kMult[j] / (double) kMult[k], r2 = 2.0 * (double) kR[k];
                    const double dr = 1.0 - x * x, di = r2 * x, den = dr * dr + di * di;
                    const double bRe = (r2 * x * di) / den, bIm = (r2 * x * dr) / den;   // 2R jx / (1 - x^2 + 2R jx)
                    const double qRe = (r2 * dr) / den, qIm = (-r2 * di) / den;          // 2R / (1 - x^2 + 2R jx)
                    a[j][k] = bRe;     a[j][n + k] = qRe;
                    a[n + j][k] = bIm; a[n + j][n + k] = qIm;
                }
            for (int j = 0; j < N; ++j)
                for (int k = 0; k < n; ++k)
                    a[j][N + k] = j == k ? 1.0 : 0.0;   // right-hand sides: unit weight on overtone k, no quadrature
            for (int col = 0; col < N; ++col)            // Gauss-Jordan with partial pivoting
            {
                int piv = col;
                for (int r = col + 1; r < N; ++r)
                    if (std::fabs (a[r][col]) > std::fabs (a[piv][col])) piv = r;
                if (piv != col)
                    for (int c = 0; c < N + n; ++c) std::swap (a[piv][c], a[col][c]);
                const double d = a[col][col];
                for (int c = 0; c < N + n; ++c) a[col][c] /= d;
                for (int r = 0; r < N; ++r)
                    if (r != col)
                    {
                        const double f = a[r][col];
                        if (std::fabs (f) > 0.0)
                            for (int c = 0; c < N + n; ++c) a[r][c] -= f * a[col][c];
                    }
            }
            for (int i = 0; i < 2 * kBands; ++i)
                for (int k = 0; k < kBands; ++k)
                    m[i][k] = 0.0f;
            for (int i = 0; i < N; ++i)
                for (int k = 0; k < n; ++k)
                    m[i][k] = (float) a[i][N + k];
        }

        // wanted per-band weights w -> in-phase weights wp and quadrature weights wq
        void apply (const float* w, float* wp, float* wq) const noexcept
        {
            for (int i = 0; i < n; ++i)
            {
                float p = 0.0f, q = 0.0f;
                for (int k = 0; k < n; ++k)
                {
                    p += m[i][k] * w[k];
                    q += m[n + i][k] * w[k];
                }
                wp[i] = p;
                wq[i] = q;
            }
        }
    };
}

// IMAGER: a Linkwitz-Riley (24 dB / octave) split into four bands at three crossovers
struct ImgSplit
{
    Biquad lp[3][2], hp[3][2];
    void design (const float* x, double fs) noexcept
    {
        for (int i = 0; i < 3; ++i)
            for (int k = 0; k < 2; ++k)
            {
                lp[i][k].lowpass (x[i], 0.70710678, fs);
                hp[i][k].highpass (x[i], 0.70710678, fs);
            }
    }
    void reset() noexcept
    {
        for (auto& r : lp) for (auto& b : r) b.reset();
        for (auto& r : hp) for (auto& b : r) b.reset();
    }
    inline void split (float x, float* b) noexcept
    {
        b[0] = lp[0][1].process (lp[0][0].process (x));
        const float r1 = hp[0][1].process (hp[0][0].process (x));
        b[1] = lp[1][1].process (lp[1][0].process (r1));
        const float r2 = hp[1][1].process (hp[1][0].process (r1));
        b[2] = lp[2][1].process (lp[2][0].process (r2));
        b[3] = hp[2][1].process (hp[2][0].process (r2));
    }
};

// ... and the bands summed back with their own gains, through the all-passes of the crossovers a band did not
// pass through: with every gain at 1 the sum is exactly one all-pass of the input (only phase, the same for all)
struct ImgSum
{
    Biquad ap[2];
    void design (const float* x, double fs) noexcept { ap[0].allpass (x[1], 0.70710678, fs); ap[1].allpass (x[2], 0.70710678, fs); }
    void reset() noexcept { ap[0].reset(); ap[1].reset(); }
    inline float sum (const float* b, const float* g) noexcept
    {
        const float lo = ap[0].process (g[0] * b[0]) + g[1] * b[1];
        return ap[1].process (lo) + g[2] * b[2] + g[3] * b[3];
    }
};

// IMAGER: a phase-dispersion decorrelator - a chain of all-passes, no echoes and no ringing: every frequency
// turns by its own angle, so it is a copy of the sound that does not line up with it
struct ImgDisperse
{
    static constexpr int kN = 6;
    Biquad ap[kN];
    void design (const float* f, double fs) noexcept { for (int i = 0; i < kN; ++i) ap[i].allpass (f[i], 0.6, fs); }
    void reset() noexcept { for (auto& b : ap) b.reset(); }
    inline float process (float x) noexcept { for (auto& b : ap) x = b.process (x); return x; }
};

// what the core added, for the GUI (filled by the active path's left channel)
struct CoreMonitor
{
    float level[InnerControls::kBands] {};  // sum of squares of what the core added at each band (before Sing / Dirt)
    float groups = 0.0f;                    // how many filter steps were summed
    float yPow = 0.0f;                      // sum of squares of the core's output over the same steps
    float u[64] {}, y[64] {};               // per base sample: the clip's input and the core's output (clip units)
    float peakIn = 0.0f;                    // largest clip input in the block
    bool on = false;
};

// static shapers - shared by the oversampled core and the level probe
inline float shapeLamp (float x, float k, float c, float tc) noexcept  { return fastTanh (k * x + c) - tc; }
inline float shapeGlass (float x, float k) noexcept                    { return fastTanh (k * x); }
inline float shapeFold (float x, float k, float s, float ss) noexcept  { return fastSin (k * x + s) - ss; }
// hard clipper with a C1 knee (much less aliasing than a sharp corner)
inline float kneeClip (float u) noexcept
{
    constexpr float kk = 0.15f;
    const float a = std::fabs (u);
    if (a <= 1.0f - kk) return u;
    const float r = a >= 1.0f + kk ? 1.0f : 1.0f - (1.0f + kk - a) * (1.0f + kk - a) * (1.0f / (4.0f * kk));
    return u < 0.0f ? -r : r;
}

inline float shapeClip (float v, float in, float h, float out) noexcept
{
    const float u = v * in;
    return ((1.0f - h) * fastTanh (u) + h * kneeClip (u)) * out;
}

// THE DOUBLES' Tone: a one-pole low-pass (dark) or high-pass (bright) on what a double hears
struct DoubleTone
{
    float lp = 0.0f, hp = 0.0f;
    float process (float x, float aLp, float aHp) noexcept
    {
        lp += aLp * (x - lp);
        hp += aHp * (lp - hp);
        return lp - hp;
    }
    void reset() noexcept { lp = hp = 0.0f; }
};

struct InnerState
{
    DelayLine dl, cl;          // the lamp: the core's input after the crusher / the clean input (its reference)
    // the doubles: what each one hears (the voice + its own repeats, through its Tone) and the same clean
    // (no crusher, no ghosts: their references for the melodic distortion)
    DelayLine gw, fw, gc, fc;
    DoubleTone tG, tF, tGc, tFc;
    float gPrev = 0.0f, fPrev = 0.0f, gcPrev = 0.0f, fcPrev = 0.0f;
    float lastToneG = 9.0f, lastToneF = 9.0f, aLpG = 1.0f, aHpG = 0.0f, aLpF = 1.0f, aHpF = 0.0f;
    OnePole dopLP, dopLPf;
    float crushPhase = 0.0f, crushHold = 0.0f;
    // MELODIC DISTORTION: state-variable band-passes on every overtone, tuned to the voice
    static constexpr int kB = InnerControls::kBands;
    float hS1[kB] {}, hS2[kB] {}, hG[kB] {}, hDen[kB] {};
    float hF0 = -1.0f, hRamp = 0.0f;
    bool hLive = false;
    // running least-squares fits: the linear gain of every machine against its clean input, and of the clip
    float fMC[3] {}, fCC[3] {}, fYU = 0.0f, fUU = 0.0f, fDG = -1.0f, fDF = -1.0f;

    static constexpr float kMaxDoubleSec = 0.33f;   // Time up to 300 ms, plus the float, the Doppler and the stereo offset

    void prepare (float topRate)
    {
        dl.prepare ((int) std::ceil (0.020f * topRate) + 8);
        cl.prepare ((int) std::ceil (0.020f * topRate) + 8);
        for (DelayLine* d : { &gw, &fw, &gc, &fc })
            d->prepare ((int) std::ceil (kMaxDoubleSec * topRate) + 8);
        dopLP.setCutoff (900.0f, topRate);
        dopLPf.setCutoff (900.0f, topRate);
        reset();
    }
    // Tone -> one-pole coefficients at this rate (0 = exactly untouched)
    static void toneCoefs (float tone, float topRate, float& aLp, float& aHp) noexcept
    {
        aLp = 1.0f; aHp = 0.0f;
        if (tone < 0.0f)
            aLp = 1.0f - std::exp (-kTwoPi * 20000.0f * std::exp2 (4.3f * tone) / topRate);
        else if (tone > 0.0f)
            aHp = 1.0f - std::exp (-kTwoPi * 20.0f * std::exp2 (6.2f * tone) / topRate);
    }
    void resetFits() noexcept
    {
        for (int h = 0; h < kB; ++h)
            hS1[h] = hS2[h] = 0.0f;
        for (int m = 0; m < 3; ++m)
            fMC[m] = fCC[m] = 0.0f;
        fYU = fUU = 0.0f;
        fDG = fDF = -1.0f;
        hRamp = 0.0f;
    }
    void reset() noexcept
    {
        dl.reset(); cl.reset(); dopLP.reset(); dopLPf.reset(); crushPhase = 0.0f; crushHold = 0.0f;
        for (DelayLine* d : { &gw, &fw, &gc, &fc })
            d->reset();
        tG.reset(); tF.reset(); tGc.reset(); tFc.reset();
        gPrev = fPrev = gcPrev = fcPrev = 0.0f;
        lastToneG = lastToneF = 9.0f;
        resetFits();
        hF0 = -1.0f; hLive = false;
    }
};

// runs at the oversampled rate
// top: the core's input (overwritten with its output); ghost: the ghosts alone at the same rate (or nullptr), so
// the clean reference can leave them out - what they add counts as distortion too
inline void runInner (InnerState& st, float* top, const float* ghost, const InnerControls* ctl, int n, int os, float topRate,
                      float* res, CoreMonitor* mon) noexcept
{
    const float msToS = 0.001f * topRate, invTop = 1.0f / topRate;
    // the overtone filters and the fits run at twice the base rate (plenty below 9 kHz): one step per hStep top samples
    const int hStep = std::max (1, os / 2);
    const float invHRate = (float) hStep * invTop, invStep = 1.0f / (float) hStep;
    const float aFit = 1.0f - std::exp (-invHRate / 0.015f);          // 15 ms
    const float aDop = 1.0f - std::exp (-invHRate / 0.010f);          // 10 ms
    const float aRamp = invHRate / 0.010f;                            // 10 ms fade-in after the filters wake up
    const float half = 0.5f * (float) (hStep - 1);                    // centre of a filter step, in top samples
    for (int j = 0; j < n; ++j)
    {
        const InnerControls& c = ctl[j];
        const float dL = c.dLamp * msToS, dG = c.dGlass * msToS, dD = c.dDop * msToS, dF = c.dFold * msToS;
        const float dDF = c.dDopF * msToS;
        const float crushInc = std::min (1.0f, c.crushRate * invTop);
        float* t = top + j * os;
        float resAcc = 0.0f, sL = 0.0f, sG = 0.0f, sF = 0.0f, sY = 0.0f, sDG = 0.0f, sDF = 0.0f, uLast = 0.0f;
        if (differs (c.toneG, st.lastToneG)) { st.lastToneG = c.toneG; InnerState::toneCoefs (c.toneG, topRate, st.aLpG, st.aHpG); }
        if (differs (c.toneF, st.lastToneF)) { st.lastToneF = c.toneF; InnerState::toneCoefs (c.toneF, topRate, st.aLpF, st.aHpF); }
        const bool toneG = st.aLpG < 1.0f || st.aHpG > 0.0f, toneF = st.aLpF < 1.0f || st.aHpF > 0.0f;
        const bool repG = c.fbG > 0.0f, repF = c.fbF > 0.0f;
        int hCount = 0;
        const bool hOn = c.hOn > 0.0f;
        if (hOn)
        {
            if (! st.hLive) { st.resetFits(); st.hLive = true; }
            if (differs (c.hF0, st.hF0))
            {
                st.hF0 = c.hF0;
                for (int h = 0; h < InnerControls::kBands; ++h)
                {
                    const float g = std::tan (kPi * std::min (inkey::kMult[h] * c.hF0 * invHRate, 0.2f));
                    st.hG[h] = g;
                    st.hDen[h] = 1.0f / (1.0f + 2.0f * inkey::kR[h] * g + g * g);
                }
            }
        }
        else
            st.hLive = false;
        const int nb = c.hHalf > 0.0f ? InnerControls::kBands : InnerControls::kHarm;
        const float* gh = ghost != nullptr ? ghost + j * os : nullptr;
        for (int k = 0; k < os; ++k)
        {
            float in = t[k];
            const float clean = gh != nullptr ? in - gh[k] : in;
            st.cl.push (clean);
            if (c.crushMix > 0.0f)
            {
                // Harmonic crush: the hold clock runs at N x the voice's pitch and restarts on every period,
                // so the stepped wave repeats exactly with the voice - every product is a harmonic
                if (k == 0 && c.crushSync > 0.0f)
                {
                    // the period started (crushSync - 1) base samples ago: place the grid exactly there
                    st.crushPhase = (c.crushSync - 1.0f) * (float) os * crushInc;
                    st.crushHold = in;
                }
                else
                    st.crushPhase += crushInc;
                if (st.crushPhase >= 1.0f)
                {
                    st.crushPhase -= std::floor (st.crushPhase);
                    st.crushHold = in;
                }
                const float q = c.crushStep > 0.0f ? c.crushStep * std::floor (st.crushHold / c.crushStep + 0.5f) : st.crushHold;
                in += c.crushMix * (q - in);
            }
            st.dl.push (in);
            // what the doubles hear: the voice plus their own repeats (what they read a sample ago), through their Tone;
            // their clean twins (the references) the same, repeating at the doubles' average delay
            {
                const float xg = repG ? in + c.fbG * st.gPrev : in, xf = repF ? in + c.fbF * st.fPrev : in;
                const float cg = repG ? clean + c.fbG * st.gcPrev : clean, cf = repF ? clean + c.fbF * st.fcPrev : clean;
                st.gw.push (toneG ? st.tG.process (xg, st.aLpG, st.aHpG) : xg);
                st.fw.push (toneF ? st.tF.process (xf, st.aLpF, st.aHpF) : xf);
                st.gc.push (toneG ? st.tGc.process (cg, st.aLpG, st.aHpG) : cg);
                st.fc.push (toneF ? st.tFc.process (cf, st.aLpF, st.aHpF) : cf);
            }
            // Lamp: asymmetric tube whose bias breathes with the clock
            const float lamp = shapeLamp (st.dl.read (dL), c.kLamp, c.cLamp, c.tcLamp);
            // Doppler Glass: the voice moves its own "speaker cone" (self-modulated delay)
            const float g0 = st.gw.read (dG);
            const float m = fastTanh (st.dopLP.lp (g0) * c.dopSens);
            const float dGm = dG + dD * (1.0f + m);
            const float gx = st.gw.read (dGm);
            const float glass = shapeGlass (gx, c.kGlass);
            // Fold: diode wavefolder (with its own Doppler, if thrown)
            float dFm = dF;
            if (dDF > 0.0f)
            {
                const float mf = fastTanh (st.dopLPf.lp (st.fw.read (dF)) * c.dopSens);
                dFm = dF + dDF * (1.0f + mf);
            }
            const float fx = st.fw.read (dFm);
            const float fold = shapeFold (fx, c.kFold, c.sFold, c.ssFold);
            if (repG)
            {
                st.gPrev = gx;
                st.gcPrev = st.gc.read (st.fDG >= 0.0f ? st.fDG : dG + dD);
            }
            if (repF)
            {
                st.fPrev = fx;
                st.fcPrev = st.fc.read (st.fDF >= 0.0f ? st.fDF : dF + dDF);
            }
            // CLIP: on vowels the positive ceiling sits lower (asymmetric = even harmonics), consonants stay symmetric
            const float u = (c.gLamp * lamp + c.gGlass * glass + c.gFold * fold) * c.clipIn;
            float y;
            if (u > 0.0f)
            {
                const float uu = u / c.clipPos;
                y = c.clipPos * ((1.0f - c.clipH) * fastTanh (uu) + c.clipH * kneeClip (uu));
            }
            else
                y = (1.0f - c.clipH) * fastTanh (u) + c.clipH * kneeClip (u);
            resAcc += std::fabs (u - y);
            uLast = u;
            t[k] = y * c.clipOut;
            if (hOn)
            {
                // MELODIC DISTORTION: what did the whole core ADD? Every machine is compared with the clean voice it
                // read (before the crusher), at its own running linear gain, and the clip with the sum of those -
                // so a 30 dB push does not count as change, only new overtones and the reshaping of the voice's own.
                // That difference is split by the voice's overtones; each overtone the core adds is then taken
                // back, kept or raised: the notes of the song's chord sing (Sing), the rest is dirt (Dirt).
                sL += lamp; sG += glass; sF += fold; sY += y; sDG += dGm; sDF += dFm;
                if (++hCount == hStep)
                {
                    // the clean voice every machine saw, at the centre of this step
                    // (the glass at its slow average delay: what the Doppler's self-modulation adds counts as distortion)
                    st.fDG = st.fDG < 0.0f ? sDG * invStep : st.fDG + (sDG * invStep - st.fDG) * aDop;
                    st.fDF = st.fDF < 0.0f ? sDF * invStep : st.fDF + (sDF * invStep - st.fDF) * aDop;
                    // (each double against its clean twin: the voice and its repeats, through the same Tone)
                    const float cL = st.cl.read (dL + half), cG = st.gc.read (st.fDG + half);
                    const float cF = st.fc.read ((dDF > 0.0f ? st.fDF : dF) + half);
                    const float mL = sL * invStep, mG = sG * invStep, mF = sF * invStep, mY = sY * invStep;
                    st.fMC[0] += (mL * cL - st.fMC[0]) * aFit; st.fCC[0] += (cL * cL - st.fCC[0]) * aFit;
                    st.fMC[1] += (mG * cG - st.fMC[1]) * aFit; st.fCC[1] += (cG * cG - st.fCC[1]) * aFit;
                    st.fMC[2] += (mF * cF - st.fMC[2]) * aFit; st.fCC[2] += (cF * cF - st.fCC[2]) * aFit;
                    const float uLin = c.clipIn * (c.gLamp * cL * st.fMC[0] / (st.fCC[0] + 1.0e-12f)
                                                   + c.gGlass * cG * st.fMC[1] / (st.fCC[1] + 1.0e-12f)
                                                   + c.gFold * cF * st.fMC[2] / (st.fCC[2] + 1.0e-12f));
                    st.fYU += (mY * uLin - st.fYU) * aFit;
                    st.fUU += (uLin * uLin - st.fUU) * aFit;
                    const float d = mY - uLin * (st.fYU / (st.fUU + 1.0e-12f));
                    float corr = 0.0f;
                    for (int h = 0; h < nb; ++h)
                    {
                        const float g = st.hG[h], R = inkey::kR[h];
                        const float hp = (d - (2.0f * R + g) * st.hS1[h] - st.hS2[h]) * st.hDen[h];
                        const float bp = g * hp + st.hS1[h];
                        st.hS1[h] = g * hp + bp;
                        const float lp = g * bp + st.hS2[h];
                        st.hS2[h] = g * bp + lp;
                        const float band = (2.0f * R) * bp;
                        corr += c.hW[h] * band + c.hQ[h] * (2.0f * R) * lp;
                        if (mon != nullptr)
                            mon->level[h] += band * band;
                    }
                    if (mon != nullptr)
                    {
                        mon->groups += 1.0f;
                        mon->yPow += mY * mY;
                    }
                    st.hRamp = std::min (1.0f, st.hRamp + aRamp);
                    corr *= c.clipOut * st.hRamp;
                    for (int q = k - hStep + 1; q <= k; ++q)   // centred on the same samples it was measured on
                        t[q] -= corr;
                    sL = sG = sF = sY = sDG = sDF = 0.0f;
                    hCount = 0;
                }
            }
        }
        if (res != nullptr)
            res[j] = resAcc * c.clipOut / (float) os;   // what the clipper cut off: it strikes the harp and throws echoes
        if (mon != nullptr)
        {
            mon->u[j] = uLast;
            mon->y[j] = t[os - 1] / c.clipOut;
            mon->peakIn = std::max (mon->peakIn, std::fabs (uLast));
        }
    }
}

//==============================================================================
// One oversampling factor: up -> core -> down, padded to the common latency.
struct CorePath
{
    int factor = 1, basePad = 0;
    float topRate = 44100.0f;
    Oversampler os[2], gos[2];   // the core's input / the ghosts alone (identical filters: they line up exactly)
    InnerState inner[2];
    DelayLine pad[2];
    bool ghostLive[2] {};

    int prepareOversampling (double fs, int numStages, int maxBlock)
    {
        factor = 1 << numStages;
        topRate = (float) (fs * factor);
        for (int c = 0; c < 2; ++c)
        {
            os[c].prepare (numStages, maxBlock);
            gos[c].prepare (numStages, maxBlock);
            inner[c].prepare (topRate);
        }
        return os[0].latency();
    }

    void setCommonLatency (int latencyMax)
    {
        basePad = latencyMax - os[0].latency();
        for (int c = 0; c < 2; ++c)
            pad[c].prepare (basePad + 4);
    }

    void reset() noexcept
    {
        for (int c = 0; c < 2; ++c)
        {
            os[c].reset();
            gos[c].reset();
            inner[c].reset();
            pad[c].reset();
            ghostLive[c] = false;
        }
    }

    void process (int ch, const float* in, const float* ghost, const InnerControls* ctl, float* out, float* res, int n,
                  CoreMonitor* mon = nullptr) noexcept
    {
        float* top = os[ch].up (in, n);
        const float* gtop = nullptr;
        if (ghost != nullptr)
        {
            if (! ghostLive[ch]) { gos[ch].reset(); ghostLive[ch] = true; }
            gtop = gos[ch].up (ghost, n);
        }
        else
            ghostLive[ch] = false;
        runInner (inner[ch], top, gtop, ctl, n, factor, topRate, res, mon);
        os[ch].down (top, out, n);
        if (basePad > 0)
            for (int i = 0; i < n; ++i)
            {
                pad[ch].push (out[i]);
                out[i] = pad[ch].tap (basePad);
            }
    }
};

//==============================================================================
// "Harp in the lamp": saturating strings tuned to the song key, excited by the distorted voice.
class Harp
{
public:
    static constexpr int kStrings = 6;

    void prepare (float sampleRate)
    {
        fs = sampleRate;
        for (auto& s : str)
            s.dl.prepare ((int) (fs / 30.0f) + 16);
        reset();
    }

    void reset() noexcept
    {
        for (auto& s : str)
        {
            s.dl.reset();
            s.z = 0.0f;
        }
    }

    void configure (const float* hz, float decaySec, float bright, float drive) noexcept
    {
        const float fc = 1200.0f * std::exp2 (bright * 3.6f);
        aLP = std::exp (-kTwoPi * std::min (fc, 0.45f * fs) / fs);
        sat = 0.9f / (1.0f + 6.0f * drive);
        invSat = 1.0f / sat;
        for (int i = 0; i < kStrings; ++i)
        {
            Str& s = str[i];
            const float f = clampf (hz[i], 30.0f, 0.2f * fs);
            const float w = kTwoPi * f / fs;
            const float pd = std::atan2 (aLP * std::sin (w), 1.0f - aLP * std::cos (w)) / w;   // LP phase delay
            s.len = std::max (1.0f, fs / f - 1.0f - pd);
            const float mag = (1.0f - aLP) / std::sqrt (1.0f - 2.0f * aLP * std::cos (w) + aLP * aLP);
            s.g = std::min (0.9995f, std::pow (10.0f, -3.0f / (f * std::max (decaySec, 0.05f))) / std::max (mag, 1.0e-3f));
            s.inG = 0.9f * std::sqrt (std::max (1.0e-6f, 1.0f - s.g * s.g));
            s.shSign = (i & 1) ? -1.0f : 1.0f;
        }
    }

    inline float process (float exc, float shimmer) noexcept
    {
        float sum = 0.0f;
        for (auto& s : str)
        {
            const float y = s.dl.read (s.len * (1.0f + shimmer * s.shSign));
            s.z = y + aLP * (s.z - y);
            const float v = exc * s.inG + s.g * s.z;
            s.dl.push (sat * fastTanh (v * invSat));
            sum += y;
        }
        return sum;
    }

private:
    struct Str
    {
        DelayLine dl;
        float len = 100.0f, g = 0.9f, inG = 0.1f, z = 0.0f, shSign = 1.0f;
    };
    Str str[kStrings];
    float fs = 44100.0f, aLP = 0.5f, sat = 0.9f, invSat = 1.0f / 0.9f;
};

//==============================================================================
class Engine
{
public:
    Engine()
    {
        // the GUI may look before the host prepares the engine
        for (int h = 0; h < InnerControls::kBands; ++h)
        {
            guiOtLevel[h].store (-120.0f);
            guiOtGain[h].store (1.0f);
            guiOtFit[h].store (0.0f);
        }
        for (int i = 0; i < kScope; ++i)
        {
            scopeIn[i].store (0.0f);
            scopeOut[i].store (0.0f);
        }
        for (auto& g : guiPhase) g.store (0.0f);
        for (auto& g : guiDuck) g.store (1.0f);
        for (int i = 0; i < kImg; ++i)
        {
            imgPreL[i].store (0.0f); imgPreR[i].store (0.0f);
            imgPostL[i].store (0.0f); imgPostR[i].store (0.0f);
        }
    }

    static constexpr int   kSub   = 64;      // internal block
    static constexpr int   kPaths = 5;       // 1x, 2x, 4x, 8x, 16x
    static constexpr float kRefDb = -12.0f;  // consistency reference (peak envelope)
    static constexpr float kPredict = 0.7f;  // pitch: how far ahead of the last period (in periods)

    void prepare (double sampleRate, int numChannels);
    void reset() noexcept;

    void setParams (const float* values) noexcept { std::copy (values, values + P_COUNT, prm); }
    void setTransport (bool isPlaying, double bpm, double ppqAtBlockStart) noexcept
    {
        hostPlaying = isPlaying;
        if (bpm > 1.0 && bpm < 2000.0)
            hostBpm = bpm;
        hostPpq = ppqAtBlockStart;
        blockPos = 0;
    }
    void setNonRealtime (bool b) noexcept { offline = b; }

    int   getLatency() const noexcept           { return totalLat; }
    int   getActiveFactor() const noexcept      { return guiFactor.load (std::memory_order_relaxed); }
    float getClockPhase (int i) const noexcept  { return guiPhase[i & 3].load (std::memory_order_relaxed); }
    float getInkStroke() const noexcept         { return guiInk.load (std::memory_order_relaxed); }
    float getVoiced() const noexcept            { return guiVoiced.load (std::memory_order_relaxed); }
    float getSlantMs() const noexcept           { return guiSlant.load (std::memory_order_relaxed); }
    float getInputDb() const noexcept           { return guiLevel.load (std::memory_order_relaxed); }
    float getRelativeDb() const noexcept        { return guiRel.load (std::memory_order_relaxed); }
    float getDuckGain (int band) const noexcept { return guiDuck[std::min (std::max (band, 0), DuckBands::kBands - 1)].load (std::memory_order_relaxed); }
    // how hard the Pressure clip bites (0..2), peak since the GUI last asked
    float takeClipHit() noexcept                { return guiClip.exchange (0.0f, std::memory_order_relaxed); }

    // MELODIC DISTORTION, for the overtone map: what the core adds at every overtone of the voice right now
    struct OvertoneView
    {
        float f0 = 0.0f, conf = 0.0f, push = 0.0f;
        // per band (overtones 2 .. 12, then the half overtones 1.5 .. 7.5 of the sub-octave ghost)
        float level[InnerControls::kBands] {};  // dB re the core's level: what ordinary distortion would add there
        float gain[InnerControls::kBands] {};   // what MAVKA keeps of it (linear, 0 .. ~8)
        float fit[InnerControls::kBands] {};    // 1 = a note of the song's chord, 0 = dirt
        bool halves = false;                    // the half-overtone bands are running
        bool plain = false;                     // Sing 100 / Dirt 100 / Force 0 / Prism 0: the ordinary distortion
    };
    void getOvertones (OvertoneView& o) const noexcept
    {
        o.f0 = guiF0.load (std::memory_order_relaxed);
        o.conf = guiConf.load (std::memory_order_relaxed);
        o.push = guiPush.load (std::memory_order_relaxed);
        o.halves = guiHalves.load (std::memory_order_relaxed);
        o.plain = guiPlain.load (std::memory_order_relaxed);
        for (int h = 0; h < InnerControls::kBands; ++h)
        {
            o.level[h] = guiOtLevel[h].load (std::memory_order_relaxed);
            o.gain[h] = guiOtGain[h].load (std::memory_order_relaxed);
            o.fit[h] = guiOtFit[h].load (std::memory_order_relaxed);
        }
    }
    // the clip as it is right now, for the transfer curve
    struct ClipView { float drive = 1.0f, knee = 0.0f, pos = 1.0f, peakIn = 0.0f; };
    void getClipView (ClipView& c) const noexcept
    {
        c.drive = guiClipDrive.load (std::memory_order_relaxed);
        c.knee = guiClipKnee.load (std::memory_order_relaxed);
        c.pos = guiClipPos.load (std::memory_order_relaxed);
        c.peakIn = guiClipPeak.load (std::memory_order_relaxed);
    }
    // the last kScope base samples of the clip's input and the core's output (clip units), oldest first
    static constexpr int kScope = 1024;
    void copyScope (float* in, float* out) const noexcept
    {
        const int w = scopePos.load (std::memory_order_acquire);
        for (int i = 0; i < kScope; ++i)
        {
            const int r = (w + i) & (kScope - 1);
            in[i] = scopeIn[r].load (std::memory_order_relaxed);
            out[i] = scopeOut[r].load (std::memory_order_relaxed);
        }
    }
    // the GUI is looking at the overtone map / scope: the left channel's core is metered
    void setMetering (bool b) noexcept { metering.store (b, std::memory_order_relaxed); }

    // IMAGER: the last kImg base samples of the wet sound before and after the imager (left, right), oldest first
    static constexpr int kImg = 2048;
    void copyImager (float* preL, float* preR, float* postL, float* postR) const noexcept
    {
        const int w = imgPos.load (std::memory_order_acquire);
        for (int i = 0; i < kImg; ++i)
        {
            const int r = (w + i) & (kImg - 1);
            preL[i] = imgPreL[r].load (std::memory_order_relaxed);
            preR[i] = imgPreR[r].load (std::memory_order_relaxed);
            postL[i] = imgPostL[r].load (std::memory_order_relaxed);
            postR[i] = imgPostR[r].load (std::memory_order_relaxed);
        }
    }
    void setImagerMetering (bool b) noexcept { imgMetering.store (b, std::memory_order_relaxed); }
    bool isImagerActive() const noexcept { return guiImgOn.load (std::memory_order_relaxed); }
    double getSampleRate() const noexcept { return fs; }

    void process (float* const* io, int numCh, int n) noexcept
    {
        numCh = std::min (numCh, 2);
        for (int off = 0; off < n; off += kSub)
            processSub (io, numCh, off, std::min (kSub, n - off));
    }

    void processBypassed (float* const* io, int numCh, int n) noexcept
    {
        numCh = std::min (numCh, 2);
        for (int c = 0; c < numCh; ++c)
            for (int i = 0; i < n; ++i)
            {
                ch[c].dry.push (io[c][i]);
                io[c][i] = ch[c].dry.tap (totalLat);
            }
    }

private:
    void processSub (float* const* io, int numCh, int off, int n) noexcept;
    void resetImager() noexcept
    {
        imgSplitS.reset(); imgSplitD.reset();
        imgSumS.reset(); imgSumD.reset(); imgSumR.reset();
        for (auto& b : imgApM) b.reset();
        for (auto& c : imgApDry) for (auto& b : c) b.reset();
        imgSpread.reset(); imgRecover.reset();
        imgLive = imgSpreadLive = imgRecLive = false;
    }
    void updateBlock() noexcept;
    void computeProbe() noexcept;
    int  desiredPath (int n) noexcept;

    static float panGain (float pan, int c) noexcept
    {
        const float th = (clampf (pan, -1.0f, 1.0f) + 1.0f) * (0.25f * kPi);
        return 1.41421356f * (c == 0 ? std::cos (th) : std::sin (th));
    }

    //--------------------------------------------------------------------------
    double fs = 44100.0;
    float fsf = 44100.0f;
    float prm[P_COUNT] {};
    bool offline = false, first = true, snap = true;

    // transport / chord clock
    bool hostPlaying = false;
    double hostBpm = 120.0, hostPpq = 0.0, freePpq = 0.0;
    long blockPos = 0;

    // oversampled core
    CorePath paths[kPaths];
    int latencyMax = 0, lampPre = 1, totalLat = 0;
    float lampBaseMs = 0.8f;
    int activePath = 0, nextPath = -1, warmLeft = 0, fadeLeft = 0, warmLen = 1, fadeLen = 1;
    int autoChoice = 0, autoPending = 0;
    long autoStable = 0;

    struct Chan
    {
        DelayLine dry;
        Biquad preHP, preLP, tiltLo, tiltHi;
        float apZ[4] {};
        Biquad ghostLP, ghostSubLP, ghostUpLP;
        Envelope ghostEnv;
        OnePole ghostUpHP;
        float zPrev = 0.0f, flip = 1.0f;
        bool armed = false;
        int crossCount = 0;
        float f0 = 150.0f, fMeas = 150.0f, lastDelta = 0.0f, pitchConf = 0.0f, noteSemi = -9.9f;
        bool newPeriod = false;
        Envelope crushEnv;
        float dLampS = 0.8f, dGlassS = 23.0f, dFoldS = 37.0f, dDopS = 0.0f, dDopFS = 0.0f;
        OnePole dc;
        Biquad smLP;
        Envelope smEnvLo, smEnvHi;
        float smGain = 1.0f;
        Biquad harpHP, harpLP;
        Harp harp;
        Biquad toneLo, toneHi;
        DelayLine fl;
        OnePole flLP;
        float flT = 50.0f;
        bool flLive = false;
        Envelope clipEnv;
        DelayLine ctlRestore, ctlFocus, ctlInkWet, ctlInkHarp;
        // MELODIC DISTORTION: which overtones of the voice are notes of the song's chord (cached per pitch)
        float fit[InnerControls::kBands] {}, fitOk[InnerControls::kBands] {};
        float fitF0 = -1.0f;
        int fitVersion = -1;
        bool compDirty = true;
        float wantA[InnerControls::kBands] {}, wantB[InnerControls::kBands] {};
        float xaP[InnerControls::kBands] {}, xaQ[InnerControls::kBands] {}, xbP[InnerControls::kBands] {}, xbQ[InnerControls::kBands] {};
        // REVERB SAFE: the wet sound that goes on into a reverb
        Biquad rsHP, rsShelf;
    };
    Chan ch[2];
    OnePole rsSideHP;
    float rsMix = 0.0f;
    bool rsLive = false;

    // mono analysis
    Envelope envCons, inkFast, inkSlow, vEnvLo, vEnvHi;
    Biquad vLP, vHP;
    float gNdb = 0.0f, voiced = 1.0f, strokeRem = 0.0f, lastEnvDb = -100.0f;
    long sinceOnset = 1L << 30, minOnsetGap = 3000;
    float inkDriveDb = 0.0f, inkDop = 1.0f, inkWet = 1.0f, inkHarp = 1.0f;
    float warpMix = 0.0f;

    // Slant: per-stroke pitch gesture of the doubles (gamma-shaped delay bump, continuous)
    float slS1 = 0.0f, slS2 = 0.0f;
    // Ligature: tempo-synced ping-pong echo that only speaks between words
    DelayLine ligL, ligR;
    OnePole ligLP[2], ligHP[2];
    float ligTL = 0.0f, ligTR = 0.0f, ligDuck = 1.0f, aLigT = 0.001f, aDuckAtk = 0.9f, aDuckRel = 0.99f;
    float duckBuf[kSub] {};
    float loudRef = -70.0f, loudFall = 0.0f;

    // AURA: nonlinear send effects
    AuraEcho aEcho;
    AuraForest aForest;
    AuraTwin aTwin;
    Biquad fxHP, fxLP;
    float tEcho = 0.0f, sEcho = 0.0f, tForest = 0.0f, sForest = 0.0f, tTwin = 0.0f, sTwin = 0.0f;
    bool echoLive = false, forestLive = false, twinLive = false;
    float harpHz[6] {};
    int harpVersion = 0, forestVersion = -1, lastForestType = -1;
    float lastForestDecay = -1.0f, lastForestTune = -1.0f;
    float auraA[kSub] {}, auraB[kSub] {}, auraC[kSub] {}, breathe[kSub] {};
    // spectral ducking (Ligature + Aura echo/forest)
    DuckKey duckKey;
    DuckFx duckLig, duckAura;

    // smoothing
    float aS = 0.01f, aCtl = 0.01f, aInk = 0.01f, aGn = 0.01f, aWarp = 0.01f;
    float aVfast = 0.9f, aVslow = 0.99f, aSmAtk = 0.9f, aSmRel = 0.99f;
    float tDrive = 1.0f, sDrive = 1.0f, tDryG = 1.0f, sDryG = 1.0f, tWetG = 0.0f, sWetG = 0.0f, tOut = 1.0f, sOut = 1.0f;
    float tLamp = 1.0f, sLamp = 1.0f, tGlass = 0.0f, sGlass = 0.0f, tFold = 0.0f, sFold = 0.0f;
    float tComp = 1.0f, sComp = 1.0f, tHarpLvl = 0.0f, sHarpLvl = 0.0f, tHarpExc = 0.0f, sHarpExc = 0.0f;
    float compLamp = 1.0f, compGlass = 1.0f, compFold = 1.0f, mixNorm = 1.0f;
    float smThr = 0.5f, smExp = 1.0f, smStatic = 1.0f, harpShim = 0.0f;

    // cached settings
    float lastLowCut = -1.0f, lastHighCut = -1.0f, lastTilt = 1.0e9f, lastTone = 1.0e9f;
    float lastDecay = -1.0f, lastBright = -1.0f, lastHarpDrive = -1.0f, lastWidth = -1.0f, lastDetune = -1.0e9f;
    float tuneSemis = 0.0f;   // Detune in semitones (the tuning reference of everything that plays in key)
    int lastKey = -1, lastChord = -1, lastReg = -1;
    float chordPc[6] {};   // pitch classes of the song's chord (0 = C), for the melodic distortion
    int numChordPc = 1;
    float aConf = 0.01f;
    float pushS = 0.0f, aPushAtk = 0.1f, aPushRel = 0.01f;   // Force: how hard the voice pushes right now (0..1)
    // THE DOUBLES: their slow float (phase) and their attack doors (delayed by each double's own time)
    float floatPhG = 0.0f, floatPhF = 0.25f, doorGs = 1.0f, doorFs = 1.0f, aDoor = 0.1f;
    DelayLine doorG, doorF;
    // IMAGER: the side of the wet sound in four bands; the mid and the dry get the very same all-pass, so
    // nothing moves against anything and the mono sum never changes
    ImgSplit imgSplitS, imgSplitD;
    ImgSum imgSumS, imgSumD, imgSumR;
    Biquad imgApM[3], imgApDry[2][3];
    ImgDisperse imgSpread, imgRecover;
    float imgG[4] = { 1.0f, 1.0f, 1.0f, 1.0f }, imgSp[4] {}, imgRec = 0.0f, imgX[3] = { -1.0f, -1.0f, -1.0f };
    float imgMix = 0.0f, aImg = 0.01f;
    int imgQuiet = 0;
    bool imgLive = false, imgSpreadLive = false, imgRecLive = false;
    std::atomic<float> imgPreL[kImg], imgPreR[kImg], imgPostL[kImg], imgPostR[kImg];
    std::atomic<int> imgPos { 0 };
    std::atomic<bool> imgMetering { false }, guiImgOn { false };
    inkey::Comp compInt, compAll;   // leakage compensation: the overtone bands alone / with the half-overtone bands
    float gNslowDb = 0.0f, aGnSlow = 0.001f;                 // POLISH density: the slow part of the leveling
    CoreMonitor mon;
    float otLevelS[InnerControls::kBands] {}, otGainAcc[InnerControls::kBands] {};
    float ghostBuf[2][kSub] {};   // the ghosts alone (the melodic reference leaves them out)

    float probe[64] {};

    // scratch
    float coreIn[2][kSub] {}, coreOutA[2][kSub] {}, coreOutB[2][kSub] {}, dryOut[2][kSub] {}, shim[2][kSub] {};
    float clipRes[2][kSub] {}, clipResB[2][kSub] {}, phase0Buf[kSub] {}, strokeBuf[kSub] {};
    float flTune[4] { 100.0f, 100.0f, 100.0f, 100.0f };
    InnerControls ctl[2][kSub] {};

    std::atomic<float> guiPhase[4];
    std::atomic<int>   guiFactor { 1 };
    std::atomic<float> guiDuck[DuckBands::kBands];
    std::atomic<float> guiInk { 0.0f }, guiVoiced { 1.0f }, guiSlant { 0.0f }, guiLevel { -100.0f }, guiRel { -60.0f };
    std::atomic<float> guiClip { 0.0f };
    std::atomic<float> guiF0 { 0.0f }, guiConf { 0.0f }, guiPush { 0.0f };
    std::atomic<float> guiOtLevel[InnerControls::kBands], guiOtGain[InnerControls::kBands], guiOtFit[InnerControls::kBands];
    std::atomic<bool> guiHalves { false }, guiPlain { false };
    std::atomic<float> guiClipDrive { 1.0f }, guiClipKnee { 0.0f }, guiClipPos { 1.0f }, guiClipPeak { 0.0f };
    std::atomic<float> scopeIn[kScope], scopeOut[kScope];
    std::atomic<int> scopePos { 0 };
    std::atomic<bool> metering { false };
};

//==============================================================================
namespace tables
{
    static const float kRatios[8][4] =
    {
        { 1.0f, 1.0f,       1.0f,       1.0f       },   // Unison (quadrature phases)
        { 1.0f, 2.0f,       4.0f,       8.0f       },   // Octaves
        { 1.0f, 1.5f,       2.0f,       3.0f       },   // Fifths 2:3
        { 1.0f, 1.25f,      1.5f,       2.0f       },   // Major 4:5:6
        { 1.0f, 1.2f,       1.5f,       2.0f       },   // Minor 10:12:15
        { 1.0f, 4.0f / 3.0f,1.5f,       2.0f       },   // Sus 6:8:9
        { 1.0f, 4.0f / 3.0f,2.0f,       8.0f / 3.0f},   // Poly 3:4
        { 1.0f, 1.618034f,  2.618034f,  4.236068f  },   // Golden (never repeats)
    };
    static const double kBeatsPerCycle[7] = { 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25 };
    static const float  kUnisonOffset[4]  = { 0.0f, 0.25f, 0.5f, 0.75f };
    static const float  kStereoOffset[4]  = { 0.25f, -0.25f, 0.25f, -0.25f };

    static const int kChords[12][6] =
    {
        { 0, 12, 24,  0, 12, 24 },   // Octaves (second set detuned)
        { 0,  7, 12, 19, 24, 31 },   // Power
        { 0,  4,  7, 12, 16, 19 },   // Major
        { 0,  3,  7, 12, 15, 19 },   // Minor
        { 0,  2,  7, 12, 14, 19 },   // Sus2
        { 0,  5,  7, 12, 17, 19 },   // Sus4
        { 0,  4,  7, 11, 14, 19 },   // Maj7
        { 0,  3,  7, 10, 12, 15 },   // Min7
        { 0,  3,  7, 10, 14, 19 },   // Min9
        { 0,  4,  7, 10, 12, 16 },   // Dom7
        { 0,  3,  6,  7, 10, 14 },   // Ukrainian Dorian colour (b3, #4, b7, 9)
        { 0,  1,  4,  7, 10, 12 },   // Hijaz / Phrygian dominant
    };
    static const float kStringDetune[6] = { 0.0f, 1.5f, -1.5f, 1.0f, -1.0f, 0.5f };   // cents
}

//==============================================================================
inline void Engine::prepare (double sampleRate, int /*numChannels*/)
{
    fs = sampleRate;
    fsf = (float) sampleRate;

    int lat = 0;
    for (int p = 0; p < kPaths; ++p)
        lat = std::max (lat, paths[p].prepareOversampling (fs, p, kSub));
    latencyMax = lat;
    for (int p = 0; p < kPaths; ++p)
        paths[p].setCommonLatency (latencyMax);
    // the Lamp (lead) voice sits on an integer pre-delay that leaves room for its vibrato;
    // the dry path and the reported latency include it, so dry and lead line up exactly
    lampPre = std::max (1, (int) std::lrint (0.0008 * fs));
    lampBaseMs = 1000.0f * (float) lampPre / fsf;
    totalLat = latencyMax + lampPre;

    for (auto& c : ch)
    {
        c.dry.prepare (totalLat + 8);
        c.ctlRestore.prepare (totalLat + kSub + 8);
        c.ctlFocus.prepare (totalLat + kSub + 8);
        c.ctlInkWet.prepare (totalLat + kSub + 8);
        c.ctlInkHarp.prepare (totalLat + kSub + 8);
        c.ghostLP.lowpass (320.0, 0.707, fs);
        c.ghostSubLP.lowpass (700.0, 0.707, fs);
        c.ghostUpLP.lowpass (1000.0, 0.707, fs);
        c.ghostEnv.set (1.0f, 50.0f, fsf);
        c.ghostUpHP.setCutoff (40.0f, fsf);
        c.crushEnv.set (1.0f, 60.0f, fsf);
        c.dc.setCutoff (20.0f, fsf);
        c.smLP.lowpass (4200.0, 0.707, fs);
        c.smEnvLo.set (1.0f, 40.0f, fsf);
        c.smEnvHi.set (1.0f, 40.0f, fsf);
        c.harpHP.highpass (150.0, 0.707, fs);
        c.harpLP.lowpass (3000.0, 0.707, fs);
        c.harp.prepare (fsf);
        c.fl.prepare ((int) (0.024f * fsf) + 16);
        c.flLP.setCutoff (6000.0f, fsf);
        c.clipEnv.set (1.0f, 80.0f, fsf);
        c.rsHP.highpass (110.0, 0.707, fs);
        c.rsShelf.highShelf (9000.0, -6.0, fs);
    }
    rsSideHP.setCutoff (250.0f, fsf);
    doorG.prepare ((int) std::ceil (0.35f * fsf) + 8);
    doorF.prepare ((int) std::ceil (0.35f * fsf) + 8);
    aDoor = smoothCoef (2.0f, fsf);
    {
        const float nyq = 0.45f * fsf;
        const float spF[ImgDisperse::kN] = { 150.0f, 420.0f, 1100.0f, 2900.0f, 6800.0f, std::min (13500.0f, nyq) };
        const float rcF[ImgDisperse::kN] = { 230.0f, 610.0f, 1600.0f, 4100.0f, 9200.0f, std::min (16500.0f, nyq) };
        imgSpread.design (spF, fs);
        imgRecover.design (rcF, fs);
    }
    aImg = smoothCoef (25.0f, fsf);
    for (int i = 0; i < kImg; ++i)
    {
        imgPreL[i].store (0.0f); imgPreR[i].store (0.0f);
        imgPostL[i].store (0.0f); imgPostR[i].store (0.0f);
    }
    compInt.design (InnerControls::kHarm);
    compAll.design (InnerControls::kBands);
    aGnSlow = smoothCoef (800.0f, fsf);
    aPushAtk = smoothCoef (5.0f, fsf);
    aPushRel = smoothCoef (80.0f, fsf);
    for (int h = 0; h < InnerControls::kBands; ++h)
    {
        guiOtLevel[h].store (-120.0f);
        guiOtGain[h].store (1.0f);
        guiOtFit[h].store (0.0f);
    }
    for (int i = 0; i < kScope; ++i)
    {
        scopeIn[i].store (0.0f);
        scopeOut[i].store (0.0f);
    }

    envCons.set (5.0f, 150.0f, fsf);
    inkFast.set (0.5f, 15.0f, fsf);
    inkSlow.set (25.0f, 150.0f, fsf);
    vEnvLo.set (1.0f, 20.0f, fsf);
    vEnvHi.set (1.0f, 20.0f, fsf);
    vLP.lowpass (1500.0, 0.707, fs);
    vHP.highpass (4500.0, 0.707, fs);

    aS     = smoothCoef (15.0f, fsf);
    aCtl   = smoothCoef (4.0f, fsf);
    aInk   = smoothCoef (3.0f, fsf);
    aGn    = smoothCoef (8.0f, fsf);
    aWarp  = smoothCoef (20.0f, fsf);
    aConf  = smoothCoef (12.0f, fsf);
    aVfast = timeCoef (2.0f, fsf);
    aVslow = timeCoef (15.0f, fsf);
    aSmAtk = timeCoef (1.0f, fsf);
    aSmRel = timeCoef (40.0f, fsf);
    minOnsetGap = (long) (0.07 * fs);
    ligL.prepare ((int) (1.6 * fs) + 8);
    ligR.prepare ((int) (1.6 * fs) + 8);
    for (int c = 0; c < 2; ++c)
    {
        ligLP[c].setCutoff (3500.0f, fsf);
        ligHP[c].setCutoff (180.0f, fsf);
    }
    aLigT = smoothCoef (80.0f, fsf);
    aEcho.prepare (fsf);
    aForest.prepare (fsf);
    aForest.modDepth = 0.00045f * fsf;
    aTwin.prepare (fsf);
    duckKey.prepare (fsf);
    duckLig.prepare (fsf);
    duckAura.prepare (fsf);
    for (auto& g : guiDuck) g.store (1.0f);
    fxHP.highpass (120.0, 0.707, fs);
    fxLP.lowpass (9000.0, 0.707, fs);
    loudFall = 8.0f / fsf;            // the loudness reference falls 8 dB per second
    forestVersion = -1;
    aDuckAtk = timeCoef (8.0f, fsf);
    aDuckRel = timeCoef (150.0f, fsf);
    warmLen = (int) (0.05 * fs) + latencyMax;
    fadeLen = std::max (1, (int) (0.03 * fs));

    // probe: a vocal-like periodic wave, peak = reference level
    {
        static const float amp[6] = { 1.0f, 0.6f, 0.45f, 0.3f, 0.2f, 0.12f };
        static const float phs[6] = { 0.0f, 0.7f, 1.9f, 0.4f, 2.6f, 1.1f };
        float peak = 0.0f;
        for (int i = 0; i < 64; ++i)
        {
            float s = 0.0f;
            for (int h = 0; h < 6; ++h)
                s += amp[h] * std::sin (kTwoPi * (float) (h + 1) * (float) i / 64.0f + phs[h]);
            probe[i] = s;
            peak = std::max (peak, std::fabs (s));
        }
        const float ref = dbToGain (kRefDb);
        for (float& p : probe)
            p *= ref / peak;
    }

    for (auto& g : guiPhase)
        g.store (0.0f);

    lastLowCut = lastHighCut = -1.0f;
    lastTilt = lastTone = 1.0e9f;
    lastDecay = lastBright = lastHarpDrive = lastWidth = -1.0f;
    lastDetune = -1.0e9f;
    lastKey = lastChord = lastReg = -1;
    reset();
}

inline void Engine::reset() noexcept
{
    for (auto& p : paths)
        p.reset();
    for (auto& c : ch)
    {
        c.dry.reset();
        c.ctlRestore.reset(); c.ctlFocus.reset(); c.ctlInkWet.reset(); c.ctlInkHarp.reset();
        c.preHP.reset(); c.preLP.reset(); c.tiltLo.reset(); c.tiltHi.reset();
        for (float& z : c.apZ) z = 0.0f;
        c.ghostLP.reset(); c.ghostSubLP.reset(); c.ghostUpLP.reset(); c.ghostEnv.reset(); c.ghostUpHP.reset();
        c.zPrev = 0.0f; c.flip = 1.0f; c.armed = false;
        c.crossCount = 0; c.f0 = 150.0f; c.fMeas = 150.0f; c.noteSemi = -9.9f; c.lastDelta = 0.0f; c.pitchConf = 0.0f; c.newPeriod = false; c.crushEnv.reset();
        c.dLampS = lampBaseMs; c.dGlassS = prm[P_GTIME] > 0.0f ? prm[P_GTIME] : 23.0f;
        c.dFoldS = prm[P_FTIME] > 0.0f ? prm[P_FTIME] : 37.0f; c.dDopS = 0.0f; c.dDopFS = 0.0f;
        c.dc.reset();
        c.smLP.reset(); c.smEnvLo.reset(); c.smEnvHi.reset(); c.smGain = 1.0f;
        c.harpHP.reset(); c.harpLP.reset(); c.harp.reset();
        c.toneLo.reset(); c.toneHi.reset();
        c.fl.reset(); c.flLP.reset(); c.flT = 0.001f * fsf; c.flLive = false; c.clipEnv.reset();
        c.fitF0 = -1.0f; c.fitVersion = -1; c.compDirty = true;
        c.rsHP.reset(); c.rsShelf.reset();
    }
    rsSideHP.reset();
    rsMix = prm[P_REVSAFE] > 0.5f ? 1.0f : 0.0f;
    rsLive = false;
    pushS = 0.0f;
    gNslowDb = 0.0f;
    floatPhG = 0.0f; floatPhF = 0.25f; doorGs = doorFs = 1.0f;
    doorG.reset(); doorF.reset();
    resetImager();
    imgMix = 0.0f; imgQuiet = 0;
    for (int b = 0; b < 4; ++b) { imgG[b] = 1.0f; imgSp[b] = 0.0f; }
    imgRec = 0.0f;
    for (auto& x : imgX) x = -1.0f;
    mon = CoreMonitor();
    for (float& l : otLevelS) l = -120.0f;
    envCons.reset(); inkFast.reset(); inkSlow.reset(); vEnvLo.reset(); vEnvHi.reset();
    vLP.reset(); vHP.reset();
    gNdb = 0.0f; voiced = 1.0f; strokeRem = 0.0f; lastEnvDb = -100.0f;
    sinceOnset = 1L << 30;
    inkDriveDb = 0.0f; inkDop = inkWet = inkHarp = 1.0f;
    warpMix = 0.0f;
    slS1 = slS2 = 0.0f;
    ligL.reset(); ligR.reset();
    for (int c = 0; c < 2; ++c) { ligLP[c].reset(); ligHP[c].reset(); }
    ligTL = ligTR = 0.0f;
    ligDuck = 1.0f;
    loudRef = -70.0f;
    aEcho.reset(); aForest.reset(); aTwin.reset(); fxHP.reset(); fxLP.reset();
    duckKey.reset(); duckLig.reset(); duckAura.reset();
    echoLive = forestLive = twinLive = false;
    nextPath = -1; warmLeft = fadeLeft = 0;
    autoStable = 0;
    first = true;
    snap = true;
}

//==============================================================================
inline void Engine::computeProbe() noexcept
{
    const float drv = tDrive, grit = prm[P_GRIT], bias = prm[P_BIAS];
    const float c0 = 1.2f * bias, tc0 = fastTanh (c0);
    const float driveG = dbToGain (prm[P_GDRIVE]), driveF = dbToGain (prm[P_FDRIVE]);
    const float kG = 0.65f * drv * driveG;
    const float kF = std::min (24.0f * std::max (1.0f, driveF), drv * (0.6f + 2.4f * grit) * driveF);
    const float s0 = 0.45f * grit, ss0 = fastSin (s0);

    const float gl = prm[P_LAMP], gg = prm[P_GLASS], gf = prm[P_FOLD];
    const float powSum = std::sqrt (gl * gl + gg * gg + gf * gf);
    mixNorm = 1.0f / std::max (1.0f, powSum);

    if (prm[P_AUTOGAIN] < 0.5f)
    {
        compLamp = compGlass = compFold = 1.0f;
        tComp = 1.0f;
        return;
    }

    auto acRms = [] (const float* x, int n)
    {
        double s = 0.0, s2 = 0.0;
        for (int i = 0; i < n; ++i) { s += x[i]; s2 += (double) x[i] * x[i]; }
        const double m = s / n;
        return (float) std::sqrt (std::max (1.0e-12, s2 / n - m * m));
    };

    float bl[64], bg[64], bf[64];
    for (int i = 0; i < 64; ++i)
    {
        bl[i] = shapeLamp (probe[i], drv, c0, tc0);
        bg[i] = shapeGlass (probe[i], kG);
        bf[i] = shapeFold (probe[i], kF, s0, ss0);
    }
    const float rIn = acRms (probe, 64);
    compLamp  = clampf (rIn / acRms (bl, 64), 0.05f, 20.0f);
    compGlass = clampf (rIn / acRms (bg, 64), 0.05f, 20.0f);
    compFold  = clampf (rIn / acRms (bf, 64), 0.05f, 20.0f);

    // clipper on a representative waveform at the expected (decorrelated) mix level
    const float knee = 1.0f - clampf (prm[P_SHOULDER], 0.0f, 1.0f);
    const float clipGain = dbToGain (grit * 15.0f);
    const float clipBaseP = dbToGain (18.0f * prm[P_CLIP]);
    const float level = std::min (1.0f, powSum) ;
    float y[64];
    for (int i = 0; i < 64; ++i)
        y[i] = shapeClip (bl[i] * compLamp * level, 2.0f * clipGain * clipBaseP, knee, 0.5f / (clipGain * clipBaseP));
    const float rY = acRms (y, 64);
    tComp = clampf ((rIn * std::max (level, 0.05f)) / rY, 0.1f, 10.0f);
}

inline int Engine::desiredPath (int n) noexcept
{
    const int sel = std::min (std::max ((int) (prm[P_OS] + 0.5f), 0), 5);
    if (sel > 0)
        return sel - 1;

    const float dEff = clampf (prm[P_DRIVE] + (1.0f - prm[P_CONSIST]) * std::max (0.0f, lastEnvDb - kRefDb) / 36.0f, 0.0f, 1.0f);
    const float grit = prm[P_GRIT];
    const float harsh = std::max (std::max (std::max (std::max (dEff, 0.85f * grit), 0.7f * prm[P_CRUSH]), 0.75f * prm[P_CLIP]),
                                  std::max (0.6f * prm[P_DOPPLER] * prm[P_GLASS] + 0.3f * dEff,
                                            prm[P_FOLD] * (0.4f + 0.6f * grit) * (0.3f + dEff)));
    // never below ~88 kHz inside (keeps the modulated delays clean), more as the distortion grows
    const float target = harsh < 0.30f ? 96000.0f : (harsh < 0.62f ? 192000.0f : 384000.0f);

    int idx = 0;
    while (idx < kPaths - 1 && fsf * (float) (1 << idx) < 0.9f * target)
        ++idx;
    if (offline && harsh >= 0.03f && idx < kPaths - 1)
        ++idx;   // rendering: one step better

    if (first || offline)
    {
        autoChoice = autoPending = idx;
        autoStable = 0;
        return idx;
    }
    if (idx != autoPending) { autoPending = idx; autoStable = 0; }
    else if (autoStable < (1L << 30)) autoStable += n;

    const long need = (long) ((autoPending > autoChoice ? 0.05 : 1.5) * fs);
    if (autoPending != autoChoice && autoStable >= need)
        autoChoice = autoPending;
    return autoChoice;
}

inline void Engine::updateBlock() noexcept
{
    const float* v = prm;
    tDrive   = dbToGain (v[P_DRIVE] * 36.0f);
    const float mix = clampf (v[P_MIX], 0.0f, 1.0f);
    tDryG    = std::pow (1.0f - mix, 0.75f);
    tWetG    = std::pow (mix, 0.75f);
    tOut     = dbToGain (v[P_OUTPUT]);
    tLamp    = v[P_LAMP];
    tGlass   = v[P_GLASS];
    tFold    = v[P_FOLD];
    tHarpLvl = v[P_MELODY] * 1.1f;
    tEcho    = v[P_ECHO] * v[P_ECHO];
    tForest  = v[P_FOREST] * v[P_FOREST];
    tTwin    = v[P_TWIN] * v[P_TWIN];
    tHarpExc = 0.5f + v[P_MELODY];

    for (auto& c : ch)
    {
        if (differs (v[P_LOWCUT], lastLowCut))   c.preHP.highpass (v[P_LOWCUT], 0.707, fs);
        if (differs (v[P_HIGHCUT], lastHighCut)) c.preLP.lowpass (v[P_HIGHCUT], 0.707, fs);
        if (differs (v[P_PRETILT], lastTilt))
        {
            c.tiltLo.lowShelf (400.0, -0.5 * v[P_PRETILT], fs);
            c.tiltHi.highShelf (1500.0, 0.5 * v[P_PRETILT], fs);
        }
        if (differs (v[P_TONE], lastTone))
        {
            c.toneLo.lowShelf (250.0, -4.0 * v[P_TONE], fs);
            c.toneHi.highShelf (3000.0, 8.0 * v[P_TONE], fs);
        }
    }
    lastLowCut = v[P_LOWCUT]; lastHighCut = v[P_HIGHCUT]; lastTilt = v[P_PRETILT]; lastTone = v[P_TONE];

    const float sm = v[P_SMOOTH];
    smThr = lerpf (0.70f, 0.12f, sm);
    smExp = 0.6f + 0.8f * sm;
    smStatic = 1.0f - 0.3f * sm;
    harpShim = 0.0008f * v[P_MOTION];

    // harp tuning
    const int key = std::min (std::max ((int) (v[P_KEY] + 0.5f), 0), 11);
    const int chord = std::min (std::max ((int) (v[P_CHORD] + 0.5f), 0), 11);
    const int reg = std::min (std::max ((int) (v[P_REGISTER] + 0.5f), 0), 2);
    if (key != lastKey || chord != lastChord || reg != lastReg || differs (v[P_WIDTH], lastWidth)
        || differs (v[P_DECAY], lastDecay) || differs (v[P_BRIGHT], lastBright) || differs (v[P_HARPDRIVE], lastHarpDrive)
        || differs (v[P_DETUNE], lastDetune))
    {
        // Detune: the whole key grid (strings, forest, tuned flanger, the melodic distortion) moves with the beat's tuning
        tuneSemis = clampf (v[P_DETUNE], -100.0f, 100.0f) * 0.01f;
        const int root = 36 + 12 * reg + key;
        numChordPc = 0;
        for (int i = 0; i < Harp::kStrings; ++i)
        {
            const int pc = (key + tables::kChords[chord][i]) % 12;
            bool known = false;
            for (int q = 0; q < numChordPc; ++q)
                known = known || (int) chordPc[q] == pc;
            if (! known)
                chordPc[numChordPc++] = (float) pc;
        }
        for (int c = 0; c < 2; ++c)
        {
            float hz[Harp::kStrings];
            for (int i = 0; i < Harp::kStrings; ++i)
            {
                float cents = tables::kStringDetune[i];
                if (chord == 0 && i >= 3) cents += 4.0f;
                if (c == 1) cents += 5.0f * v[P_WIDTH];
                const float midi = (float) (root + tables::kChords[chord][i]) + 0.01f * cents + tuneSemis;
                hz[i] = 440.0f * std::exp2 ((midi - 69.0f) / 12.0f);
            }
            ch[c].harp.configure (hz, v[P_DECAY], v[P_BRIGHT], v[P_HARPDRIVE]);
            if (c == 0)
            {
                for (int i = 0; i < Harp::kStrings; ++i)
                    harpHz[i] = hz[i];
                // Tuned flanger: comb loop delays on the first four chord tones (minus the loop's extra delay)
                const float aLp = std::exp (-kTwoPi * 6000.0f / fsf);
                for (int i = 0; i < 4; ++i)
                {
                    const float w = kTwoPi * hz[i] / fsf;
                    const float pd = std::atan2 (aLp * std::sin (w), 1.0f - aLp * std::cos (w)) / w;
                    flTune[i] = clampf (fsf / hz[i] - 1.0f - pd, 1.0f, 0.022f * fsf);
                }
            }
        }
        ++harpVersion;
        lastKey = key; lastChord = chord; lastReg = reg; lastWidth = v[P_WIDTH];
        lastDecay = v[P_DECAY]; lastBright = v[P_BRIGHT]; lastHarpDrive = v[P_HARPDRIVE]; lastDetune = v[P_DETUNE];
    }

    const int ftype = std::min (std::max ((int) (v[P_FORESTTYPE] + 0.5f), 0), 3);
    if (forestVersion != harpVersion || ftype != lastForestType
        || differs (v[P_FORESTDECAY], lastForestDecay) || differs (v[P_FORESTTUNE], lastForestTune))
    {
        aForest.configure (ftype, v[P_FORESTDECAY], v[P_FORESTTUNE], harpHz, 6);
        forestVersion = harpVersion; lastForestType = ftype;
        lastForestDecay = v[P_FORESTDECAY]; lastForestTune = v[P_FORESTTUNE];
    }

    computeProbe();

    if (snap)
    {
        sDrive = tDrive; sDryG = tDryG; sWetG = tWetG; sOut = tOut; sLamp = tLamp; sGlass = tGlass; sFold = tFold;
        sComp = tComp; sHarpLvl = tHarpLvl; sHarpExc = tHarpExc;
        sEcho = tEcho; sForest = tForest; sTwin = tTwin;
        warpMix = (v[P_WARP] * v[P_MOTION] > 0.001f) ? 1.0f : 0.0f;
        snap = false;
    }
}

//==============================================================================
inline void Engine::processSub (float* const* io, int numCh, int off, int n) noexcept
{
    updateBlock();

    const float* v = prm;
    const float motion = v[P_MOTION];
    const float vib = v[P_VIBRATO] * motion, ens = v[P_ENSEMBLE] * motion, breath = v[P_BREATH] * motion;
    const float pulse = v[P_PULSE] * motion, orbit = v[P_ORBIT] * motion, warpD = v[P_WARP] * motion;
    const float width = v[P_WIDTH];
    const int ratioIdx = std::min (std::max ((int) (v[P_RATIO] + 0.5f), 0), 7);
    const int rateIdx = std::min (std::max ((int) (v[P_RATE] + 0.5f), 0), 6);
    const double bpc = tables::kBeatsPerCycle[rateIdx];
    const float* rr = tables::kRatios[ratioIdx];
    const double dppq = hostBpm / (60.0 * fs);
    const float consist = v[P_CONSIST], focus = v[P_FOCUS];
    const float ink = v[P_INK], inkA = std::fabs (ink);
    const float inkDecay = std::exp (-2.5f / (0.001f * std::max (v[P_INKTIME], 10.0f) * fsf));
    const float gd = v[P_GHOSTDOWN], gu = v[P_GHOSTUP];
    const float doppler = v[P_DOPPLER], bias = v[P_BIAS], grit = v[P_GRIT];
    // THE DOUBLES, finely: time (free or on the beat), repeats, float, tone, drive, attack
    static const float kSyncQuarters[7] = { 0.0f, 0.0625f, 0.125f, 1.0f / 6.0f, 0.25f, 1.0f / 3.0f, 0.5f };
    auto doubleTime = [&] (int timeP, int syncP)
    {
        const int s = std::min (std::max ((int) (v[syncP] + 0.5f), 0), 6);
        return s == 0 ? v[timeP] : std::min (300.0f, kSyncQuarters[s] * (float) (60000.0 / hostBpm));
    };
    const float timeG = doubleTime (P_GTIME, P_GSYNC), timeF = doubleTime (P_FTIME, P_FSYNC);
    const float driveG = dbToGain (v[P_GDRIVE]), driveF = dbToGain (v[P_FDRIVE]);
    // the slow float: depth in cents -> how far the read swings (ms) at that rate (a swing of 15 ms at most)
    const float floatG = std::min (15.0f, 1000.0f * v[P_GFLOAT] * 0.000578f / (kTwoPi * std::max (0.05f, v[P_GRATE])));
    const float floatF = std::min (15.0f, 1000.0f * v[P_FFLOAT] * 0.000578f / (kTwoPi * std::max (0.05f, v[P_FRATE])));
    const float floatIncG = v[P_GRATE] / fsf, floatIncF = v[P_FRATE] / fsf;
    const float attG = v[P_GATTACK], attF = v[P_FATTACK];
    const float openG = 1.0f / (0.001f * std::max (5.0f, v[P_GOPEN]) * fsf), openF = 1.0f / (0.001f * std::max (5.0f, v[P_FOPEN]) * fsf);
    const float doorDG = 0.001f * timeG * fsf, doorDF = 0.001f * timeF * fsf;
    // the door of a double after an onset: -1 = closed on the attack, opens after it; +1 = only the attack
    auto door = [] (float att, float since, float inv) noexcept
    {
        if (std::fabs (att) < 1.0e-6f) return 1.0f;
        const float d = 1.0f - std::exp (-since * inv);
        return att < 0.0f ? 1.0f + att * (1.0f - d) : (1.0f + att * (1.0f - d)) * (1.0f - 0.5f * att * d);
    };
    const float clipGain = dbToGain (grit * 15.0f);
    const float clipIn = 2.0f * clipGain, clipOut = 0.5f / clipGain, clipH = 1.0f - clampf (v[P_SHOULDER], 0.0f, 1.0f);
    const float clipP = v[P_CLIP], clipBase = dbToGain (18.0f * clipP);
    const float warpTarget = (warpD > 0.001f) ? 1.0f : 0.0f;
    const bool stereo = numCh > 1;
    // MELODIC DISTORTION: Sing (the chord tones the core adds), Dirt (everything else it adds), Force (the dirt
    // follows how hard the voice pushes), Prism (the chord tones refract around the head), the overtone map
    constexpr int kHarm = InnerControls::kHarm, kBands = InnerControls::kBands;
    const float sing = v[P_SING], dirt = v[P_DIRT], force = v[P_FORCE];
    const bool revSafe = v[P_REVSAFE] > 0.5f;
    // POLISH: the lower Drive or Mix, the more MAVKA turns from distortion into polish (no extra knob): the dirt
    // steps back, the lowest chord tones bloom, the wet layer gets denser, the chord tones refract a little wider
    const float lowMix = smoothstep (0.5f, 0.1f, v[P_MIX]), lowDrive = smoothstep (0.35f, 0.05f, v[P_DRIVE]);
    const float polish = std::max (lowMix, lowDrive);
    const float prism = stereo ? std::min (1.0f, v[P_PRISM] + 0.25f * polish) : 0.0f;
    const float densAmt = lowMix;
    float otUser[kBands], otPan[kBands] {};
    bool identity = std::fabs (sing - 1.0f) < 1.0e-4f && std::fabs (dirt - 1.0f) < 1.0e-4f && force < 1.0e-4f
                    && v[P_PRISM] < 1.0e-4f && ! revSafe;
    for (int h = 0; h < kBands; ++h)
    {
        otUser[h] = h < kHarm ? dbToGain (v[P_OT2 + h]) : 1.0f;
        if (h < kHarm)
            identity = identity && std::fabs (v[P_OT2 + h]) < 0.01f;
        otGainAcc[h] = 0.0f;
    }
    const float hOn = identity ? 0.0f : 1.0f;   // the plain distortion: nothing to split
    const bool meter = metering.load (std::memory_order_relaxed);
    // the filters also run (with nothing to change) while the overtone map is on screen, so it shows the plain peaks
    const float hRun = (hOn > 0.0f || meter) ? 1.0f : 0.0f;
    // the ghosts feed the core too: what they add (the sub's partials, the octave-up's overtones) is split as well
    const bool ghosts = v[P_GHOSTDOWN] > 1.0e-4f || v[P_GHOSTUP] > 1.0e-4f;
    const float hHalf = (hRun > 0.0f && v[P_GHOSTDOWN] > 1.0e-4f) ? 1.0f : 0.0f;
    float clipDyn = 1.0f;
    const float slant = v[P_SLANT];
    const float crushParam = v[P_CRUSH];
    const int crushMode = std::min (std::max ((int) (v[P_CRUSHMODE] + 0.5f), 0), 2);
    const float slantTime = std::max (0.015f, 0.001f * v[P_INKTIME] / 3.0f);
    const float aSl = std::exp (-1.0f / (slantTime * fsf));
    const float slantImpulse = 2.5f * slant * 2.718282f;   // peak of the gesture = 2.5 ms * slant

    for (int j = 0; j < n; ++j)
    {
        // ---- chord clock
        double ppq;
        if (hostPlaying) { ppq = hostPpq + (double) blockPos * dppq; freePpq = ppq; }
        else             { freePpq += dppq; ppq = freePpq; }
        ++blockPos;

        float lfoL[4], lfoR[4];
        for (int i = 0; i < 4; ++i)
        {
            double ph = ppq / bpc * (double) rr[i] + (ratioIdx == 0 ? (double) tables::kUnisonOffset[i] : 0.0);
            ph -= std::floor (ph);
            const float phf = (float) ph;
            lfoL[i] = fastSin (kTwoPi * phf);
            lfoR[i] = fastSin (kTwoPi * (phf + width * tables::kStereoOffset[i]));
            if (j == n - 1)
                guiPhase[i].store (phf, std::memory_order_relaxed);
            if (i == 0)
                phase0Buf[j] = phf;
        }
        // Prism: every overtone the core adds has its own place around the head; the pattern sways with the clock
        if (j == 0 && prism > 0.0f)
            for (int h = 0; h < kBands; ++h)
                otPan[h] = fastSin (2.39996323f * 2.0f * inkey::kMult[h] + kPi * motion * lfoL[0]);

        sDrive += (tDrive - sDrive) * aS;
        sLamp  += (tLamp - sLamp) * aS;
        sGlass += (tGlass - sGlass) * aS;
        sFold  += (tFold - sFold) * aS;

        // ---- input, latency-aligned dry, pre EQ
        float xe[2] = { 0.0f, 0.0f };
        for (int c = 0; c < numCh; ++c)
        {
            Chan& C = ch[c];
            const float x = io[c][off + j];
            C.dry.push (x);
            dryOut[c][j] = C.dry.tap (totalLat);
            xe[c] = C.tiltHi.process (C.tiltLo.process (C.preLP.process (C.preHP.process (x))));
        }
        const float mono = stereo ? 0.5f * (xe[0] + xe[1]) : xe[0];

        // ---- analysis: consistency (level-independent drive)
        const float envDb = gainToDb (envCons.process (mono));
        lastEnvDb = envDb;
        // recent loudness of this voice: quiet and loud recordings behave the same,
        // and room noise far below the voice is never pulled up
        loudRef = envDb > loudRef ? envDb : std::max (-70.0f, loudRef - loudFall);
        const float rel = envDb - loudRef;
        const float gateW = smoothstep (-66.0f, -48.0f, envDb) * smoothstep (-42.0f, -26.0f, rel);
        // Force: how hard the voice pushes right now (its loudness against its own recent loudest)
        const float pushT = smoothstep (-18.0f, -1.0f, rel);
        pushS += (pushT - pushS) * (pushT > pushS ? aPushAtk : aPushRel);
        const float gTarget = consist * clampf (kRefDb - envDb, -18.0f, 24.0f) * gateW;
        gNdb += (gTarget - gNdb) * aGn;
        gNslowDb += (gNdb - gNslowDb) * aGnSlow;
        const float gN = dbToGain (gNdb);
        // POLISH density (low Mix): the wet layer keeps part of the fast leveling, like parallel compression -
        // the quiet syllables get denser, the dry voice keeps its dynamics
        const float restore = (1.0f / gN) * (densAmt > 0.0f ? dbToGain (densAmt * (gNdb - gNslowDb)) : 1.0f);

        // ---- vowel / consonant detector
        const float eLo = vEnvLo.process (vLP.process (mono));
        const float eHi = vEnvHi.process (vHP.process (mono));
        const float rr2 = eHi / (eLo + eHi + 1.0e-7f);
        const float vt = 1.0f - smoothstep (0.22f, 0.55f, rr2);
        voiced = vt + (vt < voiced ? aVfast : aVslow) * (voiced - vt);

        // ---- Ink: syllable onsets drive the handwriting stroke
        const float dF = gainToDb (inkFast.process (mono));
        const float dS = gainToDb (inkSlow.process (mono));
        if (sinceOnset > minOnsetGap && dF > std::max (-60.0f, loudRef - 30.0f) && dF - dS > 7.0f)
        {
            sinceOnset = 0;
            strokeRem = 1.0f;
            slS1 += slantImpulse;
        }
        else
        {
            if (sinceOnset < (1L << 30))
                ++sinceOnset;
            strokeRem *= inkDecay;
        }
        strokeBuf[j] = strokeRem;
        // THE DOUBLES: the slow float, and each double's door - delayed by its own time, so it shapes
        // the double's own copy of the attack
        floatPhG += floatIncG; floatPhG -= std::floor (floatPhG);
        floatPhF += floatIncF; floatPhF -= std::floor (floatPhF);
        doorG.push (door (attG, (float) sinceOnset, openG));
        doorF.push (door (attF, (float) sinceOnset, openF));
        doorGs += ((std::fabs (attG) < 1.0e-6f ? 1.0f : doorG.read (std::max (1.0f, doorDG))) - doorGs) * aDoor;
        doorFs += ((std::fabs (attF) < 1.0e-6f ? 1.0f : doorF.read (std::max (1.0f, doorDF))) - doorFs) * aDoor;
        if (std::fabs (attG) < 1.0e-6f && std::fabs (doorGs - 1.0f) < 1.0e-5f) doorGs = 1.0f;
        if (std::fabs (attF) < 1.0e-6f && std::fabs (doorFs - 1.0f) < 1.0e-5f) doorFs = 1.0f;
        const float flSinG = floatG > 0.0f ? std::sin (kTwoPi * floatPhG) : 0.0f, flCosG = floatG > 0.0f ? std::cos (kTwoPi * floatPhG) : 0.0f;
        const float flSinF = floatF > 0.0f ? std::sin (kTwoPi * floatPhF) : 0.0f, flCosF = floatF > 0.0f ? std::cos (kTwoPi * floatPhF) : 0.0f;
        slS1 *= aSl;
        slS2 = aSl * slS2 + (1.0f - aSl) * slS1;
        const float slantMs = clampf (slS2, -6.0f, 6.0f);

        // ducking for Ligature / Aura: relative to the voice's own recent loudness,
        // so quiet and loud recordings behave the same
        const float duckT = 1.0f - 0.92f * smoothstep (-30.0f, -12.0f, rel) * smoothstep (-62.0f, -50.0f, envDb);
        breathe[j] = smoothstep (-24.0f, 0.0f, rel);
        auraA[j] = lfoL[1];
        auraB[j] = lfoL[2];
        auraC[j] = lfoL[0];
        ligDuck = duckT + (duckT < ligDuck ? aDuckAtk : aDuckRel) * (ligDuck - duckT);
        duckBuf[j] = ligDuck;
        float tDriveDb, tDop, tWet, tHarp;
        if (ink >= 0.0f)
        {   // Ink: starts clean, the distortion spreads over the word
            tDriveDb = -20.0f * inkA * strokeRem;
            tDop  = 1.0f - 0.85f * inkA * strokeRem;
            tWet  = 1.0f - 0.55f * inkA * strokeRem;
            tHarp = 1.0f - 0.70f * inkA * strokeRem;
        }
        else
        {   // Press: the pen hits hard, then lifts off
            const float post = 1.0f - strokeRem;
            tDriveDb = inkA * (12.0f * strokeRem - 3.0f * post);
            tDop  = 1.0f + inkA * strokeRem;
            tWet  = 1.0f + inkA * (0.5f * strokeRem - 0.15f * post);
            tHarp = 1.0f + 0.8f * inkA * strokeRem;
        }
        inkDriveDb += (tDriveDb - inkDriveDb) * aInk;
        inkDop     += (tDop - inkDop) * aInk;
        inkWet     += (tWet - inkWet) * aInk;
        inkHarp    += (tHarp - inkHarp) * aInk;
        const float drv = sDrive * dbToGain (inkDriveDb);
        // the dirt the core may add right now (Reverb safe: only on the attack of a word, so no reverb smears it)
        const float dirtNow = dirt * (1.0f - force + force * pushS) * (1.0f - 0.75f * polish)
                              * (revSafe ? smoothstep (0.05f, 0.6f, strokeRem) : 1.0f);

        warpMix += (warpTarget - warpMix) * aWarp;

        for (int c = 0; c < numCh; ++c)
        {
            Chan& C = ch[c];
            const float* lfo = (c == 0) ? lfoL : lfoR;
            float x = xe[c] * gN;

            // Warp: chord-clocked allpass chain (changes the waveform the shapers see)
            if (warpMix > 1.0e-4f)
            {
                const float fc = std::min (650.0f * std::exp2 (1.6f * warpD * lfo[0]), 0.4f * fsf);
                const float t = std::tan (kPi * fc / fsf);
                const float a = (t - 1.0f) / (t + 1.0f);
                float y = x;
                for (int s = 0; s < 4; ++s)
                {
                    const float o = a * y + C.apZ[s];
                    C.apZ[s] = y - a * o;
                    y = o;
                }
                x += warpMix * (y - x);
            }

            // Ghosts: flip-flop sub octave (with its natural fifth) + rectified octave up.
            // They go INTO the shapers, so the distortion intermodulates as a chord.
            const float z = C.ghostLP.process (x);
            const float thr = 0.12f * C.ghostEnv.process (z) + 1.0e-6f;
            if (z < -thr) C.armed = true;
            ++C.crossCount;
            C.newPeriod = false;
            if (C.armed && C.zPrev < 0.0f && z >= 0.0f)
            {
                C.flip = -C.flip;
                C.armed = false;
                // pitch of the voice from the period between upward crossings (45..520 Hz), sub-sample accurate
                const float delta = C.zPrev / (C.zPrev - z + 1.0e-12f);
                const float period = (float) C.crossCount + delta - C.lastDelta;
                if (period > fsf / 520.0f && period < fsf / 45.0f)
                {
                    // the overtone bands need the pitch within a few cents, period by period (vibrato!): take every
                    // period as it comes, a little ahead (the last period's pitch is half a period old); a period
                    // far off the current pitch is believed only once the next one agrees (a new note, not a glitch)
                    const float fNew = fsf / period, ratio = fNew / C.f0;
                    bool took = false;
                    if (ratio > 0.87f && ratio < 1.15f)
                    {
                        const float slope = std::fabs (fNew / C.fMeas - 1.0f) < 0.03f ? fNew - C.fMeas : 0.0f;
                        C.f0 = fNew + kPredict * slope;
                        took = true;
                    }
                    else if (std::fabs (fNew / C.fMeas - 1.0f) < 0.06f)
                    {
                        C.f0 = fNew;
                        took = true;
                    }
                    C.fMeas = fNew;
                    if (took)
                    {
                        C.newPeriod = true;
                        // the NOTE being sung (what decides which overtones are chord tones): vibrato and scoops
                        // average out, a real new note is taken at once
                        const float semi = 12.0f * std::log2 (fNew * (1.0f / 261.6256f));
                        if (std::fabs (semi - C.noteSemi) > 1.3f) C.noteSemi = semi;
                        else C.noteSemi += std::min (1.0f, period / (0.06f * fsf)) * (semi - C.noteSemi);
                    }
                }
                C.lastDelta = delta;
                C.crossCount = 0;
            }
            if (C.crossCount > (int) fsf) C.crossCount = (int) fsf;
            C.zPrev = z;
            // how sure we are of the pitch: a period ended not long ago, on a voiced sound
            const float fresh = (float) C.crossCount < 1.7f * fsf / C.f0 ? 1.0f : 0.0f;
            C.pitchConf += (fresh * voiced - C.pitchConf) * aConf;
            const float sub = C.ghostSubLP.process (C.flip * z);
            const float up = C.ghostUpHP.hp (std::fabs (C.ghostUpLP.process (x)));
            const float ghost = voiced * (gd * 1.3f * sub + gu * 2.2f * up);
            ghostBuf[c][j] = ghost;
            x += ghost;
            coreIn[c][j] = x;

            // crusher controls (the crush follows the handwriting stroke too)
            const float cEnv = C.crushEnv.process (x);
            const float crush = std::min (1.0f, crushParam * (ink >= 0.0f ? 1.0f - 0.8f * inkA * strokeRem
                                                                            : 1.0f + 0.6f * inkA * strokeRem));
            InnerControls& kc = ctl[c][j];
            if (crush > 0.0005f)
            {
                const float bits = 12.0f - 9.0f * crush;
                float rate;
                if (crushMode == 1)
                    rate = 16000.0f * std::exp2 (-4.0f * crush);                 // Digital: fixed 16 kHz .. 1 kHz
                else
                {
                    float mult = 1.0f;                                           // Clock: resolution jumps in octaves
                    if (crushMode == 2) mult = lfo[2] > 0.5f ? 2.0f : (lfo[2] < -0.5f ? 0.5f : 1.0f);
                    // Harmonic: a WHOLE number of steps per cycle (64 .. 4), so every product is a harmonic
                    const float steps = std::max (2.0f, std::round (64.0f * std::exp2 (-4.0f * crush) * mult));
                    rate = steps * C.f0;
                }
                kc.crushRate = rate;
                kc.crushSync = (crushMode != 1 && C.newPeriod) ? 1.0f + (1.0f - C.lastDelta) : 0.0f;
                kc.crushStep = (crushMode == 1 ? 1.0f : cEnv + 1.0e-4f) * std::exp2 (1.0f - bits);
                kc.crushMix = std::min (1.0f, 3.0f * crush);
            }
            else
            {
                kc.crushRate = 0.0f;
                kc.crushStep = 0.0f;
                kc.crushMix = 0.0f;
                kc.crushSync = 0.0f;
            }

            // core controls
            const float dLampT  = lampBaseMs + 0.55f * vib * lfo[0];
            // (the right channel floats a quarter turn behind the left: a slow stereo sway)
            const float dGlassT = (c == 0 ? timeG : timeG + 6.0f * width) + 3.0f * ens * lfo[1]
                                  + (floatG > 0.0f ? floatG * (c == 0 ? flSinG : -flCosG) : 0.0f);
            const float dFoldT  = (c == 0 ? timeF : std::max (1.0f, timeF - 7.0f * width)) + 3.0f * ens * lfo[2]
                                  + (floatF > 0.0f ? floatF * (c == 0 ? flSinF : -flCosF) : 0.0f);
            const float dDopT   = 0.32f * doppler * inkDop * (1.0f + 0.6f * motion * lfo[3]);
            const float dDopFT  = 0.32f * v[P_FTHROW] * inkDop * (1.0f + 0.6f * motion * lfo[2]);
            C.dLampS  += (dLampT - C.dLampS) * aCtl;
            C.dGlassS += (dGlassT - C.dGlassS) * aCtl;
            C.dFoldS  += (dFoldT - C.dFoldS) * aCtl;
            C.dDopS   += (dDopT - C.dDopS) * aCtl;
            C.dDopFS  += (dDopFT - C.dDopFS) * aCtl;

            InnerControls& k = ctl[c][j];
            k.dLamp  = C.dLampS;
            k.dGlass = C.dGlassS + slantMs;          // the doubles draw the stroke,
            k.dFold  = C.dFoldS + 1.4f * slantMs;    // the lamp (lead) stays in tune
            k.dDop   = std::max (0.0f, C.dDopS);
            k.dDopF  = C.dDopFS > 1.0e-5f ? C.dDopFS : 0.0f;
            k.fbG = v[P_GREPEATS];
            k.fbF = v[P_FREPEATS];
            k.toneG = v[P_GTONE];
            k.toneF = v[P_FTONE];
            k.kLamp  = drv;
            k.cLamp  = 1.2f * bias + 0.6f * breath * lfo[1];
            k.tcLamp = fastTanh (k.cLamp);
            k.kGlass = 0.65f * drv * driveG;
            k.dopSens = 4.0f;
            k.kFold  = std::min (24.0f * std::max (1.0f, driveF), drv * (0.6f + 2.4f * grit) * driveF);
            k.sFold  = 0.45f * grit + 0.35f * grit * motion * lfo[3];
            k.ssFold = fastSin (k.sFold);

            const float panG = -0.55f * width + 0.45f * orbit * width * lfo[2];
            const float panF =  0.55f * width - 0.45f * orbit * width * lfo[1];
            const float pg = stereo ? panGain (panG, c) : 1.0f;
            const float pf = stereo ? panGain (panF, c) : 1.0f;
            const float pulG = 1.0f - pulse * (0.5f + 0.5f * lfo[2]);
            const float pulF = 1.0f - pulse * (0.5f - 0.5f * lfo[2]);
            k.gLamp  = sLamp * compLamp * mixNorm;
            k.gGlass = sGlass * compGlass * mixNorm * pg * pulG * doorGs;
            k.gFold  = sFold * compFold * mixNorm * pf * pulF * doorFs;
            // CLIP: the ceiling is pressed by the pen (Ink: bites into the body of the word, Press: slams the attack)
            // and breathes with the chord clock; level is half-compensated so the pressure is felt, not just heard louder
            const float dynDb = clipP * ((ink >= 0.0f ? 12.0f * inkA * (1.0f - strokeRem) : 12.0f * inkA * strokeRem)
                                         + 6.0f * motion * lfo[3]);
            const float dyn = dbToGain (dynDb), cd = clipBase * dyn;
            k.clipIn  = clipIn * cd;
            k.clipH   = clipH;
            k.clipOut = clipOut / cd * dbToGain (0.5f * dynDb);
            k.clipPos = 1.0f - 0.35f * clipP * voiced;
            if (c == 0)
                clipDyn = dyn;

            // MELODIC DISTORTION: tune the overtone filters to the voice; per overtone, how much of what the core
            // adds is kept - Sing for the notes of the song's chord, Dirt for the rest, times the overtone map
            k.hOn = hRun;
            k.hHalf = hHalf;
            if (hRun > 0.0f)
            {
                k.hF0 = C.f0;
                if (differs (C.noteSemi, C.fitF0) || C.fitVersion != harpVersion)
                {
                    C.fitF0 = C.noteSemi;
                    C.fitVersion = harpVersion;
                    const float pcVoice = C.noteSemi - tuneSemis;   // the sung note, semitones above C4 (on the beat's tuning)
                    for (int h = 0; h < kBands; ++h)
                    {
                        C.fitOk[h] = inkey::kMult[h] * C.f0 > inkey::kTopHz ? 0.0f : 1.0f;
                        if (inkey::isOctave (h))
                        {
                            C.fit[h] = 1.0f;   // an octave of the sung note always fits
                            continue;
                        }
                        float pc = std::fmod (pcVoice + inkey::kSemis[h], 12.0f);
                        if (pc < 0.0f) pc += 12.0f;
                        float dist = 6.0f;
                        for (int q = 0; q < numChordPc; ++q)
                        {
                            const float dd = std::fabs (pc - chordPc[q]);
                            dist = std::min (dist, std::min (dd, 12.0f - dd));
                        }
                        C.fit[h] = 1.0f - smoothstep (0.22f, 0.50f, dist);   // within ~22 cents of a chord tone
                    }
                    C.compDirty = true;
                }
                const int nb = hHalf > 0.0f ? kBands : kHarm;
                if (j == 0 || C.compDirty)
                {
                    // per block: the wanted amount per band = conf * (a - dirt * b), through the leakage compensation
                    const float side = c == 0 ? prism : -prism;
                    float a[kBands] {}, b[kBands] {};
                    for (int h = 0; h < nb; ++h)
                    {
                        const float sung = sing * (1.0f + side * otPan[h]) * (1.0f + polish * inkey::kBloom[h]);
                        a[h] = C.fitOk[h] * (1.0f - otUser[h] * C.fit[h] * sung);
                        b[h] = C.fitOk[h] * otUser[h] * (1.0f - C.fit[h]);
                        C.wantA[h] = a[h];
                        C.wantB[h] = b[h];
                    }
                    const inkey::Comp& cm = hHalf > 0.0f ? compAll : compInt;
                    cm.apply (a, C.xaP, C.xaQ);
                    cm.apply (b, C.xbP, C.xbQ);
                    C.compDirty = false;
                }
                const float conf = hOn > 0.0f ? C.pitchConf : 0.0f;   // plain: the filters only look
                for (int h = 0; h < nb; ++h)
                {
                    k.hW[h] = conf * (C.xaP[h] - dirtNow * C.xbP[h]);
                    k.hQ[h] = conf * (C.xaQ[h] - dirtNow * C.xbQ[h]);
                    if (c == 0)
                        otGainAcc[h] += 1.0f - conf * (C.wantA[h] - dirtNow * C.wantB[h]);
                }
            }

            shim[c][j] = harpShim * lfo[3];

            // controls used after the core (delayed by the core latency)
            C.ctlRestore.push (restore);
            C.ctlFocus.push (1.0f - focus * (1.0f - voiced));
            C.ctlInkWet.push (inkWet);
            C.ctlInkHarp.push (inkHarp);
        }
    }
    guiInk.store (strokeRem, std::memory_order_relaxed);
    guiVoiced.store (voiced, std::memory_order_relaxed);
    guiSlant.store (slS2, std::memory_order_relaxed);
    guiLevel.store (lastEnvDb, std::memory_order_relaxed);
    guiRel.store (lastEnvDb - loudRef, std::memory_order_relaxed);

    // ---- oversampled core with click-free factor changes (constant latency)
    const int want = desiredPath (n);
    if (first)
    {
        activePath = want;
        first = false;
    }
    if (nextPath < 0 && want != activePath)
    {
        nextPath = want;
        paths[nextPath].reset();
        warmLeft = warmLen;
        fadeLeft = fadeLen;
    }
    mon.on = meter;
    if (meter)
    {
        for (float& l : mon.level) l = 0.0f;
        mon.groups = mon.yPow = mon.peakIn = 0.0f;
    }
    for (int c = 0; c < numCh; ++c)
        paths[activePath].process (c, coreIn[c], (hRun > 0.0f && ghosts) ? ghostBuf[c] : nullptr, ctl[c], coreOutA[c],
                                   clipRes[c], n, (meter && c == 0) ? &mon : nullptr);
    if (nextPath >= 0)
    {
        for (int c = 0; c < numCh; ++c)
            paths[nextPath].process (c, coreIn[c], (hRun > 0.0f && ghosts) ? ghostBuf[c] : nullptr, ctl[c], coreOutB[c],
                                     clipResB[c], n);
        int w = warmLeft, f = fadeLeft;
        for (int j = 0; j < n; ++j)
        {
            float t;
            if (w > 0)      { t = 0.0f; --w; }
            else if (f > 0) { t = 1.0f - (float) f / (float) fadeLen; --f; }
            else            t = 1.0f;
            for (int c = 0; c < numCh; ++c)
                coreOutA[c][j] += t * (coreOutB[c][j] - coreOutA[c][j]);
        }
        warmLeft = w;
        fadeLeft = f;
        if (warmLeft == 0 && fadeLeft == 0)
        {
            activePath = nextPath;
            nextPath = -1;
        }
    }
    guiFactor.store (1 << activePath, std::memory_order_relaxed);

    // what the core added at every overtone, against the core's own level (for the overtone map)
    guiF0.store (ch[0].f0, std::memory_order_relaxed);
    guiConf.store (ch[0].pitchConf, std::memory_order_relaxed);
    guiPush.store (pushS, std::memory_order_relaxed);
    if (meter)
    {
        const float aBlk = 1.0f - std::exp (-(float) n / (0.05f * fsf));
        const float ref = 1.0f / (mon.yPow + 1.0e-12f);
        guiHalves.store (hHalf > 0.0f, std::memory_order_relaxed);
        for (int h = 0; h < kBands; ++h)
        {
            const float db = mon.groups > 0.0f ? 10.0f * std::log10 (mon.level[h] * ref + 1.0e-12f) : -120.0f;
            otLevelS[h] += (std::max (db, -120.0f) - otLevelS[h]) * aBlk;
            guiOtLevel[h].store (otLevelS[h], std::memory_order_relaxed);
            guiOtGain[h].store (hOn > 0.0f ? otGainAcc[h] / (float) n : 1.0f, std::memory_order_relaxed);
            guiPlain.store (hOn <= 0.0f, std::memory_order_relaxed);
            guiOtFit[h].store (ch[0].fit[h] * ch[0].fitOk[h], std::memory_order_relaxed);
        }
        guiClipDrive.store (clipDyn, std::memory_order_relaxed);
        guiClipKnee.store (clipH, std::memory_order_relaxed);
        guiClipPos.store (ctl[0][n - 1].clipPos, std::memory_order_relaxed);
        guiClipPeak.store (mon.peakIn / std::max (clipDyn, 1.0e-3f), std::memory_order_relaxed);
        int w = scopePos.load (std::memory_order_relaxed);
        for (int j = 0; j < n; ++j)
        {
            scopeIn[w].store (mon.u[j], std::memory_order_relaxed);
            scopeOut[w].store (mon.y[j], std::memory_order_relaxed);
            w = (w + 1) & (kScope - 1);
        }
        scopePos.store (w, std::memory_order_release);
    }

    // ---- post: anti-harsh, dynamics, focus, ink, ligature, harp, tone, mix
    const bool smOn = v[P_SMOOTH] > 0.001f;
    const float flAmt = v[P_FLANGE], flFb = v[P_FLANGEFB], flDepth = v[P_FLANGEDEPTH];
    const int flMode = std::min (std::max ((int) (v[P_FLANGEMODE] + 0.5f), 0), 2);
    const bool flOn = flAmt > 0.0005f;
    const float flNorm = std::sqrt (std::max (0.05f, 1.0f - flFb * flFb));
    const float flK = 1.0f + 3.0f * v[P_GRIT], flInvK = 1.0f / flK;
    const float aFlGlide = smoothCoef (flMode == 0 ? 30.0f : 3.0f, fsf);
    const float minFlT = std::max (1.0f, 0.00005f * fsf);
    const bool limitOn = v[P_LIMIT] > 0.5f;
    float clipPeak = 0.0f;
    const float bleed = v[P_BLEED] * std::min (1.0f, 4.0f * v[P_CLIP]);   // bleeds only when something is being clipped
    const int duckMode = std::min (std::max ((int) (v[P_DUCK] + 0.5f), 0), 2);   // 0 spectral, 1 classic, 2 off
    const bool spectral = duckMode == 0;
    // Depth: how far the echo must stay under the voice (spectral) / how deep the whole return dips (classic)
    const float depthDb = 6.0f + 18.0f * v[P_DUCKDEPTH];
    const float auraMargin = dbToGain (-depthDb), ligMargin = dbToGain (-(depthDb + 3.0f)), duckFloor = dbToGain (-30.0f);
    const float ligClassic = dbToGain (-(depthDb + 6.0f));
    float guiDuckSum = 0.0f;
    int guiDuckCount = 0;
    const float feed = v[P_FEED], spin = v[P_SPIN], forestBloom = v[P_FORESTBLOOM], echoFb = v[P_ECHOFB];
    const int twinType = std::min (std::max ((int) (v[P_TWINTYPE] + 0.5f), 0), 4);
    static const float kEchoBeats[7] = { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f, 1.5f, 2.0f };
    const int echoIdx = std::min (std::max ((int) (v[P_ECHOTIME] + 0.5f), 0), 6);
    const float echoTime = std::min (2.5f * fsf, echoIdx == 0 ? 0.085f * fsf : kEchoBeats[echoIdx] * (float) (60.0 / hostBpm) * fsf);
    const float lig = v[P_LIGATURE];
    const bool ligOn = lig > 0.0005f;
    const float ligFb = 0.2f + 0.5f * lig, ligOut = 0.9f * lig;
    const float beatS = (float) (60.0 / hostBpm) * fsf;
    const float ligTLt = std::min (0.5f * beatS, 1.5f * fsf);    // 1/8
    const float ligTRt = std::min (0.75f * beatS, 1.5f * fsf);   // dotted 1/8
    if (ligTL <= 0.0f) { ligTL = ligTLt; ligTR = ligTRt; }

    // IMAGER: what the bands should be (the crossovers keep at least a third of an octave apart)
    float imgGt[4], imgSpt[4];
    bool imgWant = false, spreadWant = false;
    for (int b = 0; b < 4; ++b)
    {
        imgGt[b] = 1.0f + clampf (v[P_IW1 + b], -1.0f, 1.0f);
        imgSpt[b] = clampf (v[P_IS1 + b], 0.0f, 1.0f);
        if (std::fabs (imgGt[b] - 1.0f) > 1.0e-4f || imgSpt[b] > 1.0e-4f) imgWant = true;
        if (imgSpt[b] > 1.0e-4f) spreadWant = true;
    }
    const float imgRect = clampf (v[P_IRECOVER], 0.0f, 1.0f);
    const bool recWant = imgRect > 1.0e-4f && (imgGt[0] < 0.9999f || imgGt[1] < 0.9999f || imgGt[2] < 0.9999f || imgGt[3] < 0.9999f);
    imgWant = imgWant && stereo;
    // engage at once; let go only after a second at neutral (so automation passing through 0 does not flip it)
    if (imgWant) imgQuiet = 0;
    else if (imgQuiet < 1 << 20) imgQuiet += n;
    const bool imgOn = imgWant || (imgLive && imgQuiet < (int) fsf);
    if (imgOn || imgLive)
    {
        float xt[3] = { clampf (v[P_IX1], 20.0f, 0.4f * fsf), clampf (v[P_IX2], 20.0f, 0.4f * fsf), clampf (v[P_IX3], 20.0f, 0.4f * fsf) };
        xt[1] = std::min (std::max (xt[1], xt[0] * 1.26f), 0.4f * fsf);
        xt[2] = std::min (std::max (xt[2], xt[1] * 1.26f), 0.42f * fsf);
        bool moved = imgX[0] < 0.0f;
        for (int i = 0; i < 3; ++i)
        {
            // glide (a fifth of the way per block), so a dragged crossover does not click
            const float nx = imgX[i] < 0.0f ? xt[i] : imgX[i] * std::pow (xt[i] / imgX[i], 0.2f);
            if (std::fabs (nx - imgX[i]) > 1.0e-3f * xt[i]) { imgX[i] = nx; moved = true; }
        }
        if (moved)
        {
            imgSplitS.design (imgX, fs); imgSplitD.design (imgX, fs);
            imgSumS.design (imgX, fs); imgSumD.design (imgX, fs); imgSumR.design (imgX, fs);
            for (int i = 0; i < 3; ++i)
            {
                imgApM[i].allpass (imgX[i], 0.70710678, fs);
                imgApDry[0][i].allpass (imgX[i], 0.70710678, fs);
                imgApDry[1][i].allpass (imgX[i], 0.70710678, fs);
            }
        }
    }
    guiImgOn.store (imgOn, std::memory_order_relaxed);
    const bool imgMeter = imgMetering.load (std::memory_order_relaxed);
    int imgW = imgPos.load (std::memory_order_relaxed);

    float check = 0.0f;
    for (int j = 0; j < n; ++j)
    {
        sComp    += (tComp - sComp) * aS;
        sDryG    += (tDryG - sDryG) * aS;
        sWetG    += (tWetG - sWetG) * aS;
        sOut     += (tOut - sOut) * aS;
        sHarpLvl += (tHarpLvl - sHarpLvl) * aS;
        sHarpExc += (tHarpExc - sHarpExc) * aS;
        const int dly = totalLat + (n - 1 - j);

        float wv[2] = { 0.0f, 0.0f };
        float clipHitMono = 0.0f;
        const float voiceGate = clampf ((1.0f - duckBuf[j]) * (1.0f / 0.92f), 0.0f, 1.0f);
        if (spectral)
            duckKey.process (stereo ? 0.5f * (dryOut[0][j] + dryOut[1][j]) : dryOut[0][j], voiceGate);
        for (int c = 0; c < numCh; ++c)
        {
            Chan& C = ch[c];
            float w = C.dc.hp (coreOutA[c][j]) * sComp;

            const float lo = C.smLP.process (w), hi = w - lo;
            const float eL = C.smEnvLo.process (lo), eH = C.smEnvHi.process (hi);
            const float ratio = eH / (eL + 1.0e-5f);
            const float gt = (smOn && ratio > smThr) ? std::pow (smThr / ratio, smExp) : 1.0f;
            C.smGain = gt + (gt < C.smGain ? aSmAtk : aSmRel) * (C.smGain - gt);
            w = lo + hi * C.smGain * smStatic;

            // FLANGE: comb with a saturating feedback loop. Tuned = resonates on the chord tones as an arpeggio
            // locked to the bars; Jet = sweeps with the clock; Stroke = one jet sweep per syllable (the pen).
            if (flOn)
            {
                if (! C.flLive) { C.fl.reset(); C.flLP.reset(); C.flLive = true; }
                float T;
                if (flMode == 0)
                {
                    static const int arp[4] = { 0, 2, 1, 3 };
                    T = flTune[arp[std::min (3, (int) (phase0Buf[j] * 4.0f))]] * (1.0f + 0.003f * flDepth * auraB[j]);
                }
                else
                {
                    const float tMax = (0.5f + 6.5f * flDepth) * 0.001f * fsf * (0.6f + 0.4f * breathe[j]);
                    const float pos = flMode == 1 ? 0.5f + 0.5f * auraB[j] : strokeBuf[j];
                    T = minFlT + (tMax - minFlT) * pos;
                }
                C.flT += (T - C.flT) * aFlGlide;
                const float m = C.fl.read (std::max (C.flT, 1.0f));
                C.fl.push (w + flFb * fastTanh (flK * C.flLP.lp (m)) * flInvK);
                w += flAmt * (0.6f * (w + m * flNorm) - w);
            }
            else
                C.flLive = false;

            wv[c] = w * C.ctlRestore.tap (dly) * C.ctlFocus.tap (dly) * C.ctlInkWet.tap (dly);
        }

        // Ligature: saturated ping-pong echo in time with the song, ducked while the voice sings
        if (ligOn)
        {
            ligTL += (ligTLt - ligTL) * aLigT;
            ligTR += (ligTRt - ligTR) * aLigT;
            const float oL = ligL.read (ligTL);
            const float oR = stereo ? ligR.read (ligTR) : 0.0f;
            const float duck = (duckMode == 1 ? 1.0f - voiceGate * (1.0f - ligClassic) : 1.0f) * ligOut;
            float rL = duck * oL, rR = stereo ? duck * oR : 0.0f;
            if (spectral)
                duckLig.process (duckKey, rL, rR, ligMargin, duckFloor);   // keep (depth + 3) dB under the voice
            if (stereo)
            {
                const float fL = fastTanh (1.5f * ligHP[0].hp (ligLP[0].lp (oR))) * (1.0f / 1.5f);
                const float fR = fastTanh (1.5f * ligHP[1].hp (ligLP[1].lp (oL))) * (1.0f / 1.5f);
                ligL.push (wv[0] + ligFb * fL);
                ligR.push (wv[1] + ligFb * fR);
                wv[0] += rL;
                wv[1] += rR;
            }
            else
            {
                const float f = fastTanh (1.5f * ligHP[0].hp (ligLP[0].lp (oL))) * (1.0f / 1.5f);
                ligL.push (wv[0] + ligFb * f);
                wv[0] += rL;
            }
        }

        for (int c = 0; c < numCh; ++c)
        {
            Chan& C = ch[c];
            float w = wv[c];
            const float clipHit = std::min (2.0f, C.clipEnv.process (clipRes[c][j]) * 6.67f);
            clipHitMono += clipHit;
            const float e = C.harpLP.process (C.harpHP.process (w)) * sHarpExc * C.ctlInkHarp.tap (dly)
                            * (1.0f + 3.0f * bleed * clipHit);   // BLEED: what the clipper cuts off strikes the strings
            const float h = C.harp.process (e, shim[c][j]);
            w += sHarpLvl * 0.7f * fastTanh (h * (0.55f / 0.7f));

            wv[c] = C.toneHi.process (C.toneLo.process (w));
        }
        clipPeak = std::max (clipPeak, clipHitMono / (float) numCh);

        // IMAGER: the width of the wet sound in four bands. Only the side changes; the mid (and the dry, below)
        // pass through the same all-pass as the bands, so the mono sum is untouched and nothing smears.
        float dryAp[2] = { dryOut[0][j], stereo ? dryOut[1][j] : 0.0f };
        const float preL = wv[0], preR = stereo ? wv[1] : wv[0];
        if (imgOn || imgLive)
        {
            if (! imgLive) { resetImager(); imgLive = true; }
            imgMix += ((imgOn ? 1.0f : 0.0f) - imgMix) * aImg;
            for (int b = 0; b < 4; ++b)
            {
                imgG[b] += (imgGt[b] - imgG[b]) * aS;
                imgSp[b] += (imgSpt[b] - imgSp[b]) * aS;
            }
            imgRec += ((recWant ? imgRect : 0.0f) - imgRec) * aS;

            const float m = 0.5f * (wv[0] + wv[1]), s = 0.5f * (wv[0] - wv[1]);
            float bs[4];
            imgSplitS.split (s, bs);
            float s2 = imgSumS.sum (bs, imgG);

            // SPREAD: a dispersed copy of the mid goes into the side of its band - width for what has none
            const bool spreadOn = spreadWant || imgSp[0] + imgSp[1] + imgSp[2] + imgSp[3] > 1.0e-5f;
            if (spreadOn)
            {
                if (! imgSpreadLive) { imgSpread.reset(); imgSplitD.reset(); imgSumD.reset(); imgSpreadLive = true; }
                float bd[4];
                imgSplitD.split (imgSpread.process (m), bd);
                s2 += 0.7f * imgSumD.sum (bd, imgSp);
            }
            else imgSpreadLive = false;

            float m2 = imgApM[2].process (imgApM[1].process (imgApM[0].process (m)));

            // RECOVER: what narrowing takes out of the side comes back in the middle (dispersed, so it adds body
            // instead of combing with the mid)
            if (recWant || imgRec > 1.0e-5f)
            {
                if (! imgRecLive) { imgRecover.reset(); imgSumR.reset(); imgRecLive = true; }
                const float rg[4] = { std::max (0.0f, 1.0f - imgG[0]), std::max (0.0f, 1.0f - imgG[1]),
                                      std::max (0.0f, 1.0f - imgG[2]), std::max (0.0f, 1.0f - imgG[3]) };
                m2 += imgRec * 0.7f * imgRecover.process (imgSumR.sum (bs, rg));
            }
            else imgRecLive = false;

            wv[0] += imgMix * (m2 + s2 - wv[0]);
            wv[1] += imgMix * (m2 - s2 - wv[1]);
            for (int c = 0; c < 2; ++c)
            {
                const float a = imgApDry[c][2].process (imgApDry[c][1].process (imgApDry[c][0].process (dryAp[c])));
                dryAp[c] += imgMix * (a - dryAp[c]);
            }
            if (! imgOn && imgMix < 1.0e-4f)
            {
                resetImager();
                imgMix = 0.0f;
            }
        }
        if (imgMeter)
        {
            imgPreL[imgW].store (preL, std::memory_order_relaxed);
            imgPreR[imgW].store (preR, std::memory_order_relaxed);
            imgPostL[imgW].store (wv[0], std::memory_order_relaxed);
            imgPostR[imgW].store (stereo ? wv[1] : wv[0], std::memory_order_relaxed);
            imgW = (imgW + 1) & (kImg - 1);
        }

        // AURA: nonlinear send effects, added after the dry/wet mix (like a send return)
        float aura[2] = { 0.0f, 0.0f };
        sEcho   += (tEcho - sEcho) * aS;
        sForest += (tForest - sForest) * aS;
        sTwin   += (tTwin - sTwin) * aS;
        const bool wantEcho = tEcho > 1.0e-7f || sEcho > 1.0e-5f;
        const bool wantForest = tForest > 1.0e-7f || sForest > 1.0e-5f;
        const bool wantTwin = tTwin > 1.0e-7f || sTwin > 1.0e-5f;
        if (wantEcho || wantForest || wantTwin)
        {
            const float d0 = dryOut[0][j], d1 = stereo ? dryOut[1][j] : d0;
            const float w0 = wv[0], w1 = stereo ? wv[1] : w0;
            const float src = fxLP.process (fxHP.process (0.5f * ((1.0f - feed) * (d0 + d1) + feed * (w0 + w1))));
            const float motionA = (0.3f + spin) * auraA[j];
            const float clipHitAvg = clipHitMono / (float) numCh;
            const float clipThrow = 1.0f + 1.5f * bleed * clipHitAvg;   // BLEED: hard-clipped words throw more echo
            float eL = 0.0f, eR = 0.0f, fL = 0.0f, fR = 0.0f, tL = 0.0f, tR = 0.0f;
            if (wantEcho)
            {
                if (! echoLive) { aEcho.reset(); echoLive = true; }
                aEcho.process (src * clipThrow, echoTime, echoFb, 0.25f + 0.75f * grit, 0.004f * motionA, eL, eR);
            }
            else echoLive = false;
            if (wantTwin)
            {
                if (! twinLive) { aTwin.reset(); twinLive = true; }
                aTwin.process (src, twinType, breathe[j], motionA, grit, tL, tR);
            }
            else twinLive = false;
            if (wantForest)
            {
                if (! forestLive) { aForest.reset(); forestLive = true; }
                // the echoes and the twins live in the same forest: one space, one piece of hardware
                const float fin = src * (1.0f + bleed * clipHitAvg)          // BLEED: clipped words bloom in the forest
                                  + 0.35f * sEcho * 0.5f * (eL + eR) + 0.25f * sTwin * 0.5f * (tL + tR);
                aForest.process (fin, 1.2f / (1.0f + 3.0f * grit), (0.3f + spin) * auraB[j], fL, fR);
            }
            else forestLive = false;

            const float duck = duckBuf[j];
            const float ge = sEcho * (duckMode == 1 ? 1.0f - voiceGate * (1.0f - auraMargin) : 1.0f);
            const float gf = sForest * 1.6f * (duckMode == 2 ? 1.0f : 1.0f - forestBloom * (1.0f - duck));
            const float gt = sTwin;
            float dL = ge * eL + gf * fL, dR = ge * eR + gf * fR;
            if (spectral)
            {
                // the echo and the forest step aside only in the frequencies the voice is using right now
                duckAura.process (duckKey, dL, dR, auraMargin, duckFloor);   // keep `depth` dB under the voice
            }
            else if (duckMode == 1)
            {
                guiDuckSum += 1.0f - voiceGate * (1.0f - auraMargin);
                ++guiDuckCount;
            }
            float aL = dL + gt * tL;      // the twins are meant to sing WITH the voice: never ducked
            float aR = dR + gt * tR;
            // Spin: the whole aura turns around the head with the clock
            const float th = spin * 0.75f * auraC[j];
            const float cs = std::cos (th), sn = std::sin (th);
            const float rL = cs * aL + sn * aR, rR = cs * aR - sn * aL;
            if (stereo) { aura[0] = rL; aura[1] = rR; }
            else        aura[0] = 0.5f * (rL + rR);
        }

        // REVERB SAFE: the wet sound is ready to go on into a reverb - no rumble, no fizz, a narrower low side
        // (the dirt itself is already kept to the attacks of the words)
        float wet[2] = { sWetG * wv[0] + aura[0], stereo ? sWetG * wv[1] + aura[1] : 0.0f };
        rsMix += ((revSafe ? 1.0f : 0.0f) - rsMix) * aS;
        if (rsMix > 1.0e-4f)
        {
            float pr[2] = { 0.0f, 0.0f };
            for (int c = 0; c < numCh; ++c)
                pr[c] = ch[c].rsShelf.process (ch[c].rsHP.process (wet[c]));
            if (stereo)
            {
                const float mid = 0.5f * (pr[0] + pr[1]);
                const float sd = 0.6f * rsSideHP.hp (0.5f * (pr[0] - pr[1]));
                pr[0] = mid + sd;
                pr[1] = mid - sd;
            }
            for (int c = 0; c < numCh; ++c)
                wet[c] += rsMix * (pr[c] - wet[c]);
            rsLive = true;
        }
        else if (rsLive)
        {
            for (int c = 0; c < 2; ++c) { ch[c].rsHP.reset(); ch[c].rsShelf.reset(); }
            rsSideHP.reset();
            rsLive = false;
        }

        for (int c = 0; c < numCh; ++c)
        {
            float out = (sDryG * dryAp[c] + wet[c]) * sOut;
            if (limitOn)
            {   // output stage: transparent below -2.5 dBFS, then a soft knee that never passes 0 dBFS
                const float a = std::fabs (out);
                if (a > 0.75f)
                    out = std::copysign (0.75f + 0.25f * fastTanh ((a - 0.75f) * 4.0f), out);
            }
            io[c][off + j] = out;
            check += out;
        }
    }

    if (imgMeter)
        imgPos.store (imgW, std::memory_order_release);
    if (clipPeak > guiClip.load (std::memory_order_relaxed))
        guiClip.store (clipPeak, std::memory_order_relaxed);

    for (int b = 0; b < DuckBands::kBands; ++b)
    {
        const float g = spectral ? duckAura.gain[b] : (guiDuckCount > 0 ? guiDuckSum / (float) guiDuckCount : 1.0f);
        guiDuck[b].store (g, std::memory_order_relaxed);
    }

    if (! std::isfinite (check))
    {
        reset();
        for (int c = 0; c < numCh; ++c)
            for (int j = 0; j < n; ++j)
                io[c][off + j] = 0.0f;
    }
}

} // namespace mavka
