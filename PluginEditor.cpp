#include "PluginEditor.h"

#if __has_include("MavkaFonts.h")
 #include "MavkaFonts.h"
 #define MAVKA_HAS_FONTS 1
#else
 #define MAVKA_HAS_FONTS 0
#endif

#include <cmath>
#include <cstring>
#include <thread>

using namespace mavkaui;

//==============================================================================
// shared typefaces
struct mavkaui::Assets
{
    Assets()
    {
       #if MAVKA_HAS_FONTS
        regular = juce::Typeface::createSystemTypefaceFor (mavkafonts::interRegular, (size_t) mavkafonts::interRegularSize);
        semi    = juce::Typeface::createSystemTypefaceFor (mavkafonts::interSemiBold, (size_t) mavkafonts::interSemiBoldSize);
       #endif
        live() = this;
    }
    ~Assets()
    {
        if (live() == this)
            live() = nullptr;
    }
    static Assets*& live() { static Assets* a = nullptr; return a; }

    juce::Typeface::Ptr regular, semi;
    TextureSet sharp;   // the sharpest textures made so far (native images), shared by every open editor
};

std::shared_ptr<Assets> mavkaui::acquireAssets()
{
    static std::weak_ptr<Assets> cache;
    if (auto a = cache.lock())
        return a;
    auto a = std::make_shared<Assets>();
    cache = a;
    return a;
}

juce::Font mavkaui::sans (float size, bool semibold, float tracking)
{
    const auto* a = Assets::live();
    const auto tf = a != nullptr ? (semibold ? a->semi : a->regular) : nullptr;
    const auto opts = tf != nullptr ? juce::FontOptions (tf).withHeight (size)
                                    : juce::FontOptions (size, semibold ? juce::Font::bold : juce::Font::plain);
    return juce::Font (opts.withKerningFactor (tracking));
}

juce::Font mavkaui::numbers (float size)
{
    const auto* a = Assets::live();
    if (a == nullptr || a->regular == nullptr)
        return sans (size);
    return juce::Font (juce::FontOptions (a->regular).withHeight (size).withFeatureEnabled (juce::FontFeatureTag ("tnum")));
}

namespace
{
    juce::String u8 (const char* s) { return juce::String::fromUTF8 (s); }

    bool isBipolar (const juce::String& id)
    {
        return id == "ink" || id == "slant" || id == "bias" || id == "tone" || id == "preTilt" || id == "output" || id == "detune";
    }

    // the plates of the faceplate: same white, different wear
    struct Plate { float x, y, w, h; juce::uint32 shade; const char* title; const char* sub; };
    const Plate kPlates[] =
    {
        { 12.0f,  64.0f,  430.0f, 250.0f, 0xfff0ede6, "HANDWRITING", "" },
        { 452.0f, 64.0f,  596.0f, 250.0f, 0xffedeae2, "CORE",        "" },
        { 12.0f,  322.0f, 250.0f, 256.0f, 0xffe8e8e3, "CHORD CLOCK", "" },
        { 272.0f, 322.0f, 776.0f, 256.0f, 0xffebe8df, "",            "" },
        { 12.0f,  586.0f, 1036.0f, 202.0f, 0xffe7e6e1, "AURA",       "send effects  \xc2\xb7  they step aside when you sing" },
    };

    constexpr int kW = 1060, kH = 800;
    constexpr float kBaseLevel = 1.25f;   // texture pixels per editor pixel of the set every editor starts with

    //------------------------------------------------------------------------------
    // noise
    inline juce::uint32 hash3 (int x, int y, juce::uint32 seed) noexcept
    {
        juce::uint32 h = (juce::uint32) x * 0x8da6b343u ^ (juce::uint32) y * 0xd8163841u ^ seed * 0xcb1ab31fu;
        h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12; h *= 0x297a2d39u; h ^= h >> 15;
        return h;
    }
    inline float rnd01 (int x, int y, juce::uint32 seed) noexcept { return (float) (hash3 (x, y, seed) >> 8) * (1.0f / 16777216.0f); }

    float valueNoise (float x, float y, juce::uint32 seed) noexcept
    {
        const float x0 = std::floor (x), y0 = std::floor (y);
        const int xi = (int) x0, yi = (int) y0;
        const float fx = x - x0, fy = y - y0;
        const float u = fx * fx * (3.0f - 2.0f * fx), v = fy * fy * (3.0f - 2.0f * fy);
        const float a = rnd01 (xi, yi, seed), b = rnd01 (xi + 1, yi, seed);
        const float c = rnd01 (xi, yi + 1, seed), d = rnd01 (xi + 1, yi + 1, seed);
        return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
    }

    float fbm (float x, float y, juce::uint32 seed, int octaves) noexcept
    {
        float sum = 0.0f, amp = 0.5f, norm = 0.0f;
        for (int i = 0; i < octaves; ++i)
        {
            sum += amp * valueNoise (x, y, seed + (juce::uint32) i * 1013u);
            norm += amp;
            x = x * 2.03f + 17.1f;
            y = y * 2.03f + 5.3f;
            amp *= 0.5f;
        }
        return sum / norm;
    }

    inline float smooth (float e0, float e1, float x) noexcept
    {
        const float t = juce::jlimit (0.0f, 1.0f, (x - e0) / (e1 - e0));
        return t * t * (3.0f - 2.0f * t);
    }

    // runs fn (row) for every row, spread over the cores (rows are independent)
    template <typename Fn>
    void parallelRows (int rows, Fn&& fn, const std::atomic<bool>* cancel = nullptr)
    {
        const int n = juce::jlimit (1, 8, (int) std::thread::hardware_concurrency());
        auto work = [&fn, n, rows, cancel] (int first)
        {
            for (int y = first; y < rows; y += n)
            {
                if (cancel != nullptr && cancel->load (std::memory_order_relaxed))
                    return;
                fn (y);
            }
        };
        std::vector<std::thread> pool;
        for (int t = 1; t < n; ++t)
            pool.emplace_back (work, t);
        work (0);
        for (auto& th : pool)
            th.join();
    }

    // a smooth field sampled on a coarse grid (editor coordinates), read back bilinearly
    struct Field
    {
        float step = 1.0f;
        int w = 0, h = 0;
        std::vector<float> v;

        template <typename Fn>
        void build (float stepBase, Fn&& fn)
        {
            step = stepBase;
            w = (int) std::ceil ((float) kW / step) + 2;
            h = (int) std::ceil ((float) kH / step) + 2;
            v.resize ((size_t) w * (size_t) h);
            parallelRows (h, [this, &fn] (int y)
            {
                for (int x = 0; x < w; ++x)
                    v[(size_t) y * (size_t) w + (size_t) x] = fn ((float) x * step, (float) y * step);
            });
        }

        float at (float bx, float by) const noexcept
        {
            const float gx = juce::jlimit (0.0f, (float) (w - 2) - 0.001f, bx / step);
            const float gy = juce::jlimit (0.0f, (float) (h - 2) - 0.001f, by / step);
            const int ix = (int) gx, iy = (int) gy;
            const float fx = gx - (float) ix, fy = gy - (float) iy;
            const float* r0 = v.data() + (size_t) iy * (size_t) w + (size_t) ix;
            const float* r1 = r0 + w;
            const float a = r0[0] + (r0[1] - r0[0]) * fx, b = r1[0] + (r1[1] - r1[0]) * fx;
            return a + (b - a) * fy;
        }
    };

    inline float sdRoundRect (float px, float py, const Plate& p, float r) noexcept
    {
        const float qx = std::abs (px - (p.x + 0.5f * p.w)) - (0.5f * p.w - r);
        const float qy = std::abs (py - (p.y + 0.5f * p.h)) - (0.5f * p.h - r);
        const float ox = std::max (qx, 0.0f), oy = std::max (qy, 0.0f);
        return std::sqrt (ox * ox + oy * oy) + std::min (std::max (qx, qy), 0.0f) - r;
    }

    //------------------------------------------------------------------------------
    // texture caches: plain pixels (native ARGB, premultiplied), so nothing JUCE-owned outlives the editors
    struct PixelCache { int w = 0, h = 0; std::vector<juce::uint32> px; };

    PixelCache cacheFromImage (const juce::Image& img)
    {
        PixelCache c;
        c.w = img.getWidth();
        c.h = img.getHeight();
        c.px.resize ((size_t) c.w * (size_t) c.h);
        const juce::Image::BitmapData bd (img, juce::Image::BitmapData::readOnly);
        for (int y = 0; y < c.h; ++y)
        {
            auto* dst = c.px.data() + (size_t) y * (size_t) c.w;
            if (bd.pixelFormat == juce::Image::ARGB && bd.pixelStride == 4)
                std::memcpy (dst, bd.getLinePointer (y), (size_t) c.w * 4);
            else
                for (int x = 0; x < c.w; ++x)
                {
                    const auto col = bd.getPixelColour (x, y);
                    juce::PixelARGB p (col.getAlpha(), col.getRed(), col.getGreen(), col.getBlue());
                    p.premultiply();
                    dst[x] = p.getInARGBMaskOrder();
                }
        }
        return c;
    }

    juce::Image imageFromCache (const PixelCache& c)
    {
        juce::Image img (juce::Image::ARGB, c.w, c.h, false);
        {
            juce::Image::BitmapData bd (img, juce::Image::BitmapData::writeOnly);
            for (int y = 0; y < c.h; ++y)
            {
                const auto* src = c.px.data() + (size_t) y * (size_t) c.w;
                if (bd.pixelFormat == juce::Image::ARGB && bd.pixelStride == 4)
                    std::memcpy (bd.getLinePointer (y), src, (size_t) c.w * 4);
                else
                    for (int x = 0; x < c.w; ++x)
                    {
                        const juce::uint32 v = src[x];   // A<<24 | R<<16 | G<<8 | B, premultiplied
                        juce::PixelARGB p ((juce::uint8) (v >> 24), (juce::uint8) (v >> 16), (juce::uint8) (v >> 8), (juce::uint8) v);
                        p.unpremultiply();
                        bd.setPixelColour (x, y, juce::Colour (p.getRed(), p.getGreen(), p.getBlue(), p.getAlpha()));
                    }
            }
        }
        return img;
    }

    //------------------------------------------------------------------------------
    // dried blood
    void blob (juce::Graphics& g, juce::Random& r, float cx, float cy, float R, float wob)
    {
        constexpr int n = 40;
        juce::Point<float> pts[n];
        const float p1 = r.nextFloat() * 6.283f, p2 = r.nextFloat() * 6.283f, p3 = r.nextFloat() * 6.283f;
        for (int i = 0; i < n; ++i)
        {
            const float a = juce::MathConstants<float>::twoPi * (float) i / (float) n;
            const float k = 1.0f + wob * (0.50f * std::sin (2.0f * a + p1) + 0.30f * std::sin (5.0f * a + p2)
                                          + 0.18f * std::sin (9.0f * a + p3) + 0.25f * (r.nextFloat() - 0.5f));
            pts[i] = { cx + R * k * std::cos (a), cy + R * k * std::sin (a) };
        }
        auto mid = [&] (int i) { return (pts[i % n] + pts[(i + 1) % n]) * 0.5f; };
        juce::Path p;
        p.startNewSubPath (mid (0));
        for (int i = 1; i <= n; ++i)
            p.quadraticTo (pts[i % n], mid (i));
        p.closeSubPath();
        g.fillPath (p);
    }

    void splatter (juce::Graphics& g, juce::Random& r, juce::Point<float> c, float R, float dir, float spread,
                   int drops, float drip)
    {
        const auto rim = theme::bloodDark, body = theme::blood;

        g.setColour (rim.withAlpha (0.86f));
        for (int k = 0; k < 6; ++k)   // streaks thrown out by the impact
        {
            const float a = dir + (r.nextFloat() - 0.5f) * spread * 1.3f;
            const float len = R * (1.3f + 1.1f * r.nextFloat());
            const float w = R * (0.09f + 0.12f * r.nextFloat());
            const juce::Point<float> u (std::cos (a), std::sin (a)), nrm (-u.y, u.x);
            juce::Path p;
            p.startNewSubPath (c + nrm * w + u * (R * 0.5f));
            p.lineTo (c + u * len);
            p.lineTo (c - nrm * w + u * (R * 0.5f));
            p.closeSubPath();
            g.fillPath (p);
            if (r.nextFloat() < 0.6f)
            {
                const float d = w * (0.8f + 0.8f * r.nextFloat());
                const auto e = c + u * (len + d * 1.6f);
                g.fillEllipse (e.x - d, e.y - d, 2.0f * d, 2.0f * d);
            }
        }

        g.setColour (rim.withAlpha (0.93f));
        blob (g, r, c.x, c.y, R, 0.20f);
        g.setColour (body.withAlpha (0.80f));
        blob (g, r, c.x + R * 0.06f, c.y - R * 0.05f, R * 0.80f, 0.16f);
        g.setColour (juce::Colour (0xff8c2a1f).withAlpha (0.30f));
        blob (g, r, c.x - R * 0.12f, c.y - R * 0.18f, R * 0.42f, 0.30f);

        for (int k = 0; k < drops; ++k)   // satellites
        {
            const float a = dir + (r.nextFloat() - 0.5f) * spread;
            const float t = std::pow (r.nextFloat(), 1.4f);
            const float dist = R * (1.5f + 4.5f * t);
            const float size = R * (0.05f + 0.20f * (1.0f - t)) * (0.5f + r.nextFloat());
            const float elong = 1.0f + 1.6f * r.nextFloat() * t;
            const auto pos = c + juce::Point<float> (std::cos (a), std::sin (a)) * dist;
            juce::Path p;
            p.addEllipse (-size * elong, -size, 2.0f * size * elong, 2.0f * size);
            p.applyTransform (juce::AffineTransform::rotation (a).translated (pos));
            g.setColour (rim.withAlpha (0.80f + 0.15f * r.nextFloat()));
            g.fillPath (p);
        }

        if (drip > 0.0f)   // gravity pulls a thin, wavering thread down from the body
        {
            const float x0 = c.x + R * 0.15f, y0 = c.y + R * 0.5f;
            const float w0 = R * 0.22f, w1 = R * 0.11f, wav = R * 0.14f;
            juce::Path p;
            p.startNewSubPath (x0 - w0, y0);
            p.cubicTo (x0 - w0 * 0.9f + wav, y0 + drip * 0.35f, x0 - w1 - wav, y0 + drip * 0.7f, x0 - w1, y0 + drip);
            p.lineTo (x0 + w1, y0 + drip);
            p.cubicTo (x0 + w1 - wav, y0 + drip * 0.7f, x0 + w0 * 0.8f + wav, y0 + drip * 0.35f, x0 + w0, y0);
            p.closeSubPath();
            g.setColour (rim.withAlpha (0.86f));
            g.fillPath (p);
            const float b = R * 0.21f;
            g.fillEllipse (x0 - b, y0 + drip - b * 0.5f, 2.0f * b, 2.3f * b);
            g.setColour (body.withAlpha (0.45f));
            g.drawLine (x0 + wav * 0.1f, y0 + 2.0f, x0, y0 + drip - 2.0f, w1 * 0.6f);
        }
    }

    //------------------------------------------------------------------------------
    // where the screws go: every plate is a module bolted to the chassis, plus the chassis corners
    std::vector<juce::Point<float>> screwPositions()
    {
        std::vector<juce::Point<float>> s;
        for (const auto& p : kPlates)
        {
            constexpr float in = 7.0f;
            s.push_back ({ p.x + in, p.y + in });
            s.push_back ({ p.x + p.w - in, p.y + in });
            s.push_back ({ p.x + in, p.y + p.h - in });
            s.push_back ({ p.x + p.w - in, p.y + p.h - in });
        }
        s.push_back ({ 6.0f, 6.0f });
        s.push_back ({ (float) kW - 6.0f, 6.0f });
        s.push_back ({ (float) kW - 6.0f, (float) kH - 6.0f });
        return s;
    }
    constexpr float kScrewR = 3.6f;

    void screw (juce::Graphics& g, juce::Random& r, juce::Point<float> c, float rad)
    {
        g.setColour (juce::Colours::black.withAlpha (0.28f));   // the countersink
        g.fillEllipse (c.x - rad - 0.9f, c.y - rad - 0.4f, 2.0f * rad + 1.8f, 2.0f * rad + 1.8f);
        g.setGradientFill (juce::ColourGradient (juce::Colour (0xffe2e1dc), c.x - rad * 0.6f, c.y - rad * 0.7f,
                                                 juce::Colour (0xff7f7d76), c.x + rad * 0.7f, c.y + rad * 0.8f, false));
        g.fillEllipse (c.x - rad, c.y - rad, 2.0f * rad, 2.0f * rad);
        g.setColour (juce::Colour (0xff4a4843).withAlpha (0.8f));
        g.drawEllipse (c.x - rad, c.y - rad, 2.0f * rad, 2.0f * rad, 0.5f);

        const float a = r.nextFloat() * juce::MathConstants<float>::pi;
        for (int k = 0; k < 2; ++k)   // a Phillips head, a bit chewed
        {
            const float ak = a + (float) k * juce::MathConstants<float>::halfPi;
            const juce::Point<float> u (std::cos (ak), std::sin (ak));
            g.setColour (juce::Colour (0xff2c2a26));
            g.drawLine (juce::Line<float> (c - u * (rad * 0.62f), c + u * (rad * 0.62f)), rad * 0.30f);
            g.setColour (juce::Colours::white.withAlpha (0.35f));
            g.drawLine (juce::Line<float> (c - u * (rad * 0.55f) + juce::Point<float> (0.35f, 0.35f),
                                           c + u * (rad * 0.55f) + juce::Point<float> (0.35f, 0.35f)), 0.35f);
        }
        if (r.nextFloat() < 0.4f)   // rust creeping out of the slot
        {
            const float rr = rad * (0.35f + 0.35f * r.nextFloat());
            const float ra = r.nextFloat() * 6.283f;
            g.setColour (juce::Colour (0xff8a4b22).withAlpha (0.30f + 0.25f * r.nextFloat()));
            g.fillEllipse (c.x + std::cos (ra) * rad * 0.35f - rr, c.y + std::sin (ra) * rad * 0.35f - rr, 2.0f * rr, 1.6f * rr);
        }
        g.setColour (juce::Colour (0xff3b3833).withAlpha (0.22f));   // grime packed around the head
        g.drawEllipse (c.x - rad - 0.7f, c.y - rad - 0.7f, 2.0f * rad + 1.4f, 2.0f * rad + 1.4f, 1.0f);
    }

    // printed rows the chips keep away from (titles, tabs, macro labels and values, the signal path, page text)
    const juce::Rectangle<float> kLabelRows[] =
    {
        { 18.0f, 4.0f, 190.0f, 54.0f },    { 462.0f, 104.0f, 580.0f, 26.0f }, { 462.0f, 192.0f, 580.0f, 24.0f },
        { 462.0f, 244.0f, 580.0f, 52.0f }, { 280.0f, 326.0f, 764.0f, 82.0f }, { 22.0f, 190.0f, 420.0f, 26.0f },
        { 22.0f, 276.0f, 420.0f, 24.0f },  { 20.0f, 66.0f, 200.0f, 24.0f },   { 460.0f, 66.0f, 120.0f, 24.0f },
        { 20.0f, 326.0f, 140.0f, 24.0f },  { 20.0f, 590.0f, 1020.0f, 48.0f }, { 20.0f, 762.0f, 1020.0f, 24.0f }
    };

    struct Dent { float x, y, r; };
    const Dent kDents[] = { { 150.0f, 610.0f, 9.0f }, { 702.0f, 548.0f, 7.0f }, { 985.0f, 333.0f, 6.0f },
                            { 44.0f, 452.0f, 8.0f }, { 600.0f, 40.0f, 6.0f }, { 236.0f, 150.0f, 5.0f } };

    // smooth fields in editor units: identical at every texture resolution, built once per process
    struct Fields
    {
        Field stain, water, mottle, grime, ragged, flakes, zones, edgeVar;
        Fields()
        {
            stain.build (20.0f,  [] (float x, float y) { return fbm (x / 210.0f, y / 210.0f, 11u, 4); });
            water.build (4.0f,   [] (float x, float y) { return fbm (x / 85.0f + 3.1f, y / 85.0f, 23u, 5); });
            mottle.build (2.0f,  [] (float x, float y) { return fbm (x / 9.0f, y / 9.0f, 37u, 3); });
            grime.build (3.0f,   [] (float x, float y) { return fbm (x / 16.0f, y / 16.0f, 51u, 3); });
            ragged.build (0.8f,  [] (float x, float y) { return fbm (x / 5.0f, y / 5.0f, 61u, 4); });    // ragged paint edges
            flakes.build (2.0f,  [] (float x, float y) { return fbm (x / 24.0f, y / 24.0f, 71u, 4); });  // where flakes came off
            zones.build (16.0f,  [] (float x, float y) { return fbm (x / 260.0f, y / 260.0f, 97u, 3); }); // the worst-hit areas
            edgeVar.build (4.0f, [] (float x, float y) { return fbm (x / 40.0f, y / 40.0f, 101u, 3); });  // edges wear in stretches
        }
    };
    const Fields& fields() { static const Fields f; return f; }   // thread-safe initialisation

    // which knobs, faders and screws touch which part of the panel (so each pixel only looks at its neighbours)
    struct Influences
    {
        static constexpr float cell = 24.0f;
        struct Item { int kind, index; };   // kind 0 = knob, 1 = fader, 2 = screw
        int cw = 0, ch = 0;
        std::vector<std::vector<Item>> cells;

        Influences (const std::vector<Scuff>& scuffs, const std::vector<juce::Point<float>>& screws)
        {
            cw = (int) std::ceil ((float) kW / cell);
            ch = (int) std::ceil ((float) kH / cell);
            cells.resize ((size_t) (cw * ch));
            for (size_t i = 0; i < scuffs.size(); ++i)
            {
                const auto& s = scuffs[i];
                const float ex = s.fader ? 18.0f : s.r * 1.95f, ey = s.fader ? s.r + 10.0f : s.r * 1.95f;
                add (s.x - ex, s.y - ey, s.x + ex, s.y + ey, { s.fader ? 1 : 0, (int) i });
            }
            for (size_t i = 0; i < screws.size(); ++i)
            {
                const float e = kScrewR * 3.0f;
                add (screws[i].x - e, screws[i].y - e, screws[i].x + e, screws[i].y + e, { 2, (int) i });
            }
        }

        void add (float x0, float y0, float x1, float y1, Item it)
        {
            const int cx0 = juce::jlimit (0, cw - 1, (int) (x0 / cell)), cx1 = juce::jlimit (0, cw - 1, (int) (x1 / cell));
            const int cy0 = juce::jlimit (0, ch - 1, (int) (y0 / cell)), cy1 = juce::jlimit (0, ch - 1, (int) (y1 / cell));
            for (int y = cy0; y <= cy1; ++y)
                for (int x = cx0; x <= cx1; ++x)
                    cells[(size_t) (y * cw + x)].push_back (it);
        }

        const std::vector<Item>& at (float bx, float by) const
        {
            const int x = juce::jlimit (0, cw - 1, (int) (bx / cell)), y = juce::jlimit (0, ch - 1, (int) (by / cell));
            return cells[(size_t) (y * cw + x)];
        }
    };

    bool cancelled (const std::atomic<bool>* cancel) noexcept { return cancel != nullptr && cancel->load(); }

    // the worn enamel-on-aluminium faceplate at any resolution (level = texture pixels per editor pixel)
    juce::Image renderPaper (float level, const std::vector<Scuff>& scuffs, const std::atomic<bool>* cancel)
    {
        const int W = juce::roundToInt ((float) kW * level), H = juce::roundToInt ((float) kH * level);
        const auto& F = fields();
        const auto screws = screwPositions();
        const Influences inf (scuffs, screws);

        // deep scratches: gear slid in and out of the rack, keys, screwdrivers that slipped
        juce::Image scratchImg (juce::Image::SingleChannel, W, H, true, juce::SoftwareImageType());
        {
            juce::Graphics g (scratchImg);
            g.addTransform (juce::AffineTransform::scale (level));
            juce::Random rs (0x53435241);
            for (int i = 0; i < 24; ++i)
            {
                // a gouge: tapered at both ends, skipping where the tool lifted
                const float x0 = rs.nextFloat() * (float) kW, y0 = rs.nextFloat() * (float) kH;
                const float len = 20.0f + 150.0f * std::pow (rs.nextFloat(), 1.5f);
                // diagonal, so a scratch under a label never reads as a strike-through
                const float ang = (rs.nextFloat() < 0.5f ? -1.0f : 1.0f) * (0.30f + 0.65f * rs.nextFloat());
                const float bend = (rs.nextFloat() - 0.5f) * len * 0.08f;
                const juce::Point<float> a (x0, y0), b (x0 + len * std::cos (ang), y0 + len * std::sin (ang));
                const juce::Point<float> m = (a + b) * 0.5f + juce::Point<float> (-std::sin (ang), std::cos (ang)) * bend;
                const float w0 = 0.5f + 0.9f * rs.nextFloat();
                constexpr int segs = 24;
                auto at = [&] (float t) { return a * ((1 - t) * (1 - t)) + m * (2 * t * (1 - t)) + b * (t * t); };
                for (int k = 0; k < segs; ++k)
                {
                    const float t0 = (float) k / segs, t1 = (float) (k + 1) / segs;
                    if (rs.nextFloat() < 0.12f)
                        continue;
                    const float taper = std::sqrt (std::sin (juce::MathConstants<float>::pi * 0.5f * (t0 + t1)));
                    g.setColour (juce::Colours::white.withAlpha (0.45f + 0.55f * rs.nextFloat()));
                    g.drawLine (juce::Line<float> (at (t0), at (t1)), juce::jmax (0.25f, w0 * taper * (0.6f + 0.4f * rs.nextFloat())));
                }
            }
            for (int i = 0; i < 150; ++i)
            {
                const float x0 = rs.nextFloat() * (float) kW, y0 = rs.nextFloat() * (float) kH;
                const float len = 1.5f + 16.0f * std::pow (rs.nextFloat(), 2.0f), ang = rs.nextFloat() * 6.283f;
                g.setColour (juce::Colours::white.withAlpha (0.5f + 0.5f * rs.nextFloat()));
                g.drawLine (x0, y0, x0 + len * std::cos (ang), y0 + len * std::sin (ang), 0.4f + 0.6f * rs.nextFloat());
            }
            for (const auto& c : screws)
                for (int k = 0, n = 2 + rs.nextInt (4); k < n; ++k)
                {
                    const float a = rs.nextFloat() * 6.283f;
                    const juce::Point<float> u (std::cos (a), std::sin (a));
                    g.setColour (juce::Colours::white.withAlpha (0.7f + 0.3f * rs.nextFloat()));
                    g.drawLine (juce::Line<float> (c + u * (kScrewR + 0.4f), c + u * (kScrewR + 2.0f + 7.0f * rs.nextFloat())),
                                0.6f + 0.5f * rs.nextFloat());
                }
        }
        if (cancelled (cancel))
            return {};

        juce::Image img (juce::Image::ARGB, W, H, false, juce::SoftwareImageType());
        {
            const juce::Image::BitmapData sb (scratchImg, juce::Image::BitmapData::readOnly);
            juce::Image::BitmapData bd (img, juce::Image::BitmapData::writeOnly);
            const juce::Colour face = theme::face;
            const float tapeCos = std::cos (0.045f), tapeSin = std::sin (0.045f);

            parallelRows (H, [&] (int py)
            {
                for (int px = 0; px < W; ++px)
                {
                    const float bx = ((float) px + 0.5f) / level, by = ((float) py + 0.5f) / level;

                    float dMin = 1.0e9f, dShadow = 1.0e9f;
                    int pi = -1;
                    for (int k = 0; k < (int) std::size (kPlates); ++k)
                    {
                        const float d = sdRoundRect (bx, by, kPlates[k], 4.0f);
                        if (d < dMin) { dMin = d; pi = k; }
                        dShadow = std::min (dShadow, sdRoundRect (bx, by - 1.8f, kPlates[k], 4.0f));
                    }
                    juce::Colour col = face;
                    if (pi >= 0)
                        col = face.interpolatedWith (juce::Colour (kPlates[pi].shade), juce::jlimit (0.0f, 1.0f, 0.5f - dMin * level));

                    float l = 1.0f, yel = 0.0f;
                    if (dMin < 0.0f)   // dirt collects along the edges of every module
                    {
                        const float gi = std::exp (dMin / 3.0f), gm = F.grime.at (bx, by);
                        l -= 0.055f * gi * (0.5f + gm);
                        yel += 0.35f * gi * gm;
                    }
                    else               // each module throws a small shadow; the chassis around them is grimier
                    {
                        l -= 0.09f * std::exp (-std::max (0.0f, dShadow) / 2.0f);
                        const float gm = F.grime.at (bx * 0.7f, by * 0.7f);
                        l -= 0.035f * gm;
                        yel += 0.25f * gm;
                    }

                    yel += 0.55f * smooth (0.42f, 0.80f, F.stain.at (bx, by));
                    const float wt = F.water.at (bx, by);
                    const float ringD = (wt - 0.69f) / 0.010f;
                    const float ring = std::exp (-ringD * ringD);
                    l -= 0.035f * ring;
                    yel += 0.30f * ring + 0.18f * smooth (0.69f, 0.80f, wt);

                    // grain in editor units (the same at every resolution) plus a touch per pixel
                    l += (F.mottle.at (bx, by) - 0.5f) * 0.05f;
                    l += (valueNoise (bx * 1.25f, by * 1.25f, 7u) - 0.5f) * 0.035f + (rnd01 (px, py, 8u) - 0.5f) * 0.012f;
                    {
                        const int qx = (int) (bx * 1.25f), qy = (int) (by * 1.25f);
                        if (rnd01 (qx, qy, 9u) > 0.9975f)
                            l -= 0.10f * rnd01 (qx, qy, 10u);
                    }

                    const float e = std::min (std::min (bx, (float) kW - bx), std::min (by, (float) kH - by));
                    yel += 0.8f * std::exp (-e / 26.0f);
                    l -= 0.07f * std::exp (-e / 12.0f);
                    l -= 0.03f * smooth (700.0f, 800.0f, by) * F.grime.at (bx * 0.5f, by * 0.5f);   // wrists

                    for (const auto& d : kDents)   // dents, lit from the top left
                    {
                        const float dx = bx - d.x, dy = by - d.y, q = (dx * dx + dy * dy) / (d.r * d.r);
                        if (q < 1.0f)
                            l += 0.09f * (dx + dy) / (d.r * 1.41f) * (1.0f - q);
                    }

                    // where a piece of tape used to be (top right of the clock module)
                    const float tx = bx - 200.0f, ty = by - 336.0f;
                    const float tu = tapeCos * tx + tapeSin * ty, tv = -tapeSin * tx + tapeCos * ty;
                    const float dTape = std::max (std::abs (tu) - 48.0f, std::abs (tv) - 8.0f) + (F.grime.at (bx * 3.0f, by * 3.0f) - 0.5f) * 3.0f;
                    if (dTape < 0.0f) { yel += 0.20f; l -= 0.01f; }
                    l -= 0.05f * std::exp (-(dTape * dTape) / 0.8f);

                    // ---- how far the paint is worn through: panel edges (the front lip most), module edges and corners, flakes
                    const float ev = smooth (0.30f, 0.70f, F.edgeVar.at (bx, by));
                    float w = 0.9f * std::exp (-e / 4.5f) * (0.5f + 0.8f * ev) + 0.5f * smooth (778.0f, 800.0f, by) * (0.4f + 0.8f * ev);
                    if (pi >= 0 && dMin < 0.0f)
                    {
                        const auto& P = kPlates[pi];
                        w += 0.90f * std::exp (dMin / 2.4f) * ev;
                        const float cxd = std::min (std::abs (bx - P.x), std::abs (bx - (P.x + P.w)));
                        const float cyd = std::min (std::abs (by - P.y), std::abs (by - (P.y + P.h)));
                        w += 0.55f * std::exp (-std::sqrt (cxd * cxd + cyd * cyd) / 9.0f) * (0.5f + 0.5f * ev);
                    }
                    // flakes chip off near the module edges (where things bump into it), not under the printed labels
                    const float zone = smooth (0.40f, 0.62f, F.zones.at (bx, by));
                    float flake = smooth (0.75f, 0.81f, F.flakes.at (bx, by)) * zone * (0.25f + 0.75f * std::exp (-std::abs (dMin) / 28.0f));
                    for (const auto& t : kLabelRows)
                        if (t.contains (bx, by))
                            flake *= 0.1f;
                    w += 1.2f * flake + 0.18f * zone;

                    // knobs, faders and screws nearby: grease, rubbed skirts, polished slots, screwdriver damage
                    for (const auto& it : inf.at (bx, by))
                    {
                        if (it.kind == 2)
                        {
                            const auto& c = screws[(size_t) it.index];
                            const float d = std::hypot (bx - c.x, by - c.y) / kScrewR;
                            w += 0.7f * (1.0f - smooth (1.3f, 2.6f, d));
                            continue;
                        }
                        const auto& s = scuffs[(size_t) it.index];
                        if (it.kind == 1)
                        {
                            const float u = std::abs (bx - s.x) / 15.0f, v = std::abs (by - s.y) / (s.r + 6.0f);
                            float m = (1.0f - smooth (0.3f, 1.0f, u)) * (1.0f - smooth (0.85f, 1.0f, v));
                            m *= (0.3f + 0.9f * F.grime.at (bx * 2.0f, by * 2.0f)) * s.strength;
                            l -= 0.05f * m;
                            yel += 0.30f * m;
                            const float bandD = (std::abs (bx - s.x) - 6.0f) / 2.5f;
                            w += 0.65f * std::exp (-bandD * bandD) * (1.0f - smooth (0.9f, 1.0f, v)) * s.strength;
                        }
                        else
                        {
                            const float dx = bx - s.x, dy = by - s.y;
                            const float d = std::sqrt (dx * dx + dy * dy) / s.r;
                            if (d > 1.95f)
                                continue;
                            const float side = 0.55f + 0.45f * std::cos (std::atan2 (dy, dx) - 2.3f);   // the thumb side
                            const float greaseM = smooth (0.92f, 1.08f, d) * (1.0f - smooth (1.15f, 1.9f, d));
                            const float m = greaseM * side * (0.35f + 0.9f * F.grime.at (bx * 1.7f, by * 1.7f)) * s.strength;
                            l -= 0.10f * m;
                            yel += 0.45f * m;
                            const float skirt = smooth (0.98f, 1.05f, d) * (1.0f - smooth (1.20f, 1.36f, d));
                            w += 0.66f * skirt * (0.40f + 0.60f * side) * s.strength;
                        }
                    }

                    // ---- paint or bare metal
                    const float scratch = (float) *sb.getPixelPointer (px, py) * (1.0f / 255.0f);
                    const float wv = w * (0.55f + 0.9f * F.ragged.at (bx, by)) + (valueNoise (bx * 2.2f, by * 2.2f, 63u) - 0.5f) * 0.08f
                                     + 2.2f * scratch;
                    const float paint = 1.0f - smooth (0.50f, 0.58f, wv);           // 1 = paint intact
                    const float thin = smooth (0.30f, 0.50f, wv);                   // wearing thin: greyer, dirtier
                    const float rimD = (wv - 0.55f) / 0.05f;
                    const float rim = std::exp (-rimD * rimD);                      // dirt caught in the paint edge

                    const juce::uint32 c = col.getARGB();
                    const float k = juce::jlimit (0.0f, 1.0f, yel) * 0.22f;
                    const float lc = juce::jlimit (0.55f, 1.12f, l) * (1.0f - 0.05f * thin);
                    float pr = ((float) ((c >> 16) & 0xff) * (1.0f - k) + 196.0f * k) * lc;
                    float pg = ((float) ((c >> 8) & 0xff)  * (1.0f - k) + 176.0f * k) * lc;
                    float pb = ((float) (c & 0xff)         * (1.0f - k) + 132.0f * k) * lc;
                    const float primer = 0.5f * thin;
                    pr += (190.0f - pr) * primer;
                    pg += (185.0f - pg) * primer;
                    pb += (173.0f - pb) * primer;

                    // brushed aluminium underneath, dulled by dirt
                    const float b1 = valueNoise (bx * 0.012f, by * 2.4f, 81u), b2 = valueNoise (bx * 0.05f, by * 4.1f, 83u);
                    const float sheen = 0.5f + 0.5f * std::sin ((bx * 0.7f + by * 1.3f) * 0.004f + 1.0f);
                    float m = (0.80f + 0.16f * b1 + 0.08f * b2 + 0.07f * sheen) * (1.0f - 0.22f * F.grime.at (bx, by));
                    m *= 1.0f - 0.5f * k;
                    const float mr = 178.0f * m, mg = 178.0f * m, mb = 173.0f * m;

                    const float dark = 1.0f - 0.30f * rim;
                    const juce::PixelARGB p (255,
                                             (juce::uint8) juce::jlimit (0.0f, 255.0f, (mr + (pr - mr) * paint) * dark),
                                             (juce::uint8) juce::jlimit (0.0f, 255.0f, (mg + (pg - mg) * paint) * dark),
                                             (juce::uint8) juce::jlimit (0.0f, 255.0f, (mb + (pb - mb) * paint) * dark));
                    if (bd.pixelFormat == juce::Image::ARGB && bd.pixelStride == 4)
                        std::memcpy (bd.getPixelPointer (px, py), &p, 4);
                    else
                        bd.setPixelColour (px, py, juce::Colour (p.getRed(), p.getGreen(), p.getBlue()));
                }
            }, cancel);
        }
        if (cancelled (cancel))
            return {};

        {
            juce::Graphics g (img);
            g.addTransform (juce::AffineTransform::scale (level));

            // each module is pressed slightly into the chassis: light top edge, dark bottom edge
            for (const auto& p : kPlates)
            {
                g.setColour (juce::Colours::white.withAlpha (0.55f));
                g.drawLine (p.x + 4.0f, p.y + 0.6f, p.x + p.w - 4.0f, p.y + 0.6f, 0.9f);
                g.setColour (juce::Colours::white.withAlpha (0.28f));
                g.drawLine (p.x + 0.6f, p.y + 4.0f, p.x + 0.6f, p.y + p.h - 4.0f, 0.9f);
                g.setColour (juce::Colours::black.withAlpha (0.12f));
                g.drawLine (p.x + 4.0f, p.y + p.h - 0.6f, p.x + p.w - 4.0f, p.y + p.h - 0.6f, 0.9f);
                g.setColour (juce::Colours::black.withAlpha (0.07f));
                g.drawLine (p.x + p.w - 0.6f, p.y + 4.0f, p.x + p.w - 0.6f, p.y + p.h - 4.0f, 0.9f);
            }

            // the fader slots
            for (const auto& s : scuffs)
                if (s.fader)
                {
                    const juce::Rectangle<float> slot (s.x - 1.6f, s.y - s.r, 3.2f, 2.0f * s.r);
                    g.setColour (juce::Colour (0xff1e1d1b).withAlpha (0.85f));
                    g.fillRoundedRectangle (slot, 1.6f);
                    g.setColour (juce::Colours::white.withAlpha (0.45f));
                    g.drawLine (slot.getX() + 0.5f, slot.getBottom() + 0.6f, slot.getRight() - 0.5f, slot.getBottom() + 0.6f, 0.7f);
                    g.drawLine (slot.getRight() + 0.5f, slot.getY() + 2.0f, slot.getRight() + 0.5f, slot.getBottom() - 1.0f, 0.5f);
                }

            // a mug was put down here more than once
            juce::Random r (0x434f4646);
            const juce::Point<float> mug (962.0f, 742.0f);
            for (int k = 0; k < 5; ++k)
            {
                const float rad = 45.0f + (r.nextFloat() - 0.5f) * 3.0f;
                const float a0 = r.nextFloat() * 6.283f, len = 2.6f + 2.4f * r.nextFloat();
                juce::Path arc;
                arc.addCentredArc (mug.x + (r.nextFloat() - 0.5f) * 2.0f, mug.y + (r.nextFloat() - 0.5f) * 2.0f,
                                   rad, rad * 0.98f, 0.0f, a0, a0 + len, true);
                g.setColour (juce::Colour (0xff6e4e2d).withAlpha (0.045f + 0.045f * r.nextFloat()));
                g.strokePath (arc, juce::PathStrokeType (1.0f + 1.8f * r.nextFloat()));
            }
            g.setColour (juce::Colour (0xff6e4e2d).withAlpha (0.022f));
            g.fillEllipse (mug.x - 44.0f, mug.y - 44.0f, 88.0f, 88.0f);

            // pencil recall marks next to a few knobs
            for (const auto& s : scuffs)
                if (! s.fader && s.mark >= 0.0f)
                {
                    const float a = juce::MathConstants<float>::pi * (1.2f + 1.6f * s.mark);
                    const juce::Point<float> u (std::sin (a), -std::cos (a));
                    const juce::Point<float> c (s.x, s.y);
                    g.setColour (juce::Colour (0xff57534d).withAlpha (0.55f));
                    g.drawLine (juce::Line<float> (c + u * (s.r + 9.0f), c + u * (s.r + 14.0f)), 1.2f);
                    g.setColour (juce::Colour (0xff57534d).withAlpha (0.25f));
                    g.drawLine (juce::Line<float> (c + u * (s.r + 9.5f) + juce::Point<float> (0.6f, 0.4f),
                                                   c + u * (s.r + 13.5f) + juce::Point<float> (0.6f, 0.4f)), 0.8f);
                }

            juce::Random rsc (0x5343524f);
            for (const auto& c : screws)
                screw (g, rsc, c, kScrewR);
        }
        return img;
    }

    const PixelCache& paperCache (const std::vector<Scuff>& scuffs)
    {
        static PixelCache cache;
        if (cache.px.empty())
            cache = cacheFromImage (renderPaper (kBaseLevel, scuffs, nullptr));
        return cache;
    }

    // dust, polish scratches, rubbed-off print, a fingerprint and dried blood, at any resolution
    juce::Image renderWear (float level, const std::atomic<bool>* cancel)
    {
        const int W = juce::roundToInt ((float) kW * level), H = juce::roundToInt ((float) kH * level);
        juce::Image img (juce::Image::ARGB, W, H, true, juce::SoftwareImageType());
        juce::Random r (0x4d41564b);
        {
            juce::Graphics g (img);
            g.addTransform (juce::AffineTransform::scale (level));

            // polish scratches from years of hands and cables sliding over it
            for (int i = 0; i < 22; ++i)
            {
                const float x0 = r.nextFloat() * (float) kW, y0 = r.nextFloat() * (float) kH;
                const float len = 6.0f + 120.0f * std::pow (r.nextFloat(), 2.2f);
                const float ang = (r.nextFloat() < 0.5f ? -1.0f : 1.0f) * (0.30f + 0.65f * r.nextFloat());
                const float bend = (r.nextFloat() - 0.5f) * len * 0.12f;
                const float x1 = x0 + len * std::cos (ang), y1 = y0 + len * std::sin (ang);
                juce::Path p;
                p.startNewSubPath (x0, y0);
                p.quadraticTo (0.5f * (x0 + x1) - bend * std::sin (ang), 0.5f * (y0 + y1) + bend * std::cos (ang), x1, y1);
                g.setColour (juce::Colours::white.withAlpha (0.22f + 0.30f * r.nextFloat()));
                g.strokePath (p, juce::PathStrokeType (0.45f + 0.5f * r.nextFloat()));
            }
            for (int i = 0; i < 26; ++i)
            {
                const float x0 = r.nextFloat() * (float) kW, y0 = r.nextFloat() * (float) kH;
                const float len = 4.0f + 50.0f * std::pow (r.nextFloat(), 2.0f);
                const float ang = r.nextFloat() * 6.283f;
                g.setColour (theme::ink.withAlpha (0.05f + 0.07f * r.nextFloat()));
                g.drawLine (x0, y0, x0 + len * std::cos (ang), y0 + len * std::sin (ang), 0.35f + 0.25f * r.nextFloat());
            }

            // dust, more of it along the edges of the plates
            for (int i = 0; i < 560; ++i)
            {
                float x = r.nextFloat() * (float) kW, y = r.nextFloat() * (float) kH;
                if (r.nextFloat() < 0.35f)
                {
                    const auto& p = kPlates[r.nextInt ((int) std::size (kPlates))];
                    const float t = r.nextFloat();
                    switch (r.nextInt (4))
                    {
                        case 0:  x = p.x + t * p.w; y = p.y; break;
                        case 1:  x = p.x + t * p.w; y = p.y + p.h; break;
                        case 2:  x = p.x; y = p.y + t * p.h; break;
                        default: x = p.x + p.w; y = p.y + t * p.h; break;
                    }
                    x += (r.nextFloat() - 0.5f) * 6.0f;
                    y += (r.nextFloat() - 0.5f) * 6.0f;
                }
                const float rad = 0.3f + 1.0f * std::pow (r.nextFloat(), 3.0f);
                g.setColour (juce::Colour (0xff2a2622).withAlpha (0.10f + 0.35f * r.nextFloat()));
                g.fillEllipse (x - rad, y - rad, 2.0f * rad, 2.0f * rad * (0.7f + 0.6f * r.nextFloat()));
            }

            // somebody touched the handwriting display
            {
                const auto t = juce::AffineTransform::rotation (0.45f).translated (396.0f, 138.0f);
                for (int k = 2; k <= 15; ++k)
                {
                    const float rx = 0.95f * (float) k, ry = 1.30f * (float) k;
                    float a = r.nextFloat() * 0.6f;
                    while (a < 6.2f)
                    {
                        const float len = 0.5f + 1.3f * r.nextFloat();
                        juce::Path arc;
                        arc.addCentredArc (0.0f, 0.0f, rx, ry, 0.0f, a, juce::jmin (6.283f, a + len), true);
                        arc.applyTransform (t);
                        g.setColour (theme::ink.withAlpha (0.075f * (1.0f - (float) k / 17.0f)));
                        g.strokePath (arc, juce::PathStrokeType (0.55f));
                        a += len + 0.12f + 0.3f * r.nextFloat();
                    }
                }
            }

            // lint and a couple of hairs
            for (int i = 0; i < 7; ++i)
            {
                const juce::Point<float> a (r.nextFloat() * (float) kW, r.nextFloat() * (float) kH);
                const float len = 8.0f + 26.0f * r.nextFloat(), ang = r.nextFloat() * 6.283f;
                const juce::Point<float> u (std::cos (ang), std::sin (ang)), n (-u.y, u.x);
                juce::Path p;
                p.startNewSubPath (a);
                p.cubicTo (a + u * (len * 0.3f) + n * (len * 0.25f * (r.nextFloat() - 0.5f)),
                           a + u * (len * 0.7f) + n * (len * 0.30f * (r.nextFloat() - 0.5f)),
                           a + u * len);
                g.setColour (juce::Colour (0xff3a3530).withAlpha (0.18f + 0.12f * r.nextFloat()));
                g.strokePath (p, juce::PathStrokeType (0.4f));
            }

            // print rubbed off where fingers rest: the title, the macro labels, the tabs
            const juce::Rectangle<float> touched[] =
            {
                { 20.0f, 6.0f, 170.0f, 38.0f }, { 462.0f, 96.0f, 576.0f, 22.0f }, { 22.0f, 186.0f, 420.0f, 20.0f },
                { 284.0f, 330.0f, 756.0f, 30.0f }, { 26.0f, 618.0f, 830.0f, 20.0f }, { 26.0f, 70.0f, 140.0f, 20.0f },
                { 466.0f, 70.0f, 80.0f, 20.0f }
            };
            for (const auto& t : touched)
            {
                const int count = (int) (t.getWidth() * t.getHeight() / 38.0f);
                for (int i = 0; i < count; ++i)
                {
                    const float x = t.getX() + r.nextFloat() * t.getWidth(), y = t.getY() + r.nextFloat() * t.getHeight();
                    const float rad = 0.25f + 0.40f * std::pow (r.nextFloat(), 2.0f);
                    g.setColour (theme::paperHi.withAlpha (0.55f + 0.40f * r.nextFloat()));
                    g.fillEllipse (x - rad, y - rad, 2.0f * rad, 2.0f * rad);
                }
            }
            for (int i = 0; i < 320; ++i)
            {
                const float x = r.nextFloat() * (float) kW, y = r.nextFloat() * (float) kH;
                const float rad = 0.25f + 0.6f * std::pow (r.nextFloat(), 2.0f);
                g.setColour (theme::paperHi.withAlpha (0.45f + 0.45f * r.nextFloat()));
                g.fillEllipse (x - rad, y - rad, 2.0f * rad, 2.0f * rad);
            }
        }

        // a little blood: one splash with a drip, two smaller ones, and a fine mist where the clipper cuts
        juce::Image blood (juce::Image::ARGB, W, H, true, juce::SoftwareImageType());
        {
            juce::Graphics g (blood);
            g.addTransform (juce::AffineTransform::scale (level));
            splatter (g, r, { 318.0f, 30.0f }, 10.0f, juce::MathConstants<float>::pi * 1.1f, 1.4f, 26, 44.0f);
            splatter (g, r, { 1036.0f, 300.0f }, 6.5f, juce::MathConstants<float>::pi, 1.2f, 14, 0.0f);
            splatter (g, r, { 11.0f, 792.0f }, 9.0f, juce::MathConstants<float>::pi * 0.8f, 1.2f, 18, 0.0f);
            for (int k = 0; k < 22; ++k)
            {
                const float a = -0.7f + 1.2f * r.nextFloat();
                const float dist = 31.0f + 18.0f * std::pow (r.nextFloat(), 0.7f);
                const float s = 0.3f + 0.7f * std::pow (r.nextFloat(), 3.0f);
                const auto pos = juce::Point<float> (686.0f, 162.0f) + juce::Point<float> (std::cos (a), std::sin (a)) * dist;
                g.setColour (theme::blood.withAlpha (0.75f + 0.2f * r.nextFloat()));
                g.fillEllipse (pos.x - s, pos.y - s, 2.0f * s, 2.0f * s);
            }
        }
        {
            // dried: grainy and uneven
            juce::Image::BitmapData bd (blood, juce::Image::BitmapData::readWrite);
            if (bd.pixelFormat == juce::Image::ARGB)
                parallelRows (H, [&] (int y)
                {
                    for (int x = 0; x < W; ++x)
                    {
                        auto* p = reinterpret_cast<juce::PixelARGB*> (bd.getPixelPointer (x, y));
                        if (p->getAlpha() == 0)
                            continue;
                        const float n = 0.78f + 0.12f * valueNoise ((float) x / level * 0.44f, (float) y / level * 0.44f, 91u)
                                        + 0.10f * rnd01 (x, y, 93u);
                        p->multiplyAlpha (juce::jmin (1.0f, n));
                    }
                }, cancel);
        }
        if (cancelled (cancel))
            return {};
        {
            juce::Graphics g (img);
            g.drawImageAt (blood, 0, 0);
        }
        return img;
    }

    const PixelCache& wearCache()
    {
        static PixelCache cache;
        if (cache.px.empty())
            cache = cacheFromImage (renderWear (kBaseLevel, nullptr));
        return cache;
    }

    void drawFlow (juce::Graphics& g)
    {
        // the signal path printed on the plate, like on a channel strip
        const char* items[] = { "IN", "CRUSH", "LAMP \xc2\xb7 GLASS \xc2\xb7 FOLD", "CLIP", "FLANGE", "HARP", "MIX", "AURA", "OUT" };
        constexpr int n = 9;
        const auto f = sans (10.5f, true, 0.12f);
        float widths[n], total = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            widths[i] = juce::GlyphArrangement::getStringWidth (f, u8 (items[i]));
            total += widths[i];
        }
        const float arrow = 13.0f, gap = 5.0f;
        total += (float) (n - 1) * (arrow + 2.0f * gap);
        float x = 452.0f + (596.0f - total) * 0.5f;
        const float y = 250.0f;
        float cx[n];
        g.setFont (f);
        for (int i = 0; i < n; ++i)
        {
            g.setColour (theme::inkSoft.withAlpha (0.9f));
            g.drawText (u8 (items[i]), juce::Rectangle<float> (x, y, widths[i] + 2.0f, 14.0f), juce::Justification::centredLeft, false);
            cx[i] = x + 0.5f * widths[i];
            x += widths[i];
            if (i < n - 1)
            {
                const float ax0 = x + gap, ax1 = x + gap + arrow, ay = y + 7.0f;
                g.setColour (theme::inkFaint);
                g.drawLine (ax0, ay, ax1, ay, 0.9f);
                g.drawLine (ax1 - 3.0f, ay - 2.5f, ax1, ay, 0.9f);
                g.drawLine (ax1 - 3.0f, ay + 2.5f, ax1, ay, 0.9f);
                x = ax1 + gap;
            }
        }

        // BLEED: what the clip cuts off runs on to the harp and the aura
        const float y0 = y + 16.0f;
        const float dash[] = { 3.0f, 2.2f };
        auto thread = [&] (float toX, float depth)
        {
            juce::Path p, dashed;
            p.startNewSubPath (cx[3], y0);
            p.cubicTo (cx[3], y0 + depth, toX, y0 + depth, toX, y0 + 1.5f);
            juce::PathStrokeType (1.0f).createDashedStroke (dashed, p, dash, 2);
            g.setColour (theme::blood.withAlpha (0.85f));
            g.fillPath (dashed);
            g.drawLine (toX - 2.5f, y0 + 4.5f, toX, y0 + 1.5f, 1.0f);
            g.drawLine (toX + 2.5f, y0 + 4.5f, toX, y0 + 1.5f, 1.0f);
        };
        thread (cx[5], 12.0f);
        thread (cx[7], 24.0f);
        g.setColour (theme::blood.withAlpha (0.9f));
        g.setFont (sans (10.0f, false, 0.06f));
        g.drawText ("bleed", juce::Rectangle<float> (0.5f * (cx[3] + cx[7]) - 30.0f, y0 + 18.5f, 60.0f, 12.0f),
                    juce::Justification::centred, false);
    }
}

juce::Image mavkaui::paperTexture (const std::vector<Scuff>& scuffs) { return imageFromCache (paperCache (scuffs)); }
juce::Image mavkaui::wearTexture() { return imageFromCache (wearCache()); }

TextureSet mavkaui::renderTextures (float level, const std::vector<Scuff>& scuffs, const std::atomic<bool>* cancel)
{
    TextureSet t;
    t.paper = renderPaper (level, scuffs, cancel);
    if (! cancelled (cancel))
        t.dirt = renderWear (level, cancel);
    if (cancelled (cancel) || ! t.paper.isValid() || ! t.dirt.isValid())
        return {};
    t.level = level;
    return t;
}

juce::Image mavkaui::toNativeImage (const juce::Image& src)
{
    if (! src.isValid())
        return {};
    juce::Image img (juce::Image::ARGB, src.getWidth(), src.getHeight(), false);
    {
        const juce::Image::BitmapData sd (src, juce::Image::BitmapData::readOnly);
        juce::Image::BitmapData dd (img, juce::Image::BitmapData::writeOnly);
        const bool raw = sd.pixelFormat == juce::Image::ARGB && dd.pixelFormat == juce::Image::ARGB
                         && sd.pixelStride == 4 && dd.pixelStride == 4;
        for (int y = 0; y < src.getHeight(); ++y)
        {
            if (raw)
                std::memcpy (dd.getLinePointer (y), sd.getLinePointer (y), (size_t) src.getWidth() * 4);
            else
                for (int x = 0; x < src.getWidth(); ++x)
                    dd.setPixelColour (x, y, sd.getPixelColour (x, y));
        }
    }
    return img;
}

float mavkaui::textureLevelFor (float physicalScale)
{
    // the base set, then three sharper steps; the last one is ~3700 x 2800 pixels (4K class)
    for (float level : { kBaseLevel, 1.75f, 2.5f, 3.5f })
        if (physicalScale <= level * 1.04f)
            return level;
    return 3.5f;
}

//==============================================================================
LookAndFeel::LookAndFeel()
{
    setColour (juce::ComboBox::backgroundColourId, juce::Colours::transparentBlack);
    setColour (juce::ComboBox::outlineColourId, theme::hair);
    setColour (juce::ComboBox::textColourId, theme::ink);
    setColour (juce::ComboBox::arrowColourId, theme::inkSoft);
    setColour (juce::PopupMenu::backgroundColourId, theme::paperHi);
    setColour (juce::PopupMenu::textColourId, theme::ink);
    setColour (juce::PopupMenu::headerTextColourId, theme::inkSoft);
    setColour (juce::PopupMenu::highlightedBackgroundColourId, juce::Colour (0xffe1ddd2));
    setColour (juce::PopupMenu::highlightedTextColourId, theme::ink);
    setColour (juce::TooltipWindow::backgroundColourId, theme::paperHi);
    setColour (juce::TooltipWindow::textColourId, theme::ink);
    setColour (juce::TooltipWindow::outlineColourId, theme::hair);
    setColour (juce::ToggleButton::textColourId, theme::inkSoft);
    setColour (juce::ToggleButton::tickColourId, theme::ink);
    setColour (juce::ToggleButton::tickDisabledColourId, theme::hair);
    setColour (juce::TextButton::buttonColourId, juce::Colours::transparentBlack);
    setColour (juce::TextButton::buttonOnColourId, juce::Colours::transparentBlack);
    setColour (juce::TextButton::textColourOffId, theme::inkFaint);
    setColour (juce::TextButton::textColourOnId, theme::ink);
    setColour (juce::Label::textColourId, theme::inkSoft);
    setColour (juce::ResizableWindow::backgroundColourId, theme::face);
}

void LookAndFeel::drawRotarySlider (juce::Graphics& g, int x, int y, int width, int height, float pos,
                                    float startA, float endA, juce::Slider& s)
{
    const auto bounds = juce::Rectangle<int> (x, y, width, height).toFloat().reduced (2.0f);
    const float R = juce::jmin (bounds.getWidth(), bounds.getHeight()) * 0.5f;
    const auto c = bounds.getCentre();
    const bool large = (bool) s.getProperties()["large"];
    const bool enabled = s.isEnabled();
    const bool blood = (bool) s.getProperties()["blood"];
    const float arcR = R - 1.5f;
    const float angle = startA + pos * (endA - startA);

    // printed scale: one hairline, a tick at each end (and the centre for bipolar controls)
    juce::Path track;
    track.addCentredArc (c.x, c.y, arcR, arcR, 0.0f, startA, endA, true);
    g.setColour (theme::hair);
    g.strokePath (track, juce::PathStrokeType (0.9f));
    const bool bipolar = (bool) s.getProperties()["bipolar"];
    auto tick = [&] (float a)
    {
        const juce::Point<float> u (std::sin (a), -std::cos (a));
        g.drawLine (juce::Line<float> (c + u * (arcR - 3.0f), c + u * (arcR + 1.5f)), 0.9f);
    };
    tick (startA);
    tick (endA);
    if (bipolar)
        tick (0.5f * (startA + endA));

    // value: a single ink stroke on the scale (dried blood for the clipper)
    const float from = bipolar ? 0.5f * (startA + endA) : startA;
    if (std::abs (angle - from) > 0.005f)
    {
        juce::Path v;
        v.addCentredArc (c.x, c.y, arcR, arcR, 0.0f, juce::jmin (from, angle), juce::jmax (from, angle), true);
        g.setColour ((blood ? theme::blood : theme::ink).withAlpha (enabled ? 0.92f : 0.35f));
        g.strokePath (v, juce::PathStrokeType (large ? 2.2f : 1.8f, juce::PathStrokeType::curved, juce::PathStrokeType::butt));
    }

    // the cap: white enamel on an aluminium knob, lit from above; the paint on its edge is worn to metal
    const float capR = R - (large ? 7.5f : 6.5f);
    g.setColour (juce::Colours::black.withAlpha (0.12f));
    g.fillEllipse (c.x - capR - 0.5f, c.y - capR + 1.6f, 2.0f * capR + 1.0f, 2.0f * capR + 1.0f);
    g.setColour (juce::Colours::black.withAlpha (0.05f));
    g.fillEllipse (c.x - capR - 1.5f, c.y - capR + 2.6f, 2.0f * capR + 3.0f, 2.0f * capR + 3.0f);
    g.setGradientFill (juce::ColourGradient (juce::Colour (0xffe6e5e0), c.x, c.y - capR, juce::Colour (0xff8e8c86), c.x, c.y + capR, false));
    g.fillEllipse (c.x - capR, c.y - capR, 2.0f * capR, 2.0f * capR);   // aluminium skirt
    const float faceR = capR - 1.6f;
    g.setGradientFill (juce::ColourGradient (theme::paperHi, c.x, c.y - faceR, juce::Colour (0xffe1ddd3), c.x, c.y + faceR, false));
    g.fillEllipse (c.x - faceR, c.y - faceR, 2.0f * faceR, 2.0f * faceR);
    g.setColour (juce::Colour (0xff3f3d39).withAlpha (0.55f));
    g.drawEllipse (c.x - capR, c.y - capR, 2.0f * capR, 2.0f * capR, 0.7f);
    g.setColour (theme::inkSoft.withAlpha (0.06f));
    g.drawEllipse (c.x - faceR + 1.6f, c.y - faceR + 1.6f, 2.0f * faceR - 3.2f, 2.0f * faceR - 3.2f, 2.2f);

    juce::Random rnd ((juce::int64) (s.getParentComponent() != nullptr ? s.getParentComponent()->getComponentID().hashCode() : 7));
    for (int k = 0; k < 3; ++k)   // enamel rubbed off the rim where fingers grip it
    {
        const float a0 = rnd.nextFloat() * 6.283f, len = 0.25f + 0.8f * rnd.nextFloat();
        juce::Path worn;
        worn.addCentredArc (c.x, c.y, faceR - 0.9f, faceR - 0.9f, 0.0f, a0, a0 + len, true);
        g.setColour (juce::Colour (0xffa4a29b).withAlpha (0.5f + 0.35f * rnd.nextFloat()));
        g.strokePath (worn, juce::PathStrokeType (1.2f + 1.4f * rnd.nextFloat(), juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
    }
    if (rnd.nextFloat() < 0.6f)   // a chip in the face
    {
        const float a = rnd.nextFloat() * 6.283f, d = faceR * (0.45f + 0.35f * rnd.nextFloat());
        const float cr = 0.8f + 1.3f * rnd.nextFloat();
        g.setColour (juce::Colour (0xff9d9b94).withAlpha (0.75f));
        g.fillEllipse (c.x + std::cos (a) * d - cr, c.y + std::sin (a) * d - cr * 0.8f, 2.0f * cr, 1.6f * cr);
    }

    // pointer
    const juce::Point<float> u (std::sin (angle), -std::cos (angle));
    juce::Path pointer;
    pointer.startNewSubPath (c + u * (capR * 0.30f));
    pointer.lineTo (c + u * (capR * 0.86f));
    g.setColour (theme::ink.withAlpha (enabled ? 0.9f : 0.4f));
    g.strokePath (pointer, juce::PathStrokeType (large ? 2.4f : 2.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
}

void LookAndFeel::drawLinearSlider (juce::Graphics& g, int x, int y, int width, int height, float sliderPos, float minPos,
                                    float maxPos, juce::Slider::SliderStyle style, juce::Slider& s)
{
    if (style != juce::Slider::LinearVertical)
    {
        LookAndFeel_V4::drawLinearSlider (g, x, y, width, height, sliderPos, minPos, maxPos, style, s);
        return;
    }
    const float cx = (float) x + (float) width * 0.5f;
    const float top = (float) y + 8.0f, bottom = (float) (y + height) - 8.0f;

    g.setColour (theme::hair);
    for (int i = 0; i <= 10; ++i)
    {
        const float ty = top + (bottom - top) * (float) i / 10.0f;
        const float len = (i % 5 == 0) ? 7.0f : 4.0f;
        g.drawLine (cx - 15.0f - len, ty, cx - 15.0f, ty, 0.8f);
    }

    // the cap rides in a slot cut into the metal (the slot is part of the panel texture)
    const float pos = juce::jlimit (top, bottom, sliderPos);
    const juce::Rectangle<float> cap (cx - 13.0f, pos - 6.5f, 26.0f, 13.0f);
    g.setColour (juce::Colours::black.withAlpha (0.16f));
    g.fillRoundedRectangle (cap.translated (0.0f, 1.8f).expanded (0.6f), 2.0f);
    g.setGradientFill (juce::ColourGradient (juce::Colour (0xffe6e5e0), 0.0f, cap.getY(), juce::Colour (0xff8e8c86), 0.0f, cap.getBottom(), false));
    g.fillRoundedRectangle (cap, 2.0f);
    const auto face = cap.reduced (1.4f);
    g.setGradientFill (juce::ColourGradient (theme::paperHi, 0.0f, face.getY(), juce::Colour (0xffe1ddd3), 0.0f, face.getBottom(), false));
    g.fillRoundedRectangle (face, 1.5f);
    g.setColour (juce::Colour (0xffa4a29b).withAlpha (0.8f));   // corners worn to the metal
    g.fillEllipse (face.getX() - 0.6f, face.getY() - 0.6f, 3.0f, 2.4f);
    g.fillEllipse (face.getRight() - 2.6f, face.getBottom() - 2.0f, 3.2f, 2.6f);
    g.setColour (juce::Colour (0xff3f3d39).withAlpha (0.55f));
    g.drawRoundedRectangle (cap, 2.0f, 0.7f);
    g.setColour (theme::ink.withAlpha (s.isEnabled() ? 0.9f : 0.4f));
    g.fillRect (face.getX() + 3.0f, pos - 0.7f, face.getWidth() - 6.0f, 1.4f);
}

void LookAndFeel::drawComboBox (juce::Graphics& g, int width, int height, bool, int, int, int, int, juce::ComboBox& box)
{
    const float w = (float) width, h = (float) height;
    g.setColour (theme::paperHi.withAlpha (box.isMouseOver (true) ? 0.55f : 0.28f));
    g.fillRect (0.0f, 0.0f, w, h - 1.0f);
    g.setColour (theme::inkSoft.withAlpha (0.6f));
    g.fillRect (0.0f, h - 1.0f, w, 1.0f);

    juce::Path chevron;
    const float ax = w - 10.0f, ay = h * 0.5f;
    chevron.startNewSubPath (ax - 3.5f, ay - 1.5f);
    chevron.lineTo (ax, ay + 2.0f);
    chevron.lineTo (ax + 3.5f, ay - 1.5f);
    g.setColour (theme::inkSoft);
    g.strokePath (chevron, juce::PathStrokeType (1.2f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
}

juce::Font LookAndFeel::getComboBoxFont (juce::ComboBox&) { return sans (12.5f); }

void LookAndFeel::positionComboBoxText (juce::ComboBox& box, juce::Label& label)
{
    label.setBounds (2, 1, box.getWidth() - 20, box.getHeight() - 3);
    label.setFont (getComboBoxFont (box));
}

juce::Font LookAndFeel::getPopupMenuFont() { return sans (14.0f); }

void LookAndFeel::drawPopupMenuBackground (juce::Graphics& g, int width, int height)
{
    g.fillAll (theme::paperHi);
    g.setColour (theme::hair);
    g.drawRect (0, 0, width, height, 1);
}

void LookAndFeel::drawButtonBackground (juce::Graphics& g, juce::Button& b, const juce::Colour&, bool over, bool down)
{
    // tabs: no box at all, just an ink underline under the open page
    const auto r = b.getLocalBounds().toFloat();
    if (b.getToggleState())
    {
        g.setColour (theme::ink.withAlpha (0.9f));
        g.fillRect (r.getX() + 6.0f, r.getBottom() - 2.0f, r.getWidth() - 12.0f, 2.0f);
    }
    else if (over || down)
    {
        g.setColour (theme::inkSoft.withAlpha (0.35f));
        g.fillRect (r.getX() + 6.0f, r.getBottom() - 1.0f, r.getWidth() - 12.0f, 1.0f);
    }
}

void LookAndFeel::drawButtonText (juce::Graphics& g, juce::TextButton& b, bool over, bool)
{
    g.setFont (sans (11.5f, true, 0.18f));
    g.setColour (b.getToggleState() ? theme::ink : (over ? theme::inkSoft : theme::inkFaint));
    g.drawFittedText (b.getButtonText(), b.getLocalBounds().withTrimmedBottom (3), juce::Justification::centred, 1, 0.8f);
}

void LookAndFeel::drawToggleButton (juce::Graphics& g, juce::ToggleButton& b, bool over, bool)
{
    const float h = (float) b.getHeight();
    const juce::Rectangle<float> box (2.0f, h * 0.5f - 5.5f, 11.0f, 11.0f);
    g.setColour (theme::paperHi.withAlpha (0.6f));
    g.fillRect (box);
    g.setColour (theme::inkSoft.withAlpha (over ? 0.9f : 0.7f));
    g.drawRect (box, 1.0f);
    if (b.getToggleState())
    {
        g.setColour (theme::ink);
        g.fillRect (box.reduced (2.5f));
    }
    g.setColour (b.getToggleState() ? theme::ink : theme::inkSoft);
    g.setFont (sans (12.0f));
    g.drawText (b.getButtonText(), juce::Rectangle<float> (19.0f, 0.0f, (float) b.getWidth() - 19.0f, h),
                juce::Justification::centredLeft, false);
}

juce::Rectangle<int> LookAndFeel::getTooltipBounds (const juce::String& tip, juce::Point<int> screenPos,
                                                     juce::Rectangle<int> parentArea)
{
    juce::AttributedString s;
    s.setJustification (juce::Justification::topLeft);
    s.append (tip, sans (13.0f), theme::ink);
    juce::TextLayout tl;
    tl.createLayoutWithBalancedLineLengths (s, 340.0f);
    const int w = (int) std::ceil (tl.getWidth()) + 20, h = (int) std::ceil (tl.getHeight()) + 14;
    return juce::Rectangle<int> (screenPos.x > parentArea.getCentreX() ? screenPos.x - (w + 12) : screenPos.x + 24,
                                 screenPos.y > parentArea.getCentreY() ? screenPos.y - (h + 6) : screenPos.y + 6, w, h)
        .constrainedWithin (parentArea);
}

void LookAndFeel::drawTooltip (juce::Graphics& g, const juce::String& text, int width, int height)
{
    g.fillAll (theme::paperHi);
    g.setColour (theme::hair);
    g.drawRect (0, 0, width, height, 1);
    juce::AttributedString s;
    s.setJustification (juce::Justification::topLeft);
    s.append (text, sans (13.0f), theme::ink);
    juce::TextLayout tl;
    tl.createLayoutWithBalancedLineLengths (s, (float) width - 20.0f);
    tl.draw (g, juce::Rectangle<float> (10.0f, 7.0f, (float) width - 20.0f, (float) height - 14.0f));
}

void LookAndFeel::drawCornerResizer (juce::Graphics& g, int w, int h, bool over, bool drag)
{
    g.setColour (theme::inkSoft.withAlpha (over || drag ? 0.8f : 0.4f));
    for (int i = 1; i <= 3; ++i)
    {
        const float d = (float) i * 4.0f;
        g.drawLine ((float) w - d, (float) h - 1.5f, (float) w - 1.5f, (float) h - d, 1.0f);
    }
}

//==============================================================================
Fader::Fader (juce::AudioProcessorValueTreeState& state, const juce::String& paramId, const juce::String& t,
              const juce::String& tooltip)
    : title (t)
{
    setComponentID (paramId);
    slider.setSliderStyle (juce::Slider::LinearVertical);
    slider.setTextBoxStyle (juce::Slider::NoTextBox, false, 0, 0);
    slider.setTooltip (tooltip);
    addAndMakeVisible (slider);
    attachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (state, paramId, slider);
    if (auto* p = state.getParameter (paramId))
        slider.setDoubleClickReturnValue (true, (double) p->convertFrom0to1 (p->getDefaultValue()));
    slider.onValueChange = [this] { repaint (valueArea); };
    entry.attach (*this, slider, 11.0f);
    slider.onTypeIn = [this] { entry.open (valueArea); };
}

void Fader::mouseDoubleClick (const juce::MouseEvent& e)
{
    if (valueArea.expanded (2).contains (e.getPosition()))
        entry.open (valueArea);
}

void Fader::resized()
{
    auto b = getLocalBounds();
    titleArea = b.removeFromTop (17);
    valueArea = b.removeFromBottom (16);
    slider.setBounds (b);
}

void Fader::paint (juce::Graphics& g)
{
    g.setColour (theme::ink.withAlpha (0.86f));
    g.setFont (sans (11.5f, true, 0.14f));
    g.drawFittedText (title.toUpperCase(), titleArea, juce::Justification::centred, 1, 0.75f);
    if (entry.isOpen())
        return;
    g.setColour (theme::inkSoft);
    g.setFont (numbers (11.0f));
    g.drawFittedText (slider.getTextFromValue (slider.getValue()), valueArea, juce::Justification::centred, 1, 0.8f);
}

//==============================================================================
void ValueEntry::attach (juce::Component& c, juce::Slider& s, float fontSize)
{
    owner = &c;
    slider = &s;
    editor.setJustification (juce::Justification::centred);
    editor.setFont (numbers (fontSize));
    editor.setColour (juce::TextEditor::backgroundColourId, theme::paperHi);
    editor.setColour (juce::TextEditor::textColourId, theme::ink);
    editor.setColour (juce::TextEditor::highlightColourId, theme::hair);
    editor.setColour (juce::TextEditor::outlineColourId, theme::inkSoft);
    editor.setColour (juce::TextEditor::focusedOutlineColourId, theme::ink);
    editor.setSelectAllWhenFocused (true);
    editor.onReturnKey = [this] { finish (true); };
    editor.onEscapeKey = [this] { finish (false); };
    editor.onFocusLost = [this] { finish (true); };
    c.addChildComponent (editor);
}

void ValueEntry::open (juce::Rectangle<int> where)
{
    if (owner == nullptr || slider == nullptr)
        return;
    editor.setBounds (where.expanded (4, 2));
    editor.setText (slider->getTextFromValue (slider->getValue()), juce::dontSendNotification);
    editor.setVisible (true);
    editor.toFront (false);
    editor.grabKeyboardFocus();
    editor.selectAll();
    owner->repaint();
}

void ValueEntry::type (const juce::String& text)
{
    if (owner == nullptr)
        return;
    open (owner->getLocalBounds().withSizeKeepingCentre (60, 16));
    editor.setText (text, juce::dontSendNotification);
    finish (true);
}

void ValueEntry::finish (bool apply)
{
    if (! editor.isVisible() || slider == nullptr)
        return;
    const auto text = editor.getText().trim();
    editor.setVisible (false);
    if (apply && text.isNotEmpty())
        slider->setValue (slider->getValueFromText (text), juce::sendNotificationSync);
    if (owner != nullptr)
        owner->repaint();
}

//==============================================================================
Knob::Knob (juce::AudioProcessorValueTreeState& state, const juce::String& paramId, const juce::String& t,
            bool isLarge, const juce::String& tooltip)
    : title (t), large (isLarge)
{
    setComponentID (paramId);
    slider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    slider.setTextBoxStyle (juce::Slider::NoTextBox, false, 0, 0);
    slider.getProperties().set ("bipolar", isBipolar (paramId));
    slider.getProperties().set ("large", large);
    slider.sensitivity = large ? 260 : 200;
    slider.setMouseDragSensitivity (slider.sensitivity);
    entry.attach (*this, slider, large ? 12.0f : 11.0f);
    slider.onTypeIn = [this] { entry.open (valueArea); };
    slider.setTooltip (tooltip);
    addAndMakeVisible (slider);

    attachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (state, paramId, slider);
    if (auto* p = state.getParameter (paramId))
        slider.setDoubleClickReturnValue (true, (double) p->convertFrom0to1 (p->getDefaultValue()));
    slider.onValueChange = [this] { repaint (valueArea); };
}

void Knob::resized()
{
    // title, knob and value stacked tightly; sizes are capped so the macros lead and the pages stay calm
    const auto b = getLocalBounds();
    const int th = large ? 18 : 16, vh = large ? 18 : 16;
    const int side = juce::jmax (16, juce::jmin (b.getWidth(), b.getHeight() - th - vh - 4, large ? 66 : 58));
    auto col = b.withSizeKeepingCentre (b.getWidth(), th + side + vh + 4);
    titleArea = col.removeFromTop (th);
    valueArea = col.removeFromBottom (vh);
    slider.setBounds (col.withSizeKeepingCentre (side, side));
}

void Knob::paint (juce::Graphics& g)
{
    g.setColour (theme::ink.withAlpha (0.86f));
    g.setFont (sans (large ? 12.0f : 11.0f, true, 0.12f));
    g.drawFittedText (title.toUpperCase(), titleArea, juce::Justification::centred, 1, 0.75f);
    if (entry.isOpen())
        return;
    g.setColour (theme::inkSoft);
    g.setFont (numbers (large ? 12.0f : 11.0f));
    g.drawFittedText (slider.getTextFromValue (slider.getValue()), valueArea, juce::Justification::centred, 1, 0.8f);
    if (getComponentID() == "detune")
    {
        // the tuning in cents as well (Auto-Tune shows it this way)
        const double c = slider.getValue();
        g.setColour (theme::inkFaint);
        g.setFont (numbers (10.0f));
        g.drawText (std::abs (c) < 0.05 ? juce::String ("A4 = 440") : (c > 0 ? "+" : "") + juce::String (c, 1) + " cents",
                    valueArea.translated (0, valueArea.getHeight()), juce::Justification::centred, false);
    }
}

void KnobSlider::mouseDown (const juce::MouseEvent& e)
{
    if (e.mods.isPopupMenu())
    {
        if (onTypeIn) onTypeIn();
        return;
    }
    setMouseDragSensitivity (e.mods.isShiftDown() ? sensitivity * 10 : sensitivity);   // Shift: fine
    juce::Slider::mouseDown (e);
}

void KnobSlider::mouseDrag (const juce::MouseEvent& e)
{
    if (! e.mods.isPopupMenu())
        juce::Slider::mouseDrag (e);
}

void KnobSlider::mouseUp (const juce::MouseEvent& e)
{
    if (e.mods.isPopupMenu())
        return;
    juce::Slider::mouseUp (e);
    setMouseDragSensitivity (sensitivity);
}

// the number under the knob (and Detune's cents line): double-click it to type a value
juce::Rectangle<int> Knob::numberArea() const
{
    return getComponentID() == "detune" ? valueArea.withHeight (valueArea.getHeight() * 2) : valueArea;
}

void Knob::mouseDoubleClick (const juce::MouseEvent& e)
{
    if (numberArea().expanded (4, 2).contains (e.getPosition()))
        entry.open (valueArea);
}

//==============================================================================
Choice::Choice (juce::AudioProcessorValueTreeState& state, const juce::String& paramId, const juce::String& t,
                const juce::String& tooltip)
    : title (t)
{
    setComponentID (paramId);
    if (auto* p = dynamic_cast<juce::AudioParameterChoice*> (state.getParameter (paramId)))
        box.addItemList (p->choices, 1);
    box.setTooltip (tooltip);
    addAndMakeVisible (box);
    attachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment> (state, paramId, box);
}

void Choice::resized()
{
    auto b = getLocalBounds();
    b.removeFromTop (16);
    box.setBounds (b.removeFromTop (26));
}

void Choice::paint (juce::Graphics& g)
{
    g.setColour (theme::inkFaint);
    g.setFont (sans (10.5f, true, 0.16f));
    g.drawText (title.toUpperCase(), getLocalBounds().removeFromTop (16), juce::Justification::centredLeft, false);
}

//==============================================================================
namespace
{
    // a strip pressed into the paper: darker, with a shadow under its top edge
    void recess (juce::Graphics& g, juce::Rectangle<float> r)
    {
        g.setColour (theme::ink.withAlpha (0.05f));
        g.fillRoundedRectangle (r, 2.0f);
        g.setGradientFill (juce::ColourGradient (juce::Colours::black.withAlpha (0.10f), 0.0f, r.getY(),
                                                 juce::Colours::black.withAlpha (0.0f), 0.0f, r.getY() + 7.0f, false));
        g.fillRect (r.withHeight (7.0f).reduced (1.0f, 0.0f));
        g.setColour (juce::Colours::white.withAlpha (0.55f));
        g.fillRect (r.getX() + 2.0f, r.getBottom() - 1.0f, r.getWidth() - 4.0f, 1.0f);
    }
}

void StrokeView::push (float stroke, float slantMs, float levelDb, float ink)
{
    const float lvl = mavka::smoothstep (-36.0f, -3.0f, levelDb);   // relative to the voice's recent loudness
    float shape;
    if (ink >= 0.0f) shape = 1.0f - 0.85f * ink * stroke;                  // thin at the start, then blooms
    else             shape = juce::jmin (1.0f, 0.55f - 0.6f * ink * stroke); // pressed at the start
    thick[(size_t) head] = 1.0f + 17.0f * lvl * shape;
    offset[(size_t) head] = juce::jlimit (-1.0f, 1.0f, slantMs / 2.5f) * 16.0f;
    head = (head + 1) % N;
}

void StrokeView::paint (juce::Graphics& g)
{
    const auto r = getLocalBounds().toFloat();
    recess (g, r);

    g.setColour (theme::hair.withAlpha (0.55f));
    for (int i = 1; i < 4; ++i)
        g.drawHorizontalLine ((int) (r.getY() + r.getHeight() * (float) i / 4.0f), r.getX() + 8.0f, r.getRight() - 8.0f);

    const float mid = r.getCentreY();
    juce::Path ribbon;
    for (int i = 0; i < N; ++i)
    {
        const int k = (head + i) % N;
        const float x = r.getX() + 6.0f + (r.getWidth() - 12.0f) * (float) i / (float) (N - 1);
        const float y = mid + offset[(size_t) k] - thick[(size_t) k];
        if (i == 0) ribbon.startNewSubPath (x, y);
        else        ribbon.lineTo (x, y);
    }
    for (int i = N - 1; i >= 0; --i)
    {
        const int k = (head + i) % N;
        const float x = r.getX() + 6.0f + (r.getWidth() - 12.0f) * (float) i / (float) (N - 1);
        ribbon.lineTo (x, mid + offset[(size_t) k] + thick[(size_t) k]);
    }
    ribbon.closeSubPath();
    g.setGradientFill (juce::ColourGradient (theme::ink.withAlpha (0.0f), r.getX(), mid,
                                             theme::ink.withAlpha (0.92f), r.getRight(), mid, false));
    g.fillPath (ribbon);

    const int last = (head + N - 1) % N;
    const float nx = r.getRight() - 6.0f, ny = mid + offset[(size_t) last];
    g.setColour (theme::ink);
    g.fillEllipse (nx - 2.5f, ny - 2.5f, 5.0f, 5.0f);
}

//==============================================================================
void ClockView::setState (const float* phases, const juce::String& chordName, const juce::String& rateName)
{
    for (int i = 0; i < 4; ++i)
        ph[i] = phases[i];
    chord = chordName;
    rate = rateName;
}

void ClockView::paint (juce::Graphics& g)
{
    const auto r = getLocalBounds().toFloat().reduced (6.0f);
    const float cx = r.getCentreX(), cy = r.getCentreY();
    const float R = 0.5f * juce::jmin (r.getWidth(), r.getHeight());
    const float twoPi = juce::MathConstants<float>::twoPi;

    g.setColour (theme::hair);
    for (int i = 0; i < 12; ++i)
    {
        const float a = twoPi * (float) i / 12.0f;
        const juce::Point<float> u (std::sin (a), -std::cos (a));
        const juce::Point<float> c (cx, cy);
        g.drawLine (juce::Line<float> (c + u * (R * (i % 3 == 0 ? 0.90f : 0.935f)), c + u * (R * 0.985f)), 0.9f);
    }

    for (int i = 0; i < 4; ++i)
    {
        const float rad = R * (0.44f + 0.15f * (float) i);
        g.setColour (theme::hair.withAlpha (0.85f));
        g.drawEllipse (cx - rad, cy - rad, rad * 2.0f, rad * 2.0f, 0.8f);

        const float a = twoPi * ph[i];
        juce::Path trail;
        trail.addCentredArc (cx, cy, rad, rad, 0.0f, a - 1.1f, a, true);
        g.setColour (theme::ink.withAlpha (0.20f));
        g.strokePath (trail, juce::PathStrokeType (2.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

        const float x = cx + rad * std::sin (a), y = cy - rad * std::cos (a);
        const float d = 4.6f - 0.4f * (float) i;
        g.setColour (theme::ink.withAlpha (0.92f));
        g.fillEllipse (x - d, y - d, 2.0f * d, 2.0f * d);
    }

    const auto name = chord.upToFirstOccurrenceOf (" ", false, false);
    const auto ratio = chord.fromFirstOccurrenceOf (" ", false, false);
    g.setColour (theme::ink.withAlpha (0.88f));
    g.setFont (sans (13.0f, true, 0.06f));
    g.drawText (name.toUpperCase(), juce::Rectangle<float> (cx - 50.0f, cy - 22.0f, 100.0f, 16.0f), juce::Justification::centred, false);
    g.setColour (theme::inkSoft);
    g.setFont (numbers (11.5f));
    g.drawText (ratio, juce::Rectangle<float> (cx - 50.0f, cy - 5.0f, 100.0f, 14.0f), juce::Justification::centred, false);
    g.setColour (theme::inkFaint);
    g.setFont (sans (10.5f));
    g.drawText ("cycle " + rate, juce::Rectangle<float> (cx - 50.0f, cy + 9.0f, 100.0f, 14.0f), juce::Justification::centred, false);
}

//==============================================================================
void DuckMap::setGains (const float* g)
{
    for (int b = 0; b < 16; ++b)
        gains[b] = g[b];
}

void DuckMap::paint (juce::Graphics& g)
{
    const auto r = getLocalBounds().toFloat();
    recess (g, r);
    const auto inner = r.reduced (5.0f, 6.0f).withTrimmedBottom (12.0f);
    const float bw = inner.getWidth() / 16.0f;
    for (int b = 0; b < 16; ++b)
    {
        const float db = juce::jlimit (0.0f, 18.0f, -20.0f * std::log10 (juce::jmax (gains[b], 1.0e-3f)));
        const float h = juce::jmax (1.0f, inner.getHeight() * db / 18.0f);
        g.setColour (theme::ink.withAlpha (0.18f + 0.72f * db / 18.0f));
        g.fillRect (juce::Rectangle<float> (inner.getX() + (float) b * bw + 0.5f, inner.getY(), bw - 1.0f, h));
    }
    g.setColour (theme::inkFaint);
    g.setFont (sans (9.5f));
    g.drawText (u8 ("low \xe2\x80\x93 high"), r.withTop (r.getBottom() - 15.0f), juce::Justification::centred, false);
}

//==============================================================================
void BleedDrop::setLevel (float bite, float bleedAmount)
{
    if (std::abs (bite - level) < 0.004f && std::abs (bleedAmount - bleed) < 0.004f)
        return;
    level = bite;
    bleed = bleedAmount;
    repaint();
}

void BleedDrop::paint (juce::Graphics& g)
{
    const auto r = getLocalBounds().toFloat().reduced (1.0f);
    const float cx = r.getCentreX(), rr = r.getWidth() * 0.5f, cyc = r.getBottom() - rr, top = r.getY();
    juce::Path drop;
    drop.startNewSubPath (cx, top);
    drop.cubicTo (cx + rr * 0.15f, top + (cyc - top) * 0.35f, cx + rr, cyc - rr * 0.75f, cx + rr, cyc);
    drop.cubicTo (cx + rr, cyc + rr * 0.552f, cx + rr * 0.552f, cyc + rr, cx, cyc + rr);
    drop.cubicTo (cx - rr * 0.552f, cyc + rr, cx - rr, cyc + rr * 0.552f, cx - rr, cyc);
    drop.cubicTo (cx - rr, cyc - rr * 0.75f, cx - rr * 0.15f, top + (cyc - top) * 0.35f, cx, top);
    drop.closeSubPath();

    const float lv = juce::jlimit (0.0f, 1.0f, level);
    if (lv > 0.005f)
    {
        juce::Graphics::ScopedSaveState save (g);
        g.reduceClipRegion (drop);
        const float fillTop = r.getBottom() - r.getHeight() * lv;
        g.setColour (theme::blood.withAlpha (0.92f));
        g.fillRect (juce::Rectangle<float> (r.getX() - 1.0f, fillTop, r.getWidth() + 2.0f, r.getBottom() - fillTop + 1.0f));
    }
    g.setColour ((lv > 0.02f ? theme::bloodDark : theme::inkSoft).withAlpha (bleed > 0.001f ? 0.85f : 0.4f));
    g.strokePath (drop, juce::PathStrokeType (1.0f));
}

//==============================================================================
namespace
{
    // the clip exactly as the engine draws it (u: the clip's input in units of the ceiling)
    float clipCurveAt (float u, float knee, float pos) noexcept
    {
        if (u > 0.0f)
        {
            const float uu = u / pos;
            return pos * ((1.0f - knee) * mavka::fastTanh (uu) + knee * mavka::kneeClip (uu));
        }
        return (1.0f - knee) * mavka::fastTanh (u) + knee * mavka::kneeClip (u);
    }

    const char* kNoteNames[12] = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };

    juce::String noteName (float hz, float a4 = 440.0f)
    {
        if (hz < 20.0f) return {};
        const int midi = (int) std::lround (69.0 + 12.0 * std::log2 ((double) hz / (double) a4));
        return juce::String (kNoteNames[((midi % 12) + 12) % 12]) + juce::String (midi / 12 - 1);
    }
}

void ClipCurve::setState (float k, float ceilingPos, float pressure, float inputPeak)
{
    knee = k;
    pos = juce::jlimit (0.3f, 1.0f, ceilingPos);
    dyn = juce::jlimit (0.05f, 20.0f, pressure);
    peak = juce::jmax (0.0f, inputPeak);
    trail[(size_t) trailHead] = peak;
    trailHead = (trailHead + 1) % (int) trail.size();
    repaint();
}

juce::Rectangle<float> ClipCurve::plot() const
{
    return getLocalBounds().toFloat().withTrimmedBottom (34.0f).reduced (1.0f);
}

void ClipCurve::paint (juce::Graphics& g)
{
    const auto r = plot();
    recess (g, r);
    const auto in = r.reduced (8.0f);
    const float xr = 2.4f, yr = 1.18f;
    auto px = [&] (float x) { return in.getCentreX() + x / xr * in.getWidth() * 0.5f; };
    auto py = [&] (float y) { return in.getCentreY() - y / yr * in.getHeight() * 0.5f; };

    // grid: zero lines, unity, the ceilings
    g.setColour (theme::hair.withAlpha (0.9f));
    g.drawLine (in.getX(), py (0.0f), in.getRight(), py (0.0f), 0.7f);
    g.drawLine (px (0.0f), in.getY(), px (0.0f), in.getBottom(), 0.7f);
    for (float v : { -1.0f, 1.0f })
    {
        const float dash[] = { 2.0f, 3.0f };
        g.drawDashedLine (juce::Line<float> (px (v), in.getY(), px (v), in.getBottom()), dash, 2, 0.6f);
    }
    g.setColour (theme::blood.withAlpha (0.55f));
    g.drawLine (in.getX(), py (pos), in.getRight(), py (pos), 0.8f);
    g.drawLine (in.getX(), py (-1.0f), in.getRight(), py (-1.0f), 0.8f);

    // the curve at rest (faint) and as the pen / the clock press it right now (ink)
    auto curve = [&] (float d)
    {
        juce::Path p;
        for (int i = 0; i <= 120; ++i)
        {
            const float x = -xr + 2.0f * xr * (float) i / 120.0f;
            const float y = clipCurveAt (x * d, knee, pos);
            if (i == 0) p.startNewSubPath (px (x), py (y)); else p.lineTo (px (x), py (y));
        }
        return p;
    };
    g.setColour (theme::inkFaint.withAlpha (0.7f));
    g.strokePath (curve (1.0f), juce::PathStrokeType (1.0f));
    g.setColour (theme::ink.withAlpha (0.9f));
    g.strokePath (curve (dyn), juce::PathStrokeType (1.8f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

    // the voice right now, riding the curve (with a short trail): blood while it is being clipped
    for (int i = 0; i < (int) trail.size(); ++i)
    {
        const int idx = (trailHead + i) % (int) trail.size();
        const float x = juce::jmin (trail[(size_t) idx], xr * 0.98f);
        if (x <= 0.002f) continue;
        const float age = (float) i / (float) trail.size();
        const bool cut = x * dyn > 0.85f;
        g.setColour ((cut ? theme::blood : theme::ink).withAlpha (0.08f + 0.5f * age));
        const float rad = 1.2f + 2.2f * age;
        g.fillEllipse (px (x) - rad, py (clipCurveAt (x * dyn, knee, pos)) - rad, 2.0f * rad, 2.0f * rad);
    }

    g.setColour (theme::inkFaint);
    g.setFont (sans (9.0f, true, 0.14f));
    g.drawText ("IN", in.withTrimmedTop (in.getHeight() - 12.0f), juce::Justification::centredRight, false);
    g.drawText ("OUT", in.withHeight (12.0f).withTrimmedLeft (2.0f), juce::Justification::centredLeft, false);

    // caption: the shoulder and how far the ceiling is pressed
    auto cap = getLocalBounds().toFloat().withTop (r.getBottom() + 3.0f);
    g.setColour (theme::ink.withAlpha (0.86f));
    g.setFont (sans (10.5f, true, 0.14f));
    const juce::String shoulder = knee < 0.2f ? "SOFT SHOULDER" : (knee < 0.65f ? "FIRM SHOULDER" : "HARD SHOULDER");
    g.drawText (shoulder, cap.removeFromTop (15.0f), juce::Justification::centred, false);
    g.setColour (theme::inkSoft);
    g.setFont (sans (10.5f));
    const float db = 20.0f * std::log10 (dyn);
    g.drawText (std::abs (db) < 0.3f ? juce::String ("ceiling at rest") : "ceiling pressed " + juce::String (-db, 1) + " dB",
                cap.removeFromTop (14.0f), juce::Justification::centred, false);
}

void ClipCurve::mouseDown (const juce::MouseEvent&)
{
    if (auto* p = apvts.getParameter ("clipShoulder"))
    {
        p->beginChangeGesture();
        dragFrom = p->getValue();
        dragging = true;
    }
}

void ClipCurve::mouseDrag (const juce::MouseEvent& e)
{
    if (! dragging) return;
    if (auto* p = apvts.getParameter ("clipShoulder"))
        p->setValueNotifyingHost (juce::jlimit (0.0f, 1.0f, dragFrom - (float) e.getDistanceFromDragStartY() / 160.0f));
}

void ClipCurve::mouseUp (const juce::MouseEvent&)
{
    if (! dragging) return;
    if (auto* p = apvts.getParameter ("clipShoulder"))
        p->endChangeGesture();
    dragging = false;
}

void ClipCurve::mouseDoubleClick (const juce::MouseEvent&)
{
    if (auto* p = apvts.getParameter ("clipShoulder"))
    {
        p->beginChangeGesture();
        p->setValueNotifyingHost (p->getDefaultValue());
        p->endChangeGesture();
    }
}

//==============================================================================
void ClipScope::setData (const float* in, const float* out, int n, float samplesPerPeriod, float ceilingPos)
{
    ceilPos = juce::jlimit (0.3f, 1.0f, ceilingPos);
    // three periods of the voice (5 .. 20 ms at 48 kHz), started on a rising zero crossing so the picture stands still
    const int span = juce::jlimit (240, n - 2, (int) std::lround (3.0f * juce::jmax (samplesPerPeriod, 1.0f)));
    int start = n - span;
    const int back = juce::jlimit (1, n - span - 1, (int) std::lround (juce::jmax (samplesPerPeriod, 80.0f)) + 2);
    for (int i = n - span; i > n - span - back && i > 1; --i)
        if (in[i - 1] < 0.0f && in[i] >= 0.0f) { start = i; break; }
    a.assign (in + start, in + juce::jmin (n, start + span));
    b.assign (out + start, out + juce::jmin (n, start + span));
    repaint();
}

void ClipScope::paint (juce::Graphics& g)
{
    const auto r = getLocalBounds().toFloat();
    recess (g, r);
    const auto in = r.reduced (6.0f, 5.0f);
    const float yr = 1.35f;
    auto py = [&] (float y) { return in.getCentreY() - juce::jlimit (-yr, yr, y) / yr * in.getHeight() * 0.5f; };
    g.setColour (theme::hair.withAlpha (0.8f));
    g.drawLine (in.getX(), in.getCentreY(), in.getRight(), in.getCentreY(), 0.6f);
    g.setColour (theme::blood.withAlpha (0.45f));
    g.drawLine (in.getX(), py (ceilPos), in.getRight(), py (ceilPos), 0.7f);
    g.drawLine (in.getX(), py (-1.0f), in.getRight(), py (-1.0f), 0.7f);

    const int n = (int) a.size();
    if (n > 2)
    {
        auto px = [&] (int i) { return in.getX() + in.getWidth() * (float) i / (float) (n - 1); };
        // what the clip cuts off (between the voice going in and the voice coming out): blood
        juce::Path cut, pin, pout;
        for (int i = 0; i < n; ++i)
        {
            if (i == 0) { pin.startNewSubPath (px (i), py (a[0])); pout.startNewSubPath (px (i), py (b[0])); }
            else { pin.lineTo (px (i), py (a[(size_t) i])); pout.lineTo (px (i), py (b[(size_t) i])); }
        }
        cut = pin;
        for (int i = n - 1; i >= 0; --i)
            cut.lineTo (px (i), py (b[(size_t) i]));
        cut.closeSubPath();
        {
            juce::Graphics::ScopedSaveState save (g);
            g.reduceClipRegion (in.toNearestInt());
            g.setColour (theme::blood.withAlpha (0.16f));
            g.fillPath (cut);
            g.setColour (theme::inkFaint.withAlpha (0.85f));
            g.strokePath (pin, juce::PathStrokeType (0.9f));
            g.setColour (theme::ink.withAlpha (0.9f));
            g.strokePath (pout, juce::PathStrokeType (1.5f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
        }
    }
    g.setColour (theme::inkFaint);
    g.setFont (sans (9.0f, true, 0.14f));
    g.drawText (u8 ("IN  \xc2\xb7  OUT"), r.reduced (7.0f, 3.0f), juce::Justification::topRight, false);
}

//==============================================================================
void OvertoneMap::setView (const mavka::Engine::OvertoneView& v, const juce::String& keyName, float polish)
{
    view = v;
    key = keyName;
    polishAmt = polish;
    // analyser ballistics: up at once, down over a few hundred ms
    for (int h = 0; h < kBands; ++h)
    {
        const float b0 = juce::jlimit (-90.0f, 0.0f, v.level[h]);
        const float a0 = juce::jlimit (-90.0f, 12.0f, b0 + 20.0f * std::log10 (juce::jmax (v.gain[h], 1.0e-4f)));
        before[h] = b0 > before[h] ? b0 : before[h] + (b0 - before[h]) * 0.25f;
        after[h] = a0 > after[h] ? a0 : after[h] + (a0 - after[h]) * 0.25f;
    }
    repaint();
}

juce::Rectangle<float> OvertoneMap::bars() const
{
    return getLocalBounds().toFloat().reduced (6.0f, 0.0f).withTrimmedTop (18.0f).withTrimmedBottom (26.0f);
}

float OvertoneMap::xOf (float overtone) const
{
    const auto b = bars();
    const float step = b.getWidth() / 11.0f;
    return b.getX() + step * (overtone - 1.5f) + step * 0.25f;
}

int OvertoneMap::columnAt (float x) const
{
    const auto b = bars();
    const float step = b.getWidth() / 11.0f;
    const int h = (int) std::lround ((x - b.getX() - 0.25f * step) / step + 1.5f) - 2;
    return h >= 0 && h < kHarm ? h : -1;
}

float OvertoneMap::userDb (int h) const
{
    if (auto* p = apvts.getRawParameterValue ("ot" + juce::String (h + 2)))
        return p->load();
    return 0.0f;
}

void OvertoneMap::paint (juce::Graphics& g)
{
    const auto r = getLocalBounds().toFloat();
    recess (g, r.withTrimmedTop (16.0f));
    const auto b = bars();
    const float step = b.getWidth() / 11.0f;
    const float lo = -66.0f, hi = -4.0f;
    auto yOf = [&] (float db) { return b.getBottom() - b.getHeight() * juce::jlimit (0.0f, 1.0f, (db - lo) / (hi - lo)); };
    const bool pitched = view.conf > 0.3f;
    const float conf = juce::jlimit (0.0f, 1.0f, view.conf);

    // header: the sung note and the song's chord, the legend
    g.setFont (sans (10.5f, true, 0.12f));
    g.setColour (pitched ? theme::ink.withAlpha (0.86f) : theme::inkFaint);
    const juce::String note = pitched ? noteName (view.f0, a4) : juce::String ("no pitch");
    const juce::String tuning = std::abs (a4 - 440.0f) > 0.04f ? u8 ("  \xc2\xb7  A ") + juce::String (a4, 1) : juce::String();
    g.drawText (note + u8 ("  \xc2\xb7  ") + key + tuning + (pitched ? juce::String() : juce::String ("  \xc2\xb7  plain clipping")),
                r.withHeight (15.0f).withTrimmedLeft (2.0f), juce::Justification::centredLeft, false);
    {
        // legend, right to left: [ink] chord tones   [blood] dirt   (PLAIN / polish)
        const auto lg = r.withHeight (15.0f).withTrimmedRight (2.0f);
        const auto f = sans (9.5f, false, 0.06f);
        g.setFont (f);
        float x = lg.getRight();
        auto item = [&] (const juce::String& text, juce::Colour sq)
        {
            const float w = juce::GlyphArrangement::getStringWidth (f, text) + 2.0f;
            x -= w;
            g.setColour (theme::inkSoft);
            g.drawText (text, juce::Rectangle<float> (x, lg.getY(), w, lg.getHeight()), juce::Justification::centredLeft, false);
            if (! sq.isTransparent())
            {
                x -= 9.0f;
                g.setColour (sq);
                g.fillRect (x, lg.getCentreY() - 3.0f, 6.0f, 6.0f);
            }
            x -= 12.0f;
        };
        item ("dirt", theme::blood.withAlpha (0.9f));
        item ("chord tones", theme::ink.withAlpha (0.85f));
        if (view.plain)
            item ("PLAIN", juce::Colours::transparentBlack);
        else if (polishAmt > 0.05f)
            item ("polish " + juce::String (juce::roundToInt (100.0f * polishAmt)) + "%", juce::Colours::transparentBlack);
    }

    // faint dB rules
    g.setColour (theme::hair.withAlpha (0.55f));
    for (float db : { -60.0f, -40.0f, -20.0f })
        g.drawLine (b.getX(), yOf (db), b.getRight(), yOf (db), 0.5f);

    // the sub-octave ghost's half overtones (thin, between the columns)
    if (view.halves)
        for (int h = kHarm; h < kBands; ++h)
        {
            const float x = xOf (mavka::inkey::kMult[h]);
            const bool fits = view.fit[h] >= 0.5f;
            const float top = yOf (after[h]);
            g.setColour ((fits ? theme::ink : theme::blood).withAlpha (0.25f + 0.45f * conf));
            g.fillRect (juce::Rectangle<float> (x - 1.2f, top, 2.4f, b.getBottom() - top));
        }

    for (int h = 0; h < kHarm; ++h)
    {
        const float x = xOf ((float) (h + 2));
        const bool fits = view.fit[h] >= 0.5f;
        const float w = step * 0.52f;
        const bool hot = h == hover || h == dragging;
        // what ordinary distortion would add here (outline) and what MAVKA adds (solid)
        const float top = yOf (after[h]), was = yOf (before[h]);
        const auto col = fits ? theme::ink : theme::blood;
        if (hot)
        {
            g.setColour (theme::ink.withAlpha (0.05f));
            g.fillRect (juce::Rectangle<float> (x - step * 0.5f, b.getY(), step, b.getHeight()));
        }
        g.setColour (col.withAlpha (pitched ? 0.30f + 0.62f * conf : 0.22f));
        g.fillRect (juce::Rectangle<float> (x - w * 0.5f, top, w, b.getBottom() - top));
        g.setColour (theme::inkSoft.withAlpha (0.55f));
        g.drawLine (x - w * 0.5f - 2.0f, was, x + w * 0.5f + 2.0f, was, 1.0f);

        // the overtone map setting
        const float u = userDb (h);
        if (std::abs (u) > 0.05f || hot)
        {
            g.setColour (hot ? theme::ink : theme::inkSoft);
            g.setFont (numbers (9.5f));
            g.drawText ((u > 0.0f ? "+" : "") + juce::String (u, std::abs (u) < 9.95f ? 1 : 0),
                        juce::Rectangle<float> (x - step * 0.5f, juce::jmin (top, was) - 13.0f, step, 12.0f),
                        juce::Justification::centred, false);
        }

        // overtone number and the note it is on
        g.setColour (fits ? theme::ink.withAlpha (0.8f) : theme::blood.withAlpha (0.85f));
        g.setFont (numbers (10.0f));
        g.drawText (juce::String (h + 2), juce::Rectangle<float> (x - step * 0.5f, b.getBottom() + 2.0f, step, 11.0f),
                    juce::Justification::centred, false);
        g.setColour (pitched ? theme::inkSoft : theme::inkFaint.withAlpha (0.6f));
        g.setFont (sans (9.0f));
        g.drawText (pitched ? noteName (view.f0 * (float) (h + 2), a4) : juce::String (u8 ("\xc2\xb7")),
                    juce::Rectangle<float> (x - step * 0.5f, b.getBottom() + 13.0f, step, 11.0f), juce::Justification::centred, false);
    }
}

void OvertoneMap::mouseMove (const juce::MouseEvent& e)
{
    const int h = columnAt (e.position.x);
    if (h != hover)
    {
        hover = h;
        repaint();
    }
}

void OvertoneMap::mouseExit (const juce::MouseEvent&)
{
    hover = -1;
    repaint();
}

void OvertoneMap::mouseDown (const juce::MouseEvent& e)
{
    dragging = columnAt (e.position.x);
    if (dragging < 0) return;
    if (auto* p = apvts.getParameter ("ot" + juce::String (dragging + 2)))
    {
        p->beginChangeGesture();
        dragDb = userDb (dragging);
        dragY = e.position.y;
    }
}

void OvertoneMap::mouseDrag (const juce::MouseEvent& e)
{
    if (dragging < 0) return;
    if (auto* p = apvts.getParameter ("ot" + juce::String (dragging + 2)))
    {
        const float fine = e.mods.isShiftDown() ? 0.25f : 1.0f;
        const float db = juce::jlimit (-30.0f, 12.0f, dragDb - (e.position.y - dragY) * 0.25f * fine);
        p->setValueNotifyingHost (p->convertTo0to1 (db));
        repaint();
    }
}

void OvertoneMap::mouseUp (const juce::MouseEvent&)
{
    if (dragging < 0) return;
    if (auto* p = apvts.getParameter ("ot" + juce::String (dragging + 2)))
        p->endChangeGesture();
    dragging = -1;
    repaint();
}

void OvertoneMap::mouseDoubleClick (const juce::MouseEvent& e)
{
    const int h = columnAt (e.position.x);
    if (h < 0) return;
    if (auto* p = apvts.getParameter ("ot" + juce::String (h + 2)))
    {
        p->beginChangeGesture();
        p->setValueNotifyingHost (p->convertTo0to1 (0.0f));
        p->endChangeGesture();
    }
    repaint();
}

//==============================================================================
void Page::paint (juce::Graphics& g)
{
    if (dividerX > 0)
    {
        g.setColour (theme::hair);
        g.drawLine ((float) dividerX + 0.5f, 52.0f, (float) dividerX + 0.5f, (float) getHeight() - 14.0f, 0.8f);
    }
    g.setColour (theme::inkSoft);
    g.setFont (sans (12.0f));
    g.drawFittedText (description, getLocalBounds().removeFromTop (36).reduced (6, 2), juce::Justification::topLeft, 2, 1.0f);

    if (notes.isNotEmpty())
    {
        auto area = notesArea;
        juce::StringArray lines;
        lines.addLines (notes);
        for (int i = 0; i < lines.size(); ++i)
        {
            const bool head = i == 0;
            g.setColour (head ? theme::ink.withAlpha (0.86f) : theme::inkSoft);
            g.setFont (head ? sans (11.0f, true, 0.16f) : sans (11.5f));
            g.drawText (lines[i], area.removeFromTop (head ? 22 : 19), juce::Justification::centredLeft, true);
        }
    }
}

void Surface::paint (juce::Graphics& g)
{
    // how many physical pixels one editor unit covers right now (window size x display scaling x host zoom)
    physicalScale = g.getInternalContext().getPhysicalPixelScaleFactor();

    if (paper.isValid())
        g.drawImage (paper, getLocalBounds().toFloat());
    else
        g.fillAll (theme::face);

    g.setColour (theme::ink.withAlpha (0.88f));
    g.setFont (sans (30.0f, true, 0.30f));
    g.drawText ("MAVKA", 22, 8, 240, 34, juce::Justification::centredLeft, false);
    g.setColour (theme::inkFaint);
    g.setFont (sans (11.5f, false, 0.04f));
    g.drawText ("vocal handwriting distortion", 24, 40, 260, 14, juce::Justification::centredLeft, false);
    g.setFont (sans (10.5f, true, 0.16f));
    g.drawText ("PRESET", 470, 4, 120, 14, juce::Justification::centredLeft, false);

    for (const auto& p : kPlates)
    {
        if (*p.title != 0)
        {
            g.setColour (theme::ink.withAlpha (0.86f));
            g.setFont (sans (12.0f, true, 0.24f));
            g.drawText (p.title, (int) p.x + 14, (int) p.y + 9, 220, 18, juce::Justification::centredLeft, false);
        }
        if (*p.sub != 0)
        {
            g.setColour (theme::inkFaint);
            g.setFont (sans (11.0f));
            g.drawText (u8 (p.sub), (int) p.x + 200, (int) p.y + 9, (int) p.w - 214, 18, juce::Justification::centredRight, false);
        }
    }

    // tab rule, AURA dividers
    g.setColour (theme::hair);
    g.drawLine (284.0f, 360.5f, 1040.0f, 360.5f, 0.8f);
    for (float sx : { 214.0f, 562.0f, 770.0f, 858.0f })
        g.drawLine (sx, 622.0f, sx, 778.0f, 0.8f);

    drawFlow (g);

    g.setColour (theme::inkFaint.withAlpha (0.8f));
    g.setFont (sans (9.5f, false, 0.20f));
    g.drawText (u8 ("\xd0\x9c\xd0\x90\xd0\x92\xd0\x9a\xd0\x90  \xc2\xb7  MK IV  \xc2\xb7  impossible hardware  \xc2\xb7  s/n 0417"),
                0, 788, 1044, 12, juce::Justification::centredRight, false);
}

void Wear::paint (juce::Graphics& g)
{
    if (dirt.isValid())
        g.drawImage (dirt, getLocalBounds().toFloat());
}

//==============================================================================
MavkaAudioProcessorEditor::MavkaAudioProcessorEditor (MavkaAudioProcessor& p)
    : AudioProcessorEditor (&p), proc (p), assets (mavkaui::acquireAssets())
{
    // the size this project was left at (a fresh instance opens at the smallest size)
    const float savedScale = juce::jlimit (minScale, maxScale, proc.uiScale.load());

    if (assets != nullptr && assets->regular != nullptr)
        lnf.setDefaultSansSerifTypeface (assets->regular);
    setLookAndFeel (&lnf);
    addAndMakeVisible (surface);
    tooltips = std::make_unique<juce::TooltipWindow> (this, 700);

    // ---- header
    int count = 0;
    const mavka::Preset* presets = mavka::factoryPresets (count);
    for (int i = 0; i < count; ++i)
        presetBox.addItem (presets[i].name, i + 1);
    presetBox.setTooltip ("Factory presets. Each one keeps MAVKA's handwriting - from silk to dirt.");
    presetBox.onChange = [this]
    {
        const int id = presetBox.getSelectedId();
        if (id > 0 && id - 1 != proc.getFactoryPresetIndex())
            proc.loadFactoryPreset (id - 1);
        lastPreset = id - 1;
    };
    presetBox.setBounds (470, 20, 210, 26);
    surface.addAndMakeVisible (presetBox);

    addChoice (surface, "os", "Oversampling", { 700, 4, 120, 44 },
               "Auto picks the internal rate from the distortion amount and your project's sample rate (and goes one step higher while rendering).");
    osRunning.setBounds (826, 22, 90, 26);
    osRunning.setFont (sans (11.5f));
    osRunning.setColour (juce::Label::textColourId, theme::inkSoft);
    surface.addAndMakeVisible (osRunning);

    autoGain.setTooltip ("Keeps the wet signal as loud as the dry one, so you judge character, not volume.");
    autoGain.setBounds (926, 10, 120, 20);
    surface.addAndMakeVisible (autoGain);
    autoGainAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (proc.apvts, "autoGain", autoGain);

    softLimit.setTooltip ("A gentle safety limiter at the very end: only peaks above -2.5 dBFS are rounded off. "
                          "Turn it off if you limit later in the chain.");
    softLimit.setBounds (926, 32, 120, 20);
    surface.addAndMakeVisible (softLimit);
    softLimitAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (proc.apvts, "limit", softLimit);

    // ---- handwriting
    stroke.setBounds (24, 94, 406, 86);
    surface.addAndMakeVisible (stroke);
    addKnob (surface, "ink", "Ink", true, { 22, 186, 104, 122 },
             "Ink: every word starts clean and the distortion spreads over it. Below zero = Press: a hard hit that lifts off. "
             "The Pressure clip and the Stroke flanger follow the same pen.");
    addKnob (surface, "slant", "Slant", true, { 128, 186, 104, 122 },
             "Slant: on every syllable the doubles draw a small pitch stroke. Scoop = dip and rise, Fall = up and settle.");
    addKnob (surface, "ligature", "Ligature", true, { 234, 186, 104, 122 },
             "Ligature: a dirty echo in time with the song that only speaks between words, joining them.");
    addKnob (surface, "inkTime", "Ink time", false, { 342, 196, 90, 106 },
             "Length of one pen stroke (Ink, Slant, the Pressure clip and the Stroke flanger).");

    // ---- core macros, in signal order
    struct M { const char* id; const char* t; const char* tip; };
    const M macros[] =
    {
        { "drive",  "Drive",  "How hard the three impossible machines are pushed." },
        { "grit",   "Grit",   "From silk to dirt: harder clipping, deeper folding, a rougher flanger and echo." },
        { "crush",  "Crush",  "Harmonic crusher: the sample-and-hold clock follows the pitch of your voice, "
                              "so the crushed voice still sings in tune. Mode in the HARDWARE tab." },
        { "clip",   "Clip",   "PRESSURE clip: its ceiling is pressed down by your pen (Ink), breathes with the chord clock "
                              "and turns asymmetric on vowels. What it cuts off BLEEDS into the harp, echo and forest. "
                              "The drop fills while it bites. Its shoulder, the melodic split of ALL of MAVKA's distortion "
                              "and the overtone map are on the CLIP tab." },
        { "flange", "Flange", "Nonlinear flanger tuned to the harp's chord (Tuned), a jet riding the clock (Jet) "
                              "or following your Ink (Stroke). Mode in the FLANGE tab." },
        { "melody", "Melody", "How loud the harp in the lamp rings." },
        { "motion", "Motion", "Depth of the whole chord clock." },
        { "mix",    "Mix",    "Dry / wet. Use 100% on a parallel bus. Turned down (and with Drive turned down) MAVKA "
                              "changes its character: less dirt, the lowest chord tones bloom, the wet layer gets denser - "
                              "polish instead of distortion." },
        { "output", "Output", "Output level." },
    };
    for (int i = 0; i < 9; ++i)
    {
        auto& k = addKnob (surface, macros[i].id, macros[i].t, true, { 462 + i * 64, 98, 64, 128 }, u8 (macros[i].tip));
        if (juce::String (macros[i].id) == "clip")
        {
            k.slider.getProperties().set ("blood", true);
            const auto v = k.getValueArea() + k.getPosition();
            bleedDrop.setBounds (v.getCentreX() + 17, v.getY() + 1, 11, 16);
        }
    }
    bleedDrop.setTooltip ("Pressure: fills with blood while the clip bites. What it cuts off bleeds into the harp, echo and forest.");
    surface.addAndMakeVisible (bleedDrop);

    // ---- chord clock visual
    clock.setBounds (32, 352, 210, 210);
    surface.addAndMakeVisible (clock);

    // ---- tabs
    const char* tabNames[numPages] = { "HARDWARE", "CLOCK", "HARP", "VOICE", "FLANGE", "CLIP" };
    for (int i = 0; i < numPages; ++i)
    {
        auto& t = tabs[(size_t) i];
        t.setButtonText (u8 (tabNames[i]));
        t.setClickingTogglesState (true);
        t.setRadioGroupId (4242);
        t.setBounds (284 + i * 126, 332, 122, 28);
        t.onClick = [this, i] { showPage (i); };
        surface.addAndMakeVisible (t);

        auto& pg = pages[(size_t) i];
        pg.setBounds (284, 366, 756, 204);
        surface.addChildComponent (pg);
    }
    pages[0].description = "Three machines that cannot exist: a lamp whose bias breathes, a speaker cone with a 20 cm throw "
                           "(glassy self-modulation) and a folding diode. Ghost octaves are fed INTO them; the crusher samples in tune with your pitch.";
    pages[1].description = "Every modulator runs at a rate taken from a chord (4:5:6 = major, 10:12:15 = minor...), locked to your bars. "
                           "The movement repeats like a groove instead of drifting.";
    pages[2].description = "Strings tuned to your song, struck by the distorted voice: a melodic halo even on rap. Set the key of your "
                           "track (like in your Auto-Tune) and, if the beat isn't tuned to 440 Hz, its tuning (Detune).";
    pages[3].description = "How MAVKA listens: the same saturation for whispers and shouts, vowels sing, consonants stay clear.";
    pages[4].description = "A flanger that plays the harp's chord. Its feedback loop saturates with Grit and it follows "
                           "the same pen (Ink), the same clock and the same key as the rest of MAVKA.";

    {
        auto& pg = pages[0];
        struct K { const char* id; const char* t; const char* tip; };
        const K ks[] =
        {
            { "lamp",      "Lamp",     "The lead: a warm asymmetric tube. Stays in tune and on time." },
            { "glass",     "Glass",    "Doppler glass double: the voice moves its own speaker cone." },
            { "fold",      "Fold",     "Folding diode double: metallic dirt." },
            { "doppler",   "Doppler",  "Throw of the impossible speaker cone: more = glassier, singing harmonics." },
            { "bias",      "Bias",     "Lamp asymmetry: even harmonics, warmth. Breathes with the clock." },
            { "ghostDown", "Ghost dn", "Octave below (with its natural fifth), fed into the machines." },
            { "ghostUp",   "Ghost up", "Octave above, fed into the machines." },
        };
        for (int i = 0; i < 7; ++i)
            addKnob (pg, ks[i].id, ks[i].t, false, { 2 + i * 92, 56, 90, 132 }, ks[i].tip);
        addChoice (pg, "crushMode", "Crush mode", { 652, 80, 100, 44 },
                   "Harmonic: steps locked to your pitch (in tune). Digital: classic fixed-rate lo-fi. "
                   "Clock: the resolution jumps in octaves with the chord clock.");
    }
    {
        auto& pg = pages[1];
        addChoice (pg, "rate", "Cycle", { 6, 46, 130, 44 }, "Length of the base cycle, in bars or notes.");
        addChoice (pg, "ratio", "Chord", { 6, 104, 130, 44 }, "The chord the four modulators play: their rates relate like the notes of this chord.");
        struct K { const char* id; const char* t; const char* tip; };
        const K ks[] =
        {
            { "width",    "Width",    "Stereo spread of the doubles, clock and harp." },
            { "vibrato",  "Vibrato",  "Micro pitch motion of the lead." },
            { "ensemble", "Ensemble", "Chorus motion of the doubles." },
            { "breath",   "Breath",   "Lamp bias breathing (even harmonics come and go)." },
            { "pulse",    "Pulse",    "The two doubles trade places in rhythm." },
            { "orbit",    "Orbit",    "The doubles circle around the lead in the stereo field." },
            { "warp",     "Warp",     "Phase warp before the machines: the timbre turns like a phaser inside the tube." },
        };
        for (int i = 0; i < 7; ++i)
            addKnob (pg, ks[i].id, ks[i].t, false, { 146 + i * 87, 56, 84, 132 }, ks[i].tip);
    }
    {
        auto& pg = pages[2];
        addChoice (pg, "key", "Key", { 6, 80, 104, 44 }, "Key of your song - set it like in your Auto-Tune. The harp, the forest's "
                                                      "Tune, the Tuned flanger and the melodic distortion (Sing / Dirt) follow it.");
        addChoice (pg, "chord", "Chord", { 118, 80, 124, 44 }, "Which notes the strings are tuned to - and which overtones count as "
                                                            "chord tones for Sing / Dirt (Minor = the cleanest, Min7 / Min9 = richer).");
        addChoice (pg, "register", "Register", { 250, 80, 100, 44 }, "Octave of the harp.");
        addKnob (pg, "detune", "Detune", false, { 358, 56, 96, 132 },
                 "Tuning of everything that plays in key, like Detune in Auto-Tune: if your beat is tuned to A = 447 Hz, "
                 "set 447 Hz (+27.3 cents) and the harp, the forest, the Tuned flanger and the melodic distortion move with it. "
                 "Double-click the number to type it exactly (447, 447 Hz, 432.5 or +27.3 ct); Shift-drag for a fine turn; "
                 "double-click the knob for 440. Presets never change it (nor Key and Chord).");
        addKnob (pg, "decay", "Decay", false, { 462, 56, 96, 132 }, "How long the strings ring.");
        addKnob (pg, "bright", "Bright", false, { 560, 56, 96, 132 }, "String brightness.");
        addKnob (pg, "harpDrive", "Harp drive", false, { 658, 56, 96, 132 }, "Overdrive inside the strings.");
    }
    {
        auto& pg = pages[3];
        struct K { const char* id; const char* t; const char* tip; };
        const K ks[] =
        {
            { "consist",  "Consistency", "Same saturation for whispers and shouts (dynamics are restored afterwards)." },
            { "focus",    "Vowel focus", "The distortion sings on vowels; consonants and S's stay clean." },
            { "smooth",   "Smooth",      "Tames harsh upper mids in the wet signal." },
            { "lowCut",   "Low cut",     "Low cut before the machines." },
            { "highCut",  "High cut",    "High cut before the machines (less fizz)." },
            { "preTilt",  "Pre tilt",    "Tilts what the machines hear: dark / bright bite." },
            { "tone",     "Tone",        "Tone of the wet signal." },
        };
        for (int i = 0; i < 7; ++i)
            addKnob (pg, ks[i].id, ks[i].t, false, { 4 + i * 107, 56, 100, 132 }, ks[i].tip);
    }
    {
        auto& pg = pages[4];
        addChoice (pg, "flangeMode", "Flange mode", { 6, 80, 112, 44 },
                   "Tuned: a comb tuned to the harp's chord that steps through it on the beat - an arpeggio inside your voice. "
                   "Jet: a classic jet sweep riding the chord clock. Stroke: the comb follows your pen, word by word.");
        addKnob (pg, "flangeFb", "Feedback", false, { 130, 56, 96, 132 },
                 "Resonance of the comb. It saturates with Grit, so it sings instead of whistling.");
        addKnob (pg, "flangeDepth", "Depth", false, { 230, 56, 96, 132 },
                 "Sweep range (Jet / Stroke) or tuning glide (Tuned). The sweep also follows how loud you sing.");
        pg.notes = u8 ("FLANGE\n"
                       "tuned \xe2\x86\x92 the comb plays the harp's chord, a note per beat\n"
                       "jet \xe2\x86\x92 sweeps with the chord clock\n"
                       "stroke \xe2\x86\x92 one sweep per syllable (your pen)\n"
                       "feedback \xe2\x86\x92 saturates with Grit, sings, never whistles");
        pg.notesArea = { 380, 48, 370, 140 };
        pg.dividerX = 350;
    }
    {
        // CLIP: the clip's curve, what goes in and out of it, and the melodic split of everything MAVKA adds
        auto& pg = pages[5];
        clipCurve.setBounds (4, 4, 158, 196);
        clipCurve.setTooltip ("The clip's curve. Faint: at rest; ink: pressed by your pen and the clock right now; the dot is "
                              "your voice riding it (blood while it is being clipped), the red lines are the ceilings "
                              "(lower on vowels). Drag up / down to change the shoulder, double-click to reset.");
        pg.addAndMakeVisible (clipCurve);
        clipScope.setBounds (170, 4, 386, 66);
        clipScope.setTooltip ("Three periods of your voice going into the clip (thin) and what comes out (ink). "
                              "Blood: what the clip cuts off - it strikes the harp and throws echoes (Bleed).");
        pg.addAndMakeVisible (clipScope);
        overtoneMap.setBounds (170, 74, 386, 128);
        overtoneMap.setTooltip ("What MAVKA's distortion adds at every overtone of your voice (the whole core: crusher, lamp, "
                                "glass, fold, clip and the ghost octaves). Ink: a note of the song's chord (Key + Chord on the "
                                "HARP tab) - Sing. Blood: dirt - Dirt. The line is what an ordinary distortion would add. "
                                "Drag a peak up / down to raise or lower that overtone (Shift: fine), double-click to reset. "
                                "Thin bars: what the sub-octave ghost adds between the overtones.");
        pg.addAndMakeVisible (overtoneMap);
        struct K { const char* id; const char* t; const char* tip; };
        const K ks[] =
        {
            { "clipShoulder", "Shoulder", "The clip's knee: 100% = round and soft (tube-like), 0% = a hard wall (with a tiny "
                                          "rounded corner). Independent of Grit now: Grit sets how hard the clip is hit." },
            { "clipSing",     "Sing",     "What the distortion adds on the notes of the song's chord: 100% = as an ordinary "
                                          "distortion would, 200% = twice. The octaves always count as chord tones." },
            { "clipDirt",     "Dirt",     "What the distortion adds OFF the chord (the major third every distortion adds to a "
                                          "minor song, the flat 7th, the 9th, the 11th...). 0% = only chord tones, 100% = as "
                                          "ordinary, 150% = dirtier. Where there is no pitch (breaths, consonants) the "
                                          "distortion stays ordinary." },
            { "clipForce",    "Force",    "How much the dirt follows how hard you sing: 100% = only the loud, pushed words get "
                                          "dirty, soft ones stay sweet; 0% = the same dirt everywhere - drive it as hard as "
                                          "you like and set Dirt to 0 for loud AND clean." },
            { "prism",        "Prism",    "The chord tones the distortion adds refract around your head: every overtone gets "
                                          "its own place in the stereo field, swaying with the chord clock (Motion). The mono "
                                          "sum stays exactly the same." },
            { "bleed",        "Bleed",    "What the clipper cuts off strikes the harp, throws more echo and blooms in the "
                                          "forest. 0% = a plain (but still pressed and breathing) clipper." },
        };
        for (int i = 0; i < 6; ++i)
            addKnob (pg, ks[i].id, ks[i].t, false, { 563 + (i % 3) * 64, (i / 3) * 102, 64, 100 }, ks[i].tip);
    }

    // ---- AURA: nonlinear send effects
    reverbSafe.setTooltip ("Reverb safe: for when MAVKA's sound goes ON INTO a reverb (MAVKA last on the vocal insert with "
                           "the reverb on a send from that track, or MAVKA's return sent to the reverb too). The dirt is kept "
                           "to the attacks of the words, nothing below 110 Hz, the top above 9 kHz 6 dB softer and the low "
                           "side narrower - Lexicon, Valhalla or Seventh Heaven tails stay clean. If your reverb is fed from "
                           "the dry voice on its own send, leave it off: the reverb never hears MAVKA.");
    reverbSafe.setBounds (92, 592, 130, 20);
    surface.addAndMakeVisible (reverbSafe);
    reverbSafeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (proc.apvts, "reverbSafe", reverbSafe);
    addFader ("echo", "Echo", { 26, 620, 64, 164 },
              "Tape echo in time with the song. Every repeat is re-saturated and narrowed, so the echoes get grainier as they fade. Ducks while you sing.");
    addChoice (surface, "echoTime", "Time", { 98, 622, 104, 44 }, "Echo time (Slap or note values at the project tempo).");
    addKnob (surface, "echoFb", "Feedback", false, { 98, 670, 100, 114 }, "How many repeats.");

    addFader ("forest", "Forest", { 224, 620, 64, 164 },
              "Reverb with saturation inside its loop: loud parts compress, tails bloom in the gaps. The echoes and twins live in it too.");
    addChoice (surface, "forestType", "Type", { 296, 622, 118, 44 }, "Room, Plate, Hall or Cathedral.");
    addKnob (surface, "forestDecay", "Decay", false, { 296, 670, 86, 114 }, "Reverb time.");
    addKnob (surface, "forestTune", "Tune", false, { 384, 670, 86, 114 },
             "Pulls the room onto the harp's key and chord: the reverb sings in tune.");
    addKnob (surface, "forestBloom", "Bloom", false, { 472, 670, 86, 114 },
             "Holds the reverb back while you sing and lets it bloom in the pauses.");

    addFader ("twin", "Twin", { 572, 620, 64, 164 },
              "Detuned doubles whose spread breathes with your voice and the clock; each twin has its own throat.");
    addChoice (surface, "twinType", "Type", { 644, 622, 114, 44 }, "Tight, Wide, Lush, Choir (4 twins) or Octave (a shadow below).");

    addFader ("spin", "Spin", { 780, 620, 64, 164 },
              "The whole aura turns around your head in time with the chord clock (also wobbles echo, forest and twins).");

    addChoice (surface, "duck", "Duck", { 870, 622, 168, 44 },
               "Spectral: the echo and forest step aside only in the frequencies your voice is using right now. "
               "Classic: the whole return ducks. Off: no ducking.");
    duckMap.setBounds (868, 674, 54, 106);
    duckMap.setTooltip ("Duck map: how far the echo / forest steps aside in each band right now (low to high).");
    surface.addAndMakeVisible (duckMap);
    addKnob (surface, "duckDepth", "Depth", false, { 924, 670, 62, 112 },
             "Spectral: how far the echo / forest must stay under your voice in the bands it uses (6..24 dB). "
             "Classic: how deep the whole return dips.");
    addKnob (surface, "feed", "Feed", false, { 986, 670, 62, 112 },
             "What goes into the aura: 0% = your clean voice, 100% = MAVKA's distorted voice.");

    tabs[0].setToggleState (true, juce::dontSendNotification);
    showPage (0);

    // ---- the worn paper: grease where fingers turn the permanent controls, pencil marks at two favourite settings
    {
        std::vector<mavkaui::Scuff> scuffs;
        for (const auto& k : knobs)
        {
            if (k->getParentComponent() != &surface)
                continue;
            const auto sb = k->slider.getBounds().toFloat() + k->getPosition().toFloat();
            const float R = juce::jmin (sb.getWidth(), sb.getHeight()) * 0.5f - 2.0f;
            const auto id = k->getComponentID();
            const bool large = (bool) k->slider.getProperties()["large"];
            const bool macro = large && k->getX() > 450;
            const float strength = (id == "drive" || id == "mix") ? 1.0f : (macro ? 0.75f : 0.55f);
            mavkaui::Scuff s { sb.getCentreX(), sb.getCentreY(), R - (large ? 7.5f : 6.5f), strength, false };
            if (id == "drive") s.mark = 0.62f;
            if (id == "mix")   s.mark = 1.0f;
            scuffs.push_back (s);
        }
        for (const auto& f : faders)
        {
            const auto sb = f->slider.getBounds().toFloat() + f->getPosition().toFloat();
            scuffs.push_back ({ sb.getCentreX(), sb.getCentreY(), 0.5f * sb.getHeight() - 8.0f, 0.7f, true });
        }
        basePaper = mavkaui::paperTexture (scuffs);
        baseDirt = mavkaui::wearTexture();
        texScuffs = std::move (scuffs);
        showTextures (basePaper, baseDirt, 1.25f);
    }
    wear.setBounds (0, 0, baseW, baseH);
    surface.addAndMakeVisible (wear);   // last: dust, scratches and blood lie on top of everything

    setResizable (true, true);
    setResizeLimits (juce::roundToInt ((float) baseW * minScale), juce::roundToInt ((float) baseH * minScale),
                     juce::roundToInt ((float) baseW * maxScale), juce::roundToInt ((float) baseH * maxScale));
    if (auto* c = getConstrainer())
        c->setFixedAspectRatio ((double) baseW / (double) baseH);
    setSize (juce::roundToInt ((float) baseW * savedScale), juce::roundToInt ((float) baseH * savedScale));

    timerCallback();
    startTimerHz (30);
}

MavkaAudioProcessorEditor::~MavkaAudioProcessorEditor()
{
    stopTimer();
    proc.engine.setMetering (false);
    texCancel = true;
    if (texJob.joinable())
        texJob.join();
    setLookAndFeel (nullptr);
}

void MavkaAudioProcessorEditor::showTextures (const juce::Image& paper, const juce::Image& dirt, float level)
{
    surface.paper = paper;
    wear.dirt = dirt;
    texLevel = level;
    surface.repaint();
}

void MavkaAudioProcessorEditor::updateTextures (bool synchronous)
{
    // a sharper set finished in the background: hand it to every open editor
    if (texJob.joinable() && texDone.load())
    {
        texJob.join();
        if (texResult.level > 0.0f && assets != nullptr && texResult.level > assets->sharp.level)
            assets->sharp = { mavkaui::toNativeImage (texResult.paper), mavkaui::toNativeImage (texResult.dirt), texResult.level };
        texResult = {};
    }

    const float need = mavkaui::textureLevelFor (surface.physicalScale);
    const auto& sharp = assets != nullptr ? assets->sharp : mavkaui::TextureSet();
    const bool sharpFits = sharp.level >= need - 0.01f && sharp.level <= need * 2.2f;

    if (need <= 1.3f)
    {
        if (std::abs (texLevel - 1.25f) > 0.001f)
            showTextures (basePaper, baseDirt, 1.25f);
        texWait = 0;
        return;
    }
    if (sharpFits)
    {
        if (std::abs (texLevel - sharp.level) > 0.001f)
            showTextures (sharp.paper, sharp.dirt, sharp.level);
        texWait = 0;
        return;
    }

    // the window shows more pixels than the textures have: make a sharper set (after the size has settled)
    if (texJob.joinable())
        return;
    if (synchronous)
    {
        auto t = mavkaui::renderTextures (need, texScuffs, nullptr);
        if (t.level > 0.0f && assets != nullptr)
            assets->sharp = { mavkaui::toNativeImage (t.paper), mavkaui::toNativeImage (t.dirt), t.level };
        if (assets != nullptr && assets->sharp.level > 0.0f)
            showTextures (assets->sharp.paper, assets->sharp.dirt, assets->sharp.level);
        return;
    }
    if (++texWait < 10)
        return;
    texWait = 0;
    texCancel = false;
    texDone = false;
    texJob = std::thread ([this, need, scuffs = texScuffs]
    {
        texResult = mavkaui::renderTextures (need, scuffs, &texCancel);
        texDone = true;
    });
}

mavkaui::Knob& MavkaAudioProcessorEditor::addKnob (juce::Component& parent, const juce::String& id, const juce::String& title,
                                                   bool large, juce::Rectangle<int> bounds, const juce::String& tip)
{
    knobs.push_back (std::make_unique<mavkaui::Knob> (proc.apvts, id, title, large, tip));
    auto& k = *knobs.back();
    k.setBounds (bounds);
    parent.addAndMakeVisible (k);
    return k;
}

mavkaui::Choice& MavkaAudioProcessorEditor::addChoice (juce::Component& parent, const juce::String& id, const juce::String& title,
                                                       juce::Rectangle<int> bounds, const juce::String& tip)
{
    choices.push_back (std::make_unique<mavkaui::Choice> (proc.apvts, id, title, tip));
    auto& c = *choices.back();
    c.setBounds (bounds);
    parent.addAndMakeVisible (c);
    return c;
}

mavkaui::Fader& MavkaAudioProcessorEditor::addFader (const juce::String& id, const juce::String& title,
                                                     juce::Rectangle<int> bounds, const juce::String& tip)
{
    faders.push_back (std::make_unique<mavkaui::Fader> (proc.apvts, id, title, tip));
    auto& f = *faders.back();
    f.setBounds (bounds);
    surface.addAndMakeVisible (f);
    return f;
}

void MavkaAudioProcessorEditor::selectPage (int index)
{
    const int i = juce::jlimit (0, numPages - 1, index);
    tabs[(size_t) i].setToggleState (true, juce::dontSendNotification);
    showPage (i);
}

void MavkaAudioProcessorEditor::showPage (int index)
{
    for (int i = 0; i < numPages; ++i)
        pages[(size_t) i].setVisible (i == index);
    currentPage = index;
    proc.engine.setMetering (index == 5);   // the overtone map and the scope cost a little: only while they are seen
}

void MavkaAudioProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll (mavkaui::theme::face);
}

void MavkaAudioProcessorEditor::resized()
{
    const float scale = (float) getWidth() / (float) baseW;
    surface.setBounds (0, 0, baseW, baseH);
    surface.setTransform (juce::AffineTransform::scale (scale));
    proc.uiScale = juce::jlimit (minScale, maxScale, scale);   // remembered with the project
}

void MavkaAudioProcessorEditor::timerCallback()
{
    auto& e = proc.engine;
    const float ink = proc.apvts.getRawParameterValue ("ink")->load();
    stroke.push (e.getInkStroke(), e.getSlantMs(), e.getRelativeDb(), ink);
    stroke.repaint();

    float ph[4];
    for (int i = 0; i < 4; ++i)
        ph[i] = e.getClockPhase (i);
    const int ratio = juce::jlimit (0, 7, (int) proc.apvts.getRawParameterValue ("ratio")->load());
    const int rate = juce::jlimit (0, 6, (int) proc.apvts.getRawParameterValue ("rate")->load());
    clock.setState (ph, mavka::kRatioChoices[ratio], mavka::kRateChoices[rate]);
    clock.repaint();

    osRunning.setText ("running " + juce::String (e.getActiveFactor()) + "x", juce::dontSendNotification);

    float dg[16];
    for (int b = 0; b < 16; ++b)
        dg[b] = e.getDuckGain (b);
    duckMap.setGains (dg);
    duckMap.repaint();

    // the drop: fills instantly when the clip bites, drains in about a quarter of a second
    const float bite = 1.0f - std::exp (-1.5f * e.takeClipHit());
    dropLevel = juce::jmax (bite, dropLevel * 0.88f);
    const float bleedAmount = proc.apvts.getRawParameterValue ("bleed")->load()
                              * juce::jmin (1.0f, 4.0f * proc.apvts.getRawParameterValue ("clip")->load());
    bleedDrop.setLevel (dropLevel, bleedAmount);

    if (currentPage == 5)
    {
        mavka::Engine::ClipView cv;
        e.getClipView (cv);
        clipCurve.setState (cv.knee, cv.pos, cv.drive, cv.peakIn);
        mavka::Engine::OvertoneView ov;
        e.getOvertones (ov);
        scopeIn.resize ((size_t) mavka::Engine::kScope);
        scopeOut.resize ((size_t) mavka::Engine::kScope);
        e.copyScope (scopeIn.data(), scopeOut.data());
        const double sr = proc.getSampleRate() > 0.0 ? proc.getSampleRate() : 48000.0;
        clipScope.setData (scopeIn.data(), scopeOut.data(), mavka::Engine::kScope,
                           ov.conf > 0.3f ? (float) (sr / juce::jmax (40.0f, ov.f0)) : 260.0f, cv.pos);
        const int key = juce::jlimit (0, 11, (int) proc.apvts.getRawParameterValue ("key")->load());
        const int chord = juce::jlimit (0, 11, (int) proc.apvts.getRawParameterValue ("chord")->load());
        const float mix = proc.apvts.getRawParameterValue ("mix")->load(), drive = proc.apvts.getRawParameterValue ("drive")->load();
        const float polish = juce::jmax (mavka::smoothstep (0.5f, 0.1f, mix), mavka::smoothstep (0.35f, 0.05f, drive));
        overtoneMap.a4 = mavka::detuneToA4 (proc.apvts.getRawParameterValue ("detune")->load());
        overtoneMap.setView (ov, juce::String (mavka::kKeyChoices[key]) + " " + juce::String (mavka::kChordChoices[chord]).toLowerCase(),
                             polish);
    }

    updateTextures (false);

    const int preset = proc.getFactoryPresetIndex();
    if (preset != lastPreset)
    {
        presetBox.setSelectedId (preset + 1, juce::dontSendNotification);
        lastPreset = preset;
    }
}
