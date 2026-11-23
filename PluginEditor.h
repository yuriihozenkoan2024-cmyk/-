#pragma once

#include "PluginProcessor.h"

#include <array>
#include <atomic>
#include <memory>
#include <thread>
#include <vector>

namespace mavkaui
{
// worn white: one paper in many shades, black ink, and a little dried blood
namespace theme
{
    inline const juce::Colour face      { 0xffe1dcd0 };   // the faceplate (bone)
    inline const juce::Colour paperHi   { 0xfff5f2eb };   // raised parts: knob caps, fader caps, menus
    inline const juce::Colour paperLo   { 0xffd5cfc2 };
    inline const juce::Colour hair      { 0xffb4ad9e };   // hairlines, tracks
    inline const juce::Colour ink       { 0xff1c1b19 };
    inline const juce::Colour inkSoft   { 0xff4a4741 };
    inline const juce::Colour inkFaint  { 0xff878174 };
    inline const juce::Colour blood     { 0xff6e1410 };
    inline const juce::Colour bloodDark { 0xff3d0907 };
}

// Inter (embedded) or the system sans if the font header is missing
juce::Font sans (float size, bool semibold = false, float tracking = 0.0f);
juce::Font numbers (float size);

struct Assets;   // shared typefaces (one set for every open editor)
std::shared_ptr<Assets> acquireAssets();

class LookAndFeel final : public juce::LookAndFeel_V4
{
public:
    LookAndFeel();
    void drawRotarySlider (juce::Graphics&, int x, int y, int width, int height, float sliderPos,
                           float rotaryStartAngle, float rotaryEndAngle, juce::Slider&) override;
    void drawLinearSlider (juce::Graphics&, int x, int y, int width, int height, float sliderPos, float minSliderPos,
                           float maxSliderPos, juce::Slider::SliderStyle, juce::Slider&) override;
    void drawComboBox (juce::Graphics&, int width, int height, bool isButtonDown, int buttonX, int buttonY,
                       int buttonW, int buttonH, juce::ComboBox&) override;
    juce::Font getComboBoxFont (juce::ComboBox&) override;
    void positionComboBoxText (juce::ComboBox&, juce::Label&) override;
    juce::Font getPopupMenuFont() override;
    void drawPopupMenuBackground (juce::Graphics&, int width, int height) override;
    void drawButtonBackground (juce::Graphics&, juce::Button&, const juce::Colour& backgroundColour,
                               bool shouldDrawButtonAsHighlighted, bool shouldDrawButtonAsDown) override;
    void drawButtonText (juce::Graphics&, juce::TextButton&, bool shouldDrawButtonAsHighlighted,
                         bool shouldDrawButtonAsDown) override;
    void drawToggleButton (juce::Graphics&, juce::ToggleButton&, bool shouldDrawButtonAsHighlighted,
                           bool shouldDrawButtonAsDown) override;
    juce::Rectangle<int> getTooltipBounds (const juce::String& tipText, juce::Point<int> screenPos,
                                           juce::Rectangle<int> parentArea) override;
    void drawTooltip (juce::Graphics&, const juce::String& text, int width, int height) override;
    void drawCornerResizer (juce::Graphics&, int w, int h, bool isMouseOver, bool isMouseDragging) override;
};

// a knob's slider: hold Shift while you grab it for a 10x finer turn; right-click to type a value
class KnobSlider final : public juce::Slider
{
public:
    std::function<void()> onTypeIn;
    int sensitivity = 200;
    void mouseDown (const juce::MouseEvent&) override;
    void mouseDrag (const juce::MouseEvent&) override;
    void mouseUp (const juce::MouseEvent&) override;
};

class Knob final : public juce::Component
{
public:
    Knob (juce::AudioProcessorValueTreeState& state, const juce::String& paramId, const juce::String& title,
          bool large, const juce::String& tooltip);
    void paint (juce::Graphics&) override;
    void resized() override;
    juce::Rectangle<int> getTitleArea() const { return titleArea; }
    juce::Rectangle<int> getValueArea() const { return valueArea; }

    KnobSlider slider;

private:
    void typeIn();
    void finishTyping (bool apply);
    juce::String title;
    bool large;
    juce::Rectangle<int> titleArea, valueArea;
    juce::TextEditor editor;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> attachment;
};

class Fader final : public juce::Component
{
public:
    Fader (juce::AudioProcessorValueTreeState& state, const juce::String& paramId, const juce::String& title,
           const juce::String& tooltip);
    void paint (juce::Graphics&) override;
    void resized() override;

    juce::Slider slider;

private:
    juce::String title;
    juce::Rectangle<int> titleArea, valueArea;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> attachment;
};

class Choice final : public juce::Component
{
public:
    Choice (juce::AudioProcessorValueTreeState& state, const juce::String& paramId, const juce::String& title,
            const juce::String& tooltip);
    void paint (juce::Graphics&) override;
    void resized() override;

    juce::ComboBox box;

private:
    juce::String title;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> attachment;
};

// the live handwriting: black ink ribbon on a recessed strip of paper
class StrokeView final : public juce::Component
{
public:
    static constexpr int N = 200;
    void push (float stroke, float slantMs, float levelDb, float ink);
    void paint (juce::Graphics&) override;

private:
    std::array<float, N> thick {}, offset {};
    int head = 0;
};

// the chord clock: four modulators orbiting at chord ratios
class ClockView final : public juce::Component
{
public:
    void setState (const float* phases, const juce::String& chordName, const juce::String& rateName);
    void paint (juce::Graphics&) override;

private:
    float ph[4] {};
    juce::String chord, rate;
};

// live duck map: how far the echo / forest steps aside in each of the 16 bands right now
class DuckMap final : public juce::Component, public juce::SettableTooltipClient
{
public:
    void setGains (const float* g);
    void paint (juce::Graphics&) override;

private:
    float gains[16] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
};

// the Pressure clip's tell-tale: a drop that fills with blood while the clip bites
class BleedDrop final : public juce::Component, public juce::SettableTooltipClient
{
public:
    void setLevel (float bite, float bleedAmount);
    void paint (juce::Graphics&) override;

private:
    float level = 0.0f, bleed = 0.0f;
};

// CLIP tab: the clip's transfer curve - its shoulder, the ceiling the pen presses down, the live input riding it.
// Drag up / down on it to change the shoulder (double-click: back to the preset's).
class ClipCurve final : public juce::Component, public juce::SettableTooltipClient
{
public:
    explicit ClipCurve (juce::AudioProcessorValueTreeState& state) : apvts (state) {}
    void setState (float knee, float ceilingPos, float pressure, float inputPeak);
    void paint (juce::Graphics&) override;
    void mouseDown (const juce::MouseEvent&) override;
    void mouseDrag (const juce::MouseEvent&) override;
    void mouseUp (const juce::MouseEvent&) override;
    void mouseDoubleClick (const juce::MouseEvent&) override;

private:
    juce::Rectangle<float> plot() const;
    juce::AudioProcessorValueTreeState& apvts;
    float knee = 0.0f, pos = 1.0f, dyn = 1.0f, peak = 0.0f, dragFrom = 0.0f;
    std::array<float, 10> trail {};
    int trailHead = 0;
    bool dragging = false;
};

// CLIP tab: the last periods of the voice going into the clip (outline) and coming out (ink); what is cut off is blood
class ClipScope final : public juce::Component, public juce::SettableTooltipClient
{
public:
    void setData (const float* in, const float* out, int n, float samplesPerPeriod, float ceilingPos);
    void paint (juce::Graphics&) override;

private:
    std::vector<float> a, b;
    float ceilPos = 1.0f;
};

// CLIP tab: what the distortion adds at every overtone of your voice - ink = notes of the song's chord, blood = dirt.
// The outline is what an ordinary distortion would add there. Drag a peak up / down to raise or lower that overtone,
// double-click to reset it.
class OvertoneMap final : public juce::Component, public juce::SettableTooltipClient
{
public:
    explicit OvertoneMap (juce::AudioProcessorValueTreeState& state) : apvts (state) {}
    void setView (const mavka::Engine::OvertoneView& v, const juce::String& keyName, float polish);
    void paint (juce::Graphics&) override;
    void mouseMove (const juce::MouseEvent&) override;
    void mouseExit (const juce::MouseEvent&) override;
    void mouseDown (const juce::MouseEvent&) override;
    void mouseDrag (const juce::MouseEvent&) override;
    void mouseUp (const juce::MouseEvent&) override;
    void mouseDoubleClick (const juce::MouseEvent&) override;
    static constexpr int kBands = mavka::InnerControls::kBands, kHarm = mavka::InnerControls::kHarm;

private:
    juce::Rectangle<float> bars() const;
    float xOf (float overtone) const;
    int columnAt (float x) const;   // overtone index 0..10 (2nd .. 12th) or -1
    float userDb (int h) const;
    juce::AudioProcessorValueTreeState& apvts;
    mavka::Engine::OvertoneView view;
    float after[kBands] {}, before[kBands] {};   // shown levels (dB, with analyser ballistics)
    juce::String key;
    float polishAmt = 0.0f, dragDb = 0.0f, dragY = 0.0f;
    int hover = -1, dragging = -1;

public:
    float a4 = 440.0f;   // the tuning reference (Detune): note names follow it
};

struct Page final : public juce::Component
{
    juce::String description, notes;   // notes: a few printed lines (one per line) inside notesArea
    juce::Rectangle<int> notesArea;
    int dividerX = 0;                  // a hairline between two groups of controls
    void paint (juce::Graphics&) override;
};

// worn white faceplate underneath everything
struct Surface final : public juce::Component
{
    juce::Image paper;
    float physicalScale = 1.0f;   // physical pixels per editor unit, measured on every paint
    void paint (juce::Graphics&) override;
};

// dust, scratches, rubbed-off print and dried blood ON TOP of everything (never catches the mouse)
struct Wear final : public juce::Component
{
    Wear() { setInterceptsMouseClicks (false, false); }
    juce::Image dirt;
    void paint (juce::Graphics&) override;
};

// procedural textures. The base set (1.25 texture pixels per editor pixel) is made once per process;
// sharper sets (up to 3.5, ~3700 x 2800, 4K class) are made in the background when the window needs them.
struct Scuff { float x, y, r, strength; bool fader; float mark = -1.0f; };   // mark: a pencil recall mark at this knob position
struct TextureSet { juce::Image paper, dirt; float level = 0.0f; };
juce::Image paperTexture (const std::vector<Scuff>& scuffs);
juce::Image wearTexture();
TextureSet renderTextures (float level, const std::vector<Scuff>& scuffs, const std::atomic<bool>* cancel);   // any thread
juce::Image toNativeImage (const juce::Image& softwareImage);                                                 // message thread
float textureLevelFor (float physicalScale);
} // namespace mavkaui

//==============================================================================
class MavkaAudioProcessorEditor final : public juce::AudioProcessorEditor,
                                        private juce::Timer
{
public:
    static constexpr int baseW = 1060, baseH = 800;
    static constexpr float minScale = 0.7f, maxScale = 1.6f;

    explicit MavkaAudioProcessorEditor (MavkaAudioProcessor&);
    ~MavkaAudioProcessorEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    void refreshDisplays() { timerCallback(); }
    void selectPage (int index);
    void updateTextures (bool synchronous);   // match the texture resolution to the screen (sync: for tests)
    float getTextureLevel() const noexcept { return texLevel; }
    static constexpr int numPages = 6;

private:
    void timerCallback() override;
    void showPage (int index);
    mavkaui::Knob& addKnob (juce::Component& parent, const juce::String& id, const juce::String& title,
                            bool large, juce::Rectangle<int> bounds, const juce::String& tip);
    mavkaui::Choice& addChoice (juce::Component& parent, const juce::String& id, const juce::String& title,
                                juce::Rectangle<int> bounds, const juce::String& tip);
    mavkaui::Fader& addFader (const juce::String& id, const juce::String& title, juce::Rectangle<int> bounds,
                              const juce::String& tip);

    MavkaAudioProcessor& proc;
    std::shared_ptr<mavkaui::Assets> assets;
    mavkaui::LookAndFeel lnf;
    mavkaui::Surface surface;
    mavkaui::Wear wear;
    std::unique_ptr<juce::TooltipWindow> tooltips;

    juce::ComboBox presetBox;
    juce::Label osRunning;
    juce::ToggleButton autoGain { "Auto gain" }, softLimit { "Soft limit" }, reverbSafe { "Reverb safe" };
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> autoGainAttachment, softLimitAttachment, reverbSafeAttachment;

    mavkaui::StrokeView stroke;
    mavkaui::ClockView clock;
    mavkaui::DuckMap duckMap;
    mavkaui::BleedDrop bleedDrop;
    mavkaui::ClipCurve clipCurve { proc.apvts };
    mavkaui::ClipScope clipScope;
    mavkaui::OvertoneMap overtoneMap { proc.apvts };
    int currentPage = 0;
    std::vector<float> scopeIn, scopeOut;
    std::array<juce::TextButton, numPages> tabs;
    std::array<mavkaui::Page, numPages> pages;

    std::vector<std::unique_ptr<mavkaui::Knob>> knobs;
    std::vector<std::unique_ptr<mavkaui::Choice>> choices;
    std::vector<std::unique_ptr<mavkaui::Fader>> faders;
    int lastPreset = -1;
    float dropLevel = 0.0f;

    // textures: the base set, plus a sharper one rendered in the background for big windows / HiDPI screens
    void showTextures (const juce::Image& paper, const juce::Image& dirt, float level);
    std::vector<mavkaui::Scuff> texScuffs;
    juce::Image basePaper, baseDirt;
    float texLevel = 0.0f;
    int texWait = 0;
    std::thread texJob;
    std::atomic<bool> texCancel { false }, texDone { false };
    mavkaui::TextureSet texResult;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MavkaAudioProcessorEditor)
};
