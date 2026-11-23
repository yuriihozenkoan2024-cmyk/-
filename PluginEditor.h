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

class Knob final : public juce::Component
{
public:
    Knob (juce::AudioProcessorValueTreeState& state, const juce::String& paramId, const juce::String& title,
          bool large, const juce::String& tooltip);
    void paint (juce::Graphics&) override;
    void resized() override;
    juce::Rectangle<int> getTitleArea() const { return titleArea; }
    juce::Rectangle<int> getValueArea() const { return valueArea; }

    juce::Slider slider;

private:
    juce::String title;
    bool large;
    juce::Rectangle<int> titleArea, valueArea;
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
    static constexpr int numPages = 5;

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
    juce::ToggleButton autoGain { "Auto gain" }, softLimit { "Soft limit" };
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> autoGainAttachment, softLimitAttachment;

    mavkaui::StrokeView stroke;
    mavkaui::ClockView clock;
    mavkaui::DuckMap duckMap;
    mavkaui::BleedDrop bleedDrop;
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
