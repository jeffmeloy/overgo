//go:build windows

package mediacapability

import (
	"encoding/json"
	"slices"
	"testing"

	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// mossSectionPresets are the MOSS Ultra workflow's section presets MOSS's catalog declares
// (moss-ultra-compare/section-presets), in the workflow's order, the order a page lists them (moss-page/sections).
var mossSectionPresets = []string{"ultra-text-to-image", "ultra-image-to-image", "ultra-depth-control", "ultra-text-to-image-turbo",
	"ultra-image-to-image-turbo", "ultra-depth-control-turbo"}

// mossSectionNames are the plain names a page offers the sections by, in mossSectionPresets' order.
var mossSectionNames = []string{"Text to image", "Image to image", "Depth control", "Text to image (Turbo)", "Image to image (Turbo)",
	"Depth control (Turbo)"}

func TestMOSSImageGenerationContract(t *testing.T) {
	root := testutil.ExtractTarGzip(t, testutil.FixturePath(t, "moss", "generator-compact.tar.gz"), "150c4263057d72476ef4399a442c2b7dbadcb955670232fc2ea869c71f07c5db")
	controls, refusal := Controls(modelrecipe.ModuleLatentImagePrepare, root)
	if refusal != "" {
		t.Fatal(refusal)
	}
	held := map[string]Control{}
	for _, c := range controls {
		held[c.Name] = c
	}
	for _, name := range []string{"prompt", "prompt_mode", "negative_prompt", "width", "height", "steps", "seed", "guidance_scale", "preset", "image", "denoise", "schedule", "canvas", "sharpen", "control", "layout"} {
		if _, ok := held[name]; !ok {
			t.Fatal("missing supported control", name)
		}
	}
	// Of MOSS's presets the page offers the MOSS Ultra workflow's six section presets, first and in the workflow's
	// order, and its authors' tutorial setting, which no reading reads, and 8-bit attention, a speed tier its readings
	// admit; no guidance preset read shows a gain (common-controls/gain-only).
	if len(held) != 16 || !slices.Equal(held["preset"].Choices, append(slices.Clone(mossSectionPresets), "comfy-tutorial", "eight-bit-attention")) || held["guidance_scale"].Type != ControlNumber || len(held["prompt_mode"].Choices) != 4 || held["prompt_mode"].Help == "" {
		t.Fatal("unsupported controls", held)
	}
	// The preset is MOSS's section, the control its page leads with, outside the disclosure, each choice by its plain
	// name (moss-page/sections).
	if section := controls[0]; section.Name != "preset" || section.Role != "section" || section.Secondary() || section.Required ||
		len(section.Names) != len(section.Choices) || section.Names["comfy-tutorial"] != "ComfyUI tutorial settings" {
		t.Fatal("MOSS's page does not lead with its section", controls[0])
	}
	// Each section shows the fields it runs that another does not, a text to image section none, and the sampling is a
	// group of its own (moss-page/groups).
	if shows := controls[0].Shows; shows["ultra-text-to-image"] == nil || len(shows["ultra-text-to-image"]) != 0 ||
		!slices.Equal(shows["ultra-image-to-image-turbo"], []string{"image", "denoise"}) || !slices.Equal(shows["ultra-depth-control"], []string{"control"}) || shows["comfy-tutorial"] != nil {
		t.Fatal("MOSS's sections show other than the fields they run", shows)
	}
	for _, name := range []string{"steps", "seed", "guidance_scale", "schedule"} {
		if held[name].Group != "sampling" {
			t.Errorf("MOSS's %s is in the group %q; want its sampling", name, held[name].Group)
		}
	}
	for index, preset := range mossSectionPresets {
		if held["preset"].Names[preset] != mossSectionNames[index] {
			t.Errorf("the section %s is offered as %q; want %q", preset, held["preset"].Names[preset], mossSectionNames[index])
		}
	}
	// Tuning sits behind the disclosure as on every page (common-controls/do), and the negative prompt beside the prompt.
	for _, name := range []string{"steps", "seed", "guidance_scale"} {
		if !held[name].Secondary() || held[name].Required {
			t.Fatal("optional tuning control is not secondary", name)
		}
	}
	if held["negative_prompt"].Secondary() || held["negative_prompt"].Required {
		t.Fatal("the negative prompt is behind the disclosure or required", held["negative_prompt"])
	}
	for name, want := range map[string]Bounds{"width": {Default: 1024, Step: 16}, "height": {Default: 1024, Step: 16}, "steps": {Default: 50, Step: 1}, "guidance_scale": {Default: 3}} {
		if held[name].Bounds == nil || *held[name].Bounds != want {
			t.Fatal("source default differs", name, held[name])
		}
	}
	raw, err := Catalog[recipe.TaskImageGen].ValidationInput(root, "  blue cube\n")
	testcheck.NoError(t, err)
	var input latentimage.Request
	if err = json.Unmarshal([]byte(raw), &input); err != nil {
		t.Fatal(err)
	}
	if input.Prompt != "  blue cube\n" || input.Width != 16 || input.Height != 16 || input.Steps != 1 || input.GuidanceScale != nil {
		t.Fatal("source-derived validation request differs", input)
	}
	krea := t.TempDir()
	testutil.WriteImageProfileFixture(t, krea)
	controls, refusal = Controls(modelrecipe.ModuleLatentImagePrepare, krea)
	if refusal != "" {
		t.Fatal(refusal)
	}
	// Krea runs unguided, so it offers no guidance scale or layout mode, and its negative prompt, which it honours by
	// NAG (common-controls/negative).
	if !slices.ContainsFunc(controls, func(c Control) bool { return c.Name == "negative_prompt" && c.Role == "negative prompt" && !c.Advanced }) {
		t.Fatal("Krea's page offers no negative prompt beside its prompt", controls)
	}
	for _, c := range controls {
		if c.Name == "guidance_scale" || c.Name == "prompt_mode" {
			t.Fatal("unsupported override offered for another profile", c)
		}
		// Krea-2-Turbo's page offers the size its publisher draws, 2048 by 2048.
		if (c.Name == "width" || c.Name == "height") && (c.Bounds == nil || c.Bounds.Default != 2048) {
			t.Fatal("the flow profile offers no publisher size", c)
		}
	}
}
