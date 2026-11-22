//go:build windows

package mediacapability

import (
	"encoding/json"
	"maps"
	"slices"
	"testing"

	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAPresetChoiceCarriesItsSettings holds what a preset's choice carries (moss-ultra-compare/preset-values): MOSS's
// preset control carries, for each preset it offers and none other, the fields the preset fills in a request that sets
// nothing, as a request states them: image to image Turbo its 8 steps at guidance 1 on ER-SDE and bong_tangent, its
// denoise of 0.72, DMD2 and the SFT adapter, its 16:9 size, its sharpening and the backup negative; depth control its
// moss-control adapter and no skipped block; and no preset's settings name a prompt, a seed or a field it leaves unset.
func TestAPresetChoiceCarriesItsSettings(t *testing.T) {
	t.Parallel()
	root := testutil.ExtractTarGzip(t, testutil.FixturePath(t, "moss", "generator-compact.tar.gz"), "150c4263057d72476ef4399a442c2b7dbadcb955670232fc2ea869c71f07c5db")
	controls, refusal := Controls(modelrecipe.ModuleLatentImagePrepare, root)
	if refusal != "" {
		t.Fatal(refusal)
	}
	at := slices.IndexFunc(controls, func(c Control) bool { return c.Name == presetControl })
	if at < 0 {
		t.Fatal("MOSS's page offers no preset")
	}
	preset := controls[at]
	if !slices.Equal(slices.Sorted(maps.Keys(preset.Settings)), slices.Sorted(slices.Values(preset.Choices))) {
		t.Fatalf("the preset control carries settings for %v; want its choices %v", slices.Sorted(maps.Keys(preset.Settings)), preset.Choices)
	}
	settings := func(name string) map[string]json.RawMessage {
		t.Helper()
		var fields map[string]json.RawMessage
		testcheck.NoError(t, json.Unmarshal(preset.Settings[name], &fields))
		return fields
	}
	for name := range preset.Settings {
		for _, unset := range []string{"prompt", "seed", "preset", "image", "prompt_mode", "control", "layout"} {
			if _, named := settings(name)[unset]; named {
				t.Errorf("%s's settings name %s, which it does not fill: %s", name, unset, preset.Settings[name])
			}
		}
	}
	turbo := settings("ultra-image-to-image-turbo")
	var section struct {
		Steps    int     `json:"steps"`
		Guidance float32 `json:"guidance_scale"`
		Schedule struct{ Sampler, Name string }
		Denoise  float64 `json:"denoise"`
		Adapters []struct {
			Name string `json:"name"`
		} `json:"adapters"`
		Width, Height int
		Sharpen       struct{ Radius int }
		Negative      string `json:"negative_prompt"`
	}
	testcheck.NoError(t, json.Unmarshal(preset.Settings["ultra-image-to-image-turbo"], &section))
	canvas := testcheck.Must(func() ([2]int, error) {
		width, height, err := media.Canvas{Aspect: "16:9", Megapixels: 3, Multiple: 32}.Size()
		return [2]int{width, height}, err
	}())(t)
	if section.Steps != 8 || section.Guidance != 1 || section.Schedule.Sampler != "er_sde" || section.Schedule.Name != "bong_tangent" || section.Denoise != 0.72 ||
		len(section.Adapters) != 2 || section.Adapters[0].Name != "8-steps-dmd2-lora" || section.Adapters[1].Name != "moss_image_sft" ||
		[2]int{section.Width, section.Height} != canvas || section.Sharpen.Radius != 1 || section.Negative == "" {
		t.Errorf("image to image Turbo carries %s; want its section's recorded settings", preset.Settings["ultra-image-to-image-turbo"])
	}
	if _, skips := turbo["skip_blocks"]; skips {
		t.Errorf("image to image Turbo names its skipped blocks %s; it skips the profile's", turbo["skip_blocks"])
	}
	depth := settings("ultra-depth-control")
	if string(depth["skip_blocks"]) != "[]" || !json.Valid(depth["adapters"]) || !slices.Contains(slices.Collect(maps.Keys(depth)), "adapters") {
		t.Errorf("depth control carries %s; want moss-control and no skipped block", preset.Settings["ultra-depth-control"])
	}
}
