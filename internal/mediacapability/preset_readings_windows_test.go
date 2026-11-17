//go:build windows

package mediacapability

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/latentimage"
	"overgo/internal/latentvideo"
	"overgo/internal/modelrecipe"
	"overgo/internal/sensenovarecipe"
	"overgo/internal/testcheck"
)

// TestOnlyAnAdmittedGuidancePresetIsOffered holds each model's presets to the record of their readings
// (guidance-ops/preset-readings): a preset the record reads on a model is offered on its page only once its verdict,
// read against that model's reference, admits it, and its pairwise verdict where it was read so, so a pending or refused
// one is not; and every preset a model declares is read but those that are no guider: Krea's draft and MOSS's
// comfy-tutorial, steps and a scale from their authors, and Wan's eight-bit-attention, a speed choice its clips' frame
// bound holds (TestWanEightBitAttentionKeepsItsClipsWithinTheirBound). The record reads only presets its model declares;
// a verdict holds two evaluation records, its model's reference the same for each of its presets, and the counts its
// words say, and its model's suite is named. SenseNova's page, whose presets its recipe declares with no model files, is
// read whole.
func TestOnlyAnAdmittedGuidancePresetIsOffered(t *testing.T) {
	t.Parallel()
	record := testcheck.Must(presetReadings())(t)
	image := func(pipeline string) []string {
		t.Helper()
		directory := t.TempDir()
		testcheck.NoError(t, os.WriteFile(filepath.Join(directory, "model_index.json"), []byte(`{"_class_name": "`+pipeline+`"}`), 0o644))
		return testcheck.Must(latentimage.PresetNames(directory))(t)
	}
	student := t.TempDir()
	testcheck.NoError(t, os.MkdirAll(filepath.Join(student, "checkpoints"), 0o755))
	testcheck.NoError(t, os.WriteFile(filepath.Join(student, "checkpoints", "self_forcing_dmd.pt"), nil, 0o644))
	type model struct {
		presets, unread []string
	}
	models := map[string]model{
		"Krea-2-Turbo":                       {image("Krea2Pipeline"), []string{"draft"}},
		"MOSS-Image-SFT":                     {image("MOSSImagePipeline"), []string{"comfy-tutorial"}},
		"Wan2.1-T2V-1.3B":                    {testcheck.Must(latentvideo.PresetNames(t.TempDir()))(t), []string{"eight-bit-attention"}},
		"Self-Forcing":                       {testcheck.Must(latentvideo.PresetNames(student))(t), nil},
		"SenseNova-U1-8B-MoT-Infographic-V3": {testcheck.Must(sensenovarecipe.PresetNames())(t), nil},
	}
	for name, declared := range models {
		offered := offeredPresets(record, name, declared.presets)
		for _, preset := range declared.presets {
			reading, read := record.Readings[name][preset]
			if !read && !slices.Contains(declared.unread, preset) {
				t.Errorf("%s declares the preset %s, which the record does not read", name, preset)
			}
			gained := read && reading.Verdict != nil && (reading.Verdict.Gained() || reading.Pairwise != nil && reading.Pairwise.Verdict.Gained())
			admitted := !read || gained && reading.Verdict.Admitted && (reading.Pairwise == nil || reading.Pairwise.Verdict.Admitted)
			if slices.Contains(offered, preset) != admitted {
				t.Errorf("%s offers %v; its preset %s, read %t with verdict %+v and pairwise %+v, is offered only unread, or gaining on a reading and admitted by both",
					name, offered, preset, read, reading.Verdict, reading.Pairwise)
			}
		}
	}

	for name, presets := range record.Readings {
		reference := ""
		for preset, reading := range presets {
			if !slices.Contains(models[name].presets, preset) {
				t.Errorf("the record reads %s's preset %s, which it does not declare", name, preset)
			}
			verdict := reading.Verdict
			if verdict == nil {
				if reading.Reference != "" || reading.Arm != "" {
					t.Errorf("%s's pending preset %s names the records %q and %q", name, preset, reading.Reference, reading.Arm)
				}
				continue
			}
			for _, evidence := range []string{reading.Reference, reading.Arm} {
				if id, err := artifact.ParseID(evidence); err != nil || id.Kind() != artifact.KindEvaluation {
					t.Errorf("%s's preset %s was read from %q, not an evaluation record (%v)", name, preset, evidence, err)
				}
			}
			if reference != "" && reference != reading.Reference || reading.Reference == reading.Arm {
				t.Errorf("%s's preset %s is read against %s, its arm %s; its other presets against %s", name, preset, reading.Reference, reading.Arm, reference)
			}
			reference = reading.Reference
			if err := verdictSaid(*verdict); err != nil {
				t.Errorf("%s's preset %s: %v", name, preset, err)
			}
			if record.Judge == "" || record.Suites[name] == "" {
				t.Errorf("%s's preset %s was read with judge %q and suite %q", name, preset, record.Judge, record.Suites[name])
			}
		}
	}

	controls, refusal := Controls(modelrecipe.ModuleRoutedImagePrepare, t.TempDir())
	if refusal != "" {
		t.Fatal(refusal)
	}
	var page []string
	if held := slices.IndexFunc(controls, func(c Control) bool { return c.Name == presetControl }); held >= 0 {
		page = controls[held].Choices
	}
	if want := offeredPresets(record, senseNovaModel, models[senseNovaModel].presets); !slices.Equal(slices.Sorted(slices.Values(page)), slices.Sorted(slices.Values(want))) {
		t.Errorf("SenseNova's page offers %v; its admitted presets are %v", page, want)
	}
}
