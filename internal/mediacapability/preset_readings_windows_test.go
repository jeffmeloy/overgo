//go:build windows

package mediacapability

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/latentimage"
	"overgo/internal/latentvideo"
	"overgo/internal/modelrecipe"
	"overgo/internal/sensenovarecipe"
	"overgo/internal/testcheck"
)

// TestOnlyAnAdmittedGuidancePresetIsOffered holds each model's presets to the record of their readings
// (guidance-ops/preset-readings): a preset the record reads is offered only once its verdict, read on its model against
// that model's reference, admits it, so a pending or refused one is not, and every preset a model declares is read but
// Krea's draft and MOSS's comfy-tutorial, steps and a scale from their authors, no guider. A reading names the one
// model of its family that declares the preset, so a family's verdict is that model's; a verdict holds two evaluation
// records, its model's reference the same for each of its presets, and the counts its words say, and its model's suite
// is named. SenseNova's page, whose presets its recipe declares with no model files, is read whole.
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
		family  string
		presets []string
	}
	models := map[string]model{
		"Krea-2-Turbo":                       {latentImagePresets, image("Krea2Pipeline")},
		"MOSS-Image-SFT":                     {latentImagePresets, image("MOSSImagePipeline")},
		"Wan2.1-T2V-1.3B":                    {latentVideoPresets, testcheck.Must(latentvideo.PresetNames(t.TempDir()))(t)},
		"Self-Forcing":                       {latentVideoPresets, testcheck.Must(latentvideo.PresetNames(student))(t)},
		"SenseNova-U1-8B-MoT-Infographic-V3": {senseNovaPresets, testcheck.Must(sensenovarecipe.PresetNames())(t)},
	}
	unread := []string{"draft", "comfy-tutorial"}
	for name, declared := range models {
		offered := offeredPresets(record, declared.family, declared.presets)
		for _, preset := range declared.presets {
			reading, read := record.Readings[declared.family][preset]
			if !read && !slices.Contains(unread, preset) {
				t.Errorf("%s declares the preset %s, which the record does not read", name, preset)
			}
			if admitted := !read || reading.Verdict != nil && reading.Verdict.Admitted; slices.Contains(offered, preset) != admitted {
				t.Errorf("%s offers %v; its preset %s, read %t with verdict %+v, is offered only unread or admitted", name, offered, preset, read, reading.Verdict)
			}
		}
	}

	references := map[string]string{}
	for family, presets := range record.Readings {
		for preset, reading := range presets {
			if declarer := models[reading.Model]; declarer.family != family || !slices.Contains(declarer.presets, preset) {
				t.Errorf("the record reads %s's preset %s on %q, which does not declare it", family, preset, reading.Model)
			}
			for name, other := range models {
				if name != reading.Model && other.family == family && slices.Contains(other.presets, preset) {
					t.Errorf("%s's preset %s is declared by %s and %s, so its verdict on %s would be offered on both", family, preset, reading.Model, name, reading.Model)
				}
			}
			verdict := reading.Verdict
			if verdict == nil {
				if reading.Reference != "" || reading.Arm != "" {
					t.Errorf("%s's pending preset %s names the records %q and %q", family, preset, reading.Reference, reading.Arm)
				}
				continue
			}
			for _, evidence := range []string{reading.Reference, reading.Arm} {
				if id, err := artifact.ParseID(evidence); err != nil || id.Kind() != artifact.KindEvaluation {
					t.Errorf("%s's preset %s was read from %q, not an evaluation record (%v)", family, preset, evidence, err)
				}
			}
			if known, seen := references[reading.Model]; seen && known != reading.Reference || reading.Reference == reading.Arm {
				t.Errorf("%s's preset %s is read against %s, its arm %s; %s's other presets against %s", family, preset, reading.Reference, reading.Arm, reading.Model, known)
			}
			references[reading.Model] = reading.Reference
			counts := fmt.Sprintf("%d wins, %d losses and %d ties over %d media", verdict.Wins, verdict.Losses, verdict.Ties, verdict.Media)
			if verdict.Wins+verdict.Losses+verdict.Ties == 0 || verdict.Media == 0 || !strings.HasSuffix(verdict.Says, counts) ||
				strings.HasPrefix(verdict.Says, "refused") == verdict.Admitted {
				t.Errorf("%s's preset %s holds %s, admitted %t, saying %q", family, preset, counts, verdict.Admitted, verdict.Says)
			}
			if record.Judge == "" || record.Suites[reading.Model] == "" {
				t.Errorf("%s's preset %s was read on %s with judge %q and suite %q", family, preset, reading.Model, record.Judge, record.Suites[reading.Model])
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
	senseNova := models["SenseNova-U1-8B-MoT-Infographic-V3"]
	if want := offeredPresets(record, senseNovaPresets, senseNova.presets); !slices.Equal(slices.Sorted(slices.Values(page)), slices.Sorted(slices.Values(want))) {
		t.Errorf("SenseNova's page offers %v; its admitted presets are %v", page, want)
	}
}
