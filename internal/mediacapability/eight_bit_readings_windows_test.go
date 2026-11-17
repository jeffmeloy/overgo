//go:build windows

package mediacapability

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestEachImageModelOffersEightBitAttentionOnItsReadings holds Krea-2-Turbo's and MOSS's eight-bit-attention to its
// readings (eight-bit-attention/images): MOSS's is read on its suite, by its questions and pairwise, against its
// reference drawn at full precision at 1024 x 1024, and its page offers it exactly where both readings find no loss
// beyond chance, a speed tier needing no gain; Krea's reading is pending, so its page holds the preset back.
func TestEachImageModelOffersEightBitAttentionOnItsReadings(t *testing.T) {
	t.Parallel()
	record := testcheck.Must(presetReadings())(t)
	krea := t.TempDir()
	testutil.WriteImageProfileFixture(t, krea)
	moss := testutil.ExtractTarGzip(t, testutil.FixturePath(t, "moss", "generator-compact.tar.gz"), "150c4263057d72476ef4399a442c2b7dbadcb955670232fc2ea869c71f07c5db")
	for model, directory := range map[string]string{"Krea-2-Turbo": krea, "MOSS-Image-SFT": moss} {
		reading, read := record.Readings[model][speedPreset]
		if !read {
			t.Errorf("the record holds no reading of %s's %s, pending or read", model, speedPreset)
			continue
		}
		controls, refusal := Controls(modelrecipe.ModuleLatentImagePrepare, directory)
		if refusal != "" {
			t.Fatal(refusal)
		}
		var page []string
		if held := slices.IndexFunc(controls, func(c Control) bool { return c.Name == presetControl }); held >= 0 {
			page = controls[held].Choices
		}
		if reading.Verdict == nil {
			if model == "MOSS-Image-SFT" {
				t.Errorf("MOSS's %s is pending; this step reads it", speedPreset)
			}
			if slices.Contains(page, speedPreset) {
				t.Errorf("%s's page offers %s, whose reading is pending", model, speedPreset)
			}
			continue
		}
		if reading.Pairwise == nil {
			t.Errorf("%s's %s is read by its questions alone", model, speedPreset)
			continue
		}
		for _, evidence := range []string{reading.Reference, reading.Arm, reading.Pairwise.Record} {
			if id, err := artifact.ParseID(evidence); err != nil || id.Kind() != artifact.KindEvaluation {
				t.Errorf("%s's %s was read from %q, not an evaluation record (%v)", model, speedPreset, evidence, err)
			}
		}
		admitted := reading.Verdict.Admitted && reading.Pairwise.Verdict.Admitted
		t.Logf("%s's %s: %s; pairwise, %s", model, speedPreset, reading.Verdict.Says, reading.Pairwise.Verdict.Says)
		if slices.Contains(page, speedPreset) != admitted {
			t.Errorf("%s's page offers %v; its %s, admitted %t by its questions and %t pairwise, is offered exactly where both admit it",
				model, page, speedPreset, reading.Verdict.Admitted, reading.Pairwise.Verdict.Admitted)
		}
	}
}
