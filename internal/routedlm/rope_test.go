package routedlm

import (
	"math/rand"
	"os"
	"testing"

	"overgo/internal/hostmath"
	"overgo/internal/safetensors"
)

const senseNovaDir = `C:\Users\jeffm\adaptive_new\models\SenseNova-U1-8B-MoT-Infographic-V3`

// Section widths must come from the checkpoint's norm tensor shapes and the
// thetas from the config keys the binding names — verified against the real
// SenseNova artifact (READ-ONLY).
func TestCompileRopePlanSenseNovaRealArtifact(t *testing.T) {
	if testing.Short() {
		t.Skip("opens the real artifact; skipped in -short")
	}
	if _, err := os.Stat(senseNovaDir); err != nil {
		t.Skipf("UNAVAILABLE: artifact absent at %s: %v", senseNovaDir, err)
	}
	binding := SenseNovaBinding()
	cfg, err := LoadConfig(senseNovaDir, binding)
	if err != nil {
		t.Fatal(err)
	}
	src, err := safetensors.OpenSource(senseNovaDir)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	plan, err := CompileRopePlan(src, cfg, binding)
	if err != nil {
		t.Fatal(err)
	}
	want := []RopeSection{
		{Width: 64, Theta: 5e6, Axis: AxisTime},
		{Width: 32, Theta: 1e4, Axis: AxisHeight},
		{Width: 32, Theta: 1e4, Axis: AxisWidth},
	}
	if len(plan.Sections) != len(want) {
		t.Fatalf("sections = %+v, want %+v", plan.Sections, want)
	}
	total := 0
	for i, section := range plan.Sections {
		if section != want[i] {
			t.Fatalf("section %d = %+v, want %+v", i, section, want[i])
		}
		if len(plan.invFreq[i]) != section.Width/2 {
			t.Fatalf("section %d invFreq len %d, want %d", i, len(plan.invFreq[i]), section.Width/2)
		}
		total += section.Width
	}
	if total != cfg.HeadDim {
		t.Fatalf("sections span %d, head_dim %d", total, cfg.HeadDim)
	}
	// hw rotations must depend on (H, W), not Time.
	rng := rand.New(rand.NewSource(7))
	base := make([]float32, cfg.HeadDim)
	for i := range base {
		base[i] = rng.Float32() - 0.5
	}
	rowA := append([]float32(nil), base...)
	rowB := append([]float32(nil), base...)
	plan.applyRotary(rowA, RowPosition{Branch: 1, Time: 3, H: 1, W: 2})
	plan.applyRotary(rowB, RowPosition{Branch: 1, Time: 3, H: 2, W: 1})
	textWidth := plan.Sections[0].Width
	sameText, sameHW := true, true
	for i := range rowA {
		if rowA[i] != rowB[i] {
			if i < textWidth {
				sameText = false
			} else {
				sameHW = false
			}
		}
	}
	if !sameText || sameHW {
		t.Fatalf("text section shared=%v hw section differs=%v, want true/true", sameText, !sameHW)
	}
}

// The degenerate single-section plan must be bit-identical to the plain
// full-width scalar-position rotation (the rxbrain path the ladder gates).
func TestRopePlanDegenerateBitIdentical(t *testing.T) {
	cfg := Config{
		HiddenSize: 512, IntermediateSize: 1024, NumHiddenLayers: 1,
		NumAttentionHeads: 4, NumKeyValueHeads: 2, HeadDim: 128,
		RopeTheta: 5e6, VocabSize: 16, MaxPositionEmbeddings: 4096,
		RMSNormEps:  1e-6,
		RopeScaling: map[string]any{"type": "dynamic", "alpha": 4.0},
	}
	plan, err := ropePlanFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Sections) != 1 || plan.Sections[0].Width != cfg.HeadDim || plan.Sections[0].Axis != AxisTime {
		t.Fatalf("degenerate plan sections %+v", plan.Sections)
	}
	base, err := RopeInvFreqBase(cfg)
	if err != nil {
		t.Fatal(err)
	}
	invFreq := hostmath.RopeInvFreq(base, cfg.HeadDim)
	rng := rand.New(rand.NewSource(11))
	for _, pos := range []int{0, 1, 17, 4095} {
		row := make([]float32, cfg.HeadDim)
		for i := range row {
			row[i] = rng.Float32()*2 - 1
		}
		reference := append([]float32(nil), row...)
		applyRotaryHalfBF16(reference, invFreq, pos)
		plan.applyRotary(row, RowPosition{Time: pos})
		for i := range row {
			if row[i] != reference[i] {
				t.Fatalf("pos %d lane %d: plan %g != reference %g", pos, i, row[i], reference[i])
			}
		}
	}
}
