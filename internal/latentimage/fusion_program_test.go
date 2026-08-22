package latentimage

import (
	"math"
	"testing"

	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestFusionProgramMasksTextPadding(t *testing.T) {
	program, err := CompileFusionProgram(
		syntheticSpec(), 1e-5,
		[]bool{true, false, true}, dtype.F32,
	)
	if err != nil {
		t.Fatal(err)
	}
	if program.keyBias == nil || len(program.keyData) != 3 || program.keyData[1] != padKeyBias {
		t.Fatalf("key bias = %v", program.keyData)
	}
}

// syntheticFusionInput builds deterministic selected hidden states
// [textSeq*TextLayers*TextHidden] at synthetic scale (the SelectedHiddenStates.Data
// layout the fusion consumes).
func syntheticFusionInput(t TransformerSpec, textSeq int) []float64 {
	enc := make([]float64, textSeq*t.TextLayers*t.TextHidden)
	for i := range enc {
		enc[i] = math.Cos(float64(i) * 0.11)
	}
	return enc
}

// The device fusion graph, run on the reference backend, must reproduce the host
// reference Denoiser.textConditioning (denoiser.go) op-for-op: layerwise blocks
// batched over the tapped-layer axis, the layer projector, refiner blocks over the
// token sequence, and txt_in (zero-centered norm, gelu-tanh). One graph
// definition, host golden vs graph, model-free at synthetic scale. Host math is
// f64, the graph f32, so parity holds to an f32 band. This is the parity spine of
// the device text-fusion port.
func TestFusionProgramMatchesHostReference(t *testing.T) {
	const textSeq = 4
	spec := syntheticSpec()
	store := syntheticStore(spec)
	d, err := NewDenoiser(spec, 1e-5, fixtureTimestepProgram(spec), store)
	if err != nil {
		t.Fatalf("NewDenoiser: %v", err)
	}
	enc := syntheticFusionInput(spec, textSeq)

	host, err := d.textConditioning(append([]float64(nil), enc...), textSeq)
	if err != nil {
		t.Fatalf("host textConditioning: %v", err)
	}
	if len(host) != textSeq*spec.Hidden {
		t.Fatalf("host fused len=%d want %d", len(host), textSeq*spec.Hidden)
	}

	prog, err := CompileFusionProgram(spec, 1e-5, attendedTextMask(textSeq), dtype.F32)
	if err != nil {
		t.Fatalf("CompileFusionProgram: %v", err)
	}
	weightAt := func(name string) ([]float32, error) { return store[name], nil }
	got, err := prog.RunHostFeed(GraphRunner(reference.Execute), weightAt, f32slice(enc))
	if err != nil {
		t.Fatalf("program RunHostFeed: %v", err)
	}
	if len(got) != len(host) {
		t.Fatalf("fused len got=%d want %d", len(got), len(host))
	}

	maxAbs, maxRel := 0.0, 0.0
	for i := range host {
		g := float64(got[i])
		if math.IsNaN(g) || math.IsInf(g, 0) {
			t.Fatalf("fused[%d] non-finite: %v", i, g)
		}
		abs := math.Abs(host[i] - g)
		if abs > maxAbs {
			maxAbs = abs
		}
		if rel := abs / (math.Abs(host[i]) + 1e-6); rel > maxRel {
			maxRel = rel
		}
	}
	t.Logf("fusion graph vs host: max_abs=%.3e max_rel=%.3e (%d fused values, textSeq=%d)", maxAbs, maxRel, len(host), textSeq)
	if maxAbs > 1e-3 {
		t.Fatalf("fusion graph/host divergence max_abs=%.3e exceeds 1e-3", maxAbs)
	}

	// determinism: identical replay.
	got2, err := prog.RunHostFeed(GraphRunner(reference.Execute), weightAt, f32slice(enc))
	if err != nil {
		t.Fatalf("program RunHostFeed(2): %v", err)
	}
	for i := range got {
		if got[i] != got2[i] {
			t.Fatalf("nondeterministic fusion graph at %d: %g vs %g", i, got[i], got2[i])
		}
	}
}
