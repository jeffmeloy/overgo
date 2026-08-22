package latentimage

import (
	"math"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// syntheticForwardInputs builds a deterministic latent + encoder-hidden pair at
// synthetic scale for the given geometry.
func syntheticForwardInputs(t TransformerSpec, textSeq, imgSeq int) (latent, enc []float64) {
	latent = make([]float64, imgSeq*t.InChannels)
	for i := range latent {
		latent[i] = math.Sin(float64(i) * 0.37)
	}
	enc = make([]float64, textSeq*t.TextLayers*t.TextHidden)
	for i := range enc {
		enc[i] = math.Cos(float64(i) * 0.11)
	}
	return latent, enc
}

func attendedTextMask(rows int) []bool {
	mask := make([]bool, rows)
	for index := range mask {
		mask[index] = true
	}
	return mask
}

func TestDenoiserProgramMasksTextPadding(t *testing.T) {
	program, err := CompileDenoiserProgram(
		syntheticSpec(), 1e-5, []bool{true, false, true}, 1, 2, dtype.F32,
	)
	if err != nil {
		t.Fatal(err)
	}
	if program.keyBias == nil || len(program.keyData) != 5 ||
		program.keyData[0] != 0 || program.keyData[1] != padKeyBias || program.keyData[2] != 0 ||
		program.keyData[3] != 0 || program.keyData[4] != 0 {
		t.Fatalf("key bias = %v", program.keyData)
	}
}

// The graph-based device forward must reproduce the host reference
// Denoiser.Forward on the reference backend: same img_in, [text,image] concat,
// 28 (here 2) gated-GQA co-attention blocks, SwiGLU, and final modulated
// projection. This is the parity spine of the device port -- one graph
// definition, host golden vs graph, model-free at synthetic scale. Host math is
// f64, the graph f32, so parity holds to an f32 band.
func TestDenoiserProgramMatchesHostReference(t *testing.T) {
	const gh, gw, textSeq = 2, 2, 3
	imgSeq := gh * gw
	spec := syntheticSpec()
	store := syntheticStore(spec)
	d, err := NewDenoiser(spec, 1e-5, fixtureTimestepProgram(spec), store)
	if err != nil {
		t.Fatalf("NewDenoiser: %v", err)
	}
	latent, enc := syntheticForwardInputs(spec, textSeq, imgSeq)
	const sigma = 0.9

	goldenPatches := mustPack(t, latent, spec.InChannels, gh, gw)
	golden, err := d.Forward(goldenPatches, enc, sigma, textSeq, gh, gw)
	if err != nil {
		t.Fatalf("host Forward: %v", err)
	}

	prog, err := CompileDenoiserProgram(spec, 1e-5, attendedTextMask(textSeq), gh, gw, dtype.F32)
	if err != nil {
		t.Fatalf("CompileDenoiserProgram: %v", err)
	}
	if len(prog.BlockOutputs) != spec.Layers {
		t.Fatalf("block taps=%d want %d", len(prog.BlockOutputs), spec.Layers)
	}
	res, err := prog.Forward(GraphRunner(reference.Execute), d, goldenPatches, enc, sigma)
	if err != nil {
		t.Fatalf("program Forward: %v", err)
	}
	if len(res.Velocity) != imgSeq*spec.InChannels {
		t.Fatalf("velocity len=%d want %d", len(res.Velocity), imgSeq*spec.InChannels)
	}
	maxAbs, maxRel := 0.0, 0.0
	for i := range golden {
		g, got := golden[i], float64(res.Velocity[i])
		if math.IsNaN(got) || math.IsInf(got, 0) {
			t.Fatalf("velocity[%d] non-finite: %v", i, got)
		}
		abs := math.Abs(g - got)
		if abs > maxAbs {
			maxAbs = abs
		}
		if rel := abs / (math.Abs(g) + 1e-6); rel > maxRel {
			maxRel = rel
		}
	}
	t.Logf("graph vs host: max_abs=%.3e max_rel=%.3e (%d image velocities, %d block taps)", maxAbs, maxRel, len(golden), len(res.BlockHidden))
	if maxAbs > 1e-3 {
		t.Fatalf("graph/host velocity divergence max_abs=%.3e exceeds 1e-3", maxAbs)
	}

	const fixtureDelta = -0.125
	feeds, err := prog.hostFeeds(d, goldenPatches, enc, sigma)
	if err != nil {
		t.Fatal(err)
	}
	feeds[prog.InDelta] = reference.Value{
		Shape: prog.InDelta.Shape, Data: []float32{fixtureDelta},
	}
	advanced, err := reference.Execute([]*tensor.Tensor{prog.NextLatent}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	for index, value := range advanced[prog.NextLatent].Data {
		want := float32(goldenPatches[index]) + fixtureDelta*res.Velocity[index]
		if value != want {
			t.Fatalf("next latent[%d] = %g, want %g", index, value, want)
		}
	}

	// determinism: identical replay.
	res2, err := prog.Forward(GraphRunner(reference.Execute), d, goldenPatches, enc, sigma)
	if err != nil {
		t.Fatalf("program Forward(2): %v", err)
	}
	for i := range res.Velocity {
		if res.Velocity[i] != res2.Velocity[i] {
			t.Fatalf("nondeterministic graph at %d: %g vs %g", i, res.Velocity[i], res2.Velocity[i])
		}
	}
	for l, hidden := range res.BlockHidden {
		if len(hidden) != prog.Seq*spec.Hidden {
			t.Fatalf("block %d hidden len=%d want %d", l, len(hidden), prog.Seq*spec.Hidden)
		}
	}
}

func mustPack(t *testing.T, latentRaw []float64, inChannels, gh, gw int) []float64 {
	t.Helper()
	// latentRaw is already the packed [imgSeq*InChannels] sequence for the
	// synthetic case (the host TestDenoiser* tests feed packed patches
	// directly), so pass it through.
	if len(latentRaw) != gh*gw*inChannels {
		t.Fatalf("packed latent len=%d want %d", len(latentRaw), gh*gw*inChannels)
	}
	return latentRaw
}
