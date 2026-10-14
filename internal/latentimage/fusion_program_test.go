package latentimage

import (
	"fmt"
	"math"
	"testing"

	"overgo/internal/graphruntime"
	"overgo/internal/tensor"
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
	got, err := prog.RunHostFeed(reference.Execute, weightAt, f32slice(enc))
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
		maxAbs = max(maxAbs, abs)
		if rel := abs / (math.Abs(host[i]) + 1e-6); rel > maxRel {
			maxRel = rel
		}
	}
	t.Logf("fusion graph vs host: max_abs=%.3e max_rel=%.3e (%d fused values, textSeq=%d)", maxAbs, maxRel, len(host), textSeq)
	if maxAbs > 1e-3 {
		t.Fatalf("fusion graph/host divergence max_abs=%.3e exceeds 1e-3", maxAbs)
	}

	// determinism: identical replay.
	got2, err := prog.RunHostFeed(reference.Execute, weightAt, f32slice(enc))
	if err != nil {
		t.Fatalf("program RunHostFeed(2): %v", err)
	}
	for i := range got {
		if got[i] != got2[i] {
			t.Fatalf("nondeterministic fusion graph at %d: %g vs %g", i, got[i], got2[i])
		}
	}
}

// FusionTensorShapes returns the name -> torch shape ([out,in] for Linear
// weights) manifest the fusion forward consumes: the layerwise + refiner blocks
// at TextHidden, the layer projector, and txt_in. Derived entirely from spec.
// This is the capability-retention contract for the text-fusion stream (a subset
// of DenoiserTensorShapes -- the same blocks, minus the 28 image co-attention
// blocks and the timestep/img_in/final tensors the denoiser step owns).
func FusionTensorShapes(t TransformerSpec) map[string][]int {
	h := t.Hidden
	th := t.TextHidden
	shapes := map[string][]int{
		"text_fusion.projector.weight": {tensor.SingletonExtent, t.TextLayers},
		"txt_in.norm.weight":           {th},
		"txt_in.linear_1.weight":       {h, th},
		"txt_in.linear_1.bias":         {h},
		"txt_in.linear_2.weight":       {h, h},
		"txt_in.linear_2.bias":         {h},
	}
	addFusionBlocks := func(kind string, n int) {
		for i := range n {
			p := fmt.Sprintf("text_fusion.%s.%d.", kind, i)
			addAttnFF(shapes, p, th, t.TextHeads*t.HeadDim, t.TextKVHeads*t.HeadDim, t.HeadDim, t.TextIntermediate, false)
		}
	}
	addFusionBlocks("layerwise_blocks", t.LayerwiseTextBlocks)
	addFusionBlocks("refiner_blocks", t.RefinerTextBlocks)
	return shapes
}

// RunHostFeed executes the fusion graph through run with F32 weights supplied by
// weightAt and the selected hiddens encoderHidden ([textSeq*TextLayers*TextHidden],
// the SelectedHiddenStates.Data layout, token-major then layer then feature). This
// is the exact-parity path (F32 weights): reference.Execute gives the host golden,
// the CUDA generic executor gives the device match. Returns the fused conditioning
// [textSeq*Hidden] (token-major), the layout the denoiser text stream consumes.
func (p *FusionProgram) RunHostFeed(run graphruntime.Runner, weightAt FusionFeed, encoderHidden []float32) ([]float32, error) {
	if p.MatmulType != dtype.F32 {
		return nil, fmt.Errorf("fusion RunHostFeed: needs F32 weights, program compiled %s", p.MatmulType)
	}
	if want := p.TextSeq * p.T.TextLayers * p.T.TextHidden; len(encoderHidden) != want {
		return nil, fmt.Errorf("fusion RunHostFeed: encoder hidden len=%d want %d", len(encoderHidden), want)
	}
	feeds := make(map[*tensor.Tensor]reference.Value, len(p.weightInputs)+2)
	if err := graphruntime.AddHostWeights(feeds, p.weightInputs, weightAt); err != nil {
		return nil, fmt.Errorf("fusion RunHostFeed: %w", err)
	}
	feeds[p.InEncoder] = reference.Value{Shape: p.InEncoder.Shape, Data: encoderHidden}
	if p.keyBias != nil {
		feeds[p.keyBias] = reference.Value{Shape: p.keyBias.Shape, Data: p.keyData}
	}

	results, err := run([]*tensor.Tensor{p.Fused}, feeds)
	if err != nil {
		return nil, fmt.Errorf("fusion RunHostFeed: %w", err)
	}
	return p.assemble(results)
}

// assemble extracts the fused conditioning [textSeq*Hidden] (token-major) from the
// graph outputs.
func (p *FusionProgram) assemble(results map[*tensor.Tensor]reference.Value) ([]float32, error) {
	val, ok := results[p.Fused]
	if !ok {
		return nil, fmt.Errorf("fusion assemble: missing fused output")
	}
	if want := p.TextSeq * p.T.Hidden; len(val.Data) != want {
		return nil, fmt.Errorf("fusion assemble: fused len=%d want %d", len(val.Data), want)
	}
	return append([]float32(nil), val.Data...), nil
}

// FusionFeed is a name->F32 weight lookup for the host-feed reference/CUDA path.
type FusionFeed func(name string) ([]float32, error)
