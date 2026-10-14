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

// syntheticEncoderEmbed builds deterministic per-token embedding rows.
func syntheticEncoderEmbed(seq, hidden int) []float64 {
	embed := make([]float64, seq*hidden)
	st := uint64(0x2545f4914f6cdd1d)
	for i := range embed {
		st = st*6364136223846793005 + 1442695040888963407
		embed[i] = (float64(st>>11)/float64(1<<53) - 0.5) * 0.2
	}
	return embed
}

func f32slice(v []float64) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(x)
	}
	return out
}

// The device encoder graph, run on the reference backend, must reproduce the host
// reference encodeSelected (textencoder.go) op-for-op: standard RMSNorm, per-head
// q/k norm, rotate-half RoPE, causal GQA, SwiGLU, and the 12-layer (here 3)
// selection/capture. One graph definition, host golden vs graph, model-free at
// synthetic scale. Host math is f64, the graph f32, so parity holds to an f32
// band. This is the parity spine of the device encoder port.
func TestEncoderProgramMatchesHostReference(t *testing.T) {
	e := syntheticEncoderSpec()
	store := encoderStore(e)
	const seq = 5
	embed := syntheticEncoderEmbed(seq, e.Hidden)

	host, err := encodeSelected(e, e.RMSNormEps, seq, e.Intermediate, append([]float64(nil), embed...), storeLayerAt(e, store), nil)
	if err != nil {
		t.Fatalf("host encodeSelected: %v", err)
	}

	prog, err := CompileEncoderProgram(e, float32(e.RMSNormEps), seq, dtype.F32)
	if err != nil {
		t.Fatalf("CompileEncoderProgram: %v", err)
	}
	if len(prog.Selected) != len(e.SelectLayers) {
		t.Fatalf("selected taps=%d want %d", len(prog.Selected), len(e.SelectLayers))
	}
	for i, v := range e.SelectLayers {
		if prog.CaptureAfter[i] != v-1 {
			t.Errorf("captureAfter[%d]=%d want %d", i, prog.CaptureAfter[i], v-1)
		}
	}
	weightAt := func(name string) ([]float32, error) { return store[name], nil }
	got, err := prog.RunHostFeed(reference.Execute, weightAt, f32slice(embed))
	if err != nil {
		t.Fatalf("program RunHostFeed: %v", err)
	}
	if got.Seq != host.Seq || got.LayerCount != host.LayerCount || got.Hidden != host.Hidden {
		t.Fatalf("geometry got [%d,%d,%d] want [%d,%d,%d]", got.Seq, got.LayerCount, got.Hidden, host.Seq, host.LayerCount, host.Hidden)
	}
	maxAbs, maxRel := 0.0, 0.0
	for i := range host.Data {
		g := got.Data[i]
		if math.IsNaN(g) || math.IsInf(g, 0) {
			t.Fatalf("selected[%d] non-finite: %v", i, g)
		}
		abs := math.Abs(host.Data[i] - g)
		maxAbs = max(maxAbs, abs)
		if rel := abs / (math.Abs(host.Data[i]) + 1e-6); rel > maxRel {
			maxRel = rel
		}
	}
	t.Logf("encoder graph vs host: max_abs=%.3e max_rel=%.3e (%d selected values, taps after %v)", maxAbs, maxRel, len(host.Data), prog.CaptureAfter)
	if maxAbs > 1e-3 {
		t.Fatalf("encoder graph/host divergence max_abs=%.3e exceeds 1e-3", maxAbs)
	}

	// determinism: identical replay.
	got2, err := prog.RunHostFeed(reference.Execute, weightAt, f32slice(embed))
	if err != nil {
		t.Fatalf("program RunHostFeed(2): %v", err)
	}
	for i := range got.Data {
		if got.Data[i] != got2.Data[i] {
			t.Fatalf("nondeterministic encoder graph at %d: %g vs %g", i, got.Data[i], got2.Data[i])
		}
	}
}

// RunHostFeed executes the encoder graph through run with F32 weights supplied by
// weightAt and the per-token embedding rows embedRows ([seq*Hidden], token-major
// F32). This is the exact-parity path (F32 weights): reference.Execute gives the
// host golden, the CUDA generic executor gives the device match. Returns the
// tapped hidden states in the [Seq, LayerCount, Hidden] layout Denoiser.
// textConditioning consumes.
func (p *EncoderProgram) RunHostFeed(run graphruntime.Runner, weightAt EncoderFeed, embedRows []float32) (*SelectedHiddenStates, error) {
	if p.MatmulType != dtype.F32 {
		return nil, fmt.Errorf("encoder RunHostFeed: needs F32 weights, program compiled %s", p.MatmulType)
	}
	embedValue, err := reference.BorrowedValue(p.Embed.Shape, embedRows)
	if err != nil {
		return nil, fmt.Errorf("encoder RunHostFeed: embed rows: %w", err)
	}
	feeds := make(map[*tensor.Tensor]reference.Value, len(p.weightInputs)+1)
	if err := graphruntime.AddHostWeights(feeds, p.weightInputs, weightAt); err != nil {
		return nil, fmt.Errorf("encoder RunHostFeed: %w", err)
	}
	feeds[p.Embed] = embedValue
	if p.keyBias != nil {
		feeds[p.keyBias] = reference.Value{Shape: p.keyBias.Shape, Data: p.keyBiasData}
	}

	results, err := run(append([]*tensor.Tensor(nil), p.Selected...), feeds)
	if err != nil {
		return nil, fmt.Errorf("encoder RunHostFeed: %w", err)
	}
	return p.assemble(results)
}

// assemble gathers the per-slot [Hidden, seq] outputs into a single
// [Seq, LayerCount, Hidden] SelectedHiddenStates (token-major, then tapped-layer,
// then feature) -- the layout Denoiser.textConditioning consumes.
func (p *EncoderProgram) assemble(results map[*tensor.Tensor]reference.Value) (*SelectedHiddenStates, error) {
	h := p.E.Hidden
	L := len(p.Selected)
	out := &SelectedHiddenStates{Seq: p.Seq, LayerCount: L, Hidden: h, Data: make([]float64, p.Seq*L*h)}
	for slot, node := range p.Selected {
		val, ok := results[node]
		if !ok {
			return nil, fmt.Errorf("encoder assemble: missing selected slot %d output", slot)
		}
		if len(val.Data) != p.Seq*h {
			return nil, fmt.Errorf("encoder assemble: slot %d len=%d want %d", slot, len(val.Data), p.Seq*h)
		}
		for tok := 0; tok < p.Seq; tok++ {
			src := val.Data[tok*h : tok*h+h]
			dst := out.Data[(tok*L+slot)*h : (tok*L+slot)*h+h]
			for c := range h {
				dst[c] = float64(src[c])
			}
		}
	}
	return out, nil
}

// EncoderFeed is a name->F32 weight lookup for the host-feed reference/CUDA path.
type EncoderFeed func(name string) ([]float32, error)
