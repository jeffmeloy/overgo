package latentimage

import (
	"math"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// TestAttentionKeyBiasMaskMath is the model-free mask-bias contract: an additive
// per-key bias of a large negative drops that KEY from the softmax (pad key ->
// ~0 probability), while a zero bias leaves attention unchanged. This is the
// numeric core of the Krea pad-key masking (adaptive runtime_causal_gqa_masked
// rule: a masked key is excluded from max, sum, and output). Pure builder +
// reference backend; no model, no CUDA.
func TestAttentionKeyBiasMaskMath(t *testing.T) {
	const headDim, tokens = 4, 3
	// deterministic small q/k/v [headDim, 1 head, tokens] token-major.
	mk := func(seed uint64) []float32 {
		out := make([]float32, headDim*tokens)
		s := seed
		for i := range out {
			s = s*6364136223846793005 + 1442695040888963407
			out[i] = float32((float64(s>>11)/float64(1<<53) - 0.5) * 0.8)
		}
		return out
	}
	q, k, v := mk(1), mk(2), mk(3)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	shape := tensor.MustShape(headDim, 1, tokens)

	run := func(padKey int, allZero bool) []float32 {
		b := tensor.NewBuilder()
		qN := b.Input("q", dtype.F32, shape)
		kN := b.Input("k", dtype.F32, shape)
		vN := b.Input("v", dtype.F32, shape)
		bias := make([]float32, tokens)
		if !allZero && padKey >= 0 {
			bias[padKey] = encoderPadKeyBias
		}
		kbN := b.Input("kb", dtype.F32, tensor.MustShape(tokens))
		out := b.AttentionWithKeyBias(qN, kN, vN, kbN, scale, true)
		if err := b.Err(); err != nil {
			t.Fatalf("build: %v", err)
		}
		res, err := reference.Execute([]*tensor.Tensor{out}, map[*tensor.Tensor]reference.Value{
			qN: {Shape: shape, Data: q}, kN: {Shape: shape, Data: k}, vN: {Shape: shape, Data: v},
			kbN: {Shape: tensor.MustShape(tokens), Data: bias},
		})
		if err != nil {
			t.Fatalf("execute: %v", err)
		}
		return res[out].Data
	}

	// hand-rolled f64 causal softmax over head 0, optionally skipping a masked key.
	oracle := func(padKey int) []float32 {
		out := make([]float32, headDim*tokens)
		for i := 0; i < tokens; i++ {
			scores := make([]float64, i+1)
			mx := math.Inf(-1)
			for j := 0; j <= i; j++ {
				if j == padKey {
					scores[j] = math.Inf(-1)
					continue
				}
				var dot float64
				for c := 0; c < headDim; c++ {
					dot += float64(q[(i)*headDim+c]) * float64(k[(j)*headDim+c])
				}
				scores[j] = dot * float64(scale)
				mx = math.Max(mx, scores[j])
			}
			var sum float64
			for j := 0; j <= i; j++ {
				if math.IsInf(scores[j], -1) {
					scores[j] = 0
					continue
				}
				scores[j] = math.Exp(scores[j] - mx)
				sum += scores[j]
			}
			for c := 0; c < headDim; c++ {
				var w float64
				for j := 0; j <= i; j++ {
					w += scores[j] / sum * float64(v[j*headDim+c])
				}
				out[i*headDim+c] = float32(w)
			}
		}
		return out
	}

	// 1. all-zero bias is a strict no-op vs unmasked (padKey=-1).
	zero, unmasked := run(-1, true), oracle(-1)
	for i := range zero {
		if d := math.Abs(float64(zero[i] - unmasked[i])); d > 1e-6 {
			t.Fatalf("all-zero key bias diverged from unmasked at %d: %v", i, d)
		}
	}

	// 2. padding key 1 drops it from every later query's softmax.
	masked, want := run(1, false), oracle(1)
	maxAbs := 0.0
	for i := range masked {
		maxAbs = math.Max(maxAbs, math.Abs(float64(masked[i]-want[i])))
	}
	if maxAbs > 1e-6 {
		t.Fatalf("masked attention vs masked oracle max_abs=%.3e exceeds 1e-6", maxAbs)
	}

	// 3. masking actually changes the result (guards a silent no-op / dead mask).
	changed := false
	for i := range masked {
		if math.Abs(float64(masked[i]-unmasked[i])) > 1e-4 {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatal("pad-key mask produced no change vs unmasked; mask is inert")
	}
	t.Logf("key-bias mask math: no-op(zero) exact, masked==oracle max_abs=%.3e, mask active", maxAbs)
}

// TestEncoderProgramMaskedMatchesHostReference proves the MASKED encoder graph
// (CompileEncoderProgramMasked, reference backend) reproduces the MASKED f64
// streaming host oracle (encodeSelected with a key mask) op-for-op, and that the
// mask changes the selected hidden states vs the maskless program. Model-free.
func TestEncoderProgramMaskedMatchesHostReference(t *testing.T) {
	e := syntheticEncoderSpec()
	store := encoderStore(e)
	const seq = 7
	embed := syntheticEncoderEmbed(seq, e.Hidden)
	// mark an interior pad region (keys 3,4 unattended), like [prefix][prompt][pad][suffix].
	mask := make([]bool, seq)
	for i := range mask {
		mask[i] = true
	}
	mask[3], mask[4] = false, false

	host, err := encodeSelected(e, e.RMSNormEps, seq, e.Intermediate, append([]float64(nil), embed...), storeLayerAt(e, store), mask)
	if err != nil {
		t.Fatalf("masked host encodeSelected: %v", err)
	}

	prog, err := CompileEncoderProgramMasked(e, float32(e.RMSNormEps), seq, dtype.F32, mask)
	if err != nil {
		t.Fatalf("CompileEncoderProgramMasked: %v", err)
	}
	weightAt := func(name string) ([]float32, error) { return store[name], nil }
	got, err := prog.RunHostFeed(GraphRunner(reference.Execute), weightAt, f32slice(embed))
	if err != nil {
		t.Fatalf("masked program RunHostFeed: %v", err)
	}
	maxAbs := 0.0
	for i := range host.Data {
		if math.IsNaN(got.Data[i]) || math.IsInf(got.Data[i], 0) {
			t.Fatalf("masked selected[%d] non-finite", i)
		}
		maxAbs = math.Max(maxAbs, math.Abs(host.Data[i]-got.Data[i]))
	}
	if maxAbs > 1e-3 {
		t.Fatalf("masked encoder graph/host divergence max_abs=%.3e exceeds 1e-3", maxAbs)
	}

	// The mask must move the answer vs the maskless program (proves it is wired).
	maskless, err := CompileEncoderProgram(e, float32(e.RMSNormEps), seq, dtype.F32)
	if err != nil {
		t.Fatalf("CompileEncoderProgram: %v", err)
	}
	plain, err := maskless.RunHostFeed(GraphRunner(reference.Execute), weightAt, f32slice(embed))
	if err != nil {
		t.Fatalf("maskless RunHostFeed: %v", err)
	}
	diff := 0.0
	for i := range plain.Data {
		diff = math.Max(diff, math.Abs(plain.Data[i]-got.Data[i]))
	}
	if diff < 1e-4 {
		t.Fatalf("masked and maskless encoder agree (max_abs=%.3e); mask is inert", diff)
	}
	t.Logf("masked encoder graph vs masked host: max_abs=%.3e; masked-vs-maskless delta=%.3e (mask active)", maxAbs, diff)
}
