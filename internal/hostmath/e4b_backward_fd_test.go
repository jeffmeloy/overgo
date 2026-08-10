// Finite-difference gates for the E4B (gemma3n) training-leg VJPs, each central-
// FD'd against the exact forward it differentiates (reusing fdTol/fdStep from
// backward_fd_test.go). No torch fixture — the forwards are host-exact.
package hostmath

import (
	"math"
	"math/rand"
	"testing"
)

func e4bRandVec(rng *rand.Rand, n int, base, s float64) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(base + rng.NormFloat64()*s)
	}
	return v
}

func TestWindowedCausalAttentionBackwardFiniteDifference(t *testing.T) {
	const seq, heads, kvHeads, headDim, window = 5, 2, 1, 3, 3
	rng := rand.New(rand.NewSource(7))
	q := e4bRandVec(rng, seq*heads*headDim, 0, 0.7)
	k := e4bRandVec(rng, seq*kvHeads*headDim, 0, 0.7)
	v := e4bRandVec(rng, seq*kvHeads*headDim, 0, 0.7)
	dOut := e4bRandVec(rng, seq*heads*headDim, 0, 1)

	loss := func() float64 {
		out := make([]float32, seq*heads*headDim)
		WindowedCausalAttention(out, q, k, v, seq, heads, kvHeads, headDim, window)
		var s float64
		for i := range out {
			s += float64(out[i]) * float64(dOut[i])
		}
		return s
	}

	dq := make([]float32, len(q))
	dk := make([]float32, len(k))
	dv := make([]float32, len(v))
	WindowedCausalAttentionBackward(dq, dk, dv, q, k, v, dOut, seq, heads, kvHeads, headDim, window)

	tol := fdTol(fdStep, seq*headDim)
	check := func(name string, vec, grad []float32) {
		t.Helper()
		for i := range vec {
			orig := vec[i]
			vec[i] = orig + fdStep
			lp := loss()
			vec[i] = orig - fdStep
			lm := loss()
			vec[i] = orig
			fd := (lp - lm) / (2 * fdStep)
			if math.Abs(fd-float64(grad[i])) > tol*(1+math.Abs(fd)) {
				t.Fatalf("%s[%d]: fd %.6g vs analytic %.6g (tol %.1e)", name, i, fd, grad[i], tol)
			}
		}
	}
	check("dq", q, dq)
	check("dk", k, dk)
	check("dv", v, dv)
}

// Windowed backward with window<=0 must equal the existing full-causal VJP.
func TestWindowedCausalAttentionBackwardReducesToFullCausal(t *testing.T) {
	const seq, heads, kvHeads, headDim = 4, 4, 2, 3
	rng := rand.New(rand.NewSource(13))
	q := e4bRandVec(rng, seq*heads*headDim, 0, 0.8)
	k := e4bRandVec(rng, seq*kvHeads*headDim, 0, 0.8)
	v := e4bRandVec(rng, seq*kvHeads*headDim, 0, 0.8)
	dOut := e4bRandVec(rng, seq*heads*headDim, 0, 1)

	dqW := make([]float32, len(q))
	dkW := make([]float32, len(k))
	dvW := make([]float32, len(v))
	WindowedCausalAttentionBackward(dqW, dkW, dvW, q, k, v, dOut, seq, heads, kvHeads, headDim, 0)

	dqF := make([]float32, len(q))
	dkF := make([]float32, len(k))
	dvF := make([]float32, len(v))
	CausalAttentionBackward(dqF, dkF, dvF, q, k, v, dOut, seq, heads, kvHeads, headDim)

	eq := func(name string, a, b []float32) {
		for i := range a {
			if math.Abs(float64(a[i]-b[i])) > 1e-6 {
				t.Fatalf("%s[%d]: windowed %.7g vs full-causal %.7g", name, i, a[i], b[i])
			}
		}
	}
	eq("dq", dqW, dqF)
	eq("dk", dkW, dkF)
	eq("dv", dvW, dvF)
}

func TestGELUTanhBackwardFiniteDifference(t *testing.T) {
	x := []float32{-3, -1.5, -0.5, 0, 0.25, 0.9, 2, 4}
	dy := []float32{1, -2, 0.5, 3, -1, 0.7, 1.3, -0.4}
	dx := make([]float32, len(x))
	GELUTanhBackward(dx, x, dy)
	tol := fdTol(fdStep, 1)
	for i := range x {
		lp := GELUTanh(float64(x[i])+fdStep) * float64(dy[i])
		lm := GELUTanh(float64(x[i])-fdStep) * float64(dy[i])
		fd := (lp - lm) / (2 * fdStep)
		if math.Abs(fd-float64(dx[i])) > tol*(1+math.Abs(fd)) {
			t.Fatalf("gelu-tanh dx[%d]: fd %.6g vs analytic %.6g", i, fd, dx[i])
		}
	}
	// Aliasing contract: dst may be dy.
	aliased := append([]float32(nil), dy...)
	GELUTanhBackward(aliased, x, aliased)
	for i := range aliased {
		if aliased[i] != dx[i] {
			t.Fatalf("aliased gelu-tanh backward diverges at %d", i)
		}
	}
}

func TestSoftcapBackwardFiniteDifference(t *testing.T) {
	const cap = 30.0
	z := []float32{-90, -40, -10, -1, 0, 2, 15, 55}
	dOut := []float32{0.5, -1, 2, 0.3, -0.7, 1.1, -0.9, 0.6}
	dst := make([]float32, len(z))
	SoftcapBackward(dst, dOut, z, cap)
	fwd := func(v float64) float64 { return cap * math.Tanh(v/cap) }
	tol := fdTol(fdStep, 1)
	for i := range z {
		lp := fwd(float64(z[i])+fdStep) * float64(dOut[i])
		lm := fwd(float64(z[i])-fdStep) * float64(dOut[i])
		fd := (lp - lm) / (2 * fdStep)
		if math.Abs(fd-float64(dst[i])) > tol*(1+math.Abs(fd)) {
			t.Fatalf("softcap dz[%d]: fd %.6g vs analytic %.6g", i, fd, dst[i])
		}
	}
}

func TestActivationSparsityBackwardFiniteDifference(t *testing.T) {
	const rows, width, stdMult = 2, 8, 0.5
	// Deterministic, well-separated rows so no element sits near its relu kink
	// (the cutoff couples the whole row, so a nearby kink would make FD noisy).
	gate := []float32{
		-2.0, -1.1, -0.4, 0.2, 0.9, 1.6, 2.3, 3.1,
		-3.0, -1.7, -0.6, 0.5, 1.2, 2.0, 2.9, 3.8,
	}
	dA := []float32{
		0.7, -1.2, 0.4, 1.5, -0.8, 0.9, -0.3, 1.1,
		-0.5, 0.6, 1.3, -1.0, 0.2, 0.8, -0.7, 0.4,
	}
	dGate := make([]float32, len(gate))
	ActivationSparsityBackward(dGate, gate, dA, rows, width, stdMult)

	loss := func() float64 {
		out := make([]float32, len(gate))
		SparseGateInto(out, gate, rows, width, stdMult)
		var s float64
		for i := range out {
			s += float64(out[i]) * float64(dA[i])
		}
		return s
	}
	tol := fdTol(fdStep, width)
	for i := range gate {
		orig := gate[i]
		gate[i] = orig + fdStep
		lp := loss()
		gate[i] = orig - fdStep
		lm := loss()
		gate[i] = orig
		fd := (lp - lm) / (2 * fdStep)
		if math.Abs(fd-float64(dGate[i])) > tol*(1+math.Abs(fd)) {
			t.Fatalf("sparsity dGate[%d]: fd %.6g vs analytic %.6g (tol %.1e)", i, fd, dGate[i], tol)
		}
	}
}

func TestEmbedInputScaleBackwardFiniteDifference(t *testing.T) {
	const d, vocab, scale = 4, 5, 3.0
	rng := rand.New(rand.NewSource(21))
	embed := e4bRandVec(rng, vocab*d, 0, 0.5)
	tokens := []int{1, 3, 1, 4} // token 1 repeats -> accumulation
	dHidden := e4bRandVec(rng, len(tokens)*d, 0, 1)

	dEmbed := make([]float32, vocab*d)
	EmbedInputScaleBackward(dEmbed, dHidden, tokens, d, scale)

	loss := func() float64 {
		var s float64
		for t, tok := range tokens {
			for i := 0; i < d; i++ {
				h := scale * float64(embed[tok*d+i])
				s += h * float64(dHidden[t*d+i])
			}
		}
		return s
	}
	tol := fdTol(fdStep, len(tokens))
	for j := range embed {
		orig := embed[j]
		embed[j] = orig + fdStep
		lp := loss()
		embed[j] = orig - fdStep
		lm := loss()
		embed[j] = orig
		fd := (lp - lm) / (2 * fdStep)
		if math.Abs(fd-float64(dEmbed[j])) > tol*(1+math.Abs(fd)) {
			t.Fatalf("embed-scale dEmbed[%d]: fd %.6g vs analytic %.6g", j, fd, dEmbed[j])
		}
	}
}
