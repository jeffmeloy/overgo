package optimizer

import (
	"math/rand"
	"testing"

	"overgo/internal/tensor/dtype"
)

// TestBF16SGDConvergesNearF32 trains a toy least-squares problem three ways --
// F32 master SGD, BF16 master with stochastic rounding (BF16SGD), and BF16
// master with round-to-nearest -- and pins the design decision: stochastic
// rounding tracks F32 to a small BF16 noise floor with a 2-byte master, while
// round-to-nearest STALLS (sub-ULP updates vanish every step). Gradients are
// computed in F32; only the master storage/rounding differs.
func TestBF16SGDConvergesNearF32(t *testing.T) {
	const n, d, steps = 64, 8, 400
	const lr = 0.5
	rng := rand.New(rand.NewSource(1))
	x := make([]float32, n*d)
	wTrue := make([]float32, d)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	for i := range wTrue {
		wTrue[i] = float32(rng.NormFloat64())
	}
	y := make([]float32, n)
	for i := 0; i < n; i++ {
		var s float64
		for j := 0; j < d; j++ {
			s += float64(x[i*d+j]) * float64(wTrue[j])
		}
		y[i] = float32(s)
	}
	// full-batch F32 gradient of 0.5/n * sum((Xw-y)^2) and its loss.
	gradLoss := func(w []float32) (grad []float32, loss float64) {
		grad = make([]float32, d)
		resid := make([]float64, n)
		for i := 0; i < n; i++ {
			var s float64
			for j := 0; j < d; j++ {
				s += float64(x[i*d+j]) * float64(w[j])
			}
			r := s - float64(y[i])
			resid[i] = r
			loss += 0.5 * r * r / float64(n)
		}
		for j := 0; j < d; j++ {
			var g float64
			for i := 0; i < n; i++ {
				g += resid[i] * float64(x[i*d+j])
			}
			grad[j] = float32(g / float64(n))
		}
		return grad, loss
	}

	w0 := make([]float32, d)
	for i := range w0 {
		w0[i] = float32(rng.NormFloat64())
	}

	// F32 master SGD.
	wf := append([]float32(nil), w0...)
	var f32First, f32Last float64
	for s := 0; s < steps; s++ {
		g, l := gradLoss(wf)
		if s == 0 {
			f32First = l
		}
		f32Last = l
		for j := range wf {
			wf[j] -= float32(lr * float64(g[j]))
		}
	}

	// BF16 master, round-to-nearest (the stalling baseline).
	wr := make([]float32, d)
	for i := range wr {
		wr[i] = dtype.RoundBF16(w0[i])
	}
	var rtnLast float64
	for s := 0; s < steps; s++ {
		g, l := gradLoss(wr)
		rtnLast = l
		for j := range wr {
			wr[j] = dtype.RoundBF16(wr[j] - float32(lr*float64(g[j])))
		}
	}

	// BF16 master, stochastic rounding (BF16SGD).
	opt := NewBF16SGD(w0, 42)
	buf := make([]float32, d)
	var stoLast float64
	for s := 0; s < steps; s++ {
		opt.WeightsInto(buf)
		g, l := gradLoss(buf)
		stoLast = l
		opt.Step(g, lr)
	}

	t.Logf("loss: init %.4e | f32 %.4e | bf16-stochastic %.4e | bf16-rtn %.4e", f32First, f32Last, stoLast, rtnLast)
	if len(opt.Master()) != d {
		t.Fatalf("master must hold %d BF16 params, got %d", d, len(opt.Master()))
	}
	if f32Last > 1e-6 {
		t.Fatalf("f32 baseline did not converge: %.3e", f32Last)
	}
	// stochastic BF16 must reach a small floor -- far below init, within reach of
	// the BF16 precision floor (not F32's, but decisively converged).
	if stoLast > 1e-3 {
		t.Fatalf("bf16-stochastic did not converge: %.3e (init %.3e)", stoLast, f32First)
	}
	// and it must beat round-to-nearest, which stalls above the stochastic floor.
	if stoLast >= rtnLast {
		t.Fatalf("stochastic (%.3e) must beat round-to-nearest (%.3e)", stoLast, rtnLast)
	}
}
