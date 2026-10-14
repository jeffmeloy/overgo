//go:build windows

package devicemath

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
)

func hostSiLU(x float64) float64 { return x / (1 + math.Exp(-x)) }

// TestSiLUGateForwardMatchesHost verifies device SiLUGateForward returns
// a = silu(gate) and hMLP = a*up matching the fp64 reference.
func TestSiLUGateForwardMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const n = 256
	rng := rand.New(rand.NewSource(51))
	gate := randSlice(rng, n)
	up := randSlice(rng, n)

	a, hMLP, err := SiLUGateForward(worker, gate, up)
	if err != nil {
		t.Fatal(err)
	}
	var maxA, maxH float64
	for i := range n {
		wantA := hostSiLU(float64(gate[i]))
		wantH := wantA * float64(up[i])
		if d := math.Abs(float64(a[i]) - wantA); d > maxA {
			maxA = d
		}
		if d := math.Abs(float64(hMLP[i]) - wantH); d > maxH {
			maxH = d
		}
	}
	t.Logf("silu-gate forward: max |a-host| %.3e |hMLP-host| %.3e", maxA, maxH)
	const tolerance = 1e-5
	if maxA > tolerance || maxH > tolerance {
		t.Fatalf("silu-gate forward a %.3e hMLP %.3e > %.1e", maxA, maxH, tolerance)
	}
}

// TestSiLUBackwardGradCheck verifies device SiLUBackward against a central
// finite difference of the host loss L = sum(dy ⊙ silu(x)). SiLU is nonlinear,
// so the FD carries O(eps^2) truncation; the tolerance covers that plus fp32.
func TestSiLUBackwardGradCheck(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const n = 64
	rng := rand.New(rand.NewSource(5))
	x := randSlice(rng, n)
	dy := randSlice(rng, n)

	dx, err := SiLUBackward(worker, x, dy)
	if err != nil {
		t.Fatal(err)
	}

	loss := func() float64 {
		var l float64
		for i := range x {
			l += float64(dy[i]) * hostSiLU(float64(x[i]))
		}
		return l
	}
	const eps = 1e-3
	var maxDiff float64
	for i := range x {
		orig := x[i]
		x[i] = orig + float32(eps)
		lp := loss()
		x[i] = orig - float32(eps)
		lm := loss()
		x[i] = orig
		num := (lp - lm) / (2 * eps)
		if d := math.Abs(num - float64(dx[i])); d > maxDiff {
			maxDiff = d
		}
	}
	t.Logf("grad-check: max |dx-fd| %.3e", maxDiff)
	const tolerance = 1e-3
	if maxDiff > tolerance {
		t.Fatalf("dx vs fd %.3e > %.1e", maxDiff, tolerance)
	}
}

// SiLUBackward computes grad_input = grad_output * silu'(x) elementwise on the
// GPU, where silu(x) = x*sigmoid(x) and silu'(x) = s*(1 + x*(1-s)), s =
// sigmoid(x). Uses the ops_f32 silu_backward_f32 kernel. This is the SiLU/SwiGLU
// activation half of the gated-MLP backward.
func SiLUBackward(worker *device.Worker, x, gradOutput []float32) ([]float32, error) {
	n := len(x)
	if n == 0 || len(gradOutput) != n {
		return nil, fmt.Errorf("SiLUBackward: length mismatch (x=%d dy=%d)", n, len(gradOutput))
	}
	if uint64(n) > math.MaxUint32 {
		return nil, fmt.Errorf("SiLUBackward: input too large for a 32-bit element count")
	}
	gradInput := make([]float32, n)
	err := withCUDA(worker, func(scope *cudaScope) error {
		fn, err := scope.function("silu_backward_f32")
		if err != nil {
			return err
		}
		dyPtr, err := scope.upload(gradOutput)
		if err != nil {
			return err
		}
		xPtr, err := scope.upload(x)
		if err != nil {
			return err
		}
		dxPtr, err := scope.alloc(n)
		if err != nil {
			return err
		}

		count := uint32(n)
		if err := scope.launch1D(fn, count,
			unsafe.Pointer(&dyPtr), unsafe.Pointer(&xPtr), unsafe.Pointer(&dxPtr), unsafe.Pointer(&count),
		); err != nil {
			return err
		}
		return scope.finish(cudaDownload{gradInput, dxPtr})
	})
	if err != nil {
		return nil, err
	}
	return gradInput, nil
}
