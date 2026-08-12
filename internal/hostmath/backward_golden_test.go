package hostmath

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"overgo/internal/testutil"
)

// readFixture loads a torch-produced grad golden; absent skips loudly.
func readFixture(t *testing.T, name string, out any) {
	t.Helper()
	path := testutil.FixturePath(t, name)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("UNAVAILABLE: %s absent; backward parity NOT verified", name)
	}
	if err := json.Unmarshal(raw, out); err != nil {
		t.Fatal(err)
	}
}

// Tolerance rationale: single-component f64-host vs the reference's f64
// autograd, differing only in f32 storage rounding.
const componentTol = 1e-4

func checkClose(t *testing.T, name string, got []float32, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if d := math.Abs(float64(got[i]) - want[i]); d > componentTol {
			t.Fatalf("%s[%d]: |%g - %g| = %g > %g", name, i, got[i], want[i], d, componentTol)
		}
	}
}

func f64To32(v []float64) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(x)
	}
	return out
}

func TestRMSNormBackwardMatchesGolden(t *testing.T) {
	var g struct {
		D            int       `json:"d"`
		X, Scale, Dy []float64 `json:"-"`
		XRaw         []float64 `json:"x"`
		ScaleRaw     []float64 `json:"scale"`
		DyRaw        []float64 `json:"dy"`
		GradX        []float64 `json:"grad_x"`
		GradScale    []float64 `json:"grad_scale"`
	}
	readFixture(t, "timesfm_rmsnorm_grad_golden.json", &g)
	dx := make([]float32, g.D)
	dscale := make([]float32, g.D)
	// The reference eps for this fixture family is the artifact's rms eps.
	RMSNormBackward(dx, dscale, f64To32(g.XRaw), f64To32(g.ScaleRaw), f64To32(g.DyRaw), 1, g.D, 1e-06, false)
	checkClose(t, "grad_x", dx, g.GradX)
	checkClose(t, "grad_scale", dscale, g.GradScale)
}

func TestRotaryHalfBackwardMatchesGolden(t *testing.T) {
	var g struct {
		HeadDim int       `json:"head_dim"`
		Pos     int       `json:"pos"`
		V       []float64 `json:"v"`
		Dout    []float64 `json:"dout"`
		GradV   []float64 `json:"grad_v"`
	}
	readFixture(t, "timesfm_rope_grad_golden.json", &g)
	dx := f64To32(g.Dout)
	RotaryHalfBackward(dx, RopeInvFreq(10000, g.HeadDim), g.Pos)
	checkClose(t, "grad_v", dx, g.GradV)
}

func TestCausalAttentionBackwardMatchesGolden(t *testing.T) {
	var g struct {
		N     int       `json:"n"`
		HD    int       `json:"hd"`
		Q     []float64 `json:"q"`
		K     []float64 `json:"k"`
		V     []float64 `json:"v"`
		Dout  []float64 `json:"dout"`
		GradQ []float64 `json:"grad_q"`
		GradK []float64 `json:"grad_k"`
		GradV []float64 `json:"grad_v"`
	}
	readFixture(t, "timesfm_attncore_grad_golden.json", &g)
	q, k, v := f64To32(g.Q), f64To32(g.K), f64To32(g.V)
	dq := make([]float32, len(q))
	dk := make([]float32, len(k))
	dv := make([]float32, len(v))
	CausalAttentionBackward(dq, dk, dv, q, k, v, f64To32(g.Dout), g.N, 1, 1, g.HD)
	checkClose(t, "grad_q", dq, g.GradQ)
	checkClose(t, "grad_k", dk, g.GradK)
	checkClose(t, "grad_v", dv, g.GradV)
}

// TestCausalAttentionBackwardForwardConsistency: finite-difference spot
// check tying backward to the forward it claims to differentiate.
func TestCausalAttentionBackwardForwardConsistency(t *testing.T) {
	seq, heads, hd := 3, 2, 4
	n := seq * heads * hd
	q := make([]float32, n)
	k := make([]float32, n)
	v := make([]float32, n)
	for i := 0; i < n; i++ {
		q[i] = float32(math.Sin(float64(i) * 0.7))
		k[i] = float32(math.Cos(float64(i) * 0.3))
		v[i] = float32(math.Sin(float64(i)*0.5 + 1))
	}
	dOut := make([]float32, n)
	for i := range dOut {
		dOut[i] = float32(math.Cos(float64(i) * 0.9))
	}
	dq := make([]float32, n)
	dk := make([]float32, n)
	dv := make([]float32, n)
	CausalAttentionBackward(dq, dk, dv, q, k, v, dOut, seq, heads, heads, hd)
	loss := func(q []float32) float64 {
		out := make([]float32, n)
		CausalAttention(out, q, k, v, seq, heads, heads, hd)
		var sum float64
		for i := range out {
			sum += float64(out[i]) * float64(dOut[i])
		}
		return sum
	}
	const h = 1e-3
	for _, index := range []int{0, 7, n - 1} {
		bumped := append([]float32(nil), q...)
		bumped[index] += h
		numeric := (loss(bumped) - loss(q)) / h
		if d := math.Abs(numeric - float64(dq[index])); d > 5e-3 {
			t.Fatalf("dq[%d] analytic %g vs numeric %g (|diff| %g)", index, dq[index], numeric, d)
		}
	}
}
