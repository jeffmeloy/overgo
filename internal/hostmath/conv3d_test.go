package hostmath

import (
	"math"
	"testing"

	"overgo/internal/testutil"
)

// TestCausalConv3DMatchesWanTorchFixture: values from the reference repo's
// torch-fixture oracle (adaptive causal_video_codec_test.go).
func TestCausalConv3DMatchesWanTorchFixture(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	w := make([]float32, 27)
	for i := range w {
		w[i] = float32(i+1) / 10
	}
	shape := Conv3DShape{
		CIn: 1, COut: 1, InT: 2, InH: 2, InW: 2,
		KT: 3, KH: 3, KW: 3,
		PadT: 1, PadH: 1, PadW: 1,
		StrideT: 1, StrideH: 1, StrideW: 1,
	}
	got := make([]float32, 8)
	if err := CausalConv3DInto(got, x, nil, w, []float32{0.25}, 0, shape); err != nil {
		t.Fatal(err)
	}
	want := []float32{25.950000762939453, 24.950000762939453, 22.950000762939453, 21.950000762939453, 82.6500015258789, 79.05000305175781, 71.8499984741211, 68.25}
	testutil.RequireSliceClose(t, "causal conv", got, want, 1e-5)
}

// TestCausalConv3DWithCacheUsesPriorFrames: a 2-frame cache fills the whole
// left pad, so the 3-tap temporal kernel sees [cache0, cache1, x].
func TestCausalConv3DWithCacheUsesPriorFrames(t *testing.T) {
	shape := Conv3DShape{
		CIn: 1, COut: 1, InT: 1, InH: 1, InW: 1,
		KT: 3, KH: 1, KW: 1, PadT: 1, PadH: 0, PadW: 0,
		StrideT: 1, StrideH: 1, StrideW: 1,
	}
	got := make([]float32, 1)
	if err := CausalConv3DInto(got, []float32{3}, []float32{1, 2}, []float32{1, 10, 100}, nil, 2, shape); err != nil {
		t.Fatal(err)
	}
	if got[0] != 321 {
		t.Fatalf("cached causal conv = %v, want [321]", got)
	}
	// One cached frame replaces one of the two pad frames: [0, cache, x].
	if err := CausalConv3DInto(got, []float32{3}, []float32{2}, []float32{1, 10, 100}, nil, 1, shape); err != nil {
		t.Fatal(err)
	}
	if got[0] != 320 {
		t.Fatalf("partially cached causal conv = %v, want [320]", got)
	}
}

func TestConv3DShapeTemporalUpsampleConvDims(t *testing.T) {
	tOut, hOut, wOut, err := (Conv3DShape{
		CIn: 384, COut: 768, InT: 1, InH: 2, InW: 3,
		KT: 3, KH: 1, KW: 1,
		PadT: 1, PadH: 0, PadW: 0,
		StrideT: 1, StrideH: 1, StrideW: 1,
	}).OutputDims()
	if err != nil {
		t.Fatal(err)
	}
	if tOut != 1 || hOut != 2 || wOut != 3 {
		t.Fatalf("dims=(%d,%d,%d)", tOut, hOut, wOut)
	}
}

func TestCausalConv3DRejectsMalformedInputs(t *testing.T) {
	shape := Conv3DShape{
		CIn: 1, COut: 1, InT: 1, InH: 1, InW: 1,
		KT: 3, KH: 1, KW: 1, PadT: 1, PadH: 0, PadW: 0,
		StrideT: 1, StrideH: 1, StrideW: 1,
	}
	out, x, w := make([]float32, 1), []float32{1}, []float32{1, 2, 3}
	for name, run := range map[string]func() error{
		"negative cache":  func() error { return CausalConv3DInto(out, x, nil, w, nil, -1, shape) },
		"oversized cache": func() error { return CausalConv3DInto(out, x, []float32{1, 2, 3}, w, nil, 3, shape) },
		"cache length":    func() error { return CausalConv3DInto(out, x, []float32{1}, w, nil, 2, shape) },
		"weight length":   func() error { return CausalConv3DInto(out, x, nil, w[:2], nil, 0, shape) },
		"bias length":     func() error { return CausalConv3DInto(out, x, nil, w, []float32{1, 2}, 0, shape) },
		"zero stride": func() error {
			bad := shape
			bad.StrideT = 0
			return CausalConv3DInto(out, x, nil, w, nil, 0, bad)
		},
	} {
		if run() == nil {
			t.Fatalf("%s accepted", name)
		}
	}
}

// TestResizeConv2DMatchesWanTorchFixture: values from the reference repo's
// torch-fixture oracle (adaptive causal_video_codec_test.go).
func TestResizeConv2DMatchesWanTorchFixture(t *testing.T) {
	const cIn, cOut, frames, height, width = 4, 2, 2, 2, 3
	x := make([]float32, cIn*frames*height*width)
	for i := range x {
		x[i] = float32(i+1)/10 - 0.8
	}
	next := 1
	take := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32(next+i)/30 - 0.4
		}
		next += n
		return out
	}
	weight, bias := take(cOut*cIn*3*3), take(cOut)
	got := make([]float32, cOut*frames*(2*height)*(2*width))
	if err := ResizeConv2DInto(got, x, weight, bias, cIn, cOut, frames, height, width); err != nil {
		t.Fatal(err)
	}
	want := []float32{
		14.220001220703125, 20.113332748413086, 20.32666778564453, 20.753334045410156,
		20.96666717529297, 14.433334350585938, 19.793333053588867, 28.25333023071289,
		28.513334274291992, 29.033334732055664, 29.2933349609375, 19.793331146240234,
		20.35333251953125, 29.03333282470703, 29.2933349609375, 29.813335418701172,
	}
	testutil.RequireSliceClose(t, "resize conv", got[:len(want)], want, 9e-6)
	var sum float64
	for _, v := range got {
		sum += float64(v)
	}
	if diff := math.Abs(sum - 4807.14697265625); diff > 3e-4 {
		t.Fatalf("sum diff=%g sum=%g", diff, sum)
	}
	if err := ResizeConv2DInto(got, x, weight[:len(weight)-1], bias, cIn, cOut, frames, height, width); err == nil {
		t.Fatal("inconsistent resize kernel was accepted")
	}
}
