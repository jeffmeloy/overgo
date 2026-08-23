package latentvideo

import (
	"math"
	"testing"

	"overgo/internal/hostmath"
	"overgo/internal/media"
	"overgo/internal/pytorchzip"
)

// Torch-fixture oracle values ported from the reference repo
// (adaptive causal_video_codec_test.go): same deterministic weight ramps,
// same want tensors, same tolerances.

func vaeFixtureInputDims(channels, t, h, w int, offset float32) []float32 {
	x := make([]float32, channels*t*h*w)
	for i := range x {
		x[i] = float32(i+1)/10 + offset
	}
	return x
}

func vaeFixtureTake(next *int, scale float32, shift float32) func(int) []float32 {
	return func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32(*next+i)/scale - shift
		}
		*next += n
		return out
	}
}

func requireVAEClose(t *testing.T, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) < len(want) {
		t.Fatalf("length %d want at least %d", len(got), len(want))
	}
	for i := range want {
		if diff := math.Abs(float64(got[i]) - float64(want[i])); diff > tolerance {
			t.Fatalf("element %d: got %v want %v (diff %g > %g)", i, got[i], want[i], diff, tolerance)
		}
	}
}

func vaeSum(values []float32) float64 {
	var sum float64
	for _, v := range values {
		sum += float64(v)
	}
	return sum
}

func TestChannelRMSNormMatchesTorchFixture(t *testing.T) {
	x := []float32{
		1, -2, 3, 4, 5, -6, 7, 8,
		-1, 2, -3, 4, -5, 6, -7, 8,
	}
	got := make([]float32, len(x))
	if err := hostmath.ChannelRMSNormF64Into(got, x, []float32{1.5, 0.5}, 2, 8); err != nil {
		t.Fatal(err)
	}
	want := []float32{
		1.4999998807907104, -1.4999998807907104, 1.5, 1.4999998807907104,
		1.4999998807907104, -1.5, 1.4999998807907104, 1.4999998807907104,
		-0.4999999701976776, 0.4999999701976776, -0.5, 0.4999999701976776,
		-0.4999999701976776, 0.5, -0.4999999701976776, 0.4999999701976776,
	}
	requireVAEClose(t, got, want, 6e-7)
}

func TestVAESiLUMatchesTorchFixture(t *testing.T) {
	got := []float32{1, -2, 3, 4, 5, -6, 7, 8, -1, 2, -3, 4, -5, 6, -7, 8}
	hostmath.SiLUInPlace(got)
	want := []float32{
		0.7310585975646973, -0.23840583860874176, 2.857722520828247, 3.9280550479888916,
		4.966535568237305, -0.014835738576948643, 6.993622779846191, 7.997317314147949,
		-0.2689414322376251, 1.7615940570831299, -0.14227761328220367, 3.9280550479888916,
		-0.033464252948760986, 5.985164642333984, -0.006377358455210924, 7.997317314147949,
	}
	requireVAEClose(t, got, want, 6e-7)
}

// residualFixtureOp: the deterministic ramp weights the torch fixture used.
func residualFixtureOp(cIn, cOut int) (media.CodecOperation[[]pytorchzip.TensorBinding], vaeLoadedWeights, *vaeOpState) {
	next := 1
	take := vaeFixtureTake(&next, 50, 0.2)
	values := [][]float32{
		take(cIn),                     // norm0 gamma
		take(cOut * cIn * 3 * 3 * 3),  // conv0 w
		take(cOut),                    // conv0 b
		take(cOut),                    // norm1 gamma
		take(cOut * cOut * 3 * 3 * 3), // conv1 w
		take(cOut),                    // conv1 b
	}
	if cIn != cOut {
		values = append(values, take(cOut*cIn), take(cOut))
	}
	return fixtureCodecOperation(media.CodecResidual, cIn, cOut), values, &vaeOpState{}
}

func fixtureCodecOperation(kind media.CodecOperator, cIn, cOut int) media.CodecOperation[[]pytorchzip.TensorBinding] {
	return media.CodecOperation[[]pytorchzip.TensorBinding]{
		Operator: kind, Name: "fixture", InputChannels: cIn, OutputChannels: cOut, BindingCount: 1,
	}
}

func TestVAEResidualBlockMatchesTorchIdentityShortcutFixture(t *testing.T) {
	x := vaeFixtureInputDims(2, 2, 2, 2, -0.5)
	op, values, state := residualFixtureOp(2, 2)
	got, frames, h, w, err := runVAEOp(op, values, state, 0, x, 2, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if frames != 2 || h != 2 || w != 2 {
		t.Fatalf("shape=(%d,%d,%d)", frames, h, w)
	}
	want := []float32{
		45.58547592163086, 45.39165115356445, 44.903995513916016, 44.710174560546875,
		82.2893295288086, 81.81338500976562, 80.76150512695312, 80.28557586669922,
		62.27212905883789, 62.07829666137695,
	}
	requireVAEClose(t, got, want, 2e-4)
	if diff := math.Abs(vaeSum(got) - 1205.91162109375); diff > 6e-4 {
		t.Fatalf("sum diff=%g sum=%g", diff, vaeSum(got))
	}
}

func TestVAEResidualBlockMatchesTorchConvShortcutFixture(t *testing.T) {
	x := vaeFixtureInputDims(2, 2, 2, 2, -0.5)
	op, values, state := residualFixtureOp(2, 3)
	got, _, _, _, err := runVAEOp(op, values, state, 0, x, 2, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{
		170.98739624023438, 171.88536071777344, 172.0513153076172, 172.94927978515625,
		315.05364990234375, 315.25006103515625, 314.01287841796875, 314.2093200683594,
		230.32135009765625, 231.22735595703125,
	}
	requireVAEClose(t, got, want, 6e-4)
	if diff := math.Abs(vaeSum(got) - 7945.80517578125); diff > 3e-3 {
		t.Fatalf("sum diff=%g sum=%g", diff, vaeSum(got))
	}
}

func TestVAESpatialAttentionMatchesTorchFixture(t *testing.T) {
	x := vaeFixtureInputDims(3, 2, 2, 2, -0.7)
	next := 1
	take := vaeFixtureTake(&next, 40, 0.3)
	op := fixtureCodecOperation(media.CodecAttention, 3, 3)
	values := vaeLoadedWeights{
		take(3),         // norm gamma
		take(3 * 3 * 3), // to_qkv w
		take(3 * 3),     // to_qkv b
		take(3 * 3),     // proj w
		take(3),         // proj b
	}
	got, _, _, _, err := runVAEOp(op, values, &vaeOpState{}, 0, x, 2, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{
		1.4898173809051514, 1.5898265838623047, 1.689834475517273, 1.7898414134979248,
		1.7219865322113037, 1.8219873905181885, 1.9219878911972046, 2.0219883918762207,
		2.4352991580963135, 2.535309076309204, 2.6353180408477783, 2.735325574874878,
		2.650195598602295, 2.7501964569091797, 2.8501970767974854, 2.95019793510437,
	}
	requireVAEClose(t, got, want, 8e-6)
	if diff := math.Abs(vaeSum(got) - 64.62611389160156); diff > 3e-5 {
		t.Fatalf("sum diff=%g sum=%g", diff, vaeSum(got))
	}
}

// TestVAETemporalCacheSemantics: reference temporal_cache_update modes.
func TestVAETemporalCacheSemantics(t *testing.T) {
	const c, spatial = 1, 1
	// Cold single-frame chunk: cache is that frame.
	cache := nextTemporalCache(vaeTemporalCache{}, []float32{7}, c, 1, spatial, false)
	if cache.frames != 1 || cache.data[0] != 7 {
		t.Fatalf("cold cache=%+v", cache)
	}
	// Next single-frame chunk joins the prior cache's last frame.
	cache = nextTemporalCache(cache, []float32{9}, c, 1, spatial, false)
	if cache.frames != 2 || cache.data[0] != 7 || cache.data[1] != 9 {
		t.Fatalf("joined cache=%+v", cache)
	}
	// Multi-frame chunk keeps its own last two frames.
	cache = nextTemporalCache(cache, []float32{1, 2, 3}, c, 3, spatial, false)
	if cache.frames != 2 || cache.data[0] != 2 || cache.data[1] != 3 {
		t.Fatalf("tail cache=%+v", cache)
	}
	// Replicate prefix ('Rep'): zero frame then the chunk.
	cache = nextTemporalCache(vaeTemporalCache{initialized: true, rep: true}, []float32{5}, c, 1, spatial, true)
	if cache.frames != 2 || cache.data[0] != 0 || cache.data[1] != 5 {
		t.Fatalf("rep cache=%+v", cache)
	}
}

func TestVAETimeInterleaveMatchesReferenceLayout(t *testing.T) {
	// convolved [2c=2][frames=2][spatial=1]: channel half selects parity.
	convolved := []float32{10, 11, 20, 21}
	out := make([]float32, 4)
	if err := timeInterleaveInto(out, convolved, 1, 2, 1); err != nil {
		t.Fatal(err)
	}
	want := []float32{10, 20, 11, 21}
	requireVAEClose(t, out, want, 0)
}

// TestVAEUpsample3DFirstChunkSkipsTimeConv: chunk 0 must resample without
// temporal doubling and arm the 'Rep' state; chunk 1 must double frames.
func TestVAEUpsample3DFirstChunkSkipsTimeConv(t *testing.T) {
	const c = 1
	timeW := make([]float32, 2*c*c*3)
	for i := range timeW {
		timeW[i] = float32(i+1) / 10
	}
	resampleW := make([]float32, c*c*3*3)
	resampleW[4] = 1 // identity tap: pure nearest upsample
	op := fixtureCodecOperation(media.CodecUpsampleSpatiotemporal, c, c)
	values := vaeLoadedWeights{timeW, make([]float32, 2*c), resampleW, make([]float32, c)}
	state := &vaeOpState{}
	out, frames, h, w, err := runVAEOp(op, values, state, 0, []float32{3}, 1, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if frames != 1 || h != 2 || w != 2 || !state.cache0.rep {
		t.Fatalf("chunk0 frames=%d h=%d w=%d rep=%t", frames, h, w, state.cache0.rep)
	}
	requireVAEClose(t, out, []float32{3, 3, 3, 3}, 0)
	out, frames, _, _, err = runVAEOp(op, values, state, 1, []float32{5}, 1, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if frames != 2 || state.cache0.rep || state.cache0.frames != 2 || state.cache0.data[0] != 0 || state.cache0.data[1] != 5 {
		t.Fatalf("chunk1 frames=%d cache=%+v", frames, state.cache0)
	}
	// Cacheless causal time conv on [0,0,5]: even half w[2]*5=1.5, odd half w[5]*5=3.
	requireVAEClose(t, out[:8], []float32{1.5, 1.5, 1.5, 1.5, 3, 3, 3, 3}, 1e-6)
}

func TestVAEDecodeRejectsInvalidInputs(t *testing.T) {
	plan := VAEDecoderPlan{vaePlanCore: vaePlanCore{
		Stride: [3]int{4, 8, 8},
		CodecProgram: media.CodecProgram[[]pytorchzip.TensorBinding]{
			Operations: []media.CodecOperation[[]pytorchzip.TensorBinding]{fixtureCodecOperation(media.CodecPointwise, 2, 2)},
		},
	}, ZDim: 2, OutputChannels: 3}
	stats := VAELatentStats{Mean: []float32{0, 0}, Std: []float32{1, 1}}
	sink := func(int, []float32, int, int) error { return nil }
	if _, err := DecodeLatentVideo("missing.pth", plan, stats, make([]float32, 2), 1, 1, 1, nil); err == nil {
		t.Fatal("nil sink accepted")
	}
	if _, err := DecodeLatentVideo("missing.pth", plan, VAELatentStats{Mean: []float32{0}, Std: []float32{1}}, make([]float32, 2), 1, 1, 1, sink); err == nil {
		t.Fatal("stats arity mismatch accepted")
	}
	if _, err := DecodeLatentVideo("missing.pth", plan, stats, make([]float32, 3), 1, 1, 1, sink); err == nil {
		t.Fatal("latent length mismatch accepted")
	}
	bad := stats
	bad.Std = []float32{1, 0}
	if _, err := DecodeLatentVideo("missing.pth", plan, bad, make([]float32, 2), 1, 1, 1, sink); err == nil {
		t.Fatal("zero std accepted")
	}
}
