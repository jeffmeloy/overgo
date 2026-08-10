package routedlm

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"overgo/internal/safetensors"
	"overgo/internal/tensor/dtype"
)

const senseNovaGenerationOracle = `C:\Users\jeffm\adaptive_new\fixtures\sensenova\generation_oracle_256.json`

// Reference formula: [cos half | sin half], angle_i = v*period^(-i/half).
func TestSinusoidalEmbeddingMatchesReference(t *testing.T) {
	got, err := SinusoidalEmbedding([]float64{0.375}, 4, 10000, 1)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{
		math.Cos(0.375), math.Cos(0.375 * math.Exp(-math.Log(10000)/2)),
		math.Sin(0.375), math.Sin(0.375 * math.Exp(-math.Log(10000)/2)),
	}
	for i, w := range want {
		if math.Abs(float64(got[i])-w) > 1e-7 {
			t.Fatalf("element %d = %g, want %g", i, got[i], w)
		}
	}
	multi, err := SinusoidalEmbedding([]float64{0, 1, 999}, 8, 10000, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(multi) != 24 {
		t.Fatalf("rows*dim = %d, want 24", len(multi))
	}
	// value 0: cos half all ones, sin half all zeros.
	for i := 0; i < 4; i++ {
		if multi[i] != 1 || multi[4+i] != 0 {
			t.Fatalf("zero-value row = %v", multi[:8])
		}
	}
	if _, err := SinusoidalEmbedding([]float64{1}, 7, 10000, 1); err == nil {
		t.Fatal("odd dimension must fail")
	}
	if _, err := SinusoidalEmbedding([]float64{math.Inf(1)}, 4, 10000, 1); err == nil {
		t.Fatal("non-finite value must fail")
	}
}

// Standard shifted schedule: t_i = 1 - shift*s/(1+(shift-1)s), s=1-i/steps.
func TestShiftedFlowTimeSchedule(t *testing.T) {
	got, err := ShiftedFlowTimeSchedule(2, 3)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{0, 0.25, 1}
	if len(got) != len(want) {
		t.Fatalf("schedule = %v, want %v", got, want)
	}
	for i, w := range want {
		if math.Abs(got[i]-w) > 1e-15 {
			t.Fatalf("knot %d = %g, want %g", i, got[i], w)
		}
	}
	uniform, err := ShiftedFlowTimeSchedule(2, 1)
	if err != nil {
		t.Fatal(err)
	}
	for i, w := range []float64{0, 0.5, 1} {
		if math.Abs(uniform[i]-w) > 1e-15 {
			t.Fatalf("shift=1 knot %d = %g, want %g", i, uniform[i], w)
		}
	}
	long, err := ShiftedFlowTimeSchedule(20, 3)
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i < len(long); i++ {
		if long[i] <= long[i-1] {
			t.Fatalf("schedule not strictly ascending at %d: %v", i, long)
		}
	}
	if long[0] != 0 || long[20] != 1 {
		t.Fatalf("endpoints = %g,%g", long[0], long[20])
	}
	if _, err := ShiftedFlowTimeSchedule(0, 3); err == nil {
		t.Fatal("zero steps must fail")
	}
	if _, err := ShiftedFlowTimeSchedule(2, 0); err == nil {
		t.Fatal("zero shift must fail")
	}
}

func testFlowPlan() FlowPlan {
	return FlowPlan{
		Hidden: 4096, VisionHidden: 1024, VisionChannels: 3, VisionPatch: 16,
		ImageMerge: 2, FrequencyDim: 256, FlowDim: 3072,
		VisionRopeTheta: 1e4, SinusoidalPeriod: 1e4,
		NoiseScaleMode: "resolution", NoiseScaleBase: 64,
		NoiseScale: 1, NoiseScaleMax: 8, TEps: 0.05,
	}
}

// scale = noise_scale*sqrt(tokens/base), capped; base 64 = the 256x256
// reference (token patch 32 -> 8x8 tokens).
func TestNoiseScaleDerivation(t *testing.T) {
	plan := testFlowPlan()
	cases := []struct {
		size int
		want float64
	}{
		{256, 1},  // tokens 64 = base
		{512, 2},  // tokens 256: sqrt(4)
		{1024, 4}, // tokens 1024: sqrt(16)
		{2048, 8}, // tokens 4096: sqrt(64) = cap boundary
		{4096, 8}, // sqrt(256)=16 -> capped
	}
	for _, c := range cases {
		shape, err := plan.ImagePlan(c.size, c.size)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(shape.NoiseScale-c.want) > 1e-12 {
			t.Fatalf("%dx%d noise scale = %g, want %g", c.size, c.size, shape.NoiseScale, c.want)
		}
	}
	sqrtPlan := plan
	sqrtPlan.NoiseScaleMode = "dynamic_sqrt"
	shape, err := sqrtPlan.ImagePlan(512, 512)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(shape.NoiseScale-math.Sqrt(2)) > 1e-12 {
		t.Fatalf("dynamic_sqrt 512 = %g, want sqrt(2)", shape.NoiseScale)
	}
	constPlan := plan
	constPlan.NoiseScaleMode = ""
	shape, err = constPlan.ImagePlan(512, 512)
	if err != nil {
		t.Fatal(err)
	}
	if shape.NoiseScale != 1 {
		t.Fatalf("constant-mode scale = %g, want 1", shape.NoiseScale)
	}
	if _, err := plan.ImagePlan(250, 256); err == nil {
		t.Fatal("misaligned width must fail")
	}
	if plan.NormalizedNoiseScale(1) != 0.125 {
		t.Fatalf("normalized scale = %g, want 0.125", plan.NormalizedNoiseScale(1))
	}
}

func bf16MatrixFromRows(rows [][]float32) BF16Matrix {
	out := BF16Matrix{In: len(rows[0]), Out: len(rows)}
	out.Data = make([]uint16, out.In*out.Out)
	for r, row := range rows {
		for c, v := range row {
			out.Data[r*out.In+c] = dtype.Float32ToBF16(v)
		}
	}
	return out
}

// Identity-weight embedder: out = bf16(silu(bf16(sinusoidal))).
func TestScalarConditionRowSmallCase(t *testing.T) {
	plan := FlowPlan{Hidden: 2, FrequencyDim: 2, SinusoidalPeriod: 1e4}
	identity := bf16MatrixFromRows([][]float32{{1, 0}, {0, 1}})
	w := FlowMLPWeights{W0: identity, B0: []float32{0, 0}, W2: identity, B2: []float32{0, 0}}
	got, err := ScalarConditionRow(w, 0.375, plan)
	if err != nil {
		t.Fatal(err)
	}
	for i, angle := range []float64{0.375, 0} { // dim=2: half=1 -> [cos v, sin v]
		freq := math.Cos(angle)
		if i == 1 {
			freq = math.Sin(0.375)
		}
		v := float64(dtype.RoundBF16(dtype.RoundBF16(float32(freq)))) // freq round + bias round
		want := dtype.RoundBF16(dtype.RoundBF16(float32(v / (1 + math.Exp(-v)))))
		if got[i] != want {
			t.Fatalf("element %d = %g, want %g", i, got[i], want)
		}
	}
	// condition row = bf16(t_emb + n_emb); broadcast add rounds the base.
	tw := FlowTerminalWeights{Timestep: w, NoiseScale: w}
	condition, err := FlowConditionRow(tw, plan, 0.375, 0.375)
	if err != nil {
		t.Fatal(err)
	}
	for i := range condition {
		want := dtype.RoundBF16(got[i] + got[i])
		if condition[i] != want {
			t.Fatalf("condition %d = %g, want %g", i, condition[i], want)
		}
	}
	hidden := []float32{0.1, 0.2, 0.3, 0.4}
	if err := AddConditionRows(hidden, condition); err != nil {
		t.Fatal(err)
	}
	if want := dtype.RoundBF16(dtype.RoundBF16(0.3) + condition[0]); hidden[2] != want {
		t.Fatalf("broadcast add = %g, want %g", hidden[2], want)
	}
	if err := AddConditionRows(hidden[:3], condition); err == nil {
		t.Fatal("non-divisible broadcast must fail")
	}
}

// v = bf16((head(x) - z) / max(1-t, t_eps)).
func TestFlowHeadVelocityClampsDenominator(t *testing.T) {
	plan := FlowPlan{Hidden: 2, FlowDim: 2, TEps: 0.05}
	identity := bf16MatrixFromRows([][]float32{{1, 0}, {0, 1}})
	w := FlowMLPWeights{W0: identity, B0: []float32{0, 0}, W2: identity, B2: []float32{0, 0}}
	hidden := []float32{2, -1}
	z := []float32{0.5, 0.5}
	geluOf := func(x float32) float32 {
		v := float64(x)
		return dtype.RoundBF16(float32(0.5 * v * (1 + math.Erf(v*0.7071067811865475))))
	}
	// t=0.99: denom = max(0.01, 0.05) = 0.05 (the clamp).
	got, err := FlowHeadVelocity(w, plan, hidden, z, 0.99)
	if err != nil {
		t.Fatal(err)
	}
	for i := range got {
		mid := dtype.RoundBF16(geluOf(dtype.RoundBF16(hidden[i]))) // linear0 round, gelu, linear2 round
		want := dtype.RoundBF16((mid - z[i]) / 0.05)
		if got[i] != want {
			t.Fatalf("velocity %d = %g, want %g", i, got[i], want)
		}
	}
	if _, err := FlowHeadVelocity(w, plan, hidden, z[:1], 0.5); err == nil {
		t.Fatal("short z must fail")
	}
	if _, err := FlowHeadVelocity(w, plan, hidden, z, 1); err == nil {
		t.Fatal("t=1 must fail")
	}
}

// Gather layouts: patch vector is [c][r][q] from planar [C,H,W]; dense vector
// is [c][dy][dx] from row-major patch embeddings.
func TestVisionGatherLayouts(t *testing.T) {
	shape := FlowImagePlan{Width: 4, Height: 4, PixelPatch: 2, TokenPatch: 4, GridWidth: 2, GridHeight: 2, TokenWidth: 1, TokenHeight: 1, Tokens: 1}
	channels := 2
	image := make([]float32, channels*4*4)
	for i := range image {
		image[i] = float32(i)
	}
	got := make([]float32, channels*2*2)
	patchVector(got, image, 3, shape, channels) // patch (py=1, px=1)
	// planar index (c*4 + 2+r)*4 + 2+q.
	want := []float32{10, 11, 14, 15, 26, 27, 30, 31}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("patch vector = %v, want %v", got, want)
		}
	}
	visionHidden := 3
	patches := make([]float32, 4*visionHidden)
	for i := range patches {
		patches[i] = float32(i)
	}
	dense := make([]float32, visionHidden*4)
	denseVector(dense, patches, 0, shape, visionHidden)
	// [c][dy][dx]: patch order (0,0)=(row0) (0,1)=(row1) (1,0)=(row2) (1,1)=(row3).
	wantDense := []float32{0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11}
	for i := range wantDense {
		if dense[i] != wantDense[i] {
			t.Fatalf("dense vector = %v, want %v", dense, wantDense)
		}
	}
}

// Rope: half per axis (x first), adjacent pairs, angle = pos*theta^(-2p/half).
func TestRopePatch2DSmallCase(t *testing.T) {
	theta := 100.0
	row := []float32{1, 0, 1, 0}
	ropePatch2D(row, 0, 0, theta) // both positions zero: identity
	for i, w := range []float32{1, 0, 1, 0} {
		if row[i] != w {
			t.Fatalf("zero-position rope changed row: %v", row)
		}
	}
	row = []float32{1, 0, 1, 0}
	ropePatch2D(row, 2, 1, theta) // dim=4, half=2: x-pair angle=1, y-pair angle=2
	wantAt := func(angle float64, a, b float32) (float32, float32) {
		c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
		return dtype.RoundBF16(a*c - b*s), dtype.RoundBF16(a*s + b*c)
	}
	x0, x1 := wantAt(1, 1, 0)
	y0, y1 := wantAt(2, 1, 0)
	for i, w := range []float32{x0, x1, y0, y1} {
		if row[i] != w {
			t.Fatalf("rope row = %v, want %v", row, []float32{x0, x1, y0, y1})
		}
	}
}

// Real-checkpoint shapes: every derived dim comes from config + tensors.
func TestSenseNovaFlowTerminalRealArtifact(t *testing.T) {
	if testing.Short() {
		t.Skip("opens the real artifact; skipped in -short")
	}
	if _, err := os.Stat(senseNovaDir); err != nil {
		t.Skipf("UNAVAILABLE: artifact absent at %s: %v", senseNovaDir, err)
	}
	binding := SenseNovaBinding()
	cfg, err := LoadConfig(senseNovaDir, binding)
	if err != nil {
		t.Fatal(err)
	}
	flowCfg, err := LoadFlowConfig(senseNovaDir)
	if err != nil {
		t.Fatal(err)
	}
	src, err := safetensors.OpenSource(senseNovaDir)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	flow := SenseNovaFlowBinding()
	plan, err := CompileFlowPlan(src, cfg, flowCfg, flow)
	if err != nil {
		t.Fatal(err)
	}
	want := testFlowPlan()
	if plan != want {
		t.Fatalf("plan = %+v, want %+v", plan, want)
	}
	// noise_scale_base_image_seq_len is the 256x256 reference token count.
	shape, err := plan.ImagePlan(256, 256)
	if err != nil {
		t.Fatal(err)
	}
	if shape.Tokens != plan.NoiseScaleBase {
		t.Fatalf("256x256 tokens = %d, base = %d", shape.Tokens, plan.NoiseScaleBase)
	}
	if shape.NoiseScale != 1 || plan.NormalizedNoiseScale(shape.NoiseScale) != 0.125 {
		t.Fatalf("256x256 noise scale = %g (normalized %g)", shape.NoiseScale, plan.NormalizedNoiseScale(shape.NoiseScale))
	}
	weights, err := LoadFlowTerminalWeights(src, plan, flow)
	if err != nil {
		t.Fatal(err)
	}
	condition, err := FlowConditionRow(weights, plan, 0, plan.NormalizedNoiseScale(shape.NoiseScale))
	if err != nil {
		t.Fatal(err)
	}
	if len(condition) != plan.Hidden {
		t.Fatalf("condition len = %d, want %d", len(condition), plan.Hidden)
	}
	nonzero := 0
	for _, v := range condition {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatal("condition row has non-finite values")
		}
		if v != 0 {
			nonzero++
		}
	}
	if nonzero == 0 {
		t.Fatal("condition row is all zeros")
	}
	for _, prefix := range []string{flow.SourceVisionPrefix, flow.GenerationVisionPrefix} {
		embedder, err := LoadVisionEmbedderWeights(src, prefix, plan)
		if err != nil {
			t.Fatal(err)
		}
		image := make([]float32, plan.VisionChannels*64*64)
		for i := range image {
			image[i] = float32(i%17)/17 - 0.5
		}
		tokens, tokenShape, err := VisionEmbedTokens(embedder, plan, image, 64, 64)
		if err != nil {
			t.Fatal(err)
		}
		if tokenShape.Tokens != 4 || len(tokens) != 4*plan.Hidden {
			t.Fatalf("%s: tokens = %d len = %d", prefix, tokenShape.Tokens, len(tokens))
		}
		for _, v := range tokens {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("%s: non-finite token embedding", prefix)
			}
		}
	}
	// flow head on one real-width row.
	hidden := make([]float32, plan.Hidden)
	z := make([]float32, plan.FlowDim)
	for i := range hidden {
		hidden[i] = dtype.RoundBF16(float32(i%13)/13 - 0.5)
	}
	velocity, err := FlowHeadVelocity(weights.Head, plan, hidden, z, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(velocity) != plan.FlowDim {
		t.Fatalf("velocity len = %d, want %d", len(velocity), plan.FlowDim)
	}
	for _, v := range velocity {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatal("velocity has non-finite values")
		}
	}
}

// Fixture-derivable module checks: the oracle's request reproduces the
// schedule knots, the z geometry, and the derived noise sigma.
func TestSenseNovaFlowScheduleMatchesGenerationOracle(t *testing.T) {
	if testing.Short() {
		t.Skip("opens the fixture; skipped in -short")
	}
	raw, err := os.ReadFile(senseNovaGenerationOracle)
	if err != nil {
		t.Skipf("UNAVAILABLE: fixture absent: %v", err)
	}
	var oracle struct {
		Request struct {
			Width, Height, Steps int
			TimestepShift        float64 `json:"timestep_shift"`
		} `json:"request"`
		Steps []struct {
			Timestep     float64 `json:"timestep"`
			NextTimestep float64 `json:"next_timestep"`
			Z            struct {
				Elements int     `json:"elements"`
				Std      float64 `json:"std"`
			} `json:"z"`
		} `json:"steps"`
	}
	if err := json.Unmarshal(raw, &oracle); err != nil {
		t.Fatal(err)
	}
	schedule, err := ShiftedFlowTimeSchedule(oracle.Request.Steps, oracle.Request.TimestepShift)
	if err != nil {
		t.Fatal(err)
	}
	if len(oracle.Steps) != oracle.Request.Steps {
		t.Fatalf("fixture steps = %d, request %d", len(oracle.Steps), oracle.Request.Steps)
	}
	for i, step := range oracle.Steps {
		if math.Abs(step.Timestep-schedule[i]) > 1e-9 || math.Abs(step.NextTimestep-schedule[i+1]) > 1e-9 {
			t.Fatalf("step %d = (%g,%g), schedule (%g,%g)", i, step.Timestep, step.NextTimestep, schedule[i], schedule[i+1])
		}
	}
	plan := testFlowPlan()
	shape, err := plan.ImagePlan(oracle.Request.Width, oracle.Request.Height)
	if err != nil {
		t.Fatal(err)
	}
	if wantZ := shape.Tokens * plan.FlowDim; oracle.Steps[0].Z.Elements != wantZ {
		t.Fatalf("fixture z elements = %d, plan tokens*flow = %d", oracle.Steps[0].Z.Elements, wantZ)
	}
	// z = derived_sigma * randn: the sample std certifies the derivation
	// (N=196608 -> std-of-std ~ sigma/sqrt(2N) ~ 0.0016).
	if sigma := shape.NoiseScale; math.Abs(oracle.Steps[0].Z.Std-sigma) > 0.01*sigma {
		t.Fatalf("fixture z std = %g, derived sigma = %g", oracle.Steps[0].Z.Std, sigma)
	}
}
