package latentvideo

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"

	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/testutil"
)

// referenceDenoiserPolicy: published diffusion-pipeline facts the checkpoint
// cannot carry (training timesteps, sinusoidal period, rotary base, VAE
// strides), mirrored by the g0 config capture.
var referenceDenoiserPolicy = DenoiserPolicy{
	NumTrainTimesteps:   1000,
	SinusoidalPeriod:    10000,
	RotaryFrequencyBase: 10000,
	VAEStride:           [3]int{4, 8, 8},
}

// Golden parity tolerances vs the CUDA capture engine (absolute, max
// element). Derivation: the goldens are f32 CUDA captures while the
// reference backend accumulates in f64, so residual disagreement is purely
// engine accumulation order (the reference repo's own real block-0
// host-vs-CUDA gate is 0.08 max-abs). Measured 2026-08-09 on this port
// (values logged verbatim by the tests): timestep conditioning bit-exact
// (0); block-0 seams worst 4.20e-5 (self_attn, both branches); 30-block
// outputs worst 1.43e-4 (block 8); step-0 branch outputs worst 3.8e-6;
// trajectory branch outputs worst 2.40e-5; guided model outputs worst
// 1.59e-4 (g4 step 1); sampler samples worst 1.33e-4; final latents 2.52e-5
// (g3) / 1.33e-4 (g4). Gates: measured worst x ~2.5.
const (
	goldenNoiseTolerance       = 2e-6 // device sinf/cosf/logf vs host correctly-rounded: |s*dsin| <= ~4*2^-22; measured 9.54e-7 (148 ulp on near-zero sines)
	goldenSeamTolerance        = 1e-4
	goldenBlockOutputTolerance = 4e-4
	goldenBranchTolerance      = 1e-5
	goldenGuidedTolerance      = 4e-4
	goldenSampleTolerance      = 4e-4
	goldenFinalTolerance       = 4e-4
)

type goldenNoisePlan struct {
	Seed          int64  `json:"seed"`
	Block         int    `json:"block"`
	Unroll        int    `json:"unroll"`
	Grid          int    `json:"grid"`
	CounterOffset uint64 `json:"counter_offset"`
	Elements      int    `json:"elements"`
}

type goldenDenoiseRequest struct {
	Steps         int     `json:"steps"`
	Shift         float64 `json:"shift"`
	GuideScale    float64 `json:"guide_scale"`
	Frames        int     `json:"frames"`
	Width         int     `json:"width"`
	Height        int     `json:"height"`
	Seed          int64   `json:"seed"`
	CondContext   string  `json:"cond_context"`
	UncondContext string  `json:"uncond_context"`
}

type goldenLatentShape struct {
	Channels     int `json:"channels"`
	LatentFrames int `json:"latent_frames"`
	LatentHeight int `json:"latent_height"`
	LatentWidth  int `json:"latent_width"`
	SeqLen       int `json:"seq_len"`
}

type goldenDenoiseManifest struct {
	FinalLatent g1Tensor             `json:"final_latent"`
	LatentShape goldenLatentShape    `json:"latent_shape"`
	NoisePlan   goldenNoisePlan      `json:"noise_plan"`
	Request     goldenDenoiseRequest `json:"request"`
	Sigmas      []float64            `json:"sigmas"`
	Timesteps   []int64              `json:"timesteps"`
	Traces      map[string]g1Tensor  `json:"traces"`
}

func loadDenoiseManifest(t *testing.T, name string) (goldenDenoiseManifest, string) {
	t.Helper()
	dir := testutil.FixturePath(t, "wan")
	raw, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		t.Fatalf("missing required golden manifest: %v", err)
	}
	var manifest goldenDenoiseManifest
	if err := json.Unmarshal(raw, &manifest); err != nil {
		t.Fatal(err)
	}
	return manifest, dir
}

func (m goldenDenoiseManifest) trace(t *testing.T, dir, name string) []float32 {
	t.Helper()
	spec, ok := m.Traces[name]
	if !ok {
		t.Fatalf("golden manifest has no trace %q", name)
	}
	return loadG1Tensor(t, dir, spec)
}

func loadRawContext(t *testing.T, dir, file string, elements int) []float32 {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(dir, filepath.FromSlash(file)))
	if err != nil {
		t.Fatalf("missing required golden context: %v", err)
	}
	if len(raw) != elements*4 {
		t.Fatalf("golden context %s bytes=%d want %d", file, len(raw), elements*4)
	}
	out := make([]float32, elements)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

type denoiserFixtureState struct {
	once    sync.Once
	config  DenoiserConfig
	weights *DenoiserWeights
	err     error
}

var denoiserFixtureShared denoiserFixtureState

// denoiserFixture: the 5.7GB F32 weight set loads once for the whole test
// binary (no duplication; both graphs reference the same slices).
func denoiserFixture(t *testing.T) (DenoiserConfig, *DenoiserWeights) {
	t.Helper()
	dir := wanModelDir(t)
	denoiserFixtureShared.once.Do(func() {
		config, err := LoadDenoiserConfig(dir, referenceDenoiserPolicy)
		if err != nil {
			denoiserFixtureShared.err = err
			return
		}
		weights, err := LoadDenoiserWeights(dir, config)
		if err != nil {
			denoiserFixtureShared.err = err
			return
		}
		denoiserFixtureShared.config, denoiserFixtureShared.weights = config, weights
	})
	if denoiserFixtureShared.err != nil {
		t.Fatalf("denoiser fixture: %v", denoiserFixtureShared.err)
	}
	return denoiserFixtureShared.config, denoiserFixtureShared.weights
}

type parityStat struct {
	Max, Mean float64
	MaxIndex  int
}

func parityStats(got, want []float32) parityStat {
	stat := parityStat{}
	var sum float64
	for i := range want {
		diff := math.Abs(float64(got[i]) - float64(want[i]))
		sum += diff
		if diff > stat.Max {
			stat.Max, stat.MaxIndex = diff, i
		}
	}
	stat.Mean = sum / float64(len(want))
	return stat
}

func requireParity(t *testing.T, name string, got, want []float32, tolerance float64) parityStat {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d want %d", name, len(got), len(want))
	}
	stat := parityStats(got, want)
	t.Logf("%s max_abs_diff=%.6g mean_abs_diff=%.6g at=%d", name, stat.Max, stat.Mean, stat.MaxIndex)
	if stat.Max > tolerance {
		t.Fatalf("%s max abs diff %.6g > %.6g", name, stat.Max, tolerance)
	}
	return stat
}

// TestUniPCScheduleGoldenG3: shifted flow schedule vs the captured g3
// schedule (fixture-only; no model artifacts).
func TestUniPCScheduleGoldenG3(t *testing.T) {
	manifest, _ := loadDenoiseManifest(t, "g3_denoise.json")
	timesteps, sigmas, err := UniPCSchedule(referenceDenoiserPolicy.NumTrainTimesteps, manifest.Request.Steps, manifest.Request.Shift)
	if err != nil {
		t.Fatal(err)
	}
	if len(timesteps) != len(manifest.Timesteps) || len(sigmas) != len(manifest.Sigmas) {
		t.Fatalf("schedule extent %d/%d want %d/%d", len(timesteps), len(sigmas), len(manifest.Timesteps), len(manifest.Sigmas))
	}
	for i := range timesteps {
		if timesteps[i] != manifest.Timesteps[i] {
			t.Fatalf("timestep[%d]=%d want %d", i, timesteps[i], manifest.Timesteps[i])
		}
	}
	for i := range sigmas {
		if sigmas[i] != float32(manifest.Sigmas[i]) {
			t.Fatalf("sigma[%d]=%.9g want %.9g", i, sigmas[i], manifest.Sigmas[i])
		}
	}
	t.Logf("schedule timesteps=%v sigmas=%v", timesteps, sigmas)
}

// TestNormalNoiseGoldenG3: Philox counter-based normal source vs the
// captured initial sample (fixture-only; no model artifacts).
func TestNormalNoiseGoldenG3(t *testing.T) {
	manifest, dir := loadDenoiseManifest(t, "g3_denoise.json")
	want := manifest.trace(t, dir, "step_00_sample_in")
	got := make([]float32, len(want))
	if err := FillNormalNoise(got, NoisePlan{
		Seed:   uint64(manifest.NoisePlan.Seed),
		Grid:   manifest.NoisePlan.Grid,
		Block:  manifest.NoisePlan.Block,
		Unroll: manifest.NoisePlan.Unroll,
	}); err != nil {
		t.Fatal(err)
	}
	exact := 0
	var maxULP uint32
	for i := range want {
		gotBits, wantBits := math.Float32bits(got[i]), math.Float32bits(want[i])
		if gotBits == wantBits {
			exact++
			continue
		}
		ulp := gotBits - wantBits
		if wantBits > gotBits {
			ulp = wantBits - gotBits
		}
		if ulp > maxULP {
			maxULP = ulp
		}
	}
	t.Logf("noise exact_bits=%d/%d max_ulp=%d", exact, len(want), maxULP)
	requireParity(t, "noise vs step_00_sample_in", got, want, goldenNoiseTolerance)
}

// TestTimestepConditioningGoldenG2: sinusoidal + embedding + projection over
// the real weights vs the g2 host capture (same engine class; tolerance 0).
func TestTimestepConditioningGoldenG2(t *testing.T) {
	if testing.Short() {
		t.Skip("loads the 5.7GB denoiser weight set; skipped in -short")
	}
	dir := testutil.FixturePath(t, "wan")
	raw, err := os.ReadFile(filepath.Join(dir, "g2_timestep_conditioning.json"))
	if err != nil {
		t.Fatalf("missing required golden manifest: %v", err)
	}
	var manifest struct {
		HeadE    g1Tensor `json:"head_e"`
		BlockE   g1Tensor `json:"block_e"`
		Schedule struct {
			Timesteps []int64 `json:"timesteps"`
		} `json:"schedule"`
	}
	if err := json.Unmarshal(raw, &manifest); err != nil {
		t.Fatal(err)
	}
	config, weights := denoiserFixture(t)
	values := make([]float64, len(manifest.Schedule.Timesteps))
	for i, timestep := range manifest.Schedule.Timesteps {
		values[i] = float64(timestep)
	}
	headE, blockE, err := CompileTimestepConditioning(values, weights.TimestepWeights(config))
	if err != nil {
		t.Fatal(err)
	}
	requireParity(t, "g2 head_e", headE, loadG1Tensor(t, dir, manifest.HeadE), 0)
	requireParity(t, "g2 block_e", blockE, loadG1Tensor(t, dir, manifest.BlockE), 0)
}

type blockSeam struct {
	name string
	node *tensor.Tensor
}

func blockSeams(result model.ConditionedDiffusionBlockResult) []blockSeam {
	return []blockSeam{
		{"self_q_projected", result.SelfQueryProjected},
		{"self_k_projected", result.SelfKeyProjected},
		{"self_v_projected", result.SelfValueProjected},
		{"self_q_norm", result.SelfQueryNormed},
		{"self_k_norm", result.SelfKeyNormed},
		{"self_q_rope", result.SelfQueryRotated},
		{"self_k_rope", result.SelfKeyRotated},
		{"self_sdpa", result.SelfAttention},
		{"self_attn", result.SelfProjected},
		{"self_residual", result.SelfResidual},
		{"cross_attn", result.CrossProjected},
		{"cross_residual", result.CrossResidual},
		{"ffn", result.FeedForward},
		{"ffn_residual", result.Output},
	}
}

func newGoldenDenoiserProgram(t *testing.T, manifest goldenDenoiseManifest) *DenoiserProgram {
	t.Helper()
	config, weights := denoiserFixture(t)
	geometry, err := config.CompileLatentGeometry(manifest.Request.Frames, manifest.Request.Width, manifest.Request.Height)
	if err != nil {
		t.Fatal(err)
	}
	shape := manifest.LatentShape
	if geometry.Channels != shape.Channels || geometry.LatentFrames != shape.LatentFrames ||
		geometry.LatentHeight != shape.LatentHeight || geometry.LatentWidth != shape.LatentWidth ||
		geometry.Seq != shape.SeqLen {
		t.Fatalf("latent geometry %+v differs from golden %+v", geometry, shape)
	}
	program, err := CompileDenoiserProgram(config, weights, geometry)
	if err != nil {
		t.Fatal(err)
	}
	return program
}

func logHeapPeak(t *testing.T, scope string) {
	t.Helper()
	var stats runtime.MemStats
	runtime.ReadMemStats(&stats)
	t.Logf("%s heap_alloc_bytes=%d total_alloc_bytes=%d", scope, stats.HeapAlloc, stats.TotalAlloc)
}

// TestDenoiserGoldenDenoiseG3: the full graph-routed denoiser vs the exact
// CUDA capture — block-0 intra-block seams, all 30 per-block outputs on
// both branches, per-step branch/guided/sampler tensors, and the final
// latent. Reference backend executes both graphs.
func TestDenoiserGoldenDenoiseG3(t *testing.T) {
	if testing.Short() {
		t.Skip("runs the full 30-block denoiser on the reference backend; skipped in -short")
	}
	manifest, dir := loadDenoiseManifest(t, "g3_denoise.json")
	program := newGoldenDenoiserProgram(t, manifest)
	run := GraphRunner(reference.Execute)
	config := program.Config
	contextElements := config.TextLen * config.Dim
	condContext := loadRawContext(t, dir, manifest.Request.CondContext, contextElements)
	uncondContext := loadRawContext(t, dir, manifest.Request.UncondContext, contextElements)
	sampleIn := manifest.trace(t, dir, "step_00_sample_in")

	timesteps, _, err := UniPCSchedule(config.Policy.NumTrainTimesteps, manifest.Request.Steps, manifest.Request.Shift)
	if err != nil {
		t.Fatal(err)
	}
	headE, blockE, err := CompileTimestepConditioning([]float64{float64(timesteps[0])}, program.weights.TimestepWeights(config))
	if err != nil {
		t.Fatal(err)
	}
	patchTokens, err := program.PatchifyLatent(sampleIn)
	if err != nil {
		t.Fatal(err)
	}

	// Step-0 per-branch probes: block-0 seams, all block outputs, head.
	seams := blockSeams(program.Blocks[0])
	outputs := []*tensor.Tensor{program.Head}
	for _, seam := range seams {
		outputs = append(outputs, seam.node)
	}
	for _, block := range program.Blocks {
		outputs = append(outputs, block.Output)
	}
	branches := []struct {
		name    string
		context []float32
	}{
		{"cond", condContext},
		{"uncond", uncondContext},
	}
	var worstSeam, worstBlock parityStat
	worstBlockIndex := -1
	blockMax := make([]float64, len(program.Blocks))
	for _, branch := range branches {
		projection, err := program.ProjectContext(run, branch.context)
		if err != nil {
			t.Fatal(err)
		}
		values, err := program.Forward(run, patchTokens, blockE, headE, projection, outputs)
		if err != nil {
			t.Fatal(err)
		}
		for _, seam := range seams {
			name := fmt.Sprintf("step_00_block_00_%s_%s", branch.name, seam.name)
			stat := requireParity(t, name, values[seam.node].Data, manifest.trace(t, dir, name), goldenSeamTolerance)
			if stat.Max > worstSeam.Max {
				worstSeam = stat
			}
		}
		for index, block := range program.Blocks {
			name := fmt.Sprintf("step_00_block_%02d_%s", index, branch.name)
			want := manifest.trace(t, dir, name)
			got := values[block.Output].Data
			if len(got) != len(want) {
				t.Fatalf("%s: length %d want %d", name, len(got), len(want))
			}
			stat := parityStats(got, want)
			blockMax[index] = math.Max(blockMax[index], stat.Max)
			if stat.Max > worstBlock.Max {
				worstBlock, worstBlockIndex = stat, index
			}
			if stat.Max > goldenBlockOutputTolerance {
				t.Fatalf("%s max abs diff %.6g > %.6g", name, stat.Max, goldenBlockOutputTolerance)
			}
		}
		branchOut, err := program.UnpatchifyLatent(values[program.Head].Data)
		if err != nil {
			t.Fatal(err)
		}
		name := "step_00_branch_output_" + branch.name
		requireParity(t, name, branchOut, manifest.trace(t, dir, name), goldenBranchTolerance)
	}
	t.Logf("step-0 worst seam max_abs_diff=%.6g; worst block=%d max_abs_diff=%.6g", worstSeam.Max, worstBlockIndex, worstBlock.Max)
	for index, value := range blockMax {
		t.Logf("block_%02d worst-branch max_abs_diff=%.6g", index, value)
	}

	// Full guided trajectory from the captured initial sample.
	result, err := program.Denoise(run, DenoiseRequest{
		Steps: manifest.Request.Steps, Shift: manifest.Request.Shift,
		GuideScale:    manifest.Request.GuideScale,
		CondContext:   condContext,
		UncondContext: uncondContext,
		InitialSample: sampleIn,
		TraceSteps:    true,
	})
	if err != nil {
		t.Fatal(err)
	}
	for i, step := range result.Steps {
		prefix := fmt.Sprintf("step_%02d_", i)
		requireParity(t, prefix+"sample_in", step.SampleIn, manifest.trace(t, dir, prefix+"sample_in"), goldenSampleTolerance)
		requireParity(t, prefix+"branch_output_cond", step.CondOutput, manifest.trace(t, dir, prefix+"branch_output_cond"), goldenGuidedTolerance)
		requireParity(t, prefix+"branch_output_uncond", step.UncondOutput, manifest.trace(t, dir, prefix+"branch_output_uncond"), goldenGuidedTolerance)
		requireParity(t, prefix+"model_output", step.ModelOutput, manifest.trace(t, dir, prefix+"model_output"), goldenGuidedTolerance)
		requireParity(t, prefix+"sample_out", step.SampleOut, manifest.trace(t, dir, prefix+"sample_out"), goldenSampleTolerance)
	}
	requireParity(t, "final_latent", result.Latent, loadG1Tensor(t, dir, manifest.FinalLatent), goldenFinalTolerance)
	logHeapPeak(t, "g3 denoise")
}

// TestDenoiserGoldenDenoiseG4: the temporal 5-frame clip — multi-frame
// rotary coordinates and a 2-step trajectory vs the g4 capture.
func TestDenoiserGoldenDenoiseG4(t *testing.T) {
	if testing.Short() {
		t.Skip("runs the full 30-block denoiser on the reference backend; skipped in -short")
	}
	manifest, dir := loadDenoiseManifest(t, "g4_denoise.json")
	program := newGoldenDenoiserProgram(t, manifest)
	run := GraphRunner(reference.Execute)
	config := program.Config
	contextElements := config.TextLen * config.Dim
	condContext := loadRawContext(t, dir, manifest.Request.CondContext, contextElements)
	uncondContext := loadRawContext(t, dir, manifest.Request.UncondContext, contextElements)
	sampleIn := manifest.trace(t, dir, "step_00_sample_in")

	result, err := program.Denoise(run, DenoiseRequest{
		Steps: manifest.Request.Steps, Shift: manifest.Request.Shift,
		GuideScale:    manifest.Request.GuideScale,
		CondContext:   condContext,
		UncondContext: uncondContext,
		InitialSample: sampleIn,
		TraceSteps:    true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Timesteps) != len(manifest.Timesteps) {
		t.Fatalf("timesteps %d want %d", len(result.Timesteps), len(manifest.Timesteps))
	}
	for i := range result.Timesteps {
		if result.Timesteps[i] != manifest.Timesteps[i] {
			t.Fatalf("timestep[%d]=%d want %d", i, result.Timesteps[i], manifest.Timesteps[i])
		}
	}
	for i, step := range result.Steps {
		prefix := fmt.Sprintf("step_%02d_", i)
		requireParity(t, prefix+"sample_in", step.SampleIn, manifest.trace(t, dir, prefix+"sample_in"), goldenSampleTolerance)
		requireParity(t, prefix+"branch_output_cond", step.CondOutput, manifest.trace(t, dir, prefix+"branch_output_cond"), goldenGuidedTolerance)
		requireParity(t, prefix+"branch_output_uncond", step.UncondOutput, manifest.trace(t, dir, prefix+"branch_output_uncond"), goldenGuidedTolerance)
		requireParity(t, prefix+"model_output", step.ModelOutput, manifest.trace(t, dir, prefix+"model_output"), goldenGuidedTolerance)
		requireParity(t, prefix+"sample_out", step.SampleOut, manifest.trace(t, dir, prefix+"sample_out"), goldenSampleTolerance)
	}
	requireParity(t, "final_latent", result.Latent, loadG1Tensor(t, dir, manifest.FinalLatent), goldenFinalTolerance)
	logHeapPeak(t, "g4 denoise")
}
