//go:build windows

package latentimage

import (
	"io"
	"math"
	"path/filepath"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/media"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// numTrainTimesteps: FlowMatchEuler num_train_timesteps for Krea2 (scheduler
// config / g0 fixture).
const numTrainTimesteps = 1000

// TestDenoiserResidentG2Distribution drives the 12.82B-param dual-stream Krea2
// denoiser on the device (resident BF16 weights, generic executor) over the g2
// 8-step FlowMatchEuler schedule at the canonical 256x256 geometry, asserting
// the g2 per-step DISTRIBUTION oracle: every per-block hidden state and the
// image velocity is finite and well-conditioned, replay is bit-deterministic,
// and shapes are exact. Bit-exact per-step latents remain a residual on the
// seed-42 native RNG (gap2) and the exact Qwen3VL text conditioning (gap1),
// both out of scope; the latent/text inputs here are deterministic synthetic
// stand-ins, so this is a distribution/finiteness/determinism oracle on REAL
// weights, not an element-exact match to the g2 latent knots.
func TestDenoiserResidentG2Distribution(t *testing.T) {
	cudatest.Require(t)
	dir := kreaDirOrSkip(t)
	spec, err := Derive(dir)
	if err != nil {
		t.Fatalf("Derive: %v", err)
	}
	tr := spec.Transformer
	if tr.ModFields == 0 {
		tr.ModFields = 6
	}
	// canonical g0 case: 256x256, vae 8x, patch 2 -> 32x32 latent, 16x16 grid.
	const gh, gw = 16, 16
	const textSeq = 16 // representative; exact prompt length is gap1 (not oracle-relevant)
	imgSeq := gh * gw

	prog, err := CompileDenoiserProgram(tr, float32(tr.NormEps), attendedTextMask(textSeq), gh, gw, dtype.BF16)
	if err != nil {
		t.Fatalf("CompileDenoiserProgram: %v", err)
	}
	t.Logf("real geometry: layers=%d hidden=%d heads=%d/%d headDim=%d imgSeq=%d textSeq=%d seq=%d inCh=%d",
		tr.Layers, tr.Hidden, tr.Heads, tr.KVHeads, tr.HeadDim, imgSeq, textSeq, prog.Seq, tr.InChannels)

	ctx := t.Context()
	outputs := append(append([]*tensor.Tensor(nil), prog.BlockOutputs...), prog.Velocity)
	rd := newResidentFixture(t, ctx, "test denoiser", filepath.Join(dir, "transformer"), prog.weightInputs, outputs...)
	t.Logf("resident weights uploaded: %.2f GiB", float64(rd.graph.ProgramBytes())/(1<<30))

	src, err := safetensors.OpenSource(dir + `\transformer`)
	if err != nil {
		t.Fatalf("open transformer: %v", err)
	}
	defer src.Close()

	// deterministic synthetic text conditioning [textSeq*Hidden] (gap1).
	text := make([]float32, textSeq*tr.Hidden)
	st := uint64(0x1234567)
	for i := range text {
		st = st*6364136223846793005 + 1442695040888963407
		text[i] = float32((float64(st>>40)/float64(1<<24) - 0.5) * 0.1)
	}

	sched, err := CompileFlowSchedule(8, numTrainTimesteps, 1.15)
	if err != nil {
		t.Fatalf("CompileFlowSchedule: %v", err)
	}

	// deterministic synthetic seed latent [z=16, 32, 32] (gap2 RNG).
	const z, lh, lw, patch = 16, 32, 32, 2
	sample := make([]float64, z*lh*lw)
	ns := uint64(0x9e3779b97f4a7c15)
	for i := range sample {
		ns = ns*6364136223846793005 + 1442695040888963407
		sample[i] = (float64(ns>>40)/float64(1<<24) - 0.5) * 2
	}

	var firstVel []float32
	for step := 0; step < sched.Steps; step++ {
		sigma := sched.Sigmas[step]
		patches, pgh, pgw, err := media.PackPlanar(sample, z, lh, lw, patch, media.PatchChannelsFirst)
		if err != nil {
			t.Fatalf("PackLatent: %v", err)
		}
		if pgh != gh || pgw != gw {
			t.Fatalf("grid %dx%d want %dx%d", pgh, pgw, gh, gw)
		}
		temb, tembMod := hostTimestep(t, src, tr, sigma)
		res := residentDenoise(t, ctx, rd, prog, dtype.Float64SliceToFloat32(patches), text, temb, tembMod)
		// per-block distribution + finiteness.
		for l, hidden := range res.BlockHidden {
			if len(hidden) != prog.Seq*tr.Hidden {
				t.Fatalf("step %d block %d hidden len=%d want %d", step, l, len(hidden), prog.Seq*tr.Hidden)
			}
			if !allFinite32(hidden) {
				t.Fatalf("step %d block %d hidden non-finite", step, l)
			}
			if l == 0 || l == tr.Layers-1 {
				m, s := meanStd(hidden)
				t.Logf("step %d block %2d: mean=%+.4e std=%.4e", step, l, m, s)
			}
		}
		if !allFinite32(res.Velocity) {
			t.Fatalf("step %d velocity non-finite", step)
		}
		vm, vs := meanStd(res.Velocity)
		t.Logf("step %d sigma=%.5f velocity mean=%+.4e std=%.4e", step, sigma, vm, vs)
		if vs <= 0 {
			t.Fatalf("step %d velocity is degenerate (std=%g)", step, vs)
		}

		// determinism: step 0 replayed must be bit-identical.
		if step == 0 {
			firstVel = append([]float32(nil), res.Velocity...)
			res2 := residentDenoise(t, ctx, rd, prog, dtype.Float64SliceToFloat32(patches), text, temb, tembMod)
			for i := range firstVel {
				if firstVel[i] != res2.Velocity[i] {
					t.Fatalf("nondeterministic device forward at %d: %g vs %g", i, firstVel[i], res2.Velocity[i])
				}
			}
			t.Logf("deterministic replay verified (%d velocities bit-identical)", len(firstVel))
		}

		velocity, err := media.UnpackPlanar(f64of(res.Velocity), z, gh, gw, patch, media.PatchChannelsFirst)
		if err != nil {
			t.Fatalf("UnpackLatent: %v", err)
		}
		if err := sched.EulerStep(sample, velocity, step); err != nil {
			t.Fatalf("EulerStep %d: %v", step, err)
		}
	}
	if !allFinite(sample) {
		t.Fatalf("final latent non-finite")
	}
	fm, fs := meanStd64(sample)
	t.Logf("final latent: mean=%+.4e std=%.4e (all %d finite)", fm, fs, len(sample))
}

// hostTimestep computes the timestep embedding + AdaLN-single modulation on the
// host exactly as Denoiser.timestepConditioning, loading only the small
// timestep tensors as F32 (no full-model host copy).
func hostTimestep(t *testing.T, src *safetensors.Source, spec TransformerSpec, sigma float64) (temb, tembMod []float32) {
	t.Helper()
	e1w := loadF32(t, src, "time_embed.linear_1.weight")
	e1b := loadF32(t, src, "time_embed.linear_1.bias")
	e2w := loadF32(t, src, "time_embed.linear_2.weight")
	e2b := loadF32(t, src, "time_embed.linear_2.bias")
	pw := loadF32(t, src, "time_mod_proj.weight")
	pb := loadF32(t, src, "time_mod_proj.bias")

	dim := spec.TimestepEmbed
	half := dim / 2
	emb := make([]float64, dim)
	for i := range half {
		freq := math.Exp(-math.Log(1e4) * float64(i) / float64(half))
		arg := sigma * 1e3 * freq
		emb[i] = math.Cos(arg)
		emb[half+i] = math.Sin(arg)
	}
	l1 := dense(emb, e1w, e1b, 1, dim, spec.Hidden)
	for i := range l1 {
		l1[i] = geluTanh(l1[i])
	}
	tembF := dense(l1, e2w, e2b, 1, spec.Hidden, spec.Hidden)
	modIn := make([]float64, spec.Hidden)
	for i := range tembF {
		modIn[i] = geluTanh(tembF[i])
	}
	modF := dense(modIn, pw, pb, 1, spec.Hidden, 6*spec.Hidden)
	return dtype.Float64SliceToFloat32(tembF), dtype.Float64SliceToFloat32(modF)
}

func loadF32(t *testing.T, src *safetensors.Source, name string) []float32 {
	t.Helper()
	tt, ok := src.Tensors[name]
	if !ok {
		t.Fatalf("missing tensor %s", name)
	}
	elements := 1
	for _, d := range tt.Shape {
		elements *= int(d)
	}
	r, err := safetensors.F32Reader(tt)
	if err != nil {
		t.Fatalf("F32Reader %s: %v", name, err)
	}
	raw := make([]byte, elements*4)
	if _, err := io.ReadFull(r, raw); err != nil {
		t.Fatalf("read %s: %v", name, err)
	}
	out := make([]float32, elements)
	for i := range out {
		out[i] = math.Float32frombits(uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 | uint32(raw[4*i+3])<<24)
	}
	return out
}

func allFinite32(v []float32) bool {
	for _, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return false
		}
	}
	return true
}

func f64of(v []float32) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}
	return out
}

func meanStd64(v []float64) (mean, std float64) {
	if len(v) == 0 {
		return 0, 0
	}
	var sum, sumSq float64
	for _, x := range v {
		sum += x
		sumSq += x * x
	}
	n := float64(len(v))
	mean = sum / n
	variance := sumSq/n - mean*mean
	if variance < 0 {
		variance = 0
	}
	return mean, math.Sqrt(variance)
}
