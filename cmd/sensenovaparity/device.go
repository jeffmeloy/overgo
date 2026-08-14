package main

// Device mode: takes the SenseNova-U1 MoT generation-lane denoise-step TERMINAL
// and its FlowMatchEuler sampler onto the CUDA device through overgo's generic
// executor, and verifies each against the strongest oracle that exists for it.
//
// What runs on device (the denoise-step terminal, per step, over the 64 latent
// tokens of a 256x256 image):
//   1. flow condition row  — timestep + noise-scale sinusoidal embedders
//      (fm_modules.{timestep,noise_scale}_embedder): freq -> W0+b0 -> SiLU ->
//      W2+b2, added and bf16-rounded. Verified DEVICE-vs-HOST (routedlm.
//      FlowConditionRow) on the real 35GB checkpoint's fm_modules weights.
//   2. flow head velocity  — fm_modules.fm_head: hidden -> W0+b0 -> exact-erf
//      GELU -> W2+b2 -> v=bf16((x-z)/max(1-t,t_eps)). Verified DEVICE-vs-HOST
//      (routedlm.FlowHeadVelocity) on the real fm_head weights, over 64 tokens.
//   3. FlowMatchEuler update — next_z = z + (t_next-t)*guided_velocity.
//      Verified DEVICE-vs-ORACLE against the REAL generation_oracle_256 (and
//      edit_oracle_v3_256) z-trajectory on the aligned probe coordinates — the
//      only exact absolute-value oracle the sparse (16/64-sample) fixtures admit
//      on the denoise trajectory. RNG-independent (uses the fixture's own z and
//      velocity).
//
// What does NOT run here (named honestly in the ladder + the report): the
// 42-layer branch-routed understanding-prefix + generation-branch cross-attn
// transformer body that PRODUCES the `hidden` boundary the flow head consumes.
// Its device port is scaffolded by prefilldevicegraph.go + CompileRopePlan but
// remains. Generation has exact seeded-state and terminal oracles but no
// intermediate-layer probes. Edit per-layer parity still needs the external
// source PNG. Multi-axis device rope is already shared in routedlm.RopePlan.

import (
	"context"
	"fmt"
	"math"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/routedlm"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// smiField: one integer nvidia-smi gpu field (memory.free / memory.used).
func smiField(query string) int {
	out, err := exec.Command("nvidia-smi", "--query-gpu="+query, "--format=csv,noheader,nounits").Output()
	if err != nil {
		return -1
	}
	fields := strings.Fields(strings.TrimSpace(string(out)))
	if len(fields) == 0 {
		return -1
	}
	v, err := strconv.Atoi(fields[0])
	if err != nil {
		return -1
	}
	return v
}

func gpuUsedMiB() int { return smiField("memory.used") }
func gpuFreeMiB() int { return smiField("memory.free") }

// deviceUploader: uploads host bytes as device feeds, tracks them for one free.
type deviceUploader struct {
	worker *device.Worker
	ctx    context.Context
	ptrs   []driver.DevicePtr
}

func (u *deviceUploader) up(b []byte) (driver.DevicePtr, error) {
	var ptr driver.DevicePtr
	err := u.worker.Do(u.ctx, func(state *device.State) error {
		q, e := state.Driver.MemAlloc(uint64(len(b)))
		if e != nil {
			return e
		}
		if e := state.Driver.MemcpyHtoD(q, b); e != nil {
			_ = state.Driver.MemFree(q)
			return e
		}
		ptr = q
		return nil
	})
	if err != nil {
		return 0, err
	}
	u.ptrs = append(u.ptrs, ptr)
	return ptr, nil
}

func (u *deviceUploader) upBF16(m routedlm.BF16Matrix) (driver.DevicePtr, error) {
	return u.up(driver.Bytes(m.Data))
}

func (u *deviceUploader) free() {
	_ = u.worker.Do(u.ctx, func(state *device.State) error {
		for _, q := range u.ptrs {
			_ = state.Driver.MemFree(q)
		}
		return nil
	})
	u.ptrs = u.ptrs[:0]
}

// worstAbs: worst |a-b| over aligned slices.
func worstAbs(a, b []float32) float64 {
	worst := 0.0
	for i := range a {
		if d := math.Abs(float64(a[i]) - float64(b[i])); d > worst {
			worst = d
		}
	}
	return worst
}

// maxAbs: max |v| over a slice (reference magnitude for a relative tolerance).
func maxAbs(a []float32) float64 {
	m := 0.0
	for _, v := range a {
		if d := math.Abs(float64(v)); d > m {
			m = d
		}
	}
	return m
}

// runDevice: the SenseNova generation-lane denoise-step terminal + sampler on
// the CUDA generic executor, appended to the ladder as device stages.
func runDevice(l *ladder, modelDir, fixturesDir string) error {
	ctx := context.Background()
	l.log(fmt.Sprintf("DEVICE denoise-terminal START gpu.free=%dMiB used=%dMiB", gpuFreeMiB(), gpuUsedMiB()))

	var edit editOracle
	if err := loadJSON(fixturesDir+string(os.PathSeparator)+"edit_oracle_v3_256.json", &edit); err != nil {
		return fmt.Errorf("load edit oracle: %w", err)
	}
	var gen genOracle
	if err := loadJSON(fixturesDir+string(os.PathSeparator)+"generation_oracle_256.json", &gen); err != nil {
		return fmt.Errorf("load gen oracle: %w", err)
	}

	binding := routedlm.SenseNovaBinding()
	flowBind := routedlm.SenseNovaFlowBinding()
	cfg, err := routedlm.LoadConfig(modelDir, binding)
	if err != nil {
		return err
	}
	flowCfg, err := routedlm.LoadFlowConfig(modelDir)
	if err != nil {
		return err
	}
	src, err := safetensors.OpenSource(modelDir)
	if err != nil {
		return err
	}
	defer src.Close()
	flowPlan, err := routedlm.CompileFlowPlan(src, cfg, flowCfg, flowBind)
	if err != nil {
		return err
	}
	weights, err := routedlm.LoadFlowTerminalWeights(src, flowPlan, flowBind)
	if err != nil {
		return err
	}
	shape256, err := flowPlan.ImagePlan(256, 256)
	if err != nil {
		return err
	}

	worker, err := device.New(0)
	if err != nil {
		return fmt.Errorf("device worker: %w", err)
	}
	defer worker.Close()
	exe, err := executor.NewWithWorker(worker)
	if err != nil {
		return fmt.Errorf("device executor: %w", err)
	}
	defer exe.Close()

	// ---- device multi-axis per-section rope (device-vs-host, real rope plan) ---
	// Closes the named body sub-gap: RoPEMulti carries a single FrequencyBase and
	// cannot express the SenseNova plan [64/5e6/Time, 32/1e4/H, 32/1e4/W]. The
	// generic executor expresses it by splicing one RoPENeoX per section
	// (GroupSlice -> RoPENeoX(theta,axis) -> Concat), no family kernel. Verified
	// device-vs-host on the REAL checkpoint's compiled plan over REAL block-causal
	// multi-axis positions. RNG/source-PNG independent.
	l.stage("device_multiaxis_rope", func() (string, float64, string, error) {
		plan, err := routedlm.CompileRopePlan(src, cfg, binding)
		if err != nil {
			return "", math.NaN(), "", err
		}
		if len(plan.Sections) != 3 {
			return "", math.NaN(), "", fmt.Errorf("expected 3 rope sections, got %d", len(plan.Sections))
		}
		// A block-causal multi-axis prompt: text rows, then one 4x4 image block,
		// then a trailing text row — exercises Time (text), and shared-Time +
		// (H,W) rasterization (image block) exactly as compileCausalPrefixInput.
		const tokenWidth = 4
		mask := []int{0, 0, 0}
		for i := 0; i < tokenWidth*tokenWidth; i++ {
			mask = append(mask, 1)
		}
		mask = append(mask, 0, 0)
		positions, err := routedlm.BlockPositions(mask, tokenWidth)
		if err != nil {
			return "", math.NaN(), "", err
		}
		tokens := len(positions)
		hd := cfg.HeadDim
		heads := cfg.NumAttentionHeads
		// deterministic bounded synthetic Q, layout [head_dim, heads, tokens]
		// flat = (t*heads+h)*head_dim + c (dim0 = channel, contiguous).
		q := make([]float32, tokens*heads*hd)
		for i := range q {
			q[i] = float32((i*37)%101)/50.5 - 1.0
		}
		f32ref := append([]float32(nil), q...)
		if err := plan.HostApplyF32Rows(f32ref, positions, heads, hd); err != nil {
			return "", math.NaN(), "", err
		}
		bf16ref := append([]float32(nil), q...)
		if err := plan.HostApplyBF16Rows(bf16ref, positions, heads, hd); err != nil {
			return "", math.NaN(), "", err
		}

		b := tensor.NewBuilder()
		qShape := tensor.MustShape(uint64(hd), uint64(heads), uint64(tokens))
		qNode := b.Input("q", dtype.F32, qShape)
		roped, err := plan.DeviceApply(b, qNode, positions)
		if err != nil {
			return "", math.NaN(), "", err
		}
		compiled, err := executor.Compile(roped)
		if err != nil {
			return "", math.NaN(), "", fmt.Errorf("compile: %w", err)
		}
		host := map[*tensor.Tensor]reference.Value{qNode: {Shape: qShape, Data: q}}
		out, err := exe.ExecuteCompiledWithDeviceFeeds(ctx, compiled, host, map[*tensor.Tensor]driver.DevicePtr{})
		if err != nil {
			return "", math.NaN(), "", fmt.Errorf("execute: %w", err)
		}
		dev := out[roped].Data
		if len(dev) != len(f32ref) {
			return "", math.NaN(), "", fmt.Errorf("device output len %d != %d", len(dev), len(f32ref))
		}
		// primary: device (f32 kernel) vs pure-f32 reference of the SAME
		// arithmetic — proves the composition + kernels are exact (f32-vs-f64
		// trig only). tight tol.
		worstF32 := worstAbs(dev, f32ref)
		if worstF32 > 1e-4 {
			return "", worstF32, "", fmt.Errorf("device rope != f32 reference worst|d|=%.3e > 1e-4 (composition/kernel bug)", worstF32)
		}
		// secondary: device vs the SenseNova bf16-stepped reference discipline
		// (applyRotary) — expected ~bf16 granularity, reported not gated.
		worstBF16 := worstAbs(dev, bf16ref)
		ref := maxAbs(bf16ref)
		return verdictWired, worstF32, fmt.Sprintf("DEVICE==HOST multi-axis rope on real plan [64/5e6/T,32/1e4/H,32/1e4/W]: tokens=%d heads=%d hd=%d block-causal(text+4x4 image) | vs-f32-ref worst|d|=%.3e (tol 1e-4) | vs-bf16-discipline worst|d|=%.3e max|ref|=%.3e (3 RoPENeoX spliced, no family kernel; RoPEMulti single-base cannot express this)", tokens, heads, hd, worstF32, worstBF16, ref), nil
	})

	H := flowPlan.Hidden
	F := flowPlan.FlowDim
	freqDim := flowPlan.FrequencyDim

	// ---- device flow condition row (device-vs-host, real fm_modules weights) --
	l.stage("device_flow_condition", func() (string, float64, string, error) {
		normNoise := flowPlan.NormalizedNoiseScale(shape256.NoiseScale)
		hostCond, err := routedlm.FlowConditionRow(weights, flowPlan, 0, normNoise)
		if err != nil {
			return "", math.NaN(), "", err
		}

		// host-side sinusoidal freq rows (bf16-rounded, as the host GEMM sees).
		freqT, err := routedlm.SinusoidalEmbedding([]float64{0}, freqDim, flowPlan.SinusoidalPeriod, 1)
		if err != nil {
			return "", math.NaN(), "", err
		}
		freqN, err := routedlm.SinusoidalEmbedding([]float64{normNoise}, freqDim, flowPlan.SinusoidalPeriod, 1)
		if err != nil {
			return "", math.NaN(), "", err
		}
		for i := range freqT {
			freqT[i] = dtype.RoundBF16(freqT[i])
			freqN[i] = dtype.RoundBF16(freqN[i])
		}

		b := tensor.NewBuilder()
		freqShape := tensor.MustShape(uint64(freqDim), 1)
		midBiasShape := tensor.MustShape(uint64(H), 1)
		// two-linear embedder as a graph node factory (freq -> H).
		embed := func(freqIn, w0, b0, w2, b2 *tensor.Tensor) *tensor.Tensor {
			mid := b.BF16Round(b.Add(b.MulMat(w0, freqIn), b0))
			mid = b.BF16Round(b.SiLU(mid))
			return b.BF16Round(b.Add(b.MulMat(w2, mid), b2))
		}
		freqTn := b.Input("freq_t", dtype.F32, freqShape)
		freqNn := b.Input("freq_n", dtype.F32, freqShape)
		tW0 := b.Input("t_w0", dtype.BF16, tensor.MustShape(uint64(freqDim), uint64(H)))
		tW2 := b.Input("t_w2", dtype.BF16, tensor.MustShape(uint64(H), uint64(H)))
		nW0 := b.Input("n_w0", dtype.BF16, tensor.MustShape(uint64(freqDim), uint64(H)))
		nW2 := b.Input("n_w2", dtype.BF16, tensor.MustShape(uint64(H), uint64(H)))
		tB0 := b.Input("t_b0", dtype.F32, midBiasShape)
		tB2 := b.Input("t_b2", dtype.F32, midBiasShape)
		nB0 := b.Input("n_b0", dtype.F32, midBiasShape)
		nB2 := b.Input("n_b2", dtype.F32, midBiasShape)
		tEmb := embed(freqTn, tW0, tB0, tW2, tB2)
		nEmb := embed(freqNn, nW0, nB0, nW2, nB2)
		cond := b.BF16Round(b.Add(tEmb, nEmb))
		compiled, err := executor.Compile(cond)
		if err != nil {
			return "", math.NaN(), "", fmt.Errorf("compile: %w", err)
		}

		up := &deviceUploader{worker: worker, ctx: ctx}
		defer up.free()
		dev := map[*tensor.Tensor]driver.DevicePtr{}
		for node, m := range map[*tensor.Tensor]routedlm.BF16Matrix{
			tW0: weights.Timestep.W0, tW2: weights.Timestep.W2,
			nW0: weights.NoiseScale.W0, nW2: weights.NoiseScale.W2,
		} {
			p, e := up.upBF16(m)
			if e != nil {
				return "", math.NaN(), "", e
			}
			dev[node] = p
		}
		host := map[*tensor.Tensor]reference.Value{
			freqTn: {Shape: freqShape, Data: freqT},
			freqNn: {Shape: freqShape, Data: freqN},
			tB0:    {Shape: midBiasShape, Data: weights.Timestep.B0},
			tB2:    {Shape: midBiasShape, Data: weights.Timestep.B2},
			nB0:    {Shape: midBiasShape, Data: weights.NoiseScale.B0},
			nB2:    {Shape: midBiasShape, Data: weights.NoiseScale.B2},
		}
		out, err := exe.ExecuteCompiledWithDeviceFeeds(ctx, compiled, host, dev)
		if err != nil {
			return "", math.NaN(), "", fmt.Errorf("execute: %w", err)
		}
		devCond := out[cond].Data
		worst := worstAbs(devCond, hostCond)
		ref := maxAbs(hostCond)
		rel := worst / ref
		// bf16-GEMM (device) vs f64-accumulate GEMM (host) through 2 linears +
		// SiLU + add: a few % relative is the expected native-bf16 gap.
		const relTol = 0.03
		if rel > relTol {
			return "", worst, "", fmt.Errorf("device condition rel=%.3e (worst|d|=%.3e, max|host|=%.3e) > %.3e", rel, worst, ref, relTol)
		}
		return verdictWired, worst, fmt.Sprintf("DEVICE==HOST (routedlm.FlowConditionRow) on real fm_modules embedders: hidden=%d t=0 norm_noise=%.5f worst|d|=%.3e max|host|=%.3e rel=%.3e tol=%.0e (native-bf16 vs f64 GEMM)", H, normNoise, worst, ref, rel, relTol), nil
	})

	// ---- device flow head velocity (device-vs-host, real fm_head weights) -----
	l.stage("device_flow_head", func() (string, float64, string, error) {
		const rows = 64 // the 256x256 image's latent-token count (denoise-step width)
		hidden := make([]float32, rows*H)
		for i := range hidden {
			hidden[i] = dtype.RoundBF16(float32(i%13)/13 - 0.5)
		}
		z := make([]float32, rows*F) // zero planar state (t=0)
		hostVel, err := routedlm.FlowHeadVelocity(weights.Head, flowPlan, hidden, z, 0)
		if err != nil {
			return "", math.NaN(), "", err
		}

		b := tensor.NewBuilder()
		hiddenShape := tensor.MustShape(uint64(H), uint64(rows))
		flowShape := tensor.MustShape(uint64(F), uint64(rows))
		hiddenT := b.Input("hidden", dtype.F32, hiddenShape)
		zT := b.Input("z", dtype.F32, flowShape)
		w0 := b.Input("w0", dtype.BF16, tensor.MustShape(uint64(H), uint64(H)))
		w2 := b.Input("w2", dtype.BF16, tensor.MustShape(uint64(H), uint64(F)))
		b0 := b.Input("b0", dtype.F32, tensor.MustShape(uint64(H), 1))
		b2 := b.Input("b2", dtype.F32, tensor.MustShape(uint64(F), 1))
		mid := b.BF16Round(b.Add(b.MulMat(w0, hiddenT), b0))
		mid = b.BF16Round(b.GELUErf(mid))
		out := b.BF16Round(b.Add(b.MulMat(w2, mid), b2))
		denom := float32(math.Max(1-0, flowPlan.TEps))
		vel := b.BF16Round(b.Scale(b.Add(out, b.Scale(zT, -1)), 1/denom))
		compiled, err := executor.Compile(vel)
		if err != nil {
			return "", math.NaN(), "", fmt.Errorf("compile: %w", err)
		}

		up := &deviceUploader{worker: worker, ctx: ctx}
		defer up.free()
		pw0, err := up.upBF16(weights.Head.W0)
		if err != nil {
			return "", math.NaN(), "", err
		}
		pw2, err := up.upBF16(weights.Head.W2)
		if err != nil {
			return "", math.NaN(), "", err
		}
		// ggml shape (dim, rows) is fed row-major [rows, dim] data directly (the
		// same layout deviceprefill feeds g.Row with [tokens, H] embeds).
		host := map[*tensor.Tensor]reference.Value{
			hiddenT: {Shape: hiddenShape, Data: hidden},
			zT:      {Shape: flowShape, Data: z},
			b0:      {Shape: tensor.MustShape(uint64(H), 1), Data: weights.Head.B0},
			b2:      {Shape: tensor.MustShape(uint64(F), 1), Data: weights.Head.B2},
		}
		dev := map[*tensor.Tensor]driver.DevicePtr{w0: pw0, w2: pw2}
		res, err := exe.ExecuteCompiledWithDeviceFeeds(ctx, compiled, host, dev)
		if err != nil {
			return "", math.NaN(), "", fmt.Errorf("execute: %w", err)
		}
		// device out shape (F, rows) is row-major [rows, F] — same layout as host.
		devVel := res[vel].Data
		worst := worstAbs(devVel, hostVel)
		ref := maxAbs(hostVel)
		rel := worst / ref
		const relTol = 0.03 // native-bf16 GEMM vs f64 accumulate, 2 linears + erf-GELU
		if rel > relTol {
			return "", worst, "", fmt.Errorf("device flow head rel=%.3e (worst|d|=%.3e, max|host|=%.3e) > %.3e", rel, worst, ref, relTol)
		}

		// measurement: denoise-step TERMINAL latency (flow head over 64 tokens).
		const warm, iters = 3, 30
		for i := 0; i < warm; i++ {
			if _, err := exe.ExecuteCompiledWithDeviceFeeds(ctx, compiled, host, dev); err != nil {
				return "", math.NaN(), "", err
			}
		}
		start := time.Now()
		for i := 0; i < iters; i++ {
			if _, err := exe.ExecuteCompiledWithDeviceFeeds(ctx, compiled, host, dev); err != nil {
				return "", math.NaN(), "", err
			}
		}
		perCall := float64(time.Since(start).Microseconds()) / float64(iters) / 1000.0
		return verdictWired, worst, fmt.Sprintf("DEVICE==HOST (routedlm.FlowHeadVelocity) on real fm_head: rows=%d hidden=%d flow_dim=%d worst|d|=%.3e max|host|=%.3e rel=%.3e tol=%.0e | head-terminal %.3fms/step gpu.used=%dMiB", rows, H, F, worst, ref, rel, relTol, perCall, gpuUsedMiB()), nil
	})

	// ---- device FlowMatchEuler update vs REAL z-trajectory oracle -------------
	eulerStage := func(dt float64, zS, vS, nzS sampledTensor) func() (string, float64, string, error) {
		return func() (string, float64, string, error) {
			zi, zv, err := zS.samples()
			if err != nil {
				return "", math.NaN(), "", fmt.Errorf("z: %w", err)
			}
			vi, vv, err := vS.samples()
			if err != nil {
				return "", math.NaN(), "", fmt.Errorf("v: %w", err)
			}
			ni, nv, err := nzS.samples()
			if err != nil {
				return "", math.NaN(), "", fmt.Errorf("nz: %w", err)
			}
			// require aligned probe coordinates across z/v/next_z.
			if len(zi) != len(vi) || len(zi) != len(ni) {
				return "", math.NaN(), "", fmt.Errorf("probe counts differ z=%d v=%d nz=%d", len(zi), len(vi), len(ni))
			}
			for k := range zi {
				if zi[k] != vi[k] || zi[k] != ni[k] {
					return "", math.NaN(), "", fmt.Errorf("probe index %d not aligned (z=%d v=%d nz=%d)", k, zi[k], vi[k], ni[k])
				}
			}
			n := len(zi)
			b := tensor.NewBuilder()
			vecShape := tensor.MustShape(uint64(n))
			zin := b.Input("z", dtype.F32, vecShape)
			vin := b.Input("v", dtype.F32, vecShape)
			next := b.Add(zin, b.Scale(vin, float32(dt))) // FlowMatchEuler
			compiled, err := executor.Compile(next)
			if err != nil {
				return "", math.NaN(), "", fmt.Errorf("compile: %w", err)
			}
			host := map[*tensor.Tensor]reference.Value{
				zin: {Shape: vecShape, Data: zv},
				vin: {Shape: vecShape, Data: vv},
			}
			out, err := exe.ExecuteCompiledWithDeviceFeeds(ctx, compiled, host, map[*tensor.Tensor]driver.DevicePtr{})
			if err != nil {
				return "", math.NaN(), "", fmt.Errorf("execute: %w", err)
			}
			devNext := out[next].Data
			worst := worstAbs(devNext, nv)
			// host self-check: the device kernel runs the same f32 z+dt*v, so any
			// residual vs the oracle is the fixture's own trajectory precision
			// (guided_velocity is bf16 out of the flow head). Confirm device==host.
			hostNext := make([]float32, n)
			for k := range zv {
				hostNext[k] = zv[k] + float32(dt)*vv[k]
			}
			devVsHost := worstAbs(devNext, hostNext)
			ref := maxAbs(nv)
			rel := worst / ref
			// bf16 relative granularity is 2^-8 ~= 3.9e-3; allow ~1.5 ulp.
			const relTol = 6e-3
			if devVsHost > 1e-6 {
				return "", devVsHost, "", fmt.Errorf("device euler != host euler by %.3e (kernel bug, not oracle precision)", devVsHost)
			}
			if rel > relTol {
				return "", worst, "", fmt.Errorf("device euler rel=%.3e (worst|d|=%.3e, max|oracle|=%.3e) > %.3e (dt=%g, %d probes)", rel, worst, ref, relTol, dt, n)
			}
			return verdictOracle, worst, fmt.Sprintf("DEVICE next_z == ORACLE next_z on %d aligned probes: next=z+%.3f*v worst|d|=%.3e max|oracle|=%.3e rel=%.3e (dev==host exact; residual=fixture bf16 trajectory) RNG-independent", n, dt, worst, ref, rel), nil
		}
	}
	for s := range gen.Steps {
		st := gen.Steps[s]
		l.stage("device_euler_gen_step"+strconv.Itoa(s), eulerStage(st.NextTimestep-st.Timestep, st.Z, st.GuidedVelocity, st.NextZ))
	}
	es := edit.Steps[0]
	l.stage("device_euler_edit_step0", eulerStage(es.NextTimestep-es.Timestep, es.Z, es.GuidedVelocity, es.NextZ))

	// ---- trajectory continuity (host, real oracle invariant) -----------------
	l.stage("gen_trajectory_continuity", func() (string, float64, string, error) {
		if len(gen.Steps) < 2 {
			return "", math.NaN(), "", fmt.Errorf("need >=2 gen steps")
		}
		i0, nz0, err := gen.Steps[0].NextZ.samples()
		if err != nil {
			return "", math.NaN(), "", err
		}
		i1, z1, err := gen.Steps[1].Z.samples()
		if err != nil {
			return "", math.NaN(), "", err
		}
		if len(i0) != len(i1) {
			return "", math.NaN(), "", fmt.Errorf("probe counts differ %d/%d", len(i0), len(i1))
		}
		for k := range i0 {
			if i0[k] != i1[k] {
				return "", math.NaN(), "", fmt.Errorf("continuity probe %d not aligned", k)
			}
		}
		worst := worstAbs(nz0, z1)
		if worst > 1e-6 {
			return "", worst, "", fmt.Errorf("step0.next_z != step1.z worst|d|=%.3e", worst)
		}
		return verdictOracle, worst, fmt.Sprintf("step0.next_z == step1.z on %d aligned probes (worst|d|=%.3e): denoise loop feeds each step's output forward", len(i0), worst), nil
	})

	l.log(fmt.Sprintf("DEVICE denoise-terminal DONE gpu.free=%dMiB used=%dMiB", gpuFreeMiB(), gpuUsedMiB()))
	if !l.failed {
		l.log("DEVICE denoise-terminal LANE GREEN (terminal+sampler verified; 42-layer body = named remainder)")
	}
	return nil
}
