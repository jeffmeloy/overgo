package main

// Device vision mode: runs the RxBrain merged-patch vision tower's pre-norm
// attention blocks (pre_block0 -> block_last) as an overgo tensor-graph on the
// CUDA device through the generic executor, with F32-resident block weights.
// Front-end (conv patch embed + interpolated pos-embed) is host-computed
// (matching adaptive's host-front-end / device-blocks split); the golden
// pre_block0 asset is the device input (ladder discipline: each stage restarts
// from the previous fixture). Verified device == host oracle
// (patchtower.BlockForwardStages, 44/44 fixture-exact) == golden at the
// reference tolerances, then measured for wall + peak MiB.

import (
	"context"
	"fmt"
	"math"
	"time"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/patchtower"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

// runDeviceVision: device vision-blocks parity + measurement.
func runDeviceVision(l *ladder) error {
	ctx := context.Background()
	baseMiB := gpuUsedMiB()
	l.log(fmt.Sprintf("DEVICE vision START gpu.used=%dMiB free=%dMiB", baseMiB, gpuFreeMiB()))

	vg, err := loadGoldenJSON[visionGolden](l.fixturesDir, "rxbrain_vqa_vision_golden.json")
	if err != nil {
		return err
	}
	if len(vg.ImageGridTHW) != 3 {
		return fmt.Errorf("vision golden grid %v", vg.ImageGridTHW)
	}
	gridT, gridH, gridW := vg.ImageGridTHW[0], vg.ImageGridTHW[1], vg.ImageGridTHW[2]
	nPatch := gridT * gridH * gridW

	spec, err := patchtower.LoadSpec(l.modelDir)
	if err != nil {
		return err
	}
	src, err := safetensors.OpenSource(l.modelDir)
	if err != nil {
		return err
	}
	defer src.Close()
	H := spec.Hidden

	// Device input: golden pre_block0 [nPatch, hidden] token-major == [hidden,
	// nPatch] column-major.
	preBlock0, err := loadTensorAsset(l.fixturesDir, vg.PreBlock0Asset, vg.VisionTensors["pre_block0"])
	if err != nil {
		return err
	}
	if len(preBlock0) != nPatch*H {
		return fmt.Errorf("pre_block0 len %d != %d", len(preBlock0), nPatch*H)
	}

	// Host oracle: block0 sub-stages + full-stack block_last for a tight
	// device-vs-host check alongside the golden probes.
	hostBlocks := make([]patchtower.BlockWeights, spec.Depth)
	for layer := 0; layer < spec.Depth; layer++ {
		if hostBlocks[layer], err = patchtower.LoadBlockWeights(src, spec, layer); err != nil {
			return err
		}
	}
	hostBlock0, err := patchtower.BlockForwardStages(preBlock0, nPatch, spec, hostBlocks[0])
	if err != nil {
		return err
	}
	hostHidden := preBlock0
	for layer := 0; layer < spec.Depth; layer++ {
		stages, err := patchtower.BlockForwardStages(hostHidden, nPatch, spec, hostBlocks[layer])
		if err != nil {
			return err
		}
		hostHidden = stages.Output
	}
	l.log(fmt.Sprintf("DEVICE vision host oracle ready depth=%d rows=%d hidden=%d", spec.Depth, nPatch, H))

	// ---- device graph -------------------------------------------------------
	g, err := patchtower.BuildDeviceVisionBlocks(spec, nPatch)
	if err != nil {
		return err
	}
	compiled, err := executor.Compile(g.Block0Norm1, g.Block0QKV, g.Block0Attn, g.Block0Out, g.BlockLast)
	if err != nil {
		return fmt.Errorf("device vision compile: %w", err)
	}

	worker, err := device.New(0)
	if err != nil {
		return fmt.Errorf("device vision worker: %w", err)
	}
	defer worker.Close()
	exe, err := executor.NewWithWorker(worker)
	if err != nil {
		return fmt.Errorf("device vision executor: %w", err)
	}
	defer exe.Close()

	// ---- upload F32 block weights as persistent device feeds -----------------
	var weightPtrs []driver.DevicePtr
	upload := func(v []float32) (driver.DevicePtr, error) {
		var ptr driver.DevicePtr
		err := worker.Do(ctx, func(state *device.State) error {
			p, e := state.Driver.MemAlloc(uint64(len(v) * 4))
			if e != nil {
				return e
			}
			if e := state.Driver.MemcpyHtoD(p, driver.Bytes(v)); e != nil {
				_ = state.Driver.MemFree(p)
				return e
			}
			ptr = p
			return nil
		})
		if err != nil {
			return 0, err
		}
		weightPtrs = append(weightPtrs, ptr)
		return ptr, nil
	}
	defer func() {
		_ = worker.Do(ctx, func(state *device.State) error {
			for _, p := range weightPtrs {
				_ = state.Driver.MemFree(p)
			}
			return nil
		})
	}()

	deviceFeeds := map[*tensor.Tensor]driver.DevicePtr{}
	bind := func(node *tensor.Tensor, v []float32) error {
		p, e := upload(v)
		if e != nil {
			return e
		}
		deviceFeeds[node] = p
		return nil
	}
	for layer := 0; layer < spec.Depth; layer++ {
		w := hostBlocks[layer]
		in := g.Inputs[layer]
		if err := firstErr(
			bind(in.Norm1W, w.Norm1Weight), bind(in.Norm1B, w.Norm1Bias),
			bind(in.QKVW, w.QKVWeight), bind(in.QKVB, w.QKVBias),
			bind(in.ProjW, w.ProjWeight), bind(in.ProjB, w.ProjBias),
			bind(in.Norm2W, w.Norm2Weight), bind(in.Norm2B, w.Norm2Bias),
			bind(in.FC1W, w.FC1Weight), bind(in.FC1B, w.FC1Bias),
			bind(in.FC2W, w.FC2Weight), bind(in.FC2B, w.FC2Bias),
		); err != nil {
			return fmt.Errorf("device vision weight upload block %d: %w", layer, err)
		}
	}
	afterLoadMiB := gpuUsedMiB()
	l.log(fmt.Sprintf("DEVICE vision residency gpu.used %d->%dMiB (delta=%dMiB) free=%dMiB",
		baseMiB, afterLoadMiB, afterLoadMiB-baseMiB, gpuFreeMiB()))

	preShape := tensor.MustShape(uint64(H), uint64(nPatch))
	hostFeeds := map[*tensor.Tensor]reference.Value{
		g.PreBlock0: {Shape: preShape, Data: preBlock0},
	}
	deviceInputs, err := compiled.BindDeviceInputs(deviceFeeds)
	if err != nil {
		return err
	}
	out, err := exe.ExecuteCompiled(ctx, compiled, hostFeeds, deviceInputs)
	if err != nil {
		return fmt.Errorf("device vision execute: %w", err)
	}

	// ---- exactness: device vs host oracle (whole-tensor worst|d|) ------------
	worst := func(name string, a, b []float32) {
		if len(a) != len(b) {
			l.log(fmt.Sprintf("DEVICE vision HOST %-13s len %d != %d", name, len(a), len(b)))
			return
		}
		var w float64
		for i := range a {
			if d := math.Abs(float64(a[i]) - float64(b[i])); d > w {
				w = d
			}
		}
		l.log(fmt.Sprintf("DEVICE vision HOST %-13s device-vs-host worst|d|=%.3e", name, w))
	}
	worst("block0_norm1", out[g.Block0Norm1].Data, hostBlock0.Norm1)
	worst("block0_qkv", out[g.Block0QKV].Data, hostBlock0.QKV)
	worst("block0_attn", out[g.Block0Attn].Data, hostBlock0.AttnProjected)
	worst("block0", out[g.Block0Out].Data, hostBlock0.Output)
	worst("block_last", out[g.BlockLast].Data, hostHidden)

	// ---- diagnostic: device vs FULL golden block_last asset -----------------
	if gl, e := loadTensorAsset(l.fixturesDir, vg.BlockLastAsset, vg.VisionTensors["block_last"]); e == nil {
		dev := out[g.BlockLast].Data
		var wDG, wHG float64
		var iDG int
		nanDev, nanHost := 0, 0
		for i := range gl {
			if math.IsNaN(float64(dev[i])) || math.IsInf(float64(dev[i]), 0) {
				nanDev++
			}
			if math.IsNaN(float64(hostHidden[i])) || math.IsInf(float64(hostHidden[i]), 0) {
				nanHost++
			}
			if d := math.Abs(float64(dev[i]) - float64(gl[i])); d > wDG {
				wDG, iDG = d, i
			}
			if d := math.Abs(float64(hostHidden[i]) - float64(gl[i])); d > wHG {
				wHG = d
			}
		}
		token, ch := iDG/H, iDG%H
		l.log(fmt.Sprintf("DEVICE vision DIAG block_last dev-vs-golden worst|d|=%.3e at token=%d ch=%d (dev=%.4f host=%.4f golden=%.4f) host-vs-golden worst|d|=%.3e nanDev=%d nanHost=%d",
			wDG, token, ch, dev[iDG], hostHidden[iDG], gl[iDG], wHG, nanDev, nanHost))
	}

	// ---- exactness: device vs golden (reference tolerances) -----------------
	type probe struct {
		name       string
		dev        []float32
		golden     goldenTensor
		atol, rtol float64
	}
	probes := []probe{
		{"block0_norm1", out[g.Block0Norm1].Data, vg.VisionTensors["block0_norm1"], 0.02, 0.02},
		{"block0_qkv", out[g.Block0QKV].Data, vg.VisionTensors["block0_qkv"], 0.08, 0.04},
		{"block0_attn", out[g.Block0Attn].Data, vg.VisionTensors["block0_attn"], 0.12, 0.06},
		{"block0", out[g.Block0Out].Data, vg.VisionTensors["block0"], 0.08, 0.04},
		{"block_last", out[g.BlockLast].Data, vg.VisionTensors["block_last"], 0.12, 0.06},
	}
	var probeErr error
	for _, p := range probes {
		// probe fail census
		fails, worstD, worstLim := 0, 0.0, 0.0
		for i, idx := range p.golden.ProbeIndex {
			g, w := float64(p.dev[idx]), p.golden.ProbeValue[i]
			d := math.Abs(g - w)
			lim := p.atol + p.rtol*math.Abs(w)
			if d > lim {
				fails++
				if d-lim > worstD-worstLim {
					worstD, worstLim = d, lim
				}
			}
		}
		if fails > 0 {
			l.log(fmt.Sprintf("DEVICE vision GOLDEN CENSUS %s: %d/%d probes over tol; worst over by %.3e (|d|=%.3e lim=%.3e)", p.name, fails, len(p.golden.ProbeIndex), worstD-worstLim, worstD, worstLim))
		}
		detail, err := probeCheck("device "+p.name, p.dev, p.golden, p.atol, p.rtol)
		if err != nil {
			l.log("DEVICE vision GOLDEN FAIL " + err.Error())
			if probeErr == nil {
				probeErr = err
			}
			continue
		}
		l.log("DEVICE vision GOLDEN " + detail)
	}
	if probeErr != nil {
		return probeErr
	}

	// ---- measurement --------------------------------------------------------
	statsBefore, _ := worker.ExecutionStats(ctx)
	const warm, iters = 3, 20
	for i := 0; i < warm; i++ {
		if _, err := exe.ExecuteCompiled(ctx, compiled, hostFeeds, deviceInputs); err != nil {
			return err
		}
	}
	start := time.Now()
	for i := 0; i < iters; i++ {
		if _, err := exe.ExecuteCompiled(ctx, compiled, hostFeeds, deviceInputs); err != nil {
			return err
		}
	}
	perCall := time.Since(start) / iters
	statsAfter, _ := worker.ExecutionStats(ctx)
	peakMiB := gpuUsedMiB()
	l.log(fmt.Sprintf("DEVICE vision MEASURE %.3f ms/tower (%d blocks, %d rows, F32 weights resident) peak gpu.used=%dMiB (delta=%dMiB vs base) free=%dMiB",
		float64(perCall.Microseconds())/1000.0, spec.Depth, nPatch, peakMiB, peakMiB-baseMiB, gpuFreeMiB()))
	l.log(fmt.Sprintf("DEVICE vision REPLAY graph_launches=%d graph_instantiations=%d graph_updates=%d over %d warm+%d measure",
		statsAfter.GraphLaunches-statsBefore.GraphLaunches,
		statsAfter.GraphInstantiations-statsBefore.GraphInstantiations,
		statsAfter.GraphUpdates-statsBefore.GraphUpdates, warm, iters))
	l.log("DEVICE vision LANE GREEN")
	return nil
}
