package main

// Device merger mode: runs the RxBrain vision merger (4-matrix score-pooled
// per-2x2-group softmax pooling + erf-GELU) as an overgo tensor-graph on the
// CUDA device through the generic executor, with F32-resident merger weights
// and the golden block_last asset as the device input (ladder discipline).
// Verified device == host oracle (patchtower.MergerRowInto) == golden merger
// tensor at the reference tolerances, then measured for wall + peak MiB.

import (
	"context"
	"fmt"
	"math"
	"time"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/deviceprobe"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/patchtower"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

func runDeviceMerger(l *ladder) error {
	ctx := context.Background()
	baseMemory, err := deviceprobe.MeasureMemory()
	if err != nil {
		return err
	}
	l.log(fmt.Sprintf("DEVICE merger START gpu.used=%dMiB free=%dMiB", baseMemory.UsedMiB, baseMemory.FreeMiB))

	vg, err := loadGoldenJSON[visionGolden](l.fixturesDir, "rxbrain_vqa_vision_golden.json")
	if err != nil {
		return err
	}
	if len(vg.ImageGridTHW) != 3 {
		return fmt.Errorf("vision golden grid %v", vg.ImageGridTHW)
	}
	gridT, gridH, gridW := vg.ImageGridTHW[0], vg.ImageGridTHW[1], vg.ImageGridTHW[2]

	spec, err := patchtower.LoadSpec(l.modelDir)
	if err != nil {
		return err
	}
	src, err := safetensors.OpenSource(l.modelDir)
	if err != nil {
		return err
	}
	defer src.Close()

	blockLast, err := loadTensorAsset(l.fixturesDir, vg.BlockLastAsset, vg.VisionTensors["block_last"])
	if err != nil {
		return err
	}
	merger, err := patchtower.LoadMergerWeights(src, spec)
	if err != nil {
		return err
	}
	imageRows, err := patchtower.ValidateMergerInputs(blockLast, gridT, gridH, gridW, spec)
	if err != nil {
		return err
	}
	O := spec.OutHidden
	nPatch := gridT * gridH * gridW

	// Host oracle: full merger over all rows.
	hostMerged := make([]float32, imageRows*O)
	scratch := patchtower.NewMergerScratch(spec)
	for row := 0; row < imageRows; row++ {
		patchtower.MergerRowInto(hostMerged[row*O:(row+1)*O], blockLast, row, gridH, gridW, spec, merger, &scratch)
	}
	l.log(fmt.Sprintf("DEVICE merger host oracle ready rows=%d out=%d vHidden=%d nPatch=%d", imageRows, O, spec.Hidden, nPatch))

	g, err := patchtower.BuildDeviceMerger(spec, imageRows, gridT, gridH, gridW)
	if err != nil {
		return err
	}
	compiled, err := executor.Compile(g.Merged)
	if err != nil {
		return fmt.Errorf("device merger compile: %w", err)
	}

	worker, err := device.New(0)
	if err != nil {
		return fmt.Errorf("device merger worker: %w", err)
	}
	defer worker.Close()
	exe, err := executor.NewWithWorker(worker)
	if err != nil {
		return fmt.Errorf("device merger executor: %w", err)
	}
	defer exe.Close()

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
	if err := firstErr(
		bind(g.Proj1W, merger.Proj1Weight), bind(g.Proj1B, merger.Proj1Bias),
		bind(g.Proj2W, merger.Proj2Weight), bind(g.Proj2B, merger.Proj2Bias),
		bind(g.Pool0W, merger.Pool0Weight), bind(g.Pool0B, merger.Pool0Bias),
		bind(g.Pool2W, merger.Pool2Weight), bind(g.Pool2B, merger.Pool2Bias),
	); err != nil {
		return fmt.Errorf("device merger weight upload: %w", err)
	}
	afterLoadMemory, err := deviceprobe.MeasureMemory()
	if err != nil {
		return err
	}
	l.log(fmt.Sprintf("DEVICE merger residency gpu.used %d->%dMiB (delta=%dMiB) free=%dMiB", baseMemory.UsedMiB, afterLoadMemory.UsedMiB, afterLoadMemory.UsedMiB-baseMemory.UsedMiB, afterLoadMemory.FreeMiB))

	blockShape := tensor.MustShape(uint64(spec.Hidden), uint64(nPatch))
	hostFeeds := map[*tensor.Tensor]reference.Value{
		g.BlockLast: {Shape: blockShape, Data: blockLast},
	}
	deviceInputs, err := compiled.BindDeviceInputs(deviceFeeds)
	if err != nil {
		return err
	}
	outVals, err := exe.ExecuteCompiled(ctx, compiled, hostFeeds, deviceInputs)
	if err != nil {
		return fmt.Errorf("device merger execute: %w", err)
	}
	devMerged := outVals[g.Merged].Data
	if len(devMerged) != len(hostMerged) {
		return fmt.Errorf("device merger len %d != host %d", len(devMerged), len(hostMerged))
	}

	// device vs host oracle (whole-tensor worst|d|).
	var worstDH float64
	for i := range devMerged {
		if d := math.Abs(float64(devMerged[i]) - float64(hostMerged[i])); d > worstDH {
			worstDH = d
		}
	}
	l.log(fmt.Sprintf("DEVICE merger HOST device-vs-host worst|d|=%.3e", worstDH))

	// device vs golden (reference tolerances 0.08/0.04, the ladder merger gate).
	detail, err := probeCheck("device merger", devMerged, vg.VisionTensors["merger"], 0.08, 0.04)
	if err != nil {
		return err
	}
	l.log("DEVICE merger GOLDEN " + detail)

	// measurement.
	statsBefore, _ := worker.ExecutionStats(ctx)
	const warm, iters = 3, 30
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
	peakMemory, err := deviceprobe.MeasureMemory()
	if err != nil {
		return err
	}
	l.log(fmt.Sprintf("DEVICE merger MEASURE %.3f ms/merge (%d rows, F32 weights resident) peak gpu.used=%dMiB (delta=%dMiB vs base) free=%dMiB",
		float64(perCall.Microseconds())/1000.0, imageRows, peakMemory.UsedMiB, peakMemory.UsedMiB-baseMemory.UsedMiB, peakMemory.FreeMiB))
	l.log(fmt.Sprintf("DEVICE merger REPLAY graph_launches=%d graph_instantiations=%d graph_updates=%d over %d warm+%d measure",
		statsAfter.GraphLaunches-statsBefore.GraphLaunches, statsAfter.GraphInstantiations-statsBefore.GraphInstantiations,
		statsAfter.GraphUpdates-statsBefore.GraphUpdates, warm, iters))
	l.log("DEVICE merger LANE GREEN")
	return nil
}
