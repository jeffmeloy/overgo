package main

// Device prefill mode: runs the RxBrain branch-routed MoT prompt layer stack as
// an overgo tensor-graph on the CUDA device through the generic executor. A
// single compiled branch-routed layer graph (dual-compute-then-select + segment
// attention) is replayed per layer with re-bound BF16/F32 weight feeds. Verified
// two ways: (1) the ladder discipline — each layer fed the previous GOLDEN
// boundary, device layer output compared to the layer golden (apples-to-apples
// with the host reference); (2) chained — all 32 layers from the host prefill
// embeds, device output-to-output, reporting the accumulated drift and the final
// terminal top token. Also exports the per-layer resident K/V for decode.

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"time"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/patchtower"
	"overgo/internal/routedlm"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/vqaserve"
)

// loadPrefillContext builds the golden-seeded prompt state: the golden
// block_last, the host merger rows spliced into the prefill embeds, and the
// golden modality mask, for the ladder and the full pipeline's verifier.
func loadPrefillContext(l *campaignContext) (*vqaserve.PrefillContext, error) {
	vg, err := loadGoldenJSON[visionGolden](l.fixturesDir, "rxbrain_vqa_vision_golden.json")
	if err != nil {
		return nil, err
	}
	fg, err := loadGoldenJSON[prefillGolden](l.fixturesDir, "rxbrain_vqa_prefill_golden.json")
	if err != nil {
		return nil, err
	}
	if len(vg.ImageGridTHW) != 3 {
		return nil, fmt.Errorf("vision golden grid %v", vg.ImageGridTHW)
	}
	gridT, gridH, gridW := vg.ImageGridTHW[tensor.FirstOffset], vg.ImageGridTHW[tensor.SingletonExtent], vg.ImageGridTHW[tensor.PairedExtent]
	spec, err := patchtower.LoadSpec(l.modelDir)
	if err != nil {
		return nil, err
	}
	cfg, err := routedlm.LoadConfig(l.modelDir, vqaserve.Binding)
	if err != nil {
		return nil, err
	}
	src, err := safetensors.OpenSource(l.modelDir)
	if err != nil {
		return nil, err
	}
	blockLast, err := loadTensorAsset(l.fixturesDir, vg.BlockLastAsset, vg.VisionTensors["block_last"])
	if err != nil {
		src.Close()
		return nil, err
	}
	merger, err := patchtower.LoadMergerWeights(src, spec)
	if err != nil {
		src.Close()
		return nil, err
	}
	imageRows, err := patchtower.ValidateMergerInputs(blockLast, gridT, gridH, gridW, spec)
	if err != nil {
		src.Close()
		return nil, err
	}
	scratch := patchtower.NewMergerScratch(spec)
	prefill, err := routedlm.PrefillValues(src, cfg, vqaserve.Binding, fg.InputIDs, fg.PrefillTensors.InputImageMaskPositions, imageRows, func(dst []float32, ordinal int) error {
		patchtower.MergerRowInto(dst, blockLast, ordinal, gridH, gridW, spec, merger, &scratch)
		return nil
	})
	if err != nil {
		src.Close()
		return nil, err
	}
	pc := &vqaserve.PrefillContext{
		Cfg: cfg, Src: src, Spec: spec, BlockLast: blockLast, Merger: merger,
		GridT: gridT, GridH: gridH, GridW: gridW, ImageRows: imageRows,
		InputIDs: fg.InputIDs, ImageMaskPositions: fg.PrefillTensors.InputImageMaskPositions,
		Mask: fg.PrefillTensors.ModalityMask, PromptLen: fg.PromptLen, Prefill: prefill,
	}
	pc.BindMask()
	return pc, nil
}

func runDevicePrefill(l *campaignContext) error {
	ctx := context.Background()
	l.Log("DEVICE prefill START")

	pc, err := loadPrefillContext(l)
	if err != nil {
		return err
	}
	defer pc.Src.Close()
	cfg := pc.Cfg
	H := cfg.HiddenSize
	l.Log(fmt.Sprintf("DEVICE prefill ctx prompt_len=%d image_rows=%d segments=%v blocks=%v", pc.PromptLen, pc.ImageRows, pc.Segments, pc.Blocks))

	g, err := routedlm.BuildDevicePrefillLayer(cfg, pc.PromptLen, pc.Blocks)
	if err != nil {
		return err
	}
	compiled, err := executor.Compile(g.Output, g.KeyKV, g.ValueKV, g.Context)
	if err != nil {
		return fmt.Errorf("device prefill compile: %w", err)
	}

	worker, err := device.New(device.DefaultOrdinal())
	if err != nil {
		return fmt.Errorf("device prefill worker: %w", err)
	}
	defer worker.Close()
	exe, err := executor.NewWithWorker(worker)
	if err != nil {
		return fmt.Errorf("device prefill executor: %w", err)
	}
	defer exe.Close()

	rowShape := tensor.MustShape(uint64(H), uint64(pc.PromptLen))
	maskShape := tensor.MustShape(tensor.SingletonExtent, uint64(pc.PromptLen))

	// ---- ladder verification: each layer fed the previous GOLDEN boundary ---
	allocations := device.NewAllocationSet(worker)
	defer func() { _ = allocations.Close(ctx) }()
	worstByLayer := make([]float64, cfg.NumHiddenLayers)
	for layer := range cfg.NumHiddenLayers {
		w, err := routedlm.LoadLayerWeights(pc.Src, cfg, vqaserve.Binding, layer)
		if err != nil {
			return err
		}
		input := pc.Prefill
		if layer > 0 {
			prev, err := loadGoldenJSON[motGolden](l.fixturesDir, "rxbrain_vqa_mot_layer"+strconv.Itoa(layer-1)+"_output_golden.json")
			if err != nil {
				return err
			}
			prevKey := "layer" + strconv.Itoa(layer-1) + "_output"
			input, err = loadTensorAsset(l.fixturesDir, prev.LayerOutputAsset, prev.MoTTensors[prevKey])
			if err != nil {
				return err
			}
		}
		feeds := map[*tensor.Tensor]driver.DevicePtr{}
		if err := vqaserve.BindLayerBranches(ctx, &allocations, feeds, g.Text, g.Vision, w); err != nil {
			return err
		}
		hostFeeds := map[*tensor.Tensor]reference.Value{
			g.Row:      {Shape: rowShape, Data: input},
			g.MaskText: {Shape: maskShape, Data: pc.MaskText},
			g.MaskVis:  {Shape: maskShape, Data: pc.MaskVis},
		}
		inputs := compiled.NewDeviceInputs()
		for node, pointer := range feeds {
			if err := inputs.Set(node, pointer); err != nil {
				return err
			}
		}
		out, err := exe.ExecuteCompiled(ctx, compiled, hostFeeds, inputs)
		if err != nil {
			_ = allocations.Close(ctx)
			return fmt.Errorf("device prefill execute layer %d: %w", layer, err)
		}
		devOut := out[g.Output].Data
		_ = allocations.Close(ctx)

		golden, err := loadGoldenJSON[motGolden](l.fixturesDir, "rxbrain_vqa_mot_layer"+strconv.Itoa(layer)+"_output_golden.json")
		if err != nil {
			return err
		}
		key := "layer" + strconv.Itoa(layer) + "_output"
		class := acceptLayer
		if layer == 0 {
			class = acceptFirstLayer
		}
		detail, err := probeCheck("layer"+strconv.Itoa(layer), devOut, golden.MoTTensors[key], class)
		if err != nil {
			l.Log(fmt.Sprintf("DEVICE prefill LADDER layer%d FAIL %v", layer, err))
			return err
		}
		// parse worst from detail is noisy; recompute worst on probes.
		gt := golden.MoTTensors[key]
		var worst float64
		for i, idx := range gt.ProbeIndex {
			if d := math.Abs(float64(devOut[idx]) - gt.ProbeValue[i]); d > worst {
				worst = d
			}
		}
		worstByLayer[layer] = worst
		if layer == 0 || layer == cfg.NumHiddenLayers-1 || layer%8 == 0 {
			l.Log(fmt.Sprintf("DEVICE prefill LADDER layer%-2d %s", layer, detail))
		}
	}
	var worstAll float64
	worstLayer := 0
	for i, w := range worstByLayer {
		if w > worstAll {
			worstAll, worstLayer = w, i
		}
	}
	l.Log(fmt.Sprintf("DEVICE prefill LADDER PASS 32/32 layers vs golden boundaries; worst|d|=%.3e at layer %d", worstAll, worstLayer))

	// ---- chained: all 32 layers from host prefill embeds (no golden anchors)-
	terminal, err := routedlm.LoadTerminalWeights(pc.Src, cfg, vqaserve.Binding)
	if err != nil {
		return err
	}
	row := append([]float32(nil), pc.Prefill...)
	var chainWorst float64
	chainStart := time.Now()
	for layer := range cfg.NumHiddenLayers {
		w, err := routedlm.LoadLayerWeights(pc.Src, cfg, vqaserve.Binding, layer)
		if err != nil {
			return err
		}
		feeds := map[*tensor.Tensor]driver.DevicePtr{}
		if err := vqaserve.BindLayerBranches(ctx, &allocations, feeds, g.Text, g.Vision, w); err != nil {
			return err
		}
		hostFeeds := map[*tensor.Tensor]reference.Value{
			g.Row:      {Shape: rowShape, Data: row},
			g.MaskText: {Shape: maskShape, Data: pc.MaskText},
			g.MaskVis:  {Shape: maskShape, Data: pc.MaskVis},
		}
		inputs := compiled.NewDeviceInputs()
		for node, pointer := range feeds {
			if err := inputs.Set(node, pointer); err != nil {
				return err
			}
		}
		out, err := exe.ExecuteCompiled(ctx, compiled, hostFeeds, inputs)
		if err != nil {
			_ = allocations.Close(ctx)
			return fmt.Errorf("device prefill chained layer %d: %w", layer, err)
		}
		row = append([]float32(nil), out[g.Output].Data...)
		_ = allocations.Close(ctx)
		golden, err := loadGoldenJSON[motGolden](l.fixturesDir, "rxbrain_vqa_mot_layer"+strconv.Itoa(layer)+"_output_golden.json")
		if err != nil {
			return err
		}
		gt := golden.MoTTensors["layer"+strconv.Itoa(layer)+"_output"]
		var worst float64
		for i, idx := range gt.ProbeIndex {
			if d := math.Abs(float64(row[idx]) - gt.ProbeValue[i]); d > worst {
				worst = d
			}
		}
		chainWorst = max(chainWorst, worst)
	}
	chainWall := time.Since(chainStart)

	// terminal top token from the chained last row.
	lastRow := row[(pc.PromptLen-1)*H : pc.PromptLen*H]
	topID, topLogit, err := routedlm.TerminalTopToken(lastRow, cfg, terminal, tensor.FirstOffset)
	if err != nil {
		return err
	}
	dg, err := loadGoldenJSON[decodeStepsGolden](l.fixturesDir, "rxbrain_vqa_decode_steps_golden.json")
	if err != nil {
		return err
	}
	l.Log(fmt.Sprintf("DEVICE prefill CHAINED 32 layers from prefill embeds: worst|d| vs golden=%.3e wall=%s", chainWorst, chainWall.Round(time.Millisecond)))
	if topID != dg.FirstToken {
		l.Log(fmt.Sprintf("DEVICE prefill CHAINED terminal top=%d logit=%.4f != golden first token=%d", topID, topLogit, dg.FirstToken))
		return fmt.Errorf("device prefill chained terminal top %d != golden first token %d", topID, dg.FirstToken)
	}
	l.Log(fmt.Sprintf("DEVICE prefill CHAINED terminal top=%d logit=%.4f == golden first token (EXACT)", topID, topLogit))
	l.Log("DEVICE prefill LANE GREEN")
	return nil
}
