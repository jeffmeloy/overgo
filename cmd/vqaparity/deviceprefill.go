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
)

// prefillContext: shared prompt state for the device prefill / full pipeline.
type prefillContext struct {
	cfg                routedlm.Config
	src                *safetensors.Source
	spec               patchtower.Spec
	blockLast          []float32
	merger             patchtower.MergerWeights
	gridT              int
	gridH              int
	gridW              int
	imageRows          int
	inputIDs           []int
	imageMaskPositions []int
	mask               []int
	promptLen          int
	prefill            []float32 // host prefill embeds [tokens, H]
	segments           [][2]int
	blocks             []routedlm.PrefillAttnBlock
	maskText           []float32 // [tokens] 1 at text rows
	maskVis            []float32 // [tokens] 1 at vision rows
}

// newPrefillContext: builds the prompt state from processor-derived inputs
// (input ids + image-mask positions + image grid) — the golden-free serving
// path. The image-feature block_last and host prefill embeds are produced
// on-device by the pipeline, so they are left nil here. imageRows is derived
// from the grid and cross-checked against the image-token count.
func newPrefillContext(modelDir string, inputIDs, imageMaskPositions []int, gridT, gridH, gridW int) (*prefillContext, error) {
	spec, err := patchtower.LoadSpec(modelDir)
	if err != nil {
		return nil, err
	}
	cfg, err := routedlm.LoadConfig(modelDir, binding)
	if err != nil {
		return nil, err
	}
	src, err := safetensors.OpenSource(modelDir)
	if err != nil {
		return nil, err
	}
	merger, err := patchtower.LoadMergerWeights(src, spec)
	if err != nil {
		src.Close()
		return nil, err
	}
	if gridH%spec.MergeSize != 0 || gridW%spec.MergeSize != 0 {
		src.Close()
		return nil, fmt.Errorf("prefill context: grid [%d,%d] not divisible by merge=%d", gridH, gridW, spec.MergeSize)
	}
	imageRows := gridT * (gridH / spec.MergeSize) * (gridW / spec.MergeSize)
	if imageRows != len(imageMaskPositions) {
		src.Close()
		return nil, fmt.Errorf("prefill context: imageRows=%d != image-token count=%d", imageRows, len(imageMaskPositions))
	}
	promptLen := len(inputIDs)
	mask := routedlm.ModalityMask(promptLen, imageMaskPositions)
	segments := routedlm.VisualSegments(mask)
	blocks := routedlm.PrefillAttnBlocks(segments, promptLen)
	maskText := make([]float32, promptLen)
	maskVis := make([]float32, promptLen)
	for i, m := range mask {
		if m == 0 {
			maskText[i] = 1
		} else {
			maskVis[i] = 1
		}
	}
	return &prefillContext{
		cfg: cfg, src: src, spec: spec, merger: merger,
		gridT: gridT, gridH: gridH, gridW: gridW, imageRows: imageRows,
		inputIDs: inputIDs, imageMaskPositions: imageMaskPositions,
		mask: mask, promptLen: promptLen, segments: segments, blocks: blocks,
		maskText: maskText, maskVis: maskVis,
	}, nil
}

func loadPrefillContext(l *campaignContext) (*prefillContext, error) {
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
	gridT, gridH, gridW := vg.ImageGridTHW[0], vg.ImageGridTHW[1], vg.ImageGridTHW[2]
	spec, err := patchtower.LoadSpec(l.modelDir)
	if err != nil {
		return nil, err
	}
	cfg, err := routedlm.LoadConfig(l.modelDir, binding)
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
	prefill, err := routedlm.PrefillValues(src, cfg, binding, fg.InputIDs, fg.PrefillTensors.InputImageMaskPositions, imageRows, func(dst []float32, ordinal int) error {
		patchtower.MergerRowInto(dst, blockLast, ordinal, gridH, gridW, spec, merger, &scratch)
		return nil
	})
	if err != nil {
		src.Close()
		return nil, err
	}
	mask := fg.PrefillTensors.ModalityMask
	promptLen := fg.PromptLen
	segments := routedlm.VisualSegments(mask)
	blocks := routedlm.PrefillAttnBlocks(segments, promptLen)
	maskText := make([]float32, promptLen)
	maskVis := make([]float32, promptLen)
	for i, m := range mask {
		if m == 0 {
			maskText[i] = 1
		} else {
			maskVis[i] = 1
		}
	}
	return &prefillContext{
		cfg: cfg, src: src, spec: spec, blockLast: blockLast, merger: merger,
		gridT: gridT, gridH: gridH, gridW: gridW, imageRows: imageRows,
		inputIDs: fg.InputIDs, imageMaskPositions: fg.PrefillTensors.InputImageMaskPositions,
		mask: mask, promptLen: promptLen, prefill: prefill, segments: segments, blocks: blocks,
		maskText: maskText, maskVis: maskVis,
	}, nil
}

func bindPrefillBranch(ctx context.Context, allocations *device.AllocationSet, feeds map[*tensor.Tensor]driver.DevicePtr, nodes routedlm.DevicePrefillBranch,
	inputNorm []float32, q, k, v, o routedlm.BF16Matrix, qNorm, kNorm, postNorm []float32, gate, up, down routedlm.BF16Matrix) error {
	bindV := func(node *tensor.Tensor, val []float32) error {
		ptr, e := allocations.Upload(ctx, driver.Bytes(val))
		if e != nil {
			return e
		}
		feeds[node] = ptr
		return nil
	}
	bindM := func(node *tensor.Tensor, m routedlm.BF16Matrix) error {
		ptr, e := allocations.Upload(ctx, driver.Bytes(m.Data))
		if e != nil {
			return e
		}
		feeds[node] = ptr
		return nil
	}
	return firstErr(
		bindV(nodes.InputNorm, inputNorm),
		bindM(nodes.Q, q), bindM(nodes.K, k), bindM(nodes.V, v), bindM(nodes.O, o),
		bindV(nodes.QNorm, qNorm), bindV(nodes.KNorm, kNorm), bindV(nodes.PostNorm, postNorm),
		bindM(nodes.Gate, gate), bindM(nodes.Up, up), bindM(nodes.Down, down),
	)
}

func runDevicePrefill(l *campaignContext) error {
	ctx := context.Background()
	l.Log("DEVICE prefill START")

	pc, err := loadPrefillContext(l)
	if err != nil {
		return err
	}
	defer pc.src.Close()
	cfg := pc.cfg
	H := cfg.HiddenSize
	l.Log(fmt.Sprintf("DEVICE prefill ctx prompt_len=%d image_rows=%d segments=%v blocks=%v", pc.promptLen, pc.imageRows, pc.segments, pc.blocks))

	g, err := routedlm.BuildDevicePrefillLayer(cfg, pc.promptLen, pc.blocks)
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

	rowShape := tensor.MustShape(uint64(H), uint64(pc.promptLen))
	maskShape := tensor.MustShape(tensor.SingletonExtent, uint64(pc.promptLen))

	// ---- ladder verification: each layer fed the previous GOLDEN boundary ---
	allocations := device.NewAllocationSet(worker)
	defer func() { _ = allocations.Close(ctx) }()
	worstByLayer := make([]float64, cfg.NumHiddenLayers)
	for layer := 0; layer < cfg.NumHiddenLayers; layer++ {
		w, err := routedlm.LoadLayerWeights(pc.src, cfg, binding, layer)
		if err != nil {
			return err
		}
		input := pc.prefill
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
		if err := bindPrefillBranch(ctx, &allocations, feeds, g.Text,
			w.InputNorm.Text, w.QKV.QText, w.QKV.KText, w.QKV.VText, w.QKV.OText,
			w.QKV.QNorm[0], w.QKV.KNorm[0], w.Output.PostText, w.Output.GateText, w.Output.UpText, w.Output.DownText); err != nil {
			return err
		}
		if err := bindPrefillBranch(ctx, &allocations, feeds, g.Vision,
			w.InputNorm.Vision, w.QKV.QVision, w.QKV.KVision, w.QKV.VVision, w.QKV.OVision,
			w.QKV.QNorm[1], w.QKV.KNorm[1], w.Output.PostVision, w.Output.GateVision, w.Output.UpVision, w.Output.DownVision); err != nil {
			return err
		}
		hostFeeds := map[*tensor.Tensor]reference.Value{
			g.Row:      {Shape: rowShape, Data: input},
			g.MaskText: {Shape: maskShape, Data: pc.maskText},
			g.MaskVis:  {Shape: maskShape, Data: pc.maskVis},
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
	terminal, err := routedlm.LoadTerminalWeights(pc.src, cfg, binding)
	if err != nil {
		return err
	}
	row := append([]float32(nil), pc.prefill...)
	var chainWorst float64
	chainStart := time.Now()
	for layer := 0; layer < cfg.NumHiddenLayers; layer++ {
		w, err := routedlm.LoadLayerWeights(pc.src, cfg, binding, layer)
		if err != nil {
			return err
		}
		feeds := map[*tensor.Tensor]driver.DevicePtr{}
		if err := bindPrefillBranch(ctx, &allocations, feeds, g.Text,
			w.InputNorm.Text, w.QKV.QText, w.QKV.KText, w.QKV.VText, w.QKV.OText,
			w.QKV.QNorm[0], w.QKV.KNorm[0], w.Output.PostText, w.Output.GateText, w.Output.UpText, w.Output.DownText); err != nil {
			return err
		}
		if err := bindPrefillBranch(ctx, &allocations, feeds, g.Vision,
			w.InputNorm.Vision, w.QKV.QVision, w.QKV.KVision, w.QKV.VVision, w.QKV.OVision,
			w.QKV.QNorm[1], w.QKV.KNorm[1], w.Output.PostVision, w.Output.GateVision, w.Output.UpVision, w.Output.DownVision); err != nil {
			return err
		}
		hostFeeds := map[*tensor.Tensor]reference.Value{
			g.Row:      {Shape: rowShape, Data: row},
			g.MaskText: {Shape: maskShape, Data: pc.maskText},
			g.MaskVis:  {Shape: maskShape, Data: pc.maskVis},
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
	lastRow := row[(pc.promptLen-1)*H : pc.promptLen*H]
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
