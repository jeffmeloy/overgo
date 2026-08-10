package main

// Device decode mode: runs the full 32-layer branch-routed MoT decode step as
// an overgo tensor-graph on the CUDA device through the generic executor, with
// BF16-resident weights and device-resident fixed-capacity KV appended in place
// across steps (WriteCache append + CacheTokenOffset; a single compiled graph
// replayed with only rope positions / attention window / append offset updated
// as runtime attributes). Decode is text-branch only (the standard GQA +
// per-head QK-norm path). Verified step-by-step against the host reference
// (routedlm.DecodeLayerStack, 44/44 fixture-exact) and the golden decode chain,
// then measured for ms/token and peak MiB against adaptive's 28-31 ms/token.

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
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// decodeHarness: host-side decode oracle state (prompt KV from golden
// boundaries) shared by the device path.
type decodeHarness struct {
	cfg        routedlm.Config
	src        *safetensors.Source
	terminal   routedlm.TerminalWeights
	weights    []routedlm.LayerWeights
	resident   []*routedlm.ResidentKV
	segments   [][2]int
	decodeMask []int
	promptLen  int
	dg         decodeStepsGolden
	tg         terminalGolden
}

// loadDecodeHarness: reproduce the host decode setup (main.go decode_steps):
// prompt embeds from the golden block_last + merger, per-layer resident KV from
// each layer's golden input boundary.
func loadDecodeHarness(l *ladder) (*decodeHarness, error) {
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

	dg, err := loadGoldenJSON[decodeStepsGolden](l.fixturesDir, "rxbrain_vqa_decode_steps_golden.json")
	if err != nil {
		src.Close()
		return nil, err
	}
	tg, err := loadGoldenJSON[terminalGolden](l.fixturesDir, "rxbrain_vqa_terminal_golden.json")
	if err != nil {
		src.Close()
		return nil, err
	}
	terminal, err := routedlm.LoadTerminalWeights(src, cfg, binding)
	if err != nil {
		src.Close()
		return nil, err
	}

	weights := make([]routedlm.LayerWeights, cfg.NumHiddenLayers)
	resident := make([]*routedlm.ResidentKV, cfg.NumHiddenLayers)
	for layer := 0; layer < cfg.NumHiddenLayers; layer++ {
		w, err := routedlm.LoadLayerWeights(src, cfg, binding, layer)
		if err != nil {
			src.Close()
			return nil, err
		}
		weights[layer] = w
		hidden := prefill
		if layer > 0 {
			prev, err := loadGoldenJSON[motGolden](l.fixturesDir, "rxbrain_vqa_mot_layer"+strconv.Itoa(layer-1)+"_output_golden.json")
			if err != nil {
				src.Close()
				return nil, err
			}
			prevKey := "layer" + strconv.Itoa(layer-1) + "_output"
			hidden, err = loadTensorAsset(l.fixturesDir, prev.LayerOutputAsset, prev.MoTTensors[prevKey])
			if err != nil {
				src.Close()
				return nil, err
			}
		}
		resident[layer], err = routedlm.PromptResidentKV(hidden, mask, cfg, w)
		if err != nil {
			src.Close()
			return nil, err
		}
	}
	segments := routedlm.VisualSegments(mask)
	decodeMask := make([]int, promptLen+len(dg.DecodeSteps)+1)
	copy(decodeMask, mask)
	return &decodeHarness{
		cfg: cfg, src: src, terminal: terminal, weights: weights, resident: resident,
		segments: segments, decodeMask: decodeMask, promptLen: promptLen, dg: dg, tg: tg,
	}, nil
}

// devResident: a persistent device KV buffer pair for one layer.
type devKV struct {
	key, value   *executor.DeviceBuffer
	keyV, valueV executor.DeviceValue
}

// runDeviceDecode: device 32-layer decode parity + measurement.
func runDeviceDecode(l *ladder) error {
	ctx := context.Background()
	baseMiB := gpuUsedMiB()
	l.log(fmt.Sprintf("DEVICE decode START gpu.used=%dMiB free=%dMiB", baseMiB, gpuFreeMiB()))

	h, err := loadDecodeHarness(l)
	if err != nil {
		return err
	}
	defer h.src.Close()
	cfg := h.cfg
	H := cfg.HiddenSize
	hd := cfg.HeadDim
	kvHeads := cfg.NumKeyValueHeads
	kvOut := kvHeads * hd
	vocab := cfg.VocabSize
	steps := len(h.dg.DecodeSteps)
	capacity := uint32(h.promptLen + steps + 1)
	l.log(fmt.Sprintf("DEVICE decode harness ready prompt_len=%d steps=%d capacity=%d layers=%d", h.promptLen, steps, capacity, cfg.NumHiddenLayers))

	g, err := routedlm.BuildDeviceDecodeGraph(cfg, capacity)
	if err != nil {
		return err
	}

	// ---- compile: logits + per-layer cache append outputs (retained in place)
	outputs := []*tensor.Tensor{g.Logits}
	for layer := range g.Nodes {
		outputs = append(outputs, g.Nodes[layer].KeyAppend, g.Nodes[layer].ValueAppend)
	}
	compiled, err := executor.Compile(outputs...)
	if err != nil {
		return fmt.Errorf("device decode compile: %w", err)
	}

	worker, err := device.New(0)
	if err != nil {
		return fmt.Errorf("device decode worker: %w", err)
	}
	defer worker.Close()
	exe, err := executor.NewWithWorker(worker)
	if err != nil {
		return fmt.Errorf("device decode executor: %w", err)
	}
	defer exe.Close()

	// ---- upload BF16/F32 weights as persistent device feeds -----------------
	var weightPtrs []driver.DevicePtr
	upload := func(b []byte) (driver.DevicePtr, error) {
		var ptr driver.DevicePtr
		err := worker.Do(ctx, func(state *device.State) error {
			p, e := state.Driver.MemAlloc(uint64(len(b)))
			if e != nil {
				return e
			}
			if e := state.Driver.MemcpyHtoD(p, b); e != nil {
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
	bindMat := func(node *tensor.Tensor, m routedlm.BF16Matrix) error {
		p, e := upload(driver.Bytes(m.Data))
		if e != nil {
			return e
		}
		deviceFeeds[node] = p
		return nil
	}
	bindVec := func(node *tensor.Tensor, v []float32) error {
		p, e := upload(driver.Bytes(v))
		if e != nil {
			return e
		}
		deviceFeeds[node] = p
		return nil
	}
	for layer := 0; layer < cfg.NumHiddenLayers; layer++ {
		w := h.weights[layer]
		in := g.Inputs[layer]
		if err := firstErr(
			bindVec(in.InputNorm, w.InputNorm.Text),
			bindMat(in.Q, w.QKV.QText), bindMat(in.K, w.QKV.KText), bindMat(in.V, w.QKV.VText), bindMat(in.O, w.QKV.OText),
			bindVec(in.QNorm, w.QKV.QNorm[0]), bindVec(in.KNorm, w.QKV.KNorm[0]),
			bindVec(in.PostNorm, w.Output.PostText),
			bindMat(in.Gate, w.Output.GateText), bindMat(in.Up, w.Output.UpText), bindMat(in.Down, w.Output.DownText),
		); err != nil {
			return fmt.Errorf("device decode weight upload layer %d: %w", layer, err)
		}
	}
	// terminal: final norm (F32) + head (native BF16).
	if err := bindVec(g.FinalNorm, h.terminal.FinalNorm[0]); err != nil {
		return err
	}
	if h.terminal.Head.DType != "BF16" {
		return fmt.Errorf("device decode: head dtype %s, want BF16", h.terminal.Head.DType)
	}
	headBytes := make([]byte, vocab*H*2)
	if _, err := h.terminal.Head.ReadAt(headBytes, 0); err != nil {
		return fmt.Errorf("device decode: head read: %w", err)
	}
	if p, e := upload(headBytes); e != nil {
		return e
	} else {
		deviceFeeds[g.Head] = p
	}

	// ---- device-resident KV: capacity buffers seeded with the prompt K/V -----
	kvShape := tensor.MustShape(uint64(hd), uint64(kvHeads), uint64(capacity))
	kvBytes, _ := kvShape.Bytes(dtype.F32)
	caches := make([]devKV, cfg.NumHiddenLayers)
	targets := compiled.NewRetainedTargets()
	for layer := 0; layer < cfg.NumHiddenLayers; layer++ {
		kb, e := exe.AllocateDeviceBuffer(ctx, kvBytes)
		if e != nil {
			return fmt.Errorf("device decode kv key alloc layer %d: %w", layer, e)
		}
		vb, e := exe.AllocateDeviceBuffer(ctx, kvBytes)
		if e != nil {
			return fmt.Errorf("device decode kv value alloc layer %d: %w", layer, e)
		}
		kv := devKV{key: kb, value: vb}
		if kv.keyV, e = kb.Value(kvShape); e != nil {
			return e
		}
		if kv.valueV, e = vb.Value(kvShape); e != nil {
			return e
		}
		// Seed prompt rows [0:promptLen). Host resident layout
		// [token*kvOut + head*hd + d] matches ggml [hd,kvHeads,cap] exactly.
		seedK := h.resident[layer].Keys[:h.promptLen*kvOut]
		seedV := h.resident[layer].Values[:h.promptLen*kvOut]
		if e := worker.Do(ctx, func(state *device.State) error {
			if e := state.Driver.MemcpyHtoD(kv.keyV.Pointer, driver.Bytes(seedK)); e != nil {
				return e
			}
			return state.Driver.MemcpyHtoD(kv.valueV.Pointer, driver.Bytes(seedV))
		}); e != nil {
			return fmt.Errorf("device decode kv seed layer %d: %w", layer, e)
		}
		caches[layer] = kv
		deviceFeeds[g.Inputs[layer].PastKey] = kv.keyV.Pointer
		deviceFeeds[g.Inputs[layer].PastValue] = kv.valueV.Pointer
		if e := targets.Set(g.Nodes[layer].KeyAppend, kv.keyV); e != nil {
			return fmt.Errorf("device decode kv key target layer %d: %w", layer, e)
		}
		if e := targets.Set(g.Nodes[layer].ValueAppend, kv.valueV); e != nil {
			return fmt.Errorf("device decode kv value target layer %d: %w", layer, e)
		}
	}
	defer func() {
		for _, kv := range caches {
			if kv.key != nil {
				_ = kv.key.Release(ctx)
			}
			if kv.value != nil {
				_ = kv.value.Release(ctx)
			}
		}
	}()
	afterLoadMiB := gpuUsedMiB()
	l.log(fmt.Sprintf("DEVICE decode residency gpu.used %d->%dMiB (delta=%dMiB) free=%dMiB kv_capacity_bytes=%d/layer",
		baseMiB, afterLoadMiB, afterLoadMiB-baseMiB, gpuFreeMiB(), kvBytes))

	// ---- per-step runtime attribute updater --------------------------------
	attrs := compiled.NewRuntimeAttributes()
	setStep := func(tokenPos int) error {
		pos := []uint32{uint32(tokenPos)}
		for layer := range g.Nodes {
			n := g.Nodes[layer]
			ra := n.QRope.Attrs.(tensor.RoPEAttributes)
			ra.Positions = pos
			if e := attrs.Set(n.QRope, ra); e != nil {
				return e
			}
			rk := n.KRope.Attrs.(tensor.RoPEAttributes)
			rk.Positions = pos
			if e := attrs.Set(n.KRope, rk); e != nil {
				return e
			}
			aa := n.Attention.Attrs.(tensor.AttentionAttributes)
			aa.QueryStart = uint32(tokenPos)
			aa.KeyValueTokens = uint32(tokenPos + 1)
			if e := attrs.Set(n.Attention, aa); e != nil {
				return e
			}
			ck := n.KeyAppend.Attrs.(tensor.CacheAppendAttributes)
			ck.Offset = uint32(tokenPos)
			if e := attrs.Set(n.KeyAppend, ck); e != nil {
				return e
			}
			cv := n.ValueAppend.Attrs.(tensor.CacheAppendAttributes)
			cv.Offset = uint32(tokenPos)
			if e := attrs.Set(n.ValueAppend, cv); e != nil {
				return e
			}
		}
		return nil
	}

	statsBefore, _ := worker.ExecutionStats(ctx)

	// ---- decode loop: device vs host oracle vs golden ----------------------
	tokenID := h.dg.FirstToken
	if tokenID != h.tg.LastLogits.TopIndex[0] {
		return fmt.Errorf("decode first token %d != terminal top %d", tokenID, h.tg.LastLogits.TopIndex[0])
	}
	generated := []int{tokenID}
	var worstDevHost float64
	for step := 0; step < steps; step++ {
		tokenPos := h.promptLen + step
		golden := h.dg.DecodeSteps[step]
		if golden.Position != tokenPos || golden.TokenIn != tokenID {
			return fmt.Errorf("step %d golden pos=%d token_in=%d have pos=%d token=%d", step, golden.Position, golden.TokenIn, tokenPos, tokenID)
		}
		embedding, err := routedlm.EmbeddingRows(h.src, cfg, binding, []int{tokenID})
		if err != nil {
			return err
		}
		// host oracle (mutates host resident cache).
		hostRow, err := routedlm.DecodeLayerStack(embedding, h.decodeMask, h.segments, h.resident, tokenPos, cfg, h.weights)
		if err != nil {
			return fmt.Errorf("host oracle step %d: %w", step, err)
		}
		hostTop, _, err := routedlm.TerminalTopToken(hostRow, cfg, h.terminal, 0)
		if err != nil {
			return err
		}

		// device step.
		if err := setStep(tokenPos); err != nil {
			return err
		}
		hostFeeds := map[*tensor.Tensor]reference.Value{
			g.Embedding: {Shape: tensor.MustShape(uint64(H), 1), Data: embedding},
		}
		retained, err := exe.ExecuteRetainedCompiledParameterized(ctx, compiled, hostFeeds, deviceFeeds, targets, attrs)
		if err != nil {
			return fmt.Errorf("device execute step %d: %w", step, err)
		}
		logitsVal, err := retained.CopyToHost(ctx, g.Logits)
		if err != nil {
			return fmt.Errorf("device logits copy step %d: %w", step, err)
		}
		_ = retained.Release(ctx)
		devLogits := logitsVal.Data
		devTop := argmaxF32(devLogits)

		// device-vs-host logits at the golden probe + top indices.
		probeIDs := append([]int{}, golden.Logits.ProbeIndex...)
		probeIDs = append(probeIDs, golden.Logits.TopIndex...)
		_, hostProbe, err := routedlm.TerminalProbeValues(hostRow, cfg, h.terminal, 0, nil, probeIDs)
		if err != nil {
			return err
		}
		for i, id := range probeIDs {
			if id < 0 || id >= len(devLogits) {
				return fmt.Errorf("step %d probe id %d out of range", step, id)
			}
			if d := math.Abs(float64(devLogits[id]) - float64(hostProbe[i])); d > worstDevHost {
				worstDevHost = d
			}
		}
		if devTop != hostTop {
			return fmt.Errorf("step %d device argmax %d != host oracle argmax %d", step, devTop, hostTop)
		}
		if devTop != golden.NextToken || devTop != h.dg.GeneratedTokens[step+1] {
			return fmt.Errorf("step %d device next token %d != golden %d/%d", step, devTop, golden.NextToken, h.dg.GeneratedTokens[step+1])
		}
		generated = append(generated, devTop)
		tokenID = devTop
	}
	if err := requireIntSliceEqual("device decode chain", generated, h.dg.GeneratedTokens); err != nil {
		return err
	}
	l.log(fmt.Sprintf("DEVICE decode EXACT 12-step chain=%v", generated))
	l.log(fmt.Sprintf("DEVICE decode device-vs-host worst|d|=%.3e over probe+top logits", worstDevHost))

	// ---- measurement: replay the final-position step ------------------------
	finalPos := h.promptLen + steps - 1
	if err := setStep(finalPos); err != nil {
		return err
	}
	lastEmbed, err := routedlm.EmbeddingRows(h.src, cfg, binding, []int{h.dg.DecodeSteps[steps-1].TokenIn})
	if err != nil {
		return err
	}
	measFeeds := map[*tensor.Tensor]reference.Value{
		g.Embedding: {Shape: tensor.MustShape(uint64(H), 1), Data: lastEmbed},
	}
	const warm, iters = 5, 50
	for i := 0; i < warm; i++ {
		r, e := exe.ExecuteRetainedCompiledParameterized(ctx, compiled, measFeeds, deviceFeeds, targets, attrs)
		if e != nil {
			return e
		}
		_ = r.Release(ctx)
	}
	start := time.Now()
	for i := 0; i < iters; i++ {
		r, e := exe.ExecuteRetainedCompiledParameterized(ctx, compiled, measFeeds, deviceFeeds, targets, attrs)
		if e != nil {
			return e
		}
		_ = r.Release(ctx)
	}
	perTok := time.Since(start) / iters
	statsAfter, _ := worker.ExecutionStats(ctx)
	dInst := statsAfter.GraphInstantiations - statsBefore.GraphInstantiations
	dUpd := statsAfter.GraphUpdates - statsBefore.GraphUpdates
	dLaunch := statsAfter.GraphLaunches - statsBefore.GraphLaunches
	peakMiB := gpuUsedMiB()
	l.log(fmt.Sprintf("DEVICE decode MEASURE %.3f ms/token (32 layers + KV resident, replayed %d x) peak gpu.used=%dMiB (delta=%dMiB vs base) free=%dMiB",
		float64(perTok.Microseconds())/1000.0, iters, peakMiB, peakMiB-baseMiB, gpuFreeMiB()))
	l.log(fmt.Sprintf("DEVICE decode REPLAY over %d steps + %d measure iters: graph_launches=%d graph_instantiations=%d graph_updates=%d (single compiled graph, per-step runtime attrs only: rope pos, attn window, cache offset)", steps, iters+warm, dLaunch, dInst, dUpd))
	l.log("DEVICE decode LANE GREEN")
	return nil
}

func firstErr(errs ...error) error {
	for _, e := range errs {
		if e != nil {
			return e
		}
	}
	return nil
}

func argmaxF32(v []float32) int {
	best, bestI := float32(math.Inf(-1)), -1
	for i, x := range v {
		if x > best {
			best, bestI = x, i
		}
	}
	return bestI
}
