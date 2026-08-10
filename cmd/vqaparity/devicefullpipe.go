package main

// Device full-pipeline mode: chains the RxBrain stages entirely on the CUDA
// device with NO golden-seeded state — device merger (image features) ->
// host embedding splice -> device branch-routed prefill (32 chained MoT layers,
// exporting the per-layer resident K/V) -> terminal (first token) -> device
// 32-layer decode seeded from the DEVICE prefill K/V (replacing the golden
// PromptResidentKV) -> 12-step generation. Verified to reproduce the exact
// answer and the golden 12-step chain, then measured end-to-end (wall + peak
// MiB) against adaptive's per-request bar (e2e 18.4-23.1s, engine peak 11.97GB).
// The lever is persistent residency: decode weights + the prefilled KV stay
// resident, so the 12 decode steps replay one compiled graph.

import (
	"context"
	"fmt"
	"time"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/hfbpe"
	"overgo/internal/patchtower"
	"overgo/internal/routedlm"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func runDeviceFull(l *ladder) error {
	ctx := context.Background()
	baseMiB := gpuUsedMiB()
	l.log(fmt.Sprintf("DEVICE full START gpu.used=%dMiB free=%dMiB", baseMiB, gpuFreeMiB()))

	pc, err := loadPrefillContext(l)
	if err != nil {
		return err
	}
	defer pc.src.Close()
	cfg := pc.cfg
	H := cfg.HiddenSize
	hd := cfg.HeadDim
	kvHeads := cfg.NumKeyValueHeads
	kvOut := kvHeads * hd
	vocab := cfg.VocabSize
	O := pc.spec.OutHidden

	terminal, err := routedlm.LoadTerminalWeights(pc.src, cfg, binding)
	if err != nil {
		return err
	}
	dg, err := loadGoldenJSON[decodeStepsGolden](l.fixturesDir, "rxbrain_vqa_decode_steps_golden.json")
	if err != nil {
		return err
	}

	worker, err := device.New(0)
	if err != nil {
		return fmt.Errorf("device full worker: %w", err)
	}
	defer worker.Close()
	exe, err := executor.NewWithWorker(worker)
	if err != nil {
		return fmt.Errorf("device full executor: %w", err)
	}
	defer exe.Close()

	binder := &prefillWeightBinder{worker: worker, ctx: ctx}
	e2eStart := time.Now()

	// ================= STAGE 0: DEVICE VISION TOWER -> block_last ============
	// Host front-end (conv patch embed + interpolated pos-embed) from the image
	// pixel_values, then the 27 pre-norm blocks on device (adaptive's
	// host-front-end / device-blocks split). Produces block_last for the merger,
	// replacing the golden asset so the pipeline runs from the image.
	nPatch := pc.gridT * pc.gridH * pc.gridW
	blockLast, err := runFullVision(l, ctx, worker, exe, pc, nPatch)
	if err != nil {
		return err
	}
	l.log(fmt.Sprintf("DEVICE full STAGE0 vision tower done nPatch=%d hidden=%d", nPatch, pc.spec.Hidden))

	// ================= STAGE 1: DEVICE MERGER -> image features ==============
	mg, err := patchtower.BuildDeviceMerger(pc.spec, pc.imageRows, pc.gridT, pc.gridH, pc.gridW)
	if err != nil {
		return err
	}
	mgCompiled, err := executor.Compile(mg.Merged)
	if err != nil {
		return fmt.Errorf("device full merger compile: %w", err)
	}
	mgFeeds := map[*tensor.Tensor]driver.DevicePtr{}
	if err := firstErr(
		bindMergerV(binder, mgFeeds, mg.Proj1W, pc.merger.Proj1Weight), bindMergerV(binder, mgFeeds, mg.Proj1B, pc.merger.Proj1Bias),
		bindMergerV(binder, mgFeeds, mg.Proj2W, pc.merger.Proj2Weight), bindMergerV(binder, mgFeeds, mg.Proj2B, pc.merger.Proj2Bias),
		bindMergerV(binder, mgFeeds, mg.Pool0W, pc.merger.Pool0Weight), bindMergerV(binder, mgFeeds, mg.Pool0B, pc.merger.Pool0Bias),
		bindMergerV(binder, mgFeeds, mg.Pool2W, pc.merger.Pool2Weight), bindMergerV(binder, mgFeeds, mg.Pool2B, pc.merger.Pool2Bias),
	); err != nil {
		return err
	}
	mgHost := map[*tensor.Tensor]reference.Value{
		mg.BlockLast: {Shape: tensor.MustShape(uint64(pc.spec.Hidden), uint64(nPatch)), Data: blockLast},
	}
	mgOut, err := exe.ExecuteCompiledWithDeviceFeeds(ctx, mgCompiled, mgHost, mgFeeds)
	if err != nil {
		return fmt.Errorf("device full merger execute: %w", err)
	}
	devMerged := append([]float32(nil), mgOut[mg.Merged].Data...) // [imageRows*O] row-major
	binder.free()
	l.log(fmt.Sprintf("DEVICE full STAGE1 merger done rows=%d out=%d", pc.imageRows, O))

	// prefill embeds = embedding-table text rows + device merger image rows.
	fg, err := loadGoldenJSON[prefillGolden](l.fixturesDir, "rxbrain_vqa_prefill_golden.json")
	if err != nil {
		return err
	}
	prefillEmbeds, err := routedlm.PrefillValues(pc.src, cfg, binding, fg.InputIDs, fg.PrefillTensors.InputImageMaskPositions, pc.imageRows, func(dst []float32, ordinal int) error {
		copy(dst, devMerged[ordinal*O:(ordinal+1)*O])
		return nil
	})
	if err != nil {
		return err
	}

	// ================= STAGE 2: DEVICE PREFILL (chained) + KV export =========
	pg, err := routedlm.BuildDevicePrefillLayer(cfg, pc.promptLen, pc.blocks)
	if err != nil {
		return err
	}
	pgCompiled, err := executor.Compile(pg.Output, pg.KeyKV, pg.ValueKV)
	if err != nil {
		return fmt.Errorf("device full prefill compile: %w", err)
	}
	rowShape := tensor.MustShape(uint64(H), uint64(pc.promptLen))
	maskShape := tensor.MustShape(1, uint64(pc.promptLen))
	seedKeys := make([][]float32, cfg.NumHiddenLayers)
	seedValues := make([][]float32, cfg.NumHiddenLayers)
	row := append([]float32(nil), prefillEmbeds...)
	for layer := 0; layer < cfg.NumHiddenLayers; layer++ {
		w, err := routedlm.LoadLayerWeights(pc.src, cfg, binding, layer)
		if err != nil {
			return err
		}
		feeds := map[*tensor.Tensor]driver.DevicePtr{}
		if err := bindPrefillBranch(binder, feeds, pg.Text,
			w.InputNorm.Text, w.QKV.QText, w.QKV.KText, w.QKV.VText, w.QKV.OText,
			w.QKV.QNorm[0], w.QKV.KNorm[0], w.Output.PostText, w.Output.GateText, w.Output.UpText, w.Output.DownText); err != nil {
			return err
		}
		if err := bindPrefillBranch(binder, feeds, pg.Vision,
			w.InputNorm.Vision, w.QKV.QVision, w.QKV.KVision, w.QKV.VVision, w.QKV.OVision,
			w.QKV.QNorm[1], w.QKV.KNorm[1], w.Output.PostVision, w.Output.GateVision, w.Output.UpVision, w.Output.DownVision); err != nil {
			return err
		}
		hostFeeds := map[*tensor.Tensor]reference.Value{
			pg.Row:      {Shape: rowShape, Data: row},
			pg.MaskText: {Shape: maskShape, Data: pc.maskText},
			pg.MaskVis:  {Shape: maskShape, Data: pc.maskVis},
		}
		out, err := exe.ExecuteCompiledWithDeviceFeeds(ctx, pgCompiled, hostFeeds, feeds)
		if err != nil {
			binder.free()
			return fmt.Errorf("device full prefill layer %d: %w", layer, err)
		}
		row = append([]float32(nil), out[pg.Output].Data...)
		seedKeys[layer] = append([]float32(nil), out[pg.KeyKV].Data[:pc.promptLen*kvOut]...)
		seedValues[layer] = append([]float32(nil), out[pg.ValueKV].Data[:pc.promptLen*kvOut]...)
		binder.free()
	}
	lastRow := row[(pc.promptLen-1)*H : pc.promptLen*H]
	firstToken, firstLogit, err := routedlm.TerminalTopToken(lastRow, cfg, terminal, 0)
	if err != nil {
		return err
	}
	if firstToken != dg.FirstToken {
		return fmt.Errorf("device full prefill terminal top %d != golden first token %d", firstToken, dg.FirstToken)
	}
	l.log(fmt.Sprintf("DEVICE full STAGE2 prefill done, terminal first token=%d logit=%.4f (EXACT vs golden) KV exported for %d layers", firstToken, firstLogit, cfg.NumHiddenLayers))

	// ================= STAGE 3: DEVICE DECODE seeded from prefill KV =========
	weights := make([]routedlm.LayerWeights, cfg.NumHiddenLayers)
	for layer := 0; layer < cfg.NumHiddenLayers; layer++ {
		if weights[layer], err = routedlm.LoadLayerWeights(pc.src, cfg, binding, layer); err != nil {
			return err
		}
	}
	steps := len(dg.DecodeSteps)
	capacity := uint32(pc.promptLen + steps + 1)

	dgraph, err := routedlm.BuildDeviceDecodeGraph(cfg, capacity)
	if err != nil {
		return err
	}
	outputs := []*tensor.Tensor{dgraph.Logits}
	for layer := range dgraph.Nodes {
		outputs = append(outputs, dgraph.Nodes[layer].KeyAppend, dgraph.Nodes[layer].ValueAppend)
	}
	dCompiled, err := executor.Compile(outputs...)
	if err != nil {
		return fmt.Errorf("device full decode compile: %w", err)
	}

	// upload decode text-branch weights (resident) + terminal.
	deco := &prefillWeightBinder{worker: worker, ctx: ctx}
	defer deco.free()
	deviceFeeds := map[*tensor.Tensor]driver.DevicePtr{}
	bindVec := func(node *tensor.Tensor, v []float32) error {
		p, e := deco.upload(driver.Bytes(v))
		if e != nil {
			return e
		}
		deviceFeeds[node] = p
		return nil
	}
	bindMat := func(node *tensor.Tensor, m routedlm.BF16Matrix) error {
		p, e := deco.upload(driver.Bytes(m.Data))
		if e != nil {
			return e
		}
		deviceFeeds[node] = p
		return nil
	}
	for layer := 0; layer < cfg.NumHiddenLayers; layer++ {
		w := weights[layer]
		in := dgraph.Inputs[layer]
		if err := firstErr(
			bindVec(in.InputNorm, w.InputNorm.Text),
			bindMat(in.Q, w.QKV.QText), bindMat(in.K, w.QKV.KText), bindMat(in.V, w.QKV.VText), bindMat(in.O, w.QKV.OText),
			bindVec(in.QNorm, w.QKV.QNorm[0]), bindVec(in.KNorm, w.QKV.KNorm[0]),
			bindVec(in.PostNorm, w.Output.PostText),
			bindMat(in.Gate, w.Output.GateText), bindMat(in.Up, w.Output.UpText), bindMat(in.Down, w.Output.DownText),
		); err != nil {
			return fmt.Errorf("device full decode weight upload layer %d: %w", layer, err)
		}
	}
	if err := bindVec(dgraph.FinalNorm, terminal.FinalNorm[0]); err != nil {
		return err
	}
	if terminal.Head.DType != "BF16" {
		return fmt.Errorf("device full: head dtype %s, want BF16", terminal.Head.DType)
	}
	headBytes := make([]byte, vocab*H*2)
	if _, err := terminal.Head.ReadAt(headBytes, 0); err != nil {
		return fmt.Errorf("device full: head read: %w", err)
	}
	if p, e := deco.upload(headBytes); e != nil {
		return e
	} else {
		deviceFeeds[dgraph.Head] = p
	}

	// device-resident KV capacity buffers seeded from the DEVICE prefill K/V.
	kvShape := tensor.MustShape(uint64(hd), uint64(kvHeads), uint64(capacity))
	kvBytes, _ := kvShape.Bytes(dtype.F32)
	caches := make([]devKV, cfg.NumHiddenLayers)
	targets := dCompiled.NewRetainedTargets()
	for layer := 0; layer < cfg.NumHiddenLayers; layer++ {
		kb, e := exe.AllocateDeviceBuffer(ctx, kvBytes)
		if e != nil {
			return fmt.Errorf("device full kv key alloc layer %d: %w", layer, e)
		}
		vb, e := exe.AllocateDeviceBuffer(ctx, kvBytes)
		if e != nil {
			return fmt.Errorf("device full kv value alloc layer %d: %w", layer, e)
		}
		kv := devKV{key: kb, value: vb}
		if kv.keyV, e = kb.Value(kvShape); e != nil {
			return e
		}
		if kv.valueV, e = vb.Value(kvShape); e != nil {
			return e
		}
		if e := worker.Do(ctx, func(state *device.State) error {
			if e := state.Driver.MemcpyHtoD(kv.keyV.Pointer, driver.Bytes(seedKeys[layer])); e != nil {
				return e
			}
			return state.Driver.MemcpyHtoD(kv.valueV.Pointer, driver.Bytes(seedValues[layer]))
		}); e != nil {
			return fmt.Errorf("device full kv seed layer %d: %w", layer, e)
		}
		caches[layer] = kv
		deviceFeeds[dgraph.Inputs[layer].PastKey] = kv.keyV.Pointer
		deviceFeeds[dgraph.Inputs[layer].PastValue] = kv.valueV.Pointer
		if e := targets.Set(dgraph.Nodes[layer].KeyAppend, kv.keyV); e != nil {
			return e
		}
		if e := targets.Set(dgraph.Nodes[layer].ValueAppend, kv.valueV); e != nil {
			return e
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
	l.log(fmt.Sprintf("DEVICE full residency gpu.used %d->%dMiB (delta=%dMiB) free=%dMiB (decode weights + prefilled KV resident)", baseMiB, afterLoadMiB, afterLoadMiB-baseMiB, gpuFreeMiB()))

	attrs := dCompiled.NewRuntimeAttributes()
	setStep := func(tokenPos int) error {
		pos := []uint32{uint32(tokenPos)}
		for layer := range dgraph.Nodes {
			n := dgraph.Nodes[layer]
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

	tokenID := firstToken
	generated := []int{tokenID}
	decodeStart := time.Now()
	for step := 0; step < steps; step++ {
		tokenPos := pc.promptLen + step
		embedding, err := routedlm.EmbeddingRows(pc.src, cfg, binding, []int{tokenID})
		if err != nil {
			return err
		}
		if err := setStep(tokenPos); err != nil {
			return err
		}
		hostFeeds := map[*tensor.Tensor]reference.Value{
			dgraph.Embedding: {Shape: tensor.MustShape(uint64(H), 1), Data: embedding},
		}
		retained, err := exe.ExecuteRetainedCompiledParameterized(ctx, dCompiled, hostFeeds, deviceFeeds, targets, attrs)
		if err != nil {
			return fmt.Errorf("device full decode step %d: %w", step, err)
		}
		logitsVal, err := retained.CopyToHost(ctx, dgraph.Logits)
		if err != nil {
			return err
		}
		_ = retained.Release(ctx)
		devTop := argmaxF32(logitsVal.Data)
		if devTop != dg.GeneratedTokens[step+1] {
			return fmt.Errorf("device full decode step %d top %d != golden %d", step, devTop, dg.GeneratedTokens[step+1])
		}
		generated = append(generated, devTop)
		tokenID = devTop
	}
	decodeWall := time.Since(decodeStart)
	e2eWall := time.Since(e2eStart)
	if err := requireIntSliceEqual("device full chain", generated, dg.GeneratedTokens); err != nil {
		return err
	}

	tok, err := hfbpe.Load(l.modelDir)
	if err != nil {
		return err
	}
	text := tok.Decode(generated)
	const want = "The stovetop holds a metal pot on the left burner"
	if text != want {
		return fmt.Errorf("device full decoded text %q != %q", text, want)
	}

	statsAfter, _ := worker.ExecutionStats(ctx)
	peakMiB := gpuUsedMiB()
	l.log(fmt.Sprintf("DEVICE full ANSWER EXACT chain=%v", generated))
	l.log(fmt.Sprintf("DEVICE full TEXT %q", text))
	l.log(fmt.Sprintf("DEVICE full MEASURE e2e=%s (vision+merger+prefill+decode) decode=%s (%d steps, %.3f ms/token) peak gpu.used=%dMiB (delta=%dMiB) free=%dMiB",
		e2eWall.Round(time.Millisecond), decodeWall.Round(time.Millisecond), steps, float64(decodeWall.Microseconds())/1000.0/float64(steps), peakMiB, peakMiB-baseMiB, gpuFreeMiB()))
	l.log(fmt.Sprintf("DEVICE full REPLAY decode graph_launches=%d graph_instantiations=%d graph_updates=%d over %d steps (single compiled decode graph, per-step runtime attrs only)",
		statsAfter.GraphLaunches-statsBefore.GraphLaunches, statsAfter.GraphInstantiations-statsBefore.GraphInstantiations, statsAfter.GraphUpdates-statsBefore.GraphUpdates, steps))
	l.log(fmt.Sprintf("DEVICE full vs ADAPTIVE bar: adaptive e2e 18.4-23.1s engine peak 11.97GB; overgo e2e=%s peak=%dMiB (golden-seeded KV REPLACED by device prefill)", e2eWall.Round(time.Millisecond), peakMiB))
	l.log("DEVICE full LANE GREEN")
	return nil
}

// runFullVision: host front-end + device 27-block vision tower from the image
// pixel_values, returning the device block_last [hidden, nPatch] (host copy).
func runFullVision(l *ladder, ctx context.Context, worker *device.Worker, exe *executor.Executor, pc *prefillContext, nPatch int) ([]float32, error) {
	pgv, err := loadGoldenJSON[processorGolden](l.fixturesDir, "rxbrain_vqa_processor_golden.json")
	if err != nil {
		return nil, err
	}
	pixelValues, err := loadTensorAsset(l.fixturesDir, pgv.PixelValuesAsset, pgv.PixelValues)
	if err != nil {
		return nil, err
	}
	patchWeights, err := patchtower.LoadPatchEmbedWeights(pc.src, pc.spec)
	if err != nil {
		return nil, err
	}
	pos, err := patchtower.LoadPosEmbedTable(pc.src, pc.spec)
	if err != nil {
		return nil, err
	}
	preBlock0, err := patchtower.PreBlock0Values(pixelValues, pc.gridT, pc.gridH, pc.gridW, pc.spec, patchWeights, pos)
	if err != nil {
		return nil, err
	}
	g, err := patchtower.BuildDeviceVisionBlocks(pc.spec, nPatch)
	if err != nil {
		return nil, err
	}
	compiled, err := executor.Compile(g.BlockLast)
	if err != nil {
		return nil, fmt.Errorf("device full vision compile: %w", err)
	}
	vb := &prefillWeightBinder{worker: worker, ctx: ctx}
	defer vb.free()
	feeds := map[*tensor.Tensor]driver.DevicePtr{}
	bind := func(node *tensor.Tensor, v []float32) error {
		ptr, e := vb.upload(driver.Bytes(v))
		if e != nil {
			return e
		}
		feeds[node] = ptr
		return nil
	}
	for layer := 0; layer < pc.spec.Depth; layer++ {
		w, err := patchtower.LoadBlockWeights(pc.src, pc.spec, layer)
		if err != nil {
			return nil, err
		}
		in := g.Inputs[layer]
		if err := firstErr(
			bind(in.Norm1W, w.Norm1Weight), bind(in.Norm1B, w.Norm1Bias),
			bind(in.QKVW, w.QKVWeight), bind(in.QKVB, w.QKVBias),
			bind(in.ProjW, w.ProjWeight), bind(in.ProjB, w.ProjBias),
			bind(in.Norm2W, w.Norm2Weight), bind(in.Norm2B, w.Norm2Bias),
			bind(in.FC1W, w.FC1Weight), bind(in.FC1B, w.FC1Bias),
			bind(in.FC2W, w.FC2Weight), bind(in.FC2B, w.FC2Bias),
		); err != nil {
			return nil, fmt.Errorf("device full vision weight upload block %d: %w", layer, err)
		}
	}
	hostFeeds := map[*tensor.Tensor]reference.Value{
		g.PreBlock0: {Shape: tensor.MustShape(uint64(pc.spec.Hidden), uint64(nPatch)), Data: preBlock0},
	}
	out, err := exe.ExecuteCompiledWithDeviceFeeds(ctx, compiled, hostFeeds, feeds)
	if err != nil {
		return nil, fmt.Errorf("device full vision execute: %w", err)
	}
	return append([]float32(nil), out[g.BlockLast].Data...), nil
}

func bindMergerV(p *prefillWeightBinder, feeds map[*tensor.Tensor]driver.DevicePtr, node *tensor.Tensor, v []float32) error {
	ptr, e := p.upload(driver.Bytes(v))
	if e != nil {
		return e
	}
	feeds[node] = ptr
	return nil
}
