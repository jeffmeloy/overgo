package main

// Device full-pipeline mode: chains the RxBrain stages entirely on the CUDA
// device with NO golden-seeded state — device vision tower -> device merger
// (image features) -> host embedding splice -> device branch-routed prefill (32
// chained MoT layers, exporting the per-layer resident K/V) -> terminal (first
// token) -> device 32-layer decode seeded from the DEVICE prefill K/V -> N-step
// generation. Verified to reproduce the exact answer and the golden 12-step
// chain, then measured end-to-end (wall + peak MiB) against adaptive's
// per-request bar (e2e 18.4-23.1s, engine peak 11.97GB). The lever is
// persistent residency: decode weights + the prefilled KV stay resident, so the
// decode steps replay one compiled graph.
//
// runFullPipeline is the single owner of the stage orchestration. The
// -device-full harness (runDeviceFull) drives it with golden pixel_values + a
// per-step golden verifier; the activated recipe serve (runRecipeServe) drives
// it with processor-derived pixel_values and a decode-until-EOS budget.

import (
	"context"
	"fmt"
	"time"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/patchtower"
	"overgo/internal/routedlm"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// fullOpts: how runFullPipeline drives the decode loop.
type fullOpts struct {
	// verify (harness parity): fixed step count = len(DecodeSteps), first token
	// and every step top asserted against the golden chain.
	verify *decodeStepsGolden
	// serve budget: maxSteps caps generation, eosIDs stop it (token excluded).
	maxSteps int
	eosIDs   []int
}

// fullResult: the pipeline outcome the caller decodes + reports.
type fullResult struct {
	firstToken int
	generated  []int
	e2eWall    time.Duration
	decodeWall time.Duration
	steps      int
	launches   uint64
	instantis  uint64
	updates    uint64
}

func runDeviceFull(l *campaignContext) error {
	ctx := context.Background()
	pc, err := loadPrefillContext(l)
	if err != nil {
		return err
	}
	defer pc.src.Close()

	// golden pixel_values (harness input) + per-step golden verifier.
	pgv, err := loadGoldenJSON[processorGolden](l.fixturesDir, "rxbrain_vqa_processor_golden.json")
	if err != nil {
		return err
	}
	pixelValues, err := loadTensorAsset(l.fixturesDir, pgv.PixelValuesAsset, pgv.PixelValues)
	if err != nil {
		return err
	}
	dg, err := loadGoldenJSON[decodeStepsGolden](l.fixturesDir, "rxbrain_vqa_decode_steps_golden.json")
	if err != nil {
		return err
	}

	worker, err := device.New(device.DefaultOrdinal())
	if err != nil {
		return fmt.Errorf("device full worker: %w", err)
	}
	defer worker.Close()
	exe, err := executor.NewWithWorker(worker)
	if err != nil {
		return fmt.Errorf("device full executor: %w", err)
	}
	defer exe.Close()

	res, err := runFullPipeline(l, ctx, worker, exe, pc, pixelValues, fullOpts{verify: &dg})
	if err != nil {
		return err
	}
	if err := requireIntSliceEqual("device full chain", res.generated, dg.GeneratedTokens); err != nil {
		return err
	}
	text, err := decodeChain(l.modelDir, res.generated)
	if err != nil {
		return err
	}
	const want = "The stovetop holds a metal pot on the left burner"
	if text != want {
		return fmt.Errorf("device full decoded text %q != %q", text, want)
	}
	l.Log(fmt.Sprintf("DEVICE full ANSWER EXACT chain=%v", res.generated))
	l.Log(fmt.Sprintf("DEVICE full TEXT %q", text))
	l.Log(fmt.Sprintf("DEVICE full MEASURE e2e=%s (vision+merger+prefill+decode) decode=%s (%d steps, %.3f ms/token)",
		res.e2eWall.Round(time.Millisecond), res.decodeWall.Round(time.Millisecond), res.steps, float64(res.decodeWall.Microseconds())/1000.0/float64(res.steps)))
	l.Log(fmt.Sprintf("DEVICE full REPLAY decode graph_launches=%d graph_instantiations=%d graph_updates=%d over %d steps (single compiled decode graph, per-step runtime attrs only)",
		res.launches, res.instantis, res.updates, res.steps))
	l.Log(fmt.Sprintf("DEVICE full vs ADAPTIVE e2e bar 18.4-23.1s: overgo=%s (golden-seeded KV REPLACED by device prefill)", res.e2eWall.Round(time.Millisecond)))
	l.Log("DEVICE full LANE GREEN")
	return nil
}

// runFullPipeline: the shared device VQA serving pipeline. pc carries the prompt
// state (from goldens in harness mode, from the processor in serve mode);
// pixelValues is the preprocessed image. Returns the generated token chain +
// timing/residency measurements. Logs the per-stage boundaries.
func runFullPipeline(
	l *campaignContext, ctx context.Context,
	worker *device.Worker,
	exe *executor.Executor,
	pc *prefillContext,
	pixelValues []float32,
	opts fullOpts,
) (fullResult, error) {
	var res fullResult
	l.Log("DEVICE full START")

	cfg := pc.cfg
	H := cfg.HiddenSize
	hd := cfg.HeadDim
	kvHeads := cfg.NumKeyValueHeads
	O := pc.spec.OutHidden

	terminal, err := routedlm.LoadTerminalWeights(pc.src, cfg, binding)
	if err != nil {
		return res, err
	}

	allocations := device.NewAllocationSet(worker)
	defer func() { _ = allocations.Close(ctx) }()
	e2eStart := time.Now()

	// ================= STAGE 0: DEVICE VISION TOWER -> block_last ============
	nPatch := pc.gridT * pc.gridH * pc.gridW
	blockLast, err := runFullVision(l, ctx, worker, exe, pc, pixelValues, nPatch)
	if err != nil {
		return res, err
	}
	l.Log(fmt.Sprintf("DEVICE full STAGE0 vision tower done nPatch=%d hidden=%d", nPatch, pc.spec.Hidden))

	// ================= STAGE 1: DEVICE MERGER -> image features ==============
	devMerged, err := runFullMerger(ctx, exe, &allocations, pc, blockLast, nPatch, O)
	if err != nil {
		return res, err
	}
	l.Log(fmt.Sprintf("DEVICE full STAGE1 merger done rows=%d out=%d", pc.imageRows, O))

	// prefill embeds = embedding-table text rows + device merger image rows.
	prefillEmbeds, err := routedlm.PrefillValues(pc.src, cfg, binding, pc.inputIDs, pc.imageMaskPositions, pc.imageRows, func(dst []float32, ordinal int) error {
		copy(dst, devMerged[ordinal*O:(ordinal+1)*O])
		return nil
	})
	if err != nil {
		return res, err
	}

	// ================= STAGE 2: DEVICE PREFILL (chained) + KV export =========
	lastRow, seedKeys, seedValues, err := runFullPrefill(ctx, exe, &allocations, pc, prefillEmbeds)
	if err != nil {
		return res, err
	}
	firstToken, firstLogit, err := routedlm.TerminalTopToken(lastRow, cfg, terminal, tensor.FirstOffset)
	if err != nil {
		return res, err
	}
	if opts.verify != nil && firstToken != opts.verify.FirstToken {
		return res, fmt.Errorf("device full prefill terminal top %d != golden first token %d", firstToken, opts.verify.FirstToken)
	}
	res.firstToken = firstToken
	l.Log(fmt.Sprintf("DEVICE full STAGE2 prefill done, terminal first token=%d logit=%.4f KV exported for %d layers", firstToken, firstLogit, cfg.NumHiddenLayers))

	// ================= STAGE 3: DEVICE DECODE seeded from prefill KV =========
	weights := make([]routedlm.LayerWeights, cfg.NumHiddenLayers)
	for layer := 0; layer < cfg.NumHiddenLayers; layer++ {
		if weights[layer], err = routedlm.LoadLayerWeights(pc.src, cfg, binding, layer); err != nil {
			return res, err
		}
	}
	// capacity: prompt + the maximum tokens we might generate.
	genBudget := opts.maxSteps
	if opts.verify != nil {
		genBudget = len(opts.verify.DecodeSteps)
	}
	capacity := uint32(pc.promptLen + genBudget + 1)

	dgraph, err := routedlm.BuildDeviceDecodeGraph(cfg, capacity)
	if err != nil {
		return res, err
	}
	outputs := []*tensor.Tensor{dgraph.Logits}
	for layer := range dgraph.Nodes {
		outputs = append(outputs, dgraph.Nodes[layer].KeyAppend, dgraph.Nodes[layer].ValueAppend)
	}
	dCompiled, err := executor.Compile(outputs...)
	if err != nil {
		return res, fmt.Errorf("device full decode compile: %w", err)
	}

	decodeAllocations := device.NewAllocationSet(worker)
	defer func() { _ = decodeAllocations.Close(ctx) }()
	deviceFeeds := map[*tensor.Tensor]driver.DevicePtr{}
	bindVec := func(node *tensor.Tensor, v []float32) error {
		p, e := decodeAllocations.Upload(ctx, driver.Bytes(v))
		if e != nil {
			return e
		}
		deviceFeeds[node] = p
		return nil
	}
	bindMat := func(node *tensor.Tensor, m routedlm.BF16Matrix) error {
		p, e := decodeAllocations.Upload(ctx, driver.Bytes(m.Data))
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
			return res, fmt.Errorf("device full decode weight upload layer %d: %w", layer, err)
		}
	}
	if err := bindVec(dgraph.FinalNorm, terminal.FinalNorm[tensor.FirstOffset]); err != nil {
		return res, err
	}
	if terminal.Head.DType != "BF16" {
		return res, fmt.Errorf("device full: head dtype %s, want BF16", terminal.Head.DType)
	}
	headBytes := make([]byte, terminal.Head.Size())
	if _, err := terminal.Head.ReadAt(headBytes, tensor.FirstOffset); err != nil {
		return res, fmt.Errorf("device full: head read: %w", err)
	}
	if p, e := decodeAllocations.Upload(ctx, headBytes); e != nil {
		return res, e
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
			return res, fmt.Errorf("device full kv key alloc layer %d: %w", layer, e)
		}
		vb, e := exe.AllocateDeviceBuffer(ctx, kvBytes)
		if e != nil {
			return res, fmt.Errorf("device full kv value alloc layer %d: %w", layer, e)
		}
		kv := devKV{key: kb, value: vb}
		if kv.keyV, e = kb.Value(kvShape); e != nil {
			return res, e
		}
		if kv.valueV, e = vb.Value(kvShape); e != nil {
			return res, e
		}
		if e := worker.Do(ctx, func(state *device.State) error {
			if e := state.Driver.MemcpyHtoD(kv.keyV.Pointer, driver.Bytes(seedKeys[layer])); e != nil {
				return e
			}
			return state.Driver.MemcpyHtoD(kv.valueV.Pointer, driver.Bytes(seedValues[layer]))
		}); e != nil {
			return res, fmt.Errorf("device full kv seed layer %d: %w", layer, e)
		}
		caches[layer] = kv
		deviceFeeds[dgraph.Inputs[layer].PastKey] = kv.keyV.Pointer
		deviceFeeds[dgraph.Inputs[layer].PastValue] = kv.valueV.Pointer
		if e := targets.Set(dgraph.Nodes[layer].KeyAppend, kv.keyV); e != nil {
			return res, e
		}
		if e := targets.Set(dgraph.Nodes[layer].ValueAppend, kv.valueV); e != nil {
			return res, e
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
	deviceInputs := dCompiled.NewDeviceInputs()
	for node, pointer := range deviceFeeds {
		if err := deviceInputs.Set(node, pointer); err != nil {
			return res, err
		}
	}
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

	eos := map[int]bool{}
	for _, id := range opts.eosIDs {
		eos[id] = true
	}
	tokenID := firstToken
	generated := []int{tokenID}
	decodeStart := time.Now()
	for step := 0; step < genBudget; step++ {
		tokenPos := pc.promptLen + step
		embedding, err := routedlm.EmbeddingRows(pc.src, cfg, binding, []int{tokenID})
		if err != nil {
			return res, err
		}
		if err := setStep(tokenPos); err != nil {
			return res, err
		}
		hostFeeds := map[*tensor.Tensor]reference.Value{
			dgraph.Embedding: {Shape: tensor.MustShape(uint64(H), tensor.SingletonExtent), Data: embedding},
		}
		retained, err := exe.ExecuteRetainedCompiled(ctx, dCompiled, hostFeeds, deviceInputs, targets, attrs)
		if err != nil {
			return res, fmt.Errorf("device full decode step %d: %w", step, err)
		}
		logitsVal, err := retained.CopyToHost(ctx, dgraph.Logits)
		if err != nil {
			return res, err
		}
		_ = retained.Release(ctx)
		devTop := argmaxF32(logitsVal.Data)
		if opts.verify != nil && devTop != opts.verify.GeneratedTokens[step+1] {
			return res, fmt.Errorf("device full decode step %d top %d != golden %d", step, devTop, opts.verify.GeneratedTokens[step+1])
		}
		if opts.verify == nil && eos[devTop] {
			// EOS reached: stop, do not emit the stop token.
			break
		}
		generated = append(generated, devTop)
		tokenID = devTop
	}
	res.decodeWall = time.Since(decodeStart)
	res.e2eWall = time.Since(e2eStart)
	res.steps = len(generated) - 1
	res.generated = generated
	statsAfter, _ := worker.ExecutionStats(ctx)
	res.launches = statsAfter.GraphLaunches - statsBefore.GraphLaunches
	res.instantis = statsAfter.GraphInstantiations - statsBefore.GraphInstantiations
	res.updates = statsAfter.GraphUpdates - statsBefore.GraphUpdates
	return res, nil
}

// runFullMerger: STAGE 1 — device merger from block_last to imageRows*O merged
// image-feature rows (row-major host copy).
func runFullMerger(
	ctx context.Context,
	exe *executor.Executor,
	allocations *device.AllocationSet,
	pc *prefillContext,
	blockLast []float32,
	nPatch, O int,
) ([]float32, error) {
	mg, err := patchtower.BuildDeviceMerger(pc.spec, pc.imageRows, pc.gridT, pc.gridH, pc.gridW)
	if err != nil {
		return nil, err
	}
	mgCompiled, err := executor.Compile(mg.Merged)
	if err != nil {
		return nil, fmt.Errorf("device full merger compile: %w", err)
	}
	mgFeeds := map[*tensor.Tensor]driver.DevicePtr{}
	if err := firstErr(
		bindMergerV(ctx, allocations, mgFeeds, mg.Proj1W, pc.merger.Proj1Weight), bindMergerV(ctx, allocations, mgFeeds, mg.Proj1B, pc.merger.Proj1Bias),
		bindMergerV(ctx, allocations, mgFeeds, mg.Proj2W, pc.merger.Proj2Weight), bindMergerV(ctx, allocations, mgFeeds, mg.Proj2B, pc.merger.Proj2Bias),
		bindMergerV(ctx, allocations, mgFeeds, mg.Pool0W, pc.merger.Pool0Weight), bindMergerV(ctx, allocations, mgFeeds, mg.Pool0B, pc.merger.Pool0Bias),
		bindMergerV(ctx, allocations, mgFeeds, mg.Pool2W, pc.merger.Pool2Weight), bindMergerV(ctx, allocations, mgFeeds, mg.Pool2B, pc.merger.Pool2Bias),
	); err != nil {
		return nil, err
	}
	mgHost := map[*tensor.Tensor]reference.Value{
		mg.BlockLast: {Shape: tensor.MustShape(uint64(pc.spec.Hidden), uint64(nPatch)), Data: blockLast},
	}
	mgInputs := mgCompiled.NewDeviceInputs()
	for node, pointer := range mgFeeds {
		if err := mgInputs.Set(node, pointer); err != nil {
			return nil, err
		}
	}
	mgOut, err := exe.ExecuteCompiled(ctx, mgCompiled, mgHost, mgInputs)
	if err != nil {
		return nil, fmt.Errorf("device full merger execute: %w", err)
	}
	merged := append([]float32(nil), mgOut[mg.Merged].Data...) // [imageRows*O] row-major
	_ = allocations.Close(ctx)
	return merged, nil
}

// runFullPrefill: STAGE 2 — chained branch-routed prefill over all layers,
// returning the last hidden row (terminal input) and the exported per-layer
// resident K/V.
func runFullPrefill(
	ctx context.Context,
	exe *executor.Executor,
	allocations *device.AllocationSet,
	pc *prefillContext,
	prefillEmbeds []float32,
) (lastRow []float32, seedKeys, seedValues [][]float32, err error) {
	cfg := pc.cfg
	H := cfg.HiddenSize
	kvOut := cfg.NumKeyValueHeads * cfg.HeadDim
	pg, err := routedlm.BuildDevicePrefillLayer(cfg, pc.promptLen, pc.blocks)
	if err != nil {
		return nil, nil, nil, err
	}
	pgCompiled, err := executor.Compile(pg.Output, pg.KeyKV, pg.ValueKV)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("device full prefill compile: %w", err)
	}
	rowShape := tensor.MustShape(uint64(H), uint64(pc.promptLen))
	maskShape := tensor.MustShape(tensor.SingletonExtent, uint64(pc.promptLen))
	seedKeys = make([][]float32, cfg.NumHiddenLayers)
	seedValues = make([][]float32, cfg.NumHiddenLayers)
	row := append([]float32(nil), prefillEmbeds...)
	for layer := 0; layer < cfg.NumHiddenLayers; layer++ {
		w, err := routedlm.LoadLayerWeights(pc.src, cfg, binding, layer)
		if err != nil {
			return nil, nil, nil, err
		}
		feeds := map[*tensor.Tensor]driver.DevicePtr{}
		if err := bindPrefillBranch(ctx, allocations, feeds, pg.Text,
			w.InputNorm.Text, w.QKV.QText, w.QKV.KText, w.QKV.VText, w.QKV.OText,
			w.QKV.QNorm[0], w.QKV.KNorm[0], w.Output.PostText, w.Output.GateText, w.Output.UpText, w.Output.DownText); err != nil {
			return nil, nil, nil, err
		}
		if err := bindPrefillBranch(ctx, allocations, feeds, pg.Vision,
			w.InputNorm.Vision, w.QKV.QVision, w.QKV.KVision, w.QKV.VVision, w.QKV.OVision,
			w.QKV.QNorm[1], w.QKV.KNorm[1], w.Output.PostVision, w.Output.GateVision, w.Output.UpVision, w.Output.DownVision); err != nil {
			return nil, nil, nil, err
		}
		hostFeeds := map[*tensor.Tensor]reference.Value{
			pg.Row:      {Shape: rowShape, Data: row},
			pg.MaskText: {Shape: maskShape, Data: pc.maskText},
			pg.MaskVis:  {Shape: maskShape, Data: pc.maskVis},
		}
		inputs := pgCompiled.NewDeviceInputs()
		for node, pointer := range feeds {
			if err := inputs.Set(node, pointer); err != nil {
				return nil, nil, nil, err
			}
		}
		out, err := exe.ExecuteCompiled(ctx, pgCompiled, hostFeeds, inputs)
		if err != nil {
			_ = allocations.Close(ctx)
			return nil, nil, nil, fmt.Errorf("device full prefill layer %d: %w", layer, err)
		}
		row = append([]float32(nil), out[pg.Output].Data...)
		seedKeys[layer] = append([]float32(nil), out[pg.KeyKV].Data[:pc.promptLen*kvOut]...)
		seedValues[layer] = append([]float32(nil), out[pg.ValueKV].Data[:pc.promptLen*kvOut]...)
		_ = allocations.Close(ctx)
	}
	lastRow = row[(pc.promptLen-1)*H : pc.promptLen*H]
	return lastRow, seedKeys, seedValues, nil
}

// runFullVision: host front-end + device 27-block vision tower from the supplied
// pixel_values, returning the device block_last [hidden, nPatch] (host copy).
func runFullVision(l *campaignContext, ctx context.Context, worker *device.Worker, exe *executor.Executor, pc *prefillContext, pixelValues []float32, nPatch int) ([]float32, error) {
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
	allocations := device.NewAllocationSet(worker)
	defer func() { _ = allocations.Close(ctx) }()
	feeds := map[*tensor.Tensor]driver.DevicePtr{}
	bind := func(node *tensor.Tensor, v []float32) error {
		ptr, e := allocations.Upload(ctx, driver.Bytes(v))
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
	inputs := compiled.NewDeviceInputs()
	for node, pointer := range feeds {
		if err := inputs.Set(node, pointer); err != nil {
			return nil, err
		}
	}
	out, err := exe.ExecuteCompiled(ctx, compiled, hostFeeds, inputs)
	if err != nil {
		return nil, fmt.Errorf("device full vision execute: %w", err)
	}
	return append([]float32(nil), out[g.BlockLast].Data...), nil
}

func bindMergerV(ctx context.Context, allocations *device.AllocationSet, feeds map[*tensor.Tensor]driver.DevicePtr, node *tensor.Tensor, v []float32) error {
	ptr, e := allocations.Upload(ctx, driver.Bytes(v))
	if e != nil {
		return e
	}
	feeds[node] = ptr
	return nil
}
