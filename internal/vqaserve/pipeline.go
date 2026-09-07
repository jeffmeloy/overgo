package vqaserve

import (
	"context"
	"fmt"
	"math"
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

// Options drive the pipeline's decode loop. Expected, when set, is the
// harness's golden chain (first token, then one token per decode step): the
// step count is fixed to it and every top token is asserted against it.
// Without it, MaxSteps caps generation and EOSIDs stop it (the stop token is
// not emitted). Log receives the stage boundaries; nil logs nothing.
type Options struct {
	Expected []int
	MaxSteps int
	EOSIDs   []int
	Log      func(string)
}

func (opts Options) log(message string) {
	if opts.Log != nil {
		opts.Log(message)
	}
}

// Result is the pipeline outcome the caller decodes and reports.
type Result struct {
	FirstToken     int
	Generated      []int
	E2EWall        time.Duration
	DecodeWall     time.Duration
	Steps          int
	Launches       uint64
	Instantiations uint64
	Updates        uint64
}

// DeviceKV is one layer's resident key and value cache on the device.
type DeviceKV struct {
	Key, Value   *executor.DeviceBuffer
	KeyV, ValueV executor.DeviceValue
}

// FirstError returns the first non-nil error of a bind sequence.
func FirstError(errs ...error) error {
	for _, e := range errs {
		if e != nil {
			return e
		}
	}
	return nil
}

// ArgmaxF32 returns the index of the largest value, -1 for an empty slice.
func ArgmaxF32(v []float32) int {
	best, bestI := float32(math.Inf(-1)), -1
	for i, x := range v {
		if x > best {
			best, bestI = x, i
		}
	}
	return bestI
}

// RunPipeline is the shared device VQA serving pipeline: device vision tower
// -> device merger (image features) -> host embedding splice -> device
// branch-routed prefill (chained layers, exporting the per-layer resident
// K/V) -> terminal (first token) -> device decode seeded from the device
// prefill K/V -> N-step generation. pc carries the prompt state (from
// goldens in harness mode, from the processor in serve mode); pixelValues
// is the preprocessed image. It returns the generated token chain with the
// timing and residency measurements and logs the stage boundaries.
func RunPipeline(
	ctx context.Context,
	worker *device.Worker,
	exe *executor.Executor,
	pc *PrefillContext,
	pixelValues []float32,
	opts Options,
) (Result, error) {
	var res Result
	opts.log("DEVICE full START")

	cfg := pc.Cfg
	H := cfg.HiddenSize
	hd := cfg.HeadDim
	kvHeads := cfg.NumKeyValueHeads
	O := pc.Spec.OutHidden

	terminal, err := routedlm.LoadTerminalWeights(pc.Src, cfg, Binding)
	if err != nil {
		return res, err
	}

	allocations := device.NewAllocationSet(worker)
	defer func() { _ = allocations.Close(ctx) }()
	e2eStart := time.Now()

	// STAGE 0: device vision tower -> block_last.
	nPatch := pc.GridT * pc.GridH * pc.GridW
	blockLast, err := runVision(ctx, worker, exe, pc, pixelValues, nPatch)
	if err != nil {
		return res, err
	}
	opts.log(fmt.Sprintf("DEVICE full STAGE0 vision tower done nPatch=%d hidden=%d", nPatch, pc.Spec.Hidden))

	// STAGE 1: device merger -> image features.
	devMerged, err := runMerger(ctx, exe, &allocations, pc, blockLast, nPatch, O)
	if err != nil {
		return res, err
	}
	opts.log(fmt.Sprintf("DEVICE full STAGE1 merger done rows=%d out=%d", pc.ImageRows, O))

	// prefill embeds = embedding-table text rows + device merger image rows.
	prefillEmbeds, err := routedlm.PrefillValues(pc.Src, cfg, Binding, pc.InputIDs, pc.ImageMaskPositions, pc.ImageRows, func(dst []float32, ordinal int) error {
		copy(dst, devMerged[ordinal*O:(ordinal+1)*O])
		return nil
	})
	if err != nil {
		return res, err
	}

	// STAGE 2: device prefill (chained) + KV export.
	lastRow, seedKeys, seedValues, err := runPrefill(ctx, exe, &allocations, pc, prefillEmbeds)
	if err != nil {
		return res, err
	}
	firstToken, firstLogit, err := routedlm.TerminalTopToken(lastRow, cfg, terminal, tensor.FirstOffset)
	if err != nil {
		return res, err
	}
	if len(opts.Expected) > 0 && firstToken != opts.Expected[0] {
		return res, fmt.Errorf("device full prefill terminal top %d != golden first token %d", firstToken, opts.Expected[0])
	}
	res.FirstToken = firstToken
	opts.log(fmt.Sprintf("DEVICE full STAGE2 prefill done, terminal first token=%d logit=%.4f KV exported for %d layers", firstToken, firstLogit, cfg.NumHiddenLayers))

	// STAGE 3: device decode seeded from the prefill KV.
	weights := make([]routedlm.LayerWeights, cfg.NumHiddenLayers)
	for layer := range cfg.NumHiddenLayers {
		if weights[layer], err = routedlm.LoadLayerWeights(pc.Src, cfg, Binding, layer); err != nil {
			return res, err
		}
	}
	// capacity: prompt + the maximum tokens the loop might generate.
	genBudget := opts.MaxSteps
	if len(opts.Expected) > 0 {
		genBudget = len(opts.Expected) - 1
	}
	capacity := uint32(pc.PromptLen + genBudget + 1)

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
	for layer := range cfg.NumHiddenLayers {
		w := weights[layer]
		in := dgraph.Inputs[layer]
		if err := FirstError(
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
	head, err := decodeAllocations.Upload(ctx, headBytes)
	if err != nil {
		return res, err
	}
	deviceFeeds[dgraph.Head] = head

	// device-resident KV capacity buffers seeded from the device prefill K/V.
	kvShape := tensor.MustShape(uint64(hd), uint64(kvHeads), uint64(capacity))
	kvBytes, _ := kvShape.Bytes(dtype.F32)
	caches := make([]DeviceKV, cfg.NumHiddenLayers)
	targets := dCompiled.NewRetainedTargets()
	defer func() {
		for _, kv := range caches {
			if kv.Key != nil {
				_ = kv.Key.Release(ctx)
			}
			if kv.Value != nil {
				_ = kv.Value.Release(ctx)
			}
		}
	}()
	for layer := range cfg.NumHiddenLayers {
		kb, e := exe.AllocateDeviceBuffer(ctx, kvBytes)
		if e != nil {
			return res, fmt.Errorf("device full kv key alloc layer %d: %w", layer, e)
		}
		vb, e := exe.AllocateDeviceBuffer(ctx, kvBytes)
		if e != nil {
			return res, fmt.Errorf("device full kv value alloc layer %d: %w", layer, e)
		}
		kv := DeviceKV{Key: kb, Value: vb}
		if kv.KeyV, e = kb.Value(kvShape); e != nil {
			return res, e
		}
		if kv.ValueV, e = vb.Value(kvShape); e != nil {
			return res, e
		}
		if e := worker.Do(ctx, func(state *device.State) error {
			if e := state.Driver.MemcpyHtoD(kv.KeyV.Pointer, driver.Bytes(seedKeys[layer])); e != nil {
				return e
			}
			return state.Driver.MemcpyHtoD(kv.ValueV.Pointer, driver.Bytes(seedValues[layer]))
		}); e != nil {
			return res, fmt.Errorf("device full kv seed layer %d: %w", layer, e)
		}
		caches[layer] = kv
		deviceFeeds[dgraph.Inputs[layer].PastKey] = kv.KeyV.Pointer
		deviceFeeds[dgraph.Inputs[layer].PastValue] = kv.ValueV.Pointer
		if e := targets.Set(dgraph.Nodes[layer].KeyAppend, kv.KeyV); e != nil {
			return res, e
		}
		if e := targets.Set(dgraph.Nodes[layer].ValueAppend, kv.ValueV); e != nil {
			return res, e
		}
	}
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
	for _, id := range opts.EOSIDs {
		eos[id] = true
	}
	tokenID := firstToken
	generated := []int{tokenID}
	decodeStart := time.Now()
	for step := range genBudget {
		tokenPos := pc.PromptLen + step
		embedding, err := routedlm.EmbeddingRows(pc.Src, cfg, Binding, []int{tokenID})
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
		devTop := ArgmaxF32(logitsVal.Data)
		if len(opts.Expected) > 0 && devTop != opts.Expected[step+1] {
			return res, fmt.Errorf("device full decode step %d top %d != golden %d", step, devTop, opts.Expected[step+1])
		}
		if len(opts.Expected) == 0 && eos[devTop] {
			// EOS reached: stop without emitting the stop token.
			break
		}
		generated = append(generated, devTop)
		tokenID = devTop
	}
	res.DecodeWall = time.Since(decodeStart)
	res.E2EWall = time.Since(e2eStart)
	res.Steps = len(generated) - 1
	res.Generated = generated
	statsAfter, _ := worker.ExecutionStats(ctx)
	res.Launches = statsAfter.GraphLaunches - statsBefore.GraphLaunches
	res.Instantiations = statsAfter.GraphInstantiations - statsBefore.GraphInstantiations
	res.Updates = statsAfter.GraphUpdates - statsBefore.GraphUpdates
	return res, nil
}

// runMerger: STAGE 1, the device merger from block_last to imageRows*O
// merged image-feature rows (row-major host copy).
func runMerger(
	ctx context.Context,
	exe *executor.Executor,
	allocations *device.AllocationSet,
	pc *PrefillContext,
	blockLast []float32,
	nPatch, O int,
) ([]float32, error) {
	mg, err := patchtower.BuildDeviceMerger(pc.Spec, pc.ImageRows, pc.GridT, pc.GridH, pc.GridW)
	if err != nil {
		return nil, err
	}
	mgCompiled, err := executor.Compile(mg.Merged)
	if err != nil {
		return nil, fmt.Errorf("device full merger compile: %w", err)
	}
	mgFeeds := map[*tensor.Tensor]driver.DevicePtr{}
	if err := FirstError(
		bindMergerV(ctx, allocations, mgFeeds, mg.Proj1W, pc.Merger.Proj1Weight), bindMergerV(ctx, allocations, mgFeeds, mg.Proj1B, pc.Merger.Proj1Bias),
		bindMergerV(ctx, allocations, mgFeeds, mg.Proj2W, pc.Merger.Proj2Weight), bindMergerV(ctx, allocations, mgFeeds, mg.Proj2B, pc.Merger.Proj2Bias),
		bindMergerV(ctx, allocations, mgFeeds, mg.Pool0W, pc.Merger.Pool0Weight), bindMergerV(ctx, allocations, mgFeeds, mg.Pool0B, pc.Merger.Pool0Bias),
		bindMergerV(ctx, allocations, mgFeeds, mg.Pool2W, pc.Merger.Pool2Weight), bindMergerV(ctx, allocations, mgFeeds, mg.Pool2B, pc.Merger.Pool2Bias),
	); err != nil {
		return nil, err
	}
	mgHost := map[*tensor.Tensor]reference.Value{
		mg.BlockLast: {Shape: tensor.MustShape(uint64(pc.Spec.Hidden), uint64(nPatch)), Data: blockLast},
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

// runPrefill: STAGE 2, the chained branch-routed prefill over all layers,
// returning the last hidden row (terminal input) and the exported per-layer
// resident K/V.
func runPrefill(
	ctx context.Context,
	exe *executor.Executor,
	allocations *device.AllocationSet,
	pc *PrefillContext,
	prefillEmbeds []float32,
) (lastRow []float32, seedKeys, seedValues [][]float32, err error) {
	cfg := pc.Cfg
	H := cfg.HiddenSize
	kvOut := cfg.NumKeyValueHeads * cfg.HeadDim
	pg, err := routedlm.BuildDevicePrefillLayer(cfg, pc.PromptLen, pc.Blocks)
	if err != nil {
		return nil, nil, nil, err
	}
	pgCompiled, err := executor.Compile(pg.Output, pg.KeyKV, pg.ValueKV)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("device full prefill compile: %w", err)
	}
	rowShape := tensor.MustShape(uint64(H), uint64(pc.PromptLen))
	maskShape := tensor.MustShape(tensor.SingletonExtent, uint64(pc.PromptLen))
	seedKeys = make([][]float32, cfg.NumHiddenLayers)
	seedValues = make([][]float32, cfg.NumHiddenLayers)
	row := append([]float32(nil), prefillEmbeds...)
	for layer := range cfg.NumHiddenLayers {
		w, err := routedlm.LoadLayerWeights(pc.Src, cfg, Binding, layer)
		if err != nil {
			return nil, nil, nil, err
		}
		feeds := map[*tensor.Tensor]driver.DevicePtr{}
		if err := BindLayerBranches(ctx, allocations, feeds, pg.Text, pg.Vision, w); err != nil {
			return nil, nil, nil, err
		}
		hostFeeds := map[*tensor.Tensor]reference.Value{
			pg.Row:      {Shape: rowShape, Data: row},
			pg.MaskText: {Shape: maskShape, Data: pc.MaskText},
			pg.MaskVis:  {Shape: maskShape, Data: pc.MaskVis},
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
		seedKeys[layer] = append([]float32(nil), out[pg.KeyKV].Data[:pc.PromptLen*kvOut]...)
		seedValues[layer] = append([]float32(nil), out[pg.ValueKV].Data[:pc.PromptLen*kvOut]...)
		_ = allocations.Close(ctx)
	}
	lastRow = row[(pc.PromptLen-1)*H : pc.PromptLen*H]
	return lastRow, seedKeys, seedValues, nil
}

// runVision: STAGE 0, the host front-end and the device vision tower from
// the supplied pixel_values, returning the device block_last
// [hidden, nPatch] (host copy).
func runVision(ctx context.Context, worker *device.Worker, exe *executor.Executor, pc *PrefillContext, pixelValues []float32, nPatch int) ([]float32, error) {
	patchWeights, err := patchtower.LoadPatchEmbedWeights(pc.Src, pc.Spec)
	if err != nil {
		return nil, err
	}
	pos, err := patchtower.LoadPosEmbedTable(pc.Src, pc.Spec)
	if err != nil {
		return nil, err
	}
	preBlock0, err := patchtower.PreBlock0Values(pixelValues, pc.GridT, pc.GridH, pc.GridW, pc.Spec, patchWeights, pos)
	if err != nil {
		return nil, err
	}
	g, err := patchtower.BuildDeviceVisionBlocks(pc.Spec, nPatch)
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
	for layer := range pc.Spec.Depth {
		w, err := patchtower.LoadBlockWeights(pc.Src, pc.Spec, layer)
		if err != nil {
			return nil, err
		}
		in := g.Inputs[layer]
		if err := FirstError(
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
		g.PreBlock0: {Shape: tensor.MustShape(uint64(pc.Spec.Hidden), uint64(nPatch)), Data: preBlock0},
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
