package executor

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
)

// record_window (verify.cpp 861-1836) at the bench's configuration: one token group (G = 1), no layer split, no
// branches, PDL or fused waits; the shared expert forked onto its own stream after the doorbell; the host serving each
// layer's plan (flags A and B), its CPU rows (the flag) and the PCIe share fetched by the copy kernel (pcie_mode 2).

// strataRoundEpsilon is verify.cpp's EPS, the RMS norms' epsilon.
const strataRoundEpsilon = float32(1e-6)

// strataRoundWindow is one recording's view: the window's tokens, the round, its kernels and the device state.
type strataRoundWindow struct {
	*StrataRound
	state     *device.State
	functions functionSet
	t         int64
}

// at is a float array's element offset.
func at(pointer driver.DevicePtr, elements int64) driver.DevicePtr {
	return pointer + driver.DevicePtr(elements*strataRoundWordBytes)
}

// rt is Rt(t): token t's residual streams.
func (w *strataRoundWindow) rt(token int64) driver.DevicePtr {
	return at(w.residual, token*strataGRStreams*strataGREmbedding)
}

// onStream runs launches on stream, as a captured launch onto the window's side stream.
func (w *strataRoundWindow) onStream(stream driver.Stream, launches func() error) error {
	saved := w.state.Stream
	w.state.Stream = stream
	defer func() { w.state.Stream = saved }()
	return launches()
}

// recordWindow is record_window for t tokens on the window's stream.
func (r *StrataRound) recordWindow(state *device.State, functions functionSet, t int) error {
	w := &strataRoundWindow{StrataRound: r, state: state, functions: functions, t: int64(t)}
	return w.onStream(r.cs, w.record)
}

func (w *strataRoundWindow) record() error {
	s := w.spec
	mt := int64(s.MaxTokens)
	// ---- the window's inputs, from mapped staging.
	if err := launchStrataCopyI32FromMapped(w.state, w.functions, w.dTok, w.tok.device, w.t); err != nil {
		return err
	}
	if err := launchStrataCopyI32FromMapped(w.state, w.functions, w.dStep, w.step.device, w.t*strataRoundStepCount); err != nil {
		return err
	}
	if err := launchStrataCopyI32FromMapped(w.state, w.functions, w.dPos, w.pos.device, mt*strataRoundPositionHeads); err != nil {
		return err
	}
	// ---- the embeddings (NativeEmbed::gather_dev over the GGUF's Q8_0 table), broadcast to the streams.
	if err := launchStrataEmbedRowsQ80(w.state, w.functions, s.Embedding, s.EmbeddingRow, w.dTok, int32(w.t),
		strataGREmbedding, w.emb); err != nil {
		return err
	}
	if err := launchStrataBroadcastStreams(w.state, w.functions, w.emb, w.residual, strataGREmbedding, strataGRStreams,
		int32(w.t)); err != nil {
		return err
	}
	// the layers' state indices, GDN and QSA numbered separately
	gdn, qsa := make([]int64, len(s.Layers)), make([]int64, len(s.Layers))
	var gi, qi int64
	for l, layer := range s.Layers {
		if layer.QSA {
			qsa[l], qi = qi, qi+1
		} else {
			gdn[l], gi = gi, gi+1
		}
	}
	if err := w.pre(0, gdn[0], qsa[0]); err != nil {
		return fmt.Errorf("strata round layer 0: %w", err)
	}
	for l := range s.Layers {
		if err := w.post(l); err != nil {
			return fmt.Errorf("strata round layer %d: %w", l, err)
		}
		if l+1 < len(s.Layers) {
			if err := w.pre(l+1, gdn[l+1], qsa[l+1]); err != nil {
				return fmt.Errorf("strata round layer %d: %w", l+1, err)
			}
		}
	}
	return w.head()
}

// grRead is gr_read_group: the tokens' hyper-connection read of a half's weights, the pending write applied where
// apply, its mixed rows to mixed_ and its inject to injectOut.
func (w *strataRoundWindow) grRead(hyper StrataRoundHyper, apply bool, injectPrevious, injectOut driver.DevicePtr) error {
	args := make([]strataFusedGRArgs, w.t)
	for t := range w.t {
		a := &args[t]
		a.R, a.ROut = w.rt(t), w.rt(t)
		if apply {
			a.Apply = 1
		}
		a.BoPrev, a.InjPrev = at(w.blockOut, t*strataGREmbedding), at(injectPrevious, t*strataGRStreams)
		a.WNorm, a.WDown, a.WUp, a.WInject = hyper.Norm, hyper.Down, hyper.Up, hyper.Inject
		a.Q8Down, a.Q8Up, a.Q8Inject = hyper.Q8Down, hyper.Q8Up, hyper.Q8Inject
		a.Eps = strataRoundEpsilon
		a.Lo, a.Rs = at(w.lo, t*strataGRRank), at(w.rs, t*strataGRStreams)
		a.InjectOut, a.Mixed = at(injectOut, t*strataGRStreams), at(w.mixed, t*strataGREmbedding)
	}
	return launchStrataFusedGRReadQ8(w.state, w.functions, args, w.xn)
}

// mm is record_window's mm: the interleaved kernel where native_mmvq_il_supported, interleaving xq_ once (n >= 2), else
// native_mmvq; il is whether xil_ holds xq_'s interleaved copy.
func (w *strataRoundWindow) mm(weight StrataRoundWeight, out driver.DevicePtr, nIn, nOut int64, il *bool) error {
	n := int32(w.t)
	if strataNativeMMVQILRows(w.major, w.minor, weight.Type, n, int32(nOut)) > 0 {
		if !*il && n >= strataNativeMultiFirst {
			if err := launchStrataNativeQ81Interleave(w.state, w.functions, w.xq, w.xil, int32(nIn), n); err != nil {
				return err
			}
			*il = true
		}
		return launchStrataNativeMMVQIL(w.state, w.functions, w.major, w.minor, weight.Type, weight.Pointer, w.xq,
			w.xil, out, int32(nIn), int32(nOut), n)
	}
	return launchStrataNativeMMVQ(w.state, w.functions, weight.Type, weight.Pointer, w.xq, out, int32(nIn), int32(nOut), n)
}

// pre is pre(l, 0): the hyper-connection read and the mixer, the FFN's read, the router, the doorbell, the shared
// expert on its stream and the experts' q8_1 image of the MoE input.
func (w *strataRoundWindow) pre(l int, gi, qi int64) error {
	layer := w.spec.Layers[l]
	// the previous layer's FFN write, folded into this layer's first read
	pending := l > 0
	if l == 1 {
		if err := w.lexical(); err != nil {
			return err
		}
		pending = false
	}
	if err := w.grRead(layer.Attention, pending, w.inject2, w.inject); err != nil {
		return err
	}
	il := false
	var err error
	if layer.QSA {
		err = w.qsaMixer(layer, qi, &il)
	} else {
		err = w.gdnMixer(layer, gi, &il)
	}
	if err != nil {
		return err
	}
	if err := w.grRead(layer.FFN, true, w.inject, w.inject2); err != nil {
		return err
	}
	return w.route(l, layer)
}

// lexical is the PLE block before layer 1's read (pre(1, 0)): the window's n-gram rows from mapped staging, then each
// token's pending write, ple_block (its key natively, its value as bf16, the post-ops), the history's advance and its
// snapshot for the commit.
func (w *strataRoundWindow) lexical() error {
	n := int64(strataGREmbedding)
	if err := launchStrataCopyFromMapped(w.state, w.functions, w.dPLE, w.ple.device, w.t*n); err != nil {
		return err
	}
	ple := w.spec.PLE
	if w.t > 1 {
		// gr_write_multi: the window's residual writes in one launch
		if err := launchStrataNativeGRPost(w.state, w.functions, w.rt(0), w.blockOut, w.inject2, w.rt(0),
			strataGREmbedding, strataGRStreams, int32(w.t)); err != nil {
			return err
		}
	}
	keys, queries, norms := w.pleScratch, at(w.pleScratch, strataPLEWidth), at(w.pleScratch, strataRoundPLENorm*strataPLEWidth)
	gated, conv := at(w.pleScratch, strataRoundPLEGated*strataPLEWidth), at(w.pleScratch, strataRoundPLEConv*strataPLEWidth)
	value := at(w.pleScratch, strataRoundPLEParts*strataPLEWidth)
	gate := at(value, n)
	for t := range w.t {
		if w.t == 1 {
			if err := launchStrataNativeGRPost(w.state, w.functions, w.rt(t), at(w.blockOut, t*n), at(w.inject2, t*strataGRStreams),
				w.rt(t), strataGREmbedding, strataGRStreams, 1); err != nil {
				return err
			}
		}
		emb := at(w.dPLE, t*n)
		if err := launchStrataNativeQuantizeQ81(w.state, w.functions, emb, w.pleKeyQ81, int32(n), 1); err != nil {
			return err
		}
		if err := launchStrataNativeMMVQ(w.state, w.functions, ple.Key.Type, ple.Key.Pointer, w.pleKeyQ81, keys, int32(n),
			strataPLEWidth, 1); err != nil {
			return err
		}
		if err := launchStrataBF16GEMV(w.state, w.functions, emb, ple.Value, value, n, n); err != nil {
			return err
		}
		weights := strataPLEWeights{NormKey: ple.NormKey, NormQuery: ple.NormQuery, NormConv: ple.NormConv, ConvF16: ple.ConvF16}
		buffers := strataPLEBuffers{Key: queries, Query: norms, Gate: gate, Gated: gated, Normalized: norms, Conv: conv, Result: w.rt(t)}
		if err := launchStrataNativePLEPostops(w.state, w.functions, keys, w.rt(t), value, w.pleHistory, weights, buffers); err != nil {
			return err
		}
		// out.normalized: the copy ple_block makes of its normalized row
		if err := w.state.Driver.MemcpyDtoDAsync(w.pleNormalized, norms, strataPLEWidth*strataRoundWordBytes, w.state.Stream); err != nil {
			return err
		}
		if err := launchStrataPLEHistoryAdvance(w.state, w.functions, w.pleHistory, w.pleNormalized); err != nil {
			return err
		}
		if err := launchStrataCopyFromMapped(w.state, w.functions, at(w.histSnap, t*strataRoundHistoryRows*strataPLEWidth),
			w.pleHistory, strataRoundHistoryRows*strataPLEWidth); err != nil {
			return err
		}
	}
	return nil
}

// strataRoundStreamMap is a streamed QSA state's residency map: its page table, then the map's per-slot arrays and
// control ints (tensor.QSAStream.Map), the slots its pools' pages.
func strataRoundStreamMap(cells StrataRoundQSAState, slots int64) strataKVStreamMap {
	array := func(index int64) driver.DevicePtr { return cells.StreamMap + driver.DevicePtr(index*slots*i32Bytes) }
	return strataKVStreamMap{
		table: cells.PageTable, slotBlock: array(qsaMapSlotBlock), slotStamp: array(qsaMapSlotStamp), slotRef: array(qsaMapSlotRef),
		missBlock: array(qsaMapMissBlock), missSlot: array(qsaMapMissSlot), ctl: array(tensor.QSAStreamSlotArrays),
		blocks: cells.MaxCells / tensor.QSABlockCells, slots: slots,
	}
}

// ple_block's workspace: five hc_dim rows (key, query, normalized, gated, conv), then the value and the gates.
const (
	strataRoundPLENorm  = 2
	strataRoundPLEGated = 3
	strataRoundPLEConv  = 4
	strataRoundPLEParts = 5
)

// gdnMixer is the gated-delta mixer: q/k/v, the conv (a one-token window committing its token), the gates, z, the
// recurrence and the output projection to bo_.
func (w *strataRoundWindow) gdnMixer(layer StrataRoundLayer, gi int64, il *bool) error {
	s := w.spec
	n, mt := w.t, int64(s.MaxTokens)
	c, zv, hv := s.ConvChannels, s.ValueDim, s.ValueHeads
	selfCommit := w.t == 1
	window := strataGatedDeltaWindow{
		qkv: at(w.qkvL, gi*mt*c), rows: at(w.hL, gi*mt*c), gate: at(w.gateL, gi*mt*hv), beta: at(w.betaL, gi*mt*hv),
		z: w.z, outputs: w.y,
	}
	if err := launchStrataNativeQuantizeQ81(w.state, w.functions, w.mixed, w.xq, strataGREmbedding, int32(n)); err != nil {
		return err
	}
	if err := w.mm(layer.QKV, window.qkv, strataGREmbedding, c, il); err != nil {
		return err
	}
	if err := launchStrataGatedDeltaConv(w.state, w.functions, layer.History, layer.SSMConv, window, int32(c), int32(s.KeyHeads),
		strataRoundEpsilon, 0, int32(n), selfCommit); err != nil {
		return err
	}
	if err := launchStrataGatedDeltaAB(w.state, w.functions, w.mixed, layer.SSMAlpha, layer.SSMBeta, layer.SSMDT, layer.SSMA,
		window.gate, window.beta, strataGREmbedding, int32(hv), int32(n)); err != nil {
		return err
	}
	if err := w.mm(layer.AttnGate, w.z, strataGREmbedding, zv, il); err != nil {
		return err
	}
	var kept driver.DevicePtr
	if selfCommit {
		kept = w.one
	}
	if err := launchStrataGatedDeltaStep(w.state, w.functions, layer.State, layer.SSMNorm, kept, window, int32(c),
		int32(s.KeyHeads), int32(hv), strataRoundEpsilon, int32(n), 0); err != nil {
		return err
	}
	if err := launchStrataNativeQuantizeQ81(w.state, w.functions, w.y, w.xq, int32(zv), int32(n)); err != nil {
		return err
	}
	*il = false
	return w.mm(layer.SSMOut, w.blockOut, zv, strataGREmbedding, il)
}

// qsaMixer is the QSA mixer: the indexer key, K and V normed and rotated, the tail's snapshot, the window's cells and
// indexer keys, the query and the indexer query, the block scores, the selection, the streamed blocks resolved, the
// attention, its gate and the output projection to bo_.
func (w *strataRoundWindow) qsaMixer(layer StrataRoundLayer, qi int64, il *bool) error {
	s := w.spec
	n, mt := w.t, int64(s.MaxTokens)
	nh, hd, nkv := int64(tensor.QSAQueryHeads), int64(tensor.QSAHeadWidth), int64(tensor.QSAKeyValueHeads)
	iq, id := int64(tensor.QSAIndexerHeads), int64(tensor.QSAIndexerWidth)
	cells := layer.Cells
	idxRaw := at(w.idxRawL, qi*mt*id)
	posK, posI := at(w.dPos, mt*nh), at(w.dPos, mt*(nh+nkv))
	batched := n > 1
	normRope := func(data driver.DevicePtr, stride int64, norm, out driver.DevicePtr, rows, cols int64, positions driver.DevicePtr) error {
		return launchStrataNativeNormRope(w.state, w.functions, data, int32(stride), norm, out, int32(rows), int32(cols),
			tensor.QSARotated, strataRoundEpsilon, w.rope, positions, w.spec.Positions)
	}
	if err := launchStrataNativeQuantizeQ81(w.state, w.functions, w.mixed, w.xq, strataGREmbedding, int32(n)); err != nil {
		return err
	}
	if batched {
		if err := launchStrataBF16GEMVMulti(w.state, w.functions, w.mixed, strataGREmbedding, layer.IndexerK, idxRaw, id,
			strataGREmbedding, id, int32(n)); err != nil {
			return err
		}
	} else if err := launchStrataBF16GEMV(w.state, w.functions, w.mixed, layer.IndexerK, idxRaw, strataGREmbedding, id); err != nil {
		return err
	}
	if err := w.mm(layer.Key, w.kCur, strataGREmbedding, nkv*hd, il); err != nil {
		return err
	}
	if err := w.mm(layer.Value, w.vCur, strataGREmbedding, nkv*hd, il); err != nil {
		return err
	}
	if batched {
		if err := normRope(w.kCur, hd, layer.KeyNorm, w.kCur, n*nkv, hd, posK); err != nil {
			return err
		}
	} else if err := normRope(w.kCur, hd, layer.KeyNorm, w.kCur, nkv, hd, w.dPos); err != nil {
		return err
	}
	ts := int64(strataRoundTailDeltas)
	if err := launchStrataCopyFromMapped(w.state, w.functions, at(w.tailSnap, qi*ts), cells.Tail, ts); err != nil {
		return err
	}
	pools := strataKVPools{key: cells.KeyCodes, value: cells.ValueCodes, keyScale: cells.KeyScales, valueScale: cells.ValueScales}
	host := strataKVPools{key: cells.HostKeyCodes, value: cells.HostValueCodes, keyScale: cells.HostKeyScales, valueScale: cells.HostValueScales}
	geometry := strataKVGeometry{kvHeads: tensor.QSAKeyValueHeads, headDim: tensor.QSAHeadWidth, pageSize: tensor.QSABlockCells}
	if err := launchStrataKVAppendSteps(w.state, w.functions, pools, host, cells.PageTable, w.dStep, w.kCur, w.vCur, geometry,
		int32(n), strataRoundStepCount, int32(nkv*hd)); err != nil {
		return err
	}
	indexer := strataIndexerBuffers{tail: cells.Tail, dead: cells.Spare, pooled: cells.Pooled, blockPos: cells.BlockPosition}
	if err := launchStrataNativeIndexerSteps(w.state, w.functions, idxRaw, w.dStep, strataRoundStepCount, int32(n), 0,
		layer.IndexerKNorm, strataRoundEpsilon, indexer, int32(cells.MaxCells), w.rope, w.spec.Positions); err != nil {
		return err
	}
	if err := w.mm(layer.Query, w.qFull, strataGREmbedding, nh*strataRoundQGate*hd, il); err != nil {
		return err
	}
	if batched {
		if err := normRope(w.qFull, strataRoundQGate*hd, layer.QueryNorm, w.qCur, n*nh, hd, w.dPos); err != nil {
			return err
		}
		if err := launchStrataBF16GEMVMulti(w.state, w.functions, w.mixed, strataGREmbedding, layer.IndexerQ, w.qIdx, iq*id,
			strataGREmbedding, iq*id, int32(n)); err != nil {
			return err
		}
		if err := normRope(w.qIdx, id, layer.IndexerQNorm, w.qIdx, n*iq, id, posI); err != nil {
			return err
		}
	} else {
		if err := normRope(w.qFull, strataRoundQGate*hd, layer.QueryNorm, w.qCur, nh, hd, w.dPos); err != nil {
			return err
		}
		if err := launchStrataBF16GEMV(w.state, w.functions, w.mixed, layer.IndexerQ, w.qIdx, strataGREmbedding, iq*id); err != nil {
			return err
		}
		if err := normRope(w.qIdx, id, layer.IndexerQNorm, w.qIdx, iq, id, w.dPos); err != nil {
			return err
		}
	}
	if err := launchStrataBlockScores(w.state, w.functions, cells.Pooled, cells.Spare, w.qIdx, w.dStep, n, w.maxBlocks,
		w.scores, -1); err != nil {
		return err
	}
	if err := launchStrataBlockTopK(w.state, w.functions, w.scores, w.dStep, n, w.maxBlocks, w.capacity, w.sel, -1); err != nil {
		return err
	}
	if cells.StreamMap != 0 {
		stream := strataRoundStreamMap(cells, cells.Slots)
		if err := launchStrataKVStreamResolve(w.state, w.functions, stream, pools, host, w.sel, w.dStep, n, w.capacity, geometry); err != nil {
			return err
		}
	}
	attention := strataAttentionPools{kv: pools, table: cells.PageTable}
	if err := launchStrataDecodeAttention(w.state, w.functions, w.qCur, attention, w.sel, w.dStep, w.capacity,
		tensor.QSAKeyValueHeads, tensor.QSABlockCells, w.attnScratch, w.attn, n, true); err != nil {
		return err
	}
	if err := launchStrataNativeQSAGate(w.state, w.functions, w.attn, w.qFull, w.attn32, int32(n*nh), int32(hd)); err != nil {
		return err
	}
	if err := launchStrataNativeQuantizeQ81(w.state, w.functions, w.attn32, w.xq, int32(nh*hd), int32(n)); err != nil {
		return err
	}
	*il = false
	return w.mm(layer.Output, w.blockOut, nh*hd, strataGREmbedding, il)
}

// route is the router, the doorbell, the shared expert forked onto its stream, and the experts' q8_1 image.
func (w *strataRoundWindow) route(l int, layer StrataRoundLayer) error {
	n := w.t
	if n > 1 {
		if err := launchStrataBF16GEMVMulti(w.state, w.functions, w.mixed, strataGREmbedding, layer.Router, w.logits,
			strataRoundExperts, strataGREmbedding, strataRoundExperts, int32(n)); err != nil {
			return err
		}
		if err := launchStrataNativeRouteMulti(w.state, w.functions, w.logits, w.idsD, w.w, int32(n)); err != nil {
			return err
		}
	} else {
		// moe_route: project_bf16 then native_router_top10
		if err := launchStrataBF16GEMV(w.state, w.functions, w.mixed, layer.Router, w.logits, strataGREmbedding, strataRoundExperts); err != nil {
			return err
		}
		if err := launchStrataNativeRoute(w.state, w.functions, w.logits, w.idsD, w.w); err != nil {
			return err
		}
	}
	resident := at(w.spec.Resident, int64(l)*strataRoundExperts)
	if err := launchStrataDoorbellPublishRes(w.state, w.functions, w.mixed, w.idsD, resident, strataRoundExperts,
		n*strataGREmbedding, n*strataRoundRouted, w.x.device, w.ids.device, w.seq.device); err != nil {
		return err
	}
	// sh_fork_late: the shared expert forks off once the doorbell is published
	fork, err := w.state.Driver.RecordEvent(w.cs)
	if err != nil {
		return err
	}
	if err := w.state.Driver.WaitEvent(w.shCS, fork); err != nil {
		return err
	}
	if err := w.onStream(w.shCS, func() error { return w.sharedExpert(layer) }); err != nil {
		return err
	}
	join, err := w.state.Driver.RecordEvent(w.shCS)
	if err != nil {
		return err
	}
	w.join = join
	return launchStrataQuantizeQ81Rows(w.state, w.functions, w.mixed, n, strataGREmbedding, w.natXQ)
}

// sharedExpert is shared_expert_multi: the input's q8_1, gate and up, SwiGLU with its q8_1, down, then the gate input's
// sigmoid scaling the row.
func (w *strataRoundWindow) sharedExpert(layer StrataRoundLayer) error {
	n, ff := int32(w.t), int32(w.spec.SharedFF)
	if err := launchStrataNativeQuantizeQ81(w.state, w.functions, w.mixed, w.shXQ, strataGREmbedding, n); err != nil {
		return err
	}
	if err := launchStrataNativeMMVQ(w.state, w.functions, layer.SharedGate.Type, layer.SharedGate.Pointer, w.shXQ, w.shGate,
		strataGREmbedding, ff, n); err != nil {
		return err
	}
	if err := launchStrataNativeMMVQ(w.state, w.functions, layer.SharedUp.Type, layer.SharedUp.Pointer, w.shXQ, w.shUp,
		strataGREmbedding, ff, n); err != nil {
		return err
	}
	if err := launchStrataNativeSwigluQuantizeQ81(w.state, w.functions, w.shGate, w.shUp, w.shXQ, ff, n); err != nil {
		return err
	}
	if err := launchStrataNativeMMVQ(w.state, w.functions, layer.SharedDown.Type, layer.SharedDown.Pointer, w.shXQ, w.shared,
		ff, strataGREmbedding, n); err != nil {
		return err
	}
	if n > 1 {
		if err := launchStrataBF16GEMVMulti(w.state, w.functions, w.mixed, strataGREmbedding, layer.SharedGateInput, w.shG, 1,
			strataGREmbedding, 1, n); err != nil {
			return err
		}
	} else if err := launchStrataBF16GEMV(w.state, w.functions, w.mixed, layer.SharedGateInput, w.shG, strataGREmbedding, 1); err != nil {
		return err
	}
	return launchStrataSigmoidScaleRows(w.state, w.functions, w.shared, w.shG, strataGREmbedding, n)
}

// The plan's parts in its words (set_plan_slot, post): counts, start, dst, tok, then (two-word aligned) the VRAM
// groups' blob pointers, the PCIe groups' and their starts.
func (w *strataRoundWindow) planParts() (counts, start, dst, tok, pointers, pcie, pcieStart driver.DevicePtr) {
	capx := int64(w.spec.MaxTokens) * strataRoundRouted
	counts = w.planD
	start = at(counts, strataRoundPlanHeader)
	dst = at(start, capx+1)
	tok = at(dst, capx)
	offset := (strataRoundPlanHeader + (capx + 1) + strataRoundPlanEntryArrays*capx + 1) &^ 1
	pointers = at(counts, offset)
	pcie = pointers + driver.DevicePtr(capx*strataRoundPointerBytes)
	pcieStart = at(counts, offset+strataRoundPlanHeader*capx)
	return counts, start, dst, tok, pointers, pcie, pcieStart
}

// strataRoundPointerBytes is a device address's bytes in the plan; strataRoundPlanEntryArrays its per-entry arrays (dst,
// tok).
const (
	strataRoundPointerBytes    = 8
	strataRoundPlanEntryArrays = 2
)

// post is post(l, 0): the plan's VRAM groups, the PCIe share staged and its groups, the CPU's rows, the hit add, the
// combine, and after the last layer its write.
func (w *strataRoundWindow) post(l int) error {
	n := w.t
	ring := uint32(l + 1)
	capacity := n * strataRoundRouted
	counts, start, dst, tok, pointers, pcie, pcieStart := w.planParts()
	if err := launchStrataWaitFlagGE(w.state, w.functions, w.flagA.device, ring); err != nil {
		return err
	}
	if err := launchStrataCopyI32FromMapped(w.state, w.functions, w.planD, w.plan.device, w.planWords); err != nil {
		return err
	}
	grouped := func(groupPointers, groupStart, groupCount driver.DevicePtr, side int64) error {
		return launchStrataNativeExpertGrouped(w.state, w.functions, w.layouts[l], groupPointers, groupStart, groupCount, dst, tok,
			capacity, capacity, w.natXQ, w.hitScratch, w.hitOut, side)
	}
	if err := grouped(pointers, start, counts, 0); err != nil {
		return err
	}
	if err := launchStrataWaitFlagGE(w.state, w.functions, w.flagB.device, ring); err != nil {
		return err
	}
	blob := int64(w.layouts[l].Bytes)
	if err := launchStrataFetchBlobs(w.state, w.functions, pcie, at(counts, strataRoundPCIeCount), w.staging, blob,
		strataRoundStagingBlobs); err != nil {
		return err
	}
	if err := launchStrataRebasePtrs(w.state, w.functions, pcie, at(counts, strataRoundPCIeCount), w.staging, blob); err != nil {
		return err
	}
	if err := grouped(pcie, pcieStart, at(counts, strataRoundPCIeCount), strataRoundPCIeGroupRows); err != nil {
		return err
	}
	if err := launchStrataWaitFlagGE(w.state, w.functions, w.flag.device, ring); err != nil {
		return err
	}
	if err := launchStrataCopyRowsFromMapped(w.state, w.functions, w.parts, w.ymiss.device, capacity, strataGREmbedding, dst,
		at(counts, strataRoundHitCount), 0, 0); err != nil {
		return err
	}
	if err := launchStrataMoEHitAdd(w.state, w.functions, w.parts, w.hitOut, dst, at(counts, strataRoundHitCount), capacity,
		strataGREmbedding); err != nil {
		return err
	}
	if err := w.state.Driver.WaitEvent(w.cs, w.join); err != nil {
		return err
	}
	// native_moe_combine_multi, or for one token moe_combine_parts' native_moe_combine: the same kernel, a row a token
	if err := launchStrataNativeMoECombineMulti(w.state, w.functions, w.parts, w.w, w.shared, w.blockOut, strataGREmbedding,
		strataRoundRouted, int(n)); err != nil {
		return err
	}
	if l == len(w.spec.Layers)-1 {
		// gr_write_multi, the window's rows in one launch
		return launchStrataNativeGRPost(w.state, w.functions, w.rt(0), w.blockOut, w.inject2, w.rt(0), strataGREmbedding,
			strataGRStreams, int32(n))
	}
	return nil
}

// The plan's counts: the VRAM groups, the entries (the GPU's rows), the PCIe groups.
const (
	strataRoundHitCount  = 1
	strataRoundPCIeCount = 2
)

// head is the head, T columns: the final hyper-connection read from the Q8_0 projections, the head's projection, the
// argmax of each row to the mapped output and the non-finite guard.
func (w *strataRoundWindow) head() error {
	s := w.spec
	n := w.t
	args := make([]strataFusedGRArgs, n)
	for t := range n {
		a := &args[t]
		a.R, a.ROut = w.rt(t), w.rt(t)
		a.WNorm, a.WDown, a.WUp = s.HeadHyper.Norm, s.HeadHyper.Down, s.HeadHyper.Up
		a.Q8Down, a.Q8Up = s.HeadHyper.Q8Down, s.HeadHyper.Q8Up
		a.Eps = strataRoundEpsilon
		a.Lo, a.Rs = at(w.lo, t*strataGRRank), at(w.rs, t*strataGRStreams)
		a.Mixed = at(w.headMixed, t*strataGREmbedding)
	}
	if err := launchStrataFusedGRReadQ8(w.state, w.functions, args, w.xn); err != nil {
		return err
	}
	if err := launchStrataNativeQuantizeQ81(w.state, w.functions, w.headMixed, w.xq, strataGREmbedding, int32(n)); err != nil {
		return err
	}
	il := false
	if err := w.mm(s.Head, w.headLogits, strataGREmbedding, s.Vocab, &il); err != nil {
		return err
	}
	if err := launchStrataArgmaxRows(w.state, w.functions, w.headLogits, int32(n), int32(s.Vocab), w.argScratch, w.out.device); err != nil {
		return err
	}
	return launchStrataLogitsNonfinite(w.state, w.functions, w.headLogits, int32(n), int32(s.Vocab), w.nonfinite.device)
}
