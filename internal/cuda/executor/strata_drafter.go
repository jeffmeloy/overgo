package executor

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
)

// Strata's draft layer (src/core/mtp.cpp's MtpDrafter, ported from Strata at 296b59d2, MIT) at the bench's
// configuration, the decode round's drafter (strata-port/decode): load's mapped staging, its own K/V (a ring of the
// window over a host copy, KV streaming) and its device arena carved as load carves them (this file); record_front and
// record_rest, the prefill, round and step graphs, prefill and draft (strata_drafter_run.go). The resolved switches:
// --native (native rope, router, combine and the shared expert's bf16 gate), no --mtp-q4, --mtp-hnorm or forcing
// (pipeline-windows 0), the draft vocabulary's head over its subset, CUDA's shared expert branch, the T == 1 fused
// norm-rope, argmax_rows (sm_89) and the split top probability.

// The drafter's fixed geometry: Flash-Next's, beside the verify window's.
const (
	// strataDrafterStepWords is a step record's words (cell, n_kv, blocks, width); a row of pos_ holds a word a query
	// head.
	strataDrafterStepWords = 4
	strataDrafterHeads     = tensor.QSAQueryHeads
	// strataDrafterRingLead is the cells past the window the ring keeps: four rows of max_t a round writes ahead of its
	// queries, and 64 (load's ring).
	strataDrafterRingRounds = 4
	strataDrafterRingLead   = 64
	// strataDrafterCapacityAlign is cap_'s rounding: the window in whole 64-cell chunks.
	strataDrafterCapacityAlign = 64
	// strataDrafterMappedPad is the 64 bytes each mapped() staging adds.
	strataDrafterMappedPad = 64
	// strataDrafterRowWords is row_'s words (the accepted row, then the step's row 0).
	strataDrafterRowWords = 4
	// strataDrafterOutPad is out_ids_' and probs_' rows past max_t.
	strataDrafterOutPad = 4
	// strataDrafterGroupCounts is grp_counts_' words (the groups, the entries).
	strataDrafterGroupCounts = 4
	// strataDrafterStepSets is the step and position records' row sets: a round's max_t catch-up rows, then its chain's
	// steps (h_step_'s 2 * max_t rows).
	strataDrafterStepSets = 2
	// strataDrafterRowRead is row_'s words a round copies to the card: the accepted row and the step's row 0.
	strataDrafterRowRead = 2
	// strataGateUpParts is an expert's gate and up halves, side by side in its rows and its scratch.
	strataGateUpParts = tensor.PairedExtent
	// strataS2ScaleCopies is the S2 grouped scratch's fp32 scale arrays: the intermediate's, then the activation's.
	strataS2ScaleCopies = 2
	// strataExpertScratchParts is the native expert scratch's row parts: gate, up and their SwiGLU.
	strataExpertScratchParts = 3
	// strataCPadWord and strataCPadByte are a C layout's padding before an 8-byte field: after a 32-bit field, and
	// after a byte.
	strataCPadWord = 4
	strataCPadByte = 7
	// strataDrafterHiddenChunk is fc_hidden's columns a launch: the streams of up to two tokens, eight rows.
	strataDrafterHiddenChunk = 8
	// strataDrafterQ81Columns is xq_'s columns at its widest (native_q8_1_bytes(NH * HD, 8)).
	strataDrafterQ81Columns = 8
	// strataDrafterGraphs is a graph array's slots: windows and steps from one to kVerifyMaxT.
	strataDrafterGraphs = strataRoundMaxTokens + 1
	// strataDrafterSharedScratch is shared_expert_scratch_bytes past its gate, up, q8_0 and q8_k parts (g and room).
	strataDrafterSharedTail = 32
	strataDrafterQ8KBytes   = 292 // a block_q8_K
	strataDrafterQ8KValues  = 256
	strataDrafterAlign16    = 16
)

// StrataDrafterSpec is everything the drafter reads outside its own memory: the draft layer's weights as the rt pack
// holds them (tools/mtp_rt.py: the large projections Q8_0, the hyper-connections, router and shared expert gate bf16,
// the norms f32 with the Gemma one added, the 512 routed experts' S2 blobs back to back), the main head's rows of the
// draft vocabulary and the vocabulary itself, the main model's embedding, the verify window's final residual rows,
// and the drafter's geometry.
type StrataDrafterSpec struct {
	FCEmbedding, FCHidden, Query, Key, Value, Output StrataRoundWeight
	SharedGate, SharedUp, SharedDown                 StrataRoundWeight
	EmbeddingNorm, HiddenNorm, QueryNorm, KeyNorm    driver.DevicePtr
	Attention, FFN, Mixer                            StrataRoundHyper
	Router, SharedGateInput                          driver.DevicePtr
	Experts                                          driver.DevicePtr // n_expert blobs of ExpertBlob bytes
	ExpertBlob                                       int64
	MainHead                                         StrataRoundWeight // the main model's head (output.weight)
	MainHeadRow                                      uint64            // a vocabulary row's bytes of it
	Vocabulary                                       []int32           // the draft head's token ids (rt/draft_vocab.bin)
	Embedding                                        driver.DevicePtr  // token_embd (Q8_0)
	EmbeddingRow                                     uint64
	SharedFF                                         int64            // the shared expert's n_ff
	WindowResidual                                   driver.DevicePtr // window_R_: the verify window's final residual rows
	MaxTokens                                        int              // max_t: the rows a round or prefill takes
	MaxDrafts                                        int              // max_drafts_
	Window                                           int64            // the cells the attention reads, 0 for all
	MaxCells                                         int64            // the session's cells
	RopeFreqBase                                     float64
}

// StrataDrafter is the MtpDrafter.
type StrataDrafter struct {
	spec StrataDrafterSpec
	// resources is its executor's, through whose reservation its allocations and graphs make room on the card.
	resources *executorResources
	rope      strataRopeScaling
	// window, capacity and attnScratchFloats are window_, cap_ and attn_scratch_floats_; prompt the prompt's length
	// (prompt_len_).
	window, capacity, attnScratchFloats, prompt int64
	// The K/V state (st_): the int8 pools of the ring's slots behind its page table, the host copy of every cell.
	kvArena, table driver.DevicePtr
	pools          strataKVPools
	hostKV         strataRoundMapped
	host           strataKVPools
	pages, slots   int64
	// The mapped staging (h_*, m_*).
	tok, step, pos, row, out, prob strataRoundMapped
	// The device arena and its parts.
	arena                                                      driver.DevicePtr
	dTok, dStep, dPos, dRow, ident, rin, residual, emb, en, e2 driver.DevicePtr
	hn, h2, mixed, inject, inject2, lo, rs, blockOut, xn, xq   driver.DevicePtr
	qFull, qCur, kCur, vCur, attn, attn32, attnScratch         driver.DevicePtr
	logits, w, ids, shared, parts, y, sample                   driver.DevicePtr
	hitSlot, hitDst, hitCount, groupPointers, groupStart       driver.DevicePtr
	groupCounts, hitXQ, hitXS, hitScratch, sharedScratch       driver.DevicePtr
	xBF16, outIDs, probs, argScratch, topScratch, dummyInject  driver.DevicePtr
	headLogits                                                 driver.DevicePtr
	// head and vocabulary are the draft head (dhead_) and its token ids (dvocab_).
	head, vocabulary driver.DevicePtr
	// The streams (cs_, side_), the shared expert's join, and the graphs: the prefill's by rows, the round's by
	// rows, the steps' by index.
	cs, side               driver.Stream
	join                   driver.Event
	prefillExec, roundExec [strataDrafterGraphs]driver.GraphExec
	stepExec               [strataDrafterGraphs]driver.GraphExec
	prefillRecords         driver.DevicePtr // pf_dev_
	prefillCapacity        int64            // pf_cap_, in words
	rounds                 int64
}

// strataDrafterRingCells is load's ring: the window and the cells a round writes ahead of its queries, or none where
// the window covers the session.
func strataDrafterRingCells(window, maxCells int64, maxTokens int) int64 {
	if window > 0 && window < maxCells {
		return window + strataDrafterRingRounds*int64(maxTokens) + strataDrafterRingLead
	}
	return 0
}

// NewStrataDrafter is MtpDrafter::load and bind on the executor's worker: the mapped staging, the K/V state, the
// arena counted then carved and cleared, the identity ids, the draft logits, and the drafter's two streams.
func (e *Executor) NewStrataDrafter(ctx context.Context, spec StrataDrafterSpec) (*StrataDrafter, error) {
	if spec.MaxTokens < 1 || spec.MaxTokens > strataRoundMaxTokens {
		return nil, errors.New("mtp: max_t out of range")
	}
	if spec.FCEmbedding.Pointer == 0 || spec.FCHidden.Pointer == 0 || spec.Query.Pointer == 0 || spec.Key.Pointer == 0 ||
		spec.Value.Pointer == 0 || spec.Output.Pointer == 0 || spec.SharedGate.Pointer == 0 || spec.SharedUp.Pointer == 0 ||
		spec.SharedDown.Pointer == 0 {
		return nil, errors.New("mtp: a projection is missing (q8_0)")
	}
	if spec.MainHead.Pointer == 0 || spec.MainHeadRow == 0 || len(spec.Vocabulary) == 0 {
		return nil, errors.New("mtp: the draft layer needs the main head and the draft vocabulary")
	}
	d := &StrataDrafter{spec: spec, resources: &e.resources, rope: strataRopeScaling{freqBase: spec.RopeFreqBase, factor: 1, attnFactor: 1}}
	err := e.worker.Do(ctx, func(state *device.State) error {
		resources, err := e.ensureResources(state, false, 0, 0, 0)
		if err != nil {
			return err
		}
		return d.load(state, resources.functions)
	})
	if err != nil {
		return nil, errors.Join(err, e.worker.Do(context.WithoutCancel(ctx), func(state *device.State) error { return d.release(state) }))
	}
	return d, nil
}

// mappedParts is load's mapped() staging in its order: tokens, step records (2 max_t rows), positions a head (2 max_t
// rows), the selected row, the drafts out and their probabilities.
func (d *StrataDrafter) mappedParts(maxT uint64) []struct {
	part  *strataRoundMapped
	bytes uint64
} {
	rows := strataDrafterStepSets * maxT
	return []struct {
		part  *strataRoundMapped
		bytes uint64
	}{
		{&d.tok, maxT*strataRoundWordBytes + strataDrafterMappedPad},
		{&d.step, rows*strataDrafterStepWords*strataRoundWordBytes + strataDrafterMappedPad},
		{&d.pos, rows*strataDrafterHeads*strataRoundWordBytes + strataDrafterMappedPad},
		{&d.row, strataDrafterMappedPad},
		{&d.out, maxT*strataRoundWordBytes + strataDrafterMappedPad},
		{&d.prob, maxT*strataRoundWordBytes + strataDrafterMappedPad},
	}
}

func (d *StrataDrafter) load(state *device.State, functions functionSet) error {
	spec := d.spec
	maxT := uint64(spec.MaxTokens)
	n, hc, k := uint64(strataGREmbedding), uint64(strataGRStreams), uint64(strataRoundRouted)
	nh, hd, nkv := uint64(tensor.QSAQueryHeads), uint64(tensor.QSAHeadWidth), uint64(tensor.QSAKeyValueHeads)
	// ---- the layer's own K/V (dense attention: no indexer state is read), int8 in a ring over a host copy
	page := int64(tensor.QSABlockCells)
	d.pages = (spec.MaxCells + page - 1) / page
	d.slots = d.pages
	if ring := strataDrafterRingCells(spec.Window, spec.MaxCells, spec.MaxTokens); ring > 0 {
		d.slots = min(d.slots, (ring+page-1)/page)
	}
	if err := d.loadKV(state, uint64(page), nkv, hd); err != nil {
		return err
	}
	// ---- buffers
	if spec.Window > 0 && spec.Window < spec.MaxCells {
		d.window = spec.Window
	}
	cells := spec.MaxCells
	if d.window > 0 {
		cells = d.window
	}
	d.capacity = (cells + strataDrafterCapacityAlign - 1) / strataDrafterCapacityAlign * strataDrafterCapacityAlign
	d.attnScratchFloats = strataAttentionScratchFloats(d.capacity, tensor.QSAQueryHeads)
	for _, part := range d.mappedParts(maxT) {
		host, device, err := state.Driver.MemHostAlloc(part.bytes)
		if err != nil {
			return fmt.Errorf("mtp: mapped staging failed: %w", err)
		}
		clear(host)
		part.part.host, part.part.device = host, device
	}
	rows := strataDrafterStepSets * maxT // step/pos rows: T catch-up rows + up to T-2 chain steps
	ff := uint64(spec.SharedFF)
	carve := func(b *strataRoundBump) {
		d.dTok, d.dStep = b.take(maxT*strataRoundWordBytes), b.take(rows*strataDrafterStepWords*strataRoundWordBytes)
		d.dPos, d.dRow = b.take(rows*nh*strataRoundWordBytes), b.take(strataDrafterRowWords*strataRoundWordBytes)
		d.ident = b.take(maxT * uint64(d.capacity) * strataRoundWordBytes)
		d.rin, d.residual = b.take(maxT*hc*n*strataRoundWordBytes), b.take(maxT*hc*n*strataRoundWordBytes)
		d.emb, d.en, d.e2 = b.take(maxT*n*strataRoundWordBytes), b.take(maxT*n*strataRoundWordBytes), b.take(maxT*n*strataRoundWordBytes)
		d.hn, d.h2 = b.take(maxT*hc*n*strataRoundWordBytes), b.take(maxT*hc*n*strataRoundWordBytes)
		d.mixed, d.inject, d.inject2 = b.take(maxT*n*strataRoundWordBytes), b.take(maxT*hc*strataRoundWordBytes), b.take(maxT*hc*strataRoundWordBytes)
		d.lo, d.rs, d.blockOut = b.take(maxT*strataGRRank*strataRoundWordBytes), b.take(maxT*hc*strataRoundWordBytes), b.take(maxT*n*strataRoundWordBytes)
		d.xn = b.take(maxT * hc * n * strataRoundWordBytes)
		d.xq = b.take(strataNativeQ81Size(int32(nh*hd), strataDrafterQ81Columns))
		d.qFull, d.qCur = b.take(maxT*nh*strataRoundQGate*hd*strataRoundWordBytes), b.take(maxT*nh*hd*strataRoundWordBytes)
		d.kCur, d.vCur = b.take(maxT*nkv*hd*strataRoundWordBytes), b.take(maxT*nkv*hd*strataRoundWordBytes)
		d.attn, d.attn32 = b.take(maxT*nh*hd*strataRoundWordBytes), b.take(maxT*nh*hd*strataRoundWordBytes)
		d.attnScratch = b.take(uint64(d.attnScratchFloats) * strataRoundWordBytes) // the full layer runs one row at a time
		d.logits, d.w, d.ids = b.take(maxT*strataRoundExperts*strataRoundWordBytes), b.take(maxT*k*strataRoundWordBytes), b.take(maxT*k*strataRoundWordBytes)
		d.shared, d.parts, d.y = b.take(maxT*n*strataRoundWordBytes), b.take(maxT*k*n*strataRoundWordBytes), b.take(maxT*n*strataRoundWordBytes)
		d.sample = b.take(maxT * n * strataRoundWordBytes)
		// own_gr_ (gr_read's scratch for the per-row reads this configuration never takes) is not carved.
		d.hitSlot, d.hitDst, d.hitCount = b.take(maxT*k*strataRoundWordBytes), b.take(maxT*k*strataRoundWordBytes), b.take(strataRoundHitCounts*strataRoundWordBytes)
		d.groupPointers, d.groupStart = b.take(maxT*k*strataPointerBytes), b.take((maxT*k+1)*strataRoundWordBytes)
		d.groupCounts = b.take(strataDrafterGroupCounts * strataRoundWordBytes)
		d.hitXQ, d.hitXS = b.take(maxT*(n/strataNativeQ8K)*strataRoundQ80Bytes), b.take(maxT*(n/strataNativeQ8K)*strataRoundWordBytes)
		d.hitScratch = b.take(uint64(strataS2GroupedScratchBytes(int64(maxT*k), int64(n), int64(ff))))
		d.sharedScratch = b.take(strataDrafterSharedScratchBytes(ff))
		d.xBF16 = b.take(n * strataRoundBF16Bytes)
		d.outIDs, d.probs = b.take((maxT+strataDrafterOutPad)*strataRoundWordBytes), b.take((maxT+strataDrafterOutPad)*strataRoundWordBytes)
		d.argScratch, d.topScratch = b.take(strataArgmaxScratchSize(int32(maxT))), b.take(maxT*strataTopRowBytes)
		d.dummyInject = b.take(hc * strataRoundWordBytes)
	}
	var count strataRoundBump
	carve(&count)
	var err error
	if d.arena, err = d.resources.allocate(state, count.used); err != nil {
		return fmt.Errorf("mtp: buffers do not fit: %w", err)
	}
	if err := state.Driver.MemsetD32Async(d.arena, 0, count.used/strataRoundWordBytes, state.Stream); err != nil {
		return err
	}
	carve(&strataRoundBump{base: d.arena})
	identity := make([]int32, maxT*uint64(d.capacity))
	for t := range maxT {
		for i := range d.capacity {
			identity[t*uint64(d.capacity)+uint64(i)] = int32(i)
		}
	}
	if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
		return err
	}
	if err := state.Driver.MemcpyHtoD(d.ident, int32Words(identity)); err != nil {
		return fmt.Errorf("mtp: the identity ids could not be set: %w", err)
	}
	// bind: the draft logits (max_t rows of the head's vocabulary; the draft head writes its subset's)
	if d.headLogits, err = d.resources.allocate(state, maxT*uint64(len(spec.Vocabulary))*strataRoundWordBytes); err != nil {
		return fmt.Errorf("mtp: the draft logits do not fit: %w", err)
	}
	if err := d.bindHead(state, functions); err != nil {
		return err
	}
	for _, stream := range []*driver.Stream{&d.cs, &d.side} {
		if *stream, err = state.Driver.StreamCreate(strataRoundStreamNonBlocking); err != nil {
			return fmt.Errorf("mtp: streams: %w", err)
		}
	}
	return nil
}

// bindHead is bind's draft head: the draft vocabulary's ids on the card and the main head's rows of them gathered
// there, back to back in the head's format.
func (d *StrataDrafter) bindHead(state *device.State, functions functionSet) error {
	spec := d.spec
	count := uint64(len(spec.Vocabulary))
	var err error
	if d.vocabulary, err = d.resources.allocate(state, count*strataRoundWordBytes); err != nil {
		return fmt.Errorf("mtp: the draft head does not fit: %w", err)
	}
	if d.head, err = d.resources.allocate(state, count*spec.MainHeadRow); err != nil {
		return fmt.Errorf("mtp: the draft head does not fit: %w", err)
	}
	if err := state.Driver.MemcpyHtoD(d.vocabulary, int32Words(spec.Vocabulary)); err != nil {
		return err
	}
	if err := launchStrataGatherRows(state, functions, spec.MainHead.Pointer, int64(spec.MainHeadRow), d.vocabulary,
		int64(count), d.head); err != nil {
		return err
	}
	return state.Driver.StreamSynchronize(state.Stream)
}

// strataPointerBytes is a device pointer's bytes (grp_ptr_'s elements).
const strataPointerBytes = 8

// strataS2GroupedScratchBytes is moe_hit_grouped_scratch_bytes: the gate and up rows, the intermediate's Q8_0 and its
// fp32 scales (twice), and the activation's scales, each 16-byte aligned.
func strataS2GroupedScratchBytes(hits, embedding, ff int64) int64 {
	if hits <= 0 {
		return 0
	}
	align := func(bytes int64) int64 { return (bytes + strataDrafterAlign16 - 1) &^ (strataDrafterAlign16 - 1) }
	gateUp := hits * strataGateUpParts * ff * strataRoundWordBytes
	blocks := hits * (ff / strataS2Warp) * strataS2Q80Bytes
	scales := hits * (ff / strataS2Warp) * strataRoundWordBytes
	activation := (embedding / strataS2Warp) * strataRoundWordBytes
	return align(gateUp) + align(blocks) + strataS2ScaleCopies*align(scales) + align(activation)
}

// strataDrafterSharedScratchBytes is shared_expert_scratch_bytes: its gate, up, q8_0 and q8_K parts and g.
func strataDrafterSharedScratchBytes(ff uint64) uint64 {
	align := func(bytes uint64) uint64 { return (bytes + strataDrafterAlign16 - 1) &^ (strataDrafterAlign16 - 1) }
	a := align(ff * strataRoundWordBytes)
	q0 := align(ff / strataS2Warp * strataS2Q80Bytes)
	qk := align(ff / strataDrafterQ8KValues * strataDrafterQ8KBytes)
	return a*strataGateUpParts + q0 + qk + strataDrafterSharedTail
}

// strataDrafterSharedParts is shared_expert's carve of its scratch: gate, up, then g past the q8_0 and q8_K parts.
func strataDrafterSharedParts(scratch driver.DevicePtr, ff uint64) (gate, up, g driver.DevicePtr) {
	align := func(bytes uint64) uint64 { return (bytes + strataDrafterAlign16 - 1) &^ (strataDrafterAlign16 - 1) }
	a := align(ff * strataRoundWordBytes)
	q0 := align(ff / strataS2Warp * strataS2Q80Bytes)
	qk := align(ff / strataDrafterQ8KValues * strataDrafterQ8KBytes)
	return scratch, scratch + driver.DevicePtr(a), scratch + driver.DevicePtr(a*strataGateUpParts+q0+qk)
}

// loadKV is qsa_state_init and qsa_state_zero for the drafter's int8 K/V: the ring's slots' codes and scales and the
// page table (a ring at block % slots, or the identity where the ring holds every page), and, for a ring, the pinned
// host copy of every cell in the identity layout.
func (d *StrataDrafter) loadKV(state *device.State, page, heads, width uint64) error {
	rows := uint64(d.slots) * heads * page
	scaleBytes := width / uint64(strataKVQ8Group) * strataKVScaleBytes
	carve := func(b *strataRoundBump, rows uint64) strataKVPools {
		return strataKVPools{
			key: b.take(rows * width), value: b.take(rows * width),
			keyScale: b.take(rows * scaleBytes), valueScale: b.take(rows * scaleBytes),
		}
	}
	var count strataRoundBump
	carve(&count, rows)
	d.table = count.take(uint64(d.pages) * strataRoundWordBytes)
	var err error
	if d.kvArena, err = d.resources.allocate(state, count.used); err != nil {
		return fmt.Errorf("mtp: the K/V state does not fit: %w", err)
	}
	if err := state.Driver.MemsetD32Async(d.kvArena, 0, count.used/strataRoundWordBytes, state.Stream); err != nil {
		return err
	}
	b := strataRoundBump{base: d.kvArena}
	d.pools = carve(&b, rows)
	d.table = b.take(uint64(d.pages) * strataRoundWordBytes)
	table := make([]int32, d.pages)
	for block := range d.pages {
		table[block] = int32(block % d.slots)
	}
	if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
		return err
	}
	if err := state.Driver.MemcpyHtoD(d.table, int32Words(table)); err != nil {
		return err
	}
	if d.slots == d.pages {
		return nil
	}
	hostRows := uint64(d.pages) * heads * page
	var hostCount strataRoundBump
	carve(&hostCount, hostRows)
	host, device, err := state.Driver.MemHostAlloc(hostCount.used)
	if err != nil {
		return fmt.Errorf("mtp: no pinned RAM left for the draft layer's K/V copy: %w", err)
	}
	d.hostKV = strataRoundMapped{host, device}
	hostBump := strataRoundBump{base: device}
	d.host = carve(&hostBump, hostRows)
	return nil
}

// strataKVScaleBytes is an fp16 scale's bytes.
const strataKVScaleBytes = 2

// release frees what load made.
func (d *StrataDrafter) release(state *device.State) error {
	var errs []error
	for _, graphs := range []*[strataDrafterGraphs]driver.GraphExec{&d.prefillExec, &d.roundExec, &d.stepExec} {
		for index, exec := range graphs {
			if exec != 0 {
				errs = append(errs, state.Driver.GraphExecDestroy(exec))
				graphs[index] = 0
			}
		}
	}
	for _, stream := range []*driver.Stream{&d.cs, &d.side} {
		if *stream != 0 {
			errs = append(errs, state.Driver.StreamDestroy(*stream))
			*stream = 0
		}
	}
	for _, pointer := range []*driver.DevicePtr{&d.arena, &d.kvArena, &d.headLogits, &d.prefillRecords, &d.head, &d.vocabulary} {
		if *pointer != 0 {
			errs = append(errs, state.Driver.MemFree(*pointer))
			*pointer = 0
		}
	}
	for _, part := range append(d.mappedParts(0), struct {
		part  *strataRoundMapped
		bytes uint64
	}{&d.hostKV, 0}) {
		if part.part.host != nil {
			errs = append(errs, state.Driver.MemFreeHost(part.part.host))
			*part.part = strataRoundMapped{}
		}
	}
	return errors.Join(errs...)
}

// CloseStrataDrafter releases the drafter on the executor's worker once the card has finished with it.
func (e *Executor) CloseStrataDrafter(ctx context.Context, d *StrataDrafter) error {
	if d == nil {
		return nil
	}
	return e.worker.Do(ctx, func(state *device.State) error {
		return errors.Join(state.Driver.StreamSynchronize(d.cs), state.Driver.StreamSynchronize(d.side), d.release(state))
	})
}

// SetStrataDrafterPrompt is set_prompt_len: prefill skips the cells the attention window can never reach again.
func (d *StrataDrafter) SetStrataDrafterPrompt(cells int64) { d.prompt = cells }

// SetStrataDrafterWindow is bind's window_R: the verify window whose final residual rows the drafter's rounds read; a
// new address drops the round graphs that read the old one.
func (e *Executor) SetStrataDrafterWindow(ctx context.Context, d *StrataDrafter, residual driver.DevicePtr) error {
	if d.spec.WindowResidual == residual {
		return nil
	}
	return e.worker.Do(ctx, func(state *device.State) error {
		errs := []error{state.Driver.StreamSynchronize(d.cs)}
		for index, exec := range d.roundExec {
			if exec != 0 {
				errs = append(errs, state.Driver.GraphExecDestroy(exec))
				d.roundExec[index] = 0
			}
		}
		d.spec.WindowResidual = residual
		return errors.Join(errs...)
	})
}

// FinalResidual is the verify window's final residual rows (final_R_all): the drafter's window_R.
func (r *StrataRound) FinalResidual() driver.DevicePtr { return r.residual }
