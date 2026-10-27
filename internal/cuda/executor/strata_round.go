package executor

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// Strata's decode round (strata-port/decode), ported as one unit from Strata at 296b59d2 (MIT): src/core/verify.cpp's
// Verifier at the bench's configuration (tmp/strata-port/verifier-notes.txt resolves each switch it reads): its device
// arena and mapped staging carved as init carves them (this file), record_window's launches in its order
// (strata_round_window.go), one captured graph a window size, the inputs and outputs through mapped memory, the host
// serving each layer's doorbell while the graph runs, and the commit graph (strata_round_run.go).

// The round's fixed geometry: Flash-Next's, which the ported kernels are compiled for and init checks
// (fused_gr_supported, ss.k == 10, ssm_state_size == 128, ssm_d_conv == 4).
const (
	strataRoundRouted      = 10  // ss.k: the routed experts a token
	strataRoundExperts     = 512 // n_expert
	strataRoundStateSize   = 128 // ssm_state_size
	strataRoundConvTaps    = 4   // ssm_d_conv
	strataRoundMaxTokens   = 8   // kVerifyMaxT, and the host pool's MAXT
	strataRoundStepCount   = tensor.QSAStepWords
	strataRoundHistoryRows = 9 // NG_HIST: the PLE history's rows a channel
	strataRoundArenaAlign  = 256
	// strataRoundWordBytes is an int32's and a float's bytes, the arena's elements; a q8_0 block is 34 bytes.
	strataRoundWordBytes  = 4
	strataRoundQ80Bytes   = 34
	strataRoundTailDeltas = tensor.QSATailRows * tensor.QSAIndexerWidth
	// strataRoundStagingBlobs and strataRoundPCIeGroupRows are verify.hpp's kStagingBlobs and kPcieGroupRows: the PCIe
	// share's most blobs a window, and its call's groups side by side.
	strataRoundStagingBlobs  = 16
	strataRoundPCIeGroupRows = 4
	// strataRoundFlagBytes is a mapped flag word's allocation (mapped(64, ...)).
	strataRoundFlagBytes = 64
	// strataRoundPlanHeader and strataRoundPlanPad are the plan's counts (4) and the 16 words verify.cpp pads each
	// group's copy with.
	strataRoundPlanHeader = 4
	strataRoundPlanPad    = 16
	// strataRoundCommitHeader is h_commit_'s n_keep and n_keep - 1 before the kept positions; strataRoundPad16 the
	// 16 bytes mapped() adds to the commit and output words.
	strataRoundCommitHeader = 2
	strataRoundPad16        = 16
	strataRoundOneWords     = 4 // one_: four words, the first 1
	strataRoundShGWords     = 4 // sh_g_'s T + 4
	strataRoundHitCounts    = 4 // hit_count_
	// strataRoundBF16Bytes is a bf16 value's bytes; strataRoundQGate the q|gate rows of attn_q's output (2 * HD a head);
	// strataRoundPlanGroups the plan's copies, one a token group of a split window.
	strataRoundBF16Bytes  = 2
	strataRoundQGate      = 2
	strataRoundPlanGroups = 2
)

// StrataRoundWeight is a weight as the round reads it: its device address and, for a native projection, its GGML type
// (WeightRef's native_data and native_type).
type StrataRoundWeight struct {
	Pointer driver.DevicePtr
	Type    int
}

// StrataRoundHyper is a hyper-connection read's weights (FusedGrArgs): the norm (f32), the down, up and inject
// projections as bf16 (w_down, w_up, w_inject) and the down and up projections as the GGUF's Q8_0 (hc_q8, STRATA_HC_Q8);
// the inject's Q8_0 where the file stores it so, else 0.
type StrataRoundHyper struct {
	Norm, Down, Up, Inject, Q8Down, Q8Up, Q8Inject driver.DevicePtr
}

// StrataRoundQSAState is a QSA layer's state (QsaState): its int8 K/V codes and scales behind its page table, the
// indexer's tail, spare (idx_dead), pooled keys and block position, the cells it holds; a streamed state adds its host
// pools and its residency map (KvHostPools, the stream's map), zero otherwise.
type StrataRoundQSAState struct {
	KeyCodes, ValueCodes, KeyScales, ValueScales, PageTable driver.DevicePtr
	Tail, Spare, Pooled, BlockPosition                      driver.DevicePtr
	HostKeyCodes, HostValueCodes, HostKeyScales             driver.DevicePtr
	HostValueScales, StreamMap                              driver.DevicePtr
	MaxCells, Slots                                         int64
}

// StrataRoundLayer is a layer's weights and state.
type StrataRoundLayer struct {
	QSA            bool
	Attention, FFN StrataRoundHyper
	// The gated-delta mixer: attn_qkv, attn_gate and ssm_out natively; ssm_alpha and ssm_beta as bf16; ssm_conv1d,
	// ssm_norm, ssm_dt and ssm_a as f32; the layer's recurrent state and conv history.
	QKV, AttnGate, SSMOut         StrataRoundWeight
	SSMAlpha, SSMBeta             driver.DevicePtr
	SSMConv, SSMNorm, SSMDT, SSMA driver.DevicePtr
	State, History                driver.DevicePtr
	// The QSA mixer: the indexer's k and q projections as bf16, attn_q, attn_k, attn_v and attn_output natively, the
	// norms as f32, and the layer's state.
	IndexerK, IndexerQ                             driver.DevicePtr
	Query, Key, Value, Output                      StrataRoundWeight
	QueryNorm, KeyNorm, IndexerQNorm, IndexerKNorm driver.DevicePtr
	Cells                                          StrataRoundQSAState
	// The MoE: the router and the shared expert's gate input as bf16, the shared expert natively, the routed experts'
	// formats (their blob: gate, up and down rows as the GGUF stores them, NativeExpertLayout).
	Router, SharedGateInput          driver.DevicePtr
	SharedGate, SharedUp, SharedDown StrataRoundWeight
	ExpertGateUp, ExpertDown         dtype.Type
}

// StrataRoundPLE is the lexical injection's weights (PleWeights): its key projection natively (key_native_data, Q8_0),
// its value projection as bf16, its norms, and its conv as f16. Its history (ss.ple.hist, a channel's nine rows
// together) is the round's own (StrataRoundHistory).
type StrataRoundPLE struct {
	Key                                   StrataRoundWeight
	Value                                 driver.DevicePtr
	NormKey, NormQuery, NormConv, ConvF16 driver.DevicePtr
}

// StrataRoundSpec is everything the round reads and writes outside its own arena: the model's geometry beyond the
// fixed one, each layer's weights and state, the embedding, the head, the experts' residency table and the PLE.
type StrataRoundSpec struct {
	Layers       []StrataRoundLayer
	Embedding    driver.DevicePtr // token_embd (Q8_0)
	EmbeddingRow uint64           // a row's bytes
	Head         StrataRoundWeight
	HeadHyper    StrataRoundHyper
	Vocab        int64
	SharedFF     int64            // n_ff
	ConvChannels int64            // ssm_conv_channels
	ValueDim     int64            // ssm_value_dim
	ValueHeads   int64            // ssm_v_heads
	KeyHeads     int64            // ssm_k_heads
	MaxTokens    int              // max_t
	Positions    driver.DevicePtr // the multi-axis position table (mrope_table)
	Resident     driver.DevicePtr // hits_.d_res: [layer][expert] int32, a card slot or -1
	ExpertFF     int64            // the routed experts' n_ff
	PLE          StrataRoundPLE
	RopeFreqBase float64
}

// strataRoundBump is verify.cpp's Bump: an arena counted, then carved, each take 256-byte aligned.
type strataRoundBump struct {
	base driver.DevicePtr
	used uint64
}

func (b *strataRoundBump) take(bytes uint64) driver.DevicePtr {
	pointer := driver.DevicePtr(0)
	if b.base != 0 {
		pointer = b.base + driver.DevicePtr(b.used)
	}
	b.used += (bytes + strataRoundArenaAlign - 1) &^ (strataRoundArenaAlign - 1)
	return pointer
}

// strataRoundMapped is a mapped staging word or array: its host view and the address the card reads it at.
type strataRoundMapped struct {
	host   []byte
	device driver.DevicePtr
}

// StrataRound is the Verifier.
type StrataRound struct {
	spec StrataRoundSpec
	// resources is its executor's, through whose reservation its allocations and graphs make room on the card.
	resources *executorResources
	// layouts are each layer's expert blob (native_expert_layout), maxBlob the largest (expert_layout().max_blob);
	// rope the QSA layers' rope scaling; major and minor the card's compute capability (the interleaved rows table).
	layouts      []strataNativeExpertLayout
	maxBlob      uint64
	rope         strataRopeScaling
	major, minor int
	// pleHistory is the sequence's PLE history (ss.ple.hist): each channel's nine rows together, the oldest first.
	pleHistory driver.DevicePtr
	// The mapped staging (h_*, m_*).
	tok, step, pos, commit, ple, out, nonfinite, x, ids, weights strataRoundMapped
	seq, flag, flagA, flagB, planErr, ymiss, plan                strataRoundMapped
	planWords                                                    int64 // plan_i32_
	// The device arena and its parts.
	arena                                                            driver.DevicePtr
	arenaBytes                                                       uint64
	dTok, dStep, dPos, dCommit, dPLE, emb, residual, mixed, blockOut driver.DevicePtr
	inject, inject2, lo, rs, xn, xq, shXQ, xil                       driver.DevicePtr
	qkvL, hL, gateL, betaL, z, y, yDummy                             driver.DevicePtr
	qFull, qCur, kCur, vCur, idxRawL, qIdx, scores, sel              driver.DevicePtr
	attn, attn32, attnScratch, tailSnap                              driver.DevicePtr
	logits, w, idsD, shared, parts, hitOut, hitSlot, hitDst          driver.DevicePtr
	hitCount, planD, staging, hitXQ, hitXS, natXQ, hitScratch        driver.DevicePtr
	headMixed, headInject, shBF16, shGate, shUp, shG, headLogits     driver.DevicePtr
	argScratch, one, histSnap, pleKey, pleVal                        driver.DevicePtr
	// ple_block's workspace (ss.ple.scratch), the normalized row after it, and the key's q8_1 (key_native_q8_1).
	pleScratch, pleNormalized, pleKeyQ81 driver.DevicePtr
	// join is the event the shared expert's stream records after a layer's shared expert (ev_join_).
	join                                   driver.Event
	capacity, maxBlocks, attnScratchFloats int64
	// The streams (cs_, sh_cs_, copy_) and the graphs a window size, and the commit's.
	cs, shCS, copies driver.Stream
	exec             [strataRoundMaxTokens + 1]driver.GraphExec
	commitExec       driver.GraphExec
	// sampling is the request's sampling, nil for greedy (set_sampling), and samplerScratch the split sampler's lists.
	sampling       *StrataSampling
	samplerScratch driver.DevicePtr
	lastT          int
	lastPos0       int64
	lastTokens     [strataRoundMaxTokens]int32
}

// strataRoundMapNames is the order the mapped staging is allocated in (init's mapped() calls).
func (r *StrataRound) mappedParts(maxT, heads uint64) []struct {
	part  *strataRoundMapped
	bytes uint64
} {
	n := uint64(strataGREmbedding)
	k := uint64(strataRoundRouted)
	return []struct {
		part  *strataRoundMapped
		bytes uint64
	}{
		{&r.tok, maxT * strataRoundWordBytes},
		{&r.step, maxT * strataRoundStepCount * strataRoundWordBytes},
		{&r.pos, maxT * heads * strataRoundWordBytes},
		{&r.commit, (strataRoundCommitHeader+maxT)*strataRoundWordBytes + strataRoundPad16},
		{&r.ple, maxT * n * strataRoundWordBytes},
		{&r.out, maxT*strataRoundWordBytes + strataRoundPad16},
		{&r.nonfinite, strataRoundMaxTokens * strataNonfiniteBlocks * strataRoundWordBytes},
		{&r.x, maxT * n * strataRoundWordBytes},
		{&r.ids, maxT * k * strataRoundWordBytes},
		{&r.weights, maxT * k * strataRoundWordBytes},
		{&r.seq, strataRoundFlagBytes},
		{&r.flag, strataRoundFlagBytes},
		{&r.flagA, strataRoundFlagBytes},
		{&r.flagB, strataRoundFlagBytes},
		{&r.planErr, strataRoundFlagBytes},
		{&r.ymiss, maxT * k * n * strataRoundWordBytes},
	}
}

// strataRoundPositionHeads is pos_'s rows a token: the query heads', the K heads' and the indexer query heads'.
const strataRoundPositionHeads = tensor.QSAQueryHeads + tensor.QSAKeyValueHeads + tensor.QSAIndexerHeads

// NewStrataRound is Verifier::init on the executor's worker: the spec checked, the mapped staging and the plan, the
// arena counted then carved, one_ set, and the window's, shared expert's and copies' streams.
func (e *Executor) NewStrataRound(ctx context.Context, spec StrataRoundSpec) (*StrataRound, error) {
	if spec.MaxTokens < strataNativeMultiFirst || spec.MaxTokens > strataRoundMaxTokens {
		return nil, fmt.Errorf("strata round: the window must hold 2..%d tokens", strataRoundMaxTokens)
	}
	if spec.Resident == 0 {
		return nil, errors.New("strata round: needs the experts' residency table")
	}
	r := &StrataRound{spec: spec, resources: &e.resources, rope: strataRopeScaling{freqBase: spec.RopeFreqBase, factor: 1, attnFactor: 1}}
	for l, layer := range spec.Layers {
		layout, err := strataNativeExpertLayoutOf(layer.ExpertGateUp, layer.ExpertDown, strataGREmbedding, spec.ExpertFF)
		if err != nil {
			return nil, fmt.Errorf("strata round layer %d's experts: %w", l, err)
		}
		r.layouts, r.maxBlob = append(r.layouts, layout), max(r.maxBlob, layout.Bytes)
	}
	err := e.worker.Do(ctx, func(state *device.State) error { return r.init(state) })
	if err != nil {
		return nil, errors.Join(err, e.worker.Do(context.WithoutCancel(ctx), func(state *device.State) error { return r.release(state) }))
	}
	return r, nil
}

func (r *StrataRound) init(state *device.State) error {
	var err error
	if r.major, r.minor, err = state.Driver.ComputeCapability(state.Device); err != nil {
		return err
	}
	spec := r.spec
	maxT := uint64(spec.MaxTokens)
	n, hc, k := uint64(strataGREmbedding), uint64(strataGRStreams), uint64(strataRoundRouted)
	c, zv, hv := uint64(spec.ConvChannels), uint64(spec.ValueDim), uint64(spec.ValueHeads)
	nh, hd, nkv := uint64(tensor.QSAQueryHeads), uint64(tensor.QSAHeadWidth), uint64(tensor.QSAKeyValueHeads)
	iq, id := uint64(tensor.QSAIndexerHeads), uint64(tensor.QSAIndexerWidth)
	var nG, nQ uint64
	for _, layer := range spec.Layers {
		if layer.QSA {
			nQ++
		} else {
			nG++
		}
	}
	hs := uint64(strataRoundHistoryRows * strataPLEWidth)
	ts := uint64(strataRoundTailDeltas)
	maxIn := int32(max(n, zv, nh*hd))
	maxCells := spec.Layers[0].Cells.MaxCells
	for _, layer := range spec.Layers {
		if layer.QSA {
			maxCells = layer.Cells.MaxCells
			break
		}
	}
	r.capacity = int64(tensor.QSASelectionWidth(strataTopKMaxCells))
	r.maxBlocks = maxCells/tensor.QSABlockCells + tensor.QSAPooledSpare
	r.attnScratchFloats = strataAttentionScratchFloats(r.capacity, tensor.QSAQueryHeads)
	for _, part := range r.mappedParts(maxT, strataRoundPositionHeads) {
		host, device, err := state.Driver.MemHostAlloc(part.bytes)
		if err != nil {
			return fmt.Errorf("strata round: mapped staging allocation failed: %w", err)
		}
		clear(host)
		part.part.host, part.part.device = host, device
	}
	// The GPU plan: counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) | ptr2(cap u64) | start2(cap+1).
	planCap := int64(maxT * k)
	words := strataRoundPlanHeader + (planCap + 1) + planCap + planCap
	pointerOffset := (words + 1) &^ 1
	r.planWords = pointerOffset + strataRoundPlanHeader*planCap + (planCap + 1) + 1
	host, device, err := state.Driver.MemHostAlloc(uint64(r.planWords)*strataRoundWordBytes*strataRoundPlanGroups + strataRoundFlagBytes)
	if err != nil {
		return fmt.Errorf("strata round: mapped plan allocation failed: %w", err)
	}
	clear(host)
	r.plan = strataRoundMapped{host, device}
	carve := func(b *strataRoundBump) {
		r.dTok, r.dStep = b.take(maxT*strataRoundWordBytes), b.take(maxT*strataRoundStepCount*strataRoundWordBytes)
		r.dPos, r.dCommit = b.take(maxT*strataRoundPositionHeads*strataRoundWordBytes), b.take((strataRoundCommitHeader+maxT)*strataRoundWordBytes)
		r.dPLE, r.emb, r.residual = b.take(maxT*n*strataRoundWordBytes), b.take(maxT*n*strataRoundWordBytes), b.take(maxT*hc*n*strataRoundWordBytes)
		r.mixed, r.blockOut = b.take(maxT*n*strataRoundWordBytes), b.take(maxT*n*strataRoundWordBytes)
		r.inject, r.inject2 = b.take(maxT*hc*strataRoundWordBytes), b.take(maxT*hc*strataRoundWordBytes)
		r.lo, r.rs, r.xn = b.take(maxT*strataGRRank*strataRoundWordBytes), b.take(maxT*hc*strataRoundWordBytes), b.take(maxT*hc*n*strataRoundWordBytes)
		r.xq, r.shXQ = b.take(strataNativeQ81Size(maxIn, int32(maxT))), b.take(strataNativeQ81Size(maxIn, int32(maxT)))
		r.xil = b.take(strataNativeQ81ILSize(maxIn, strataNativeILMaxColumns))
		r.qkvL, r.hL = b.take(nG*maxT*c*strataRoundWordBytes), b.take(nG*maxT*c*strataRoundWordBytes)
		r.gateL, r.betaL = b.take(nG*maxT*hv*strataRoundWordBytes), b.take(nG*maxT*hv*strataRoundWordBytes)
		r.z, r.y, r.yDummy = b.take(maxT*zv*strataRoundWordBytes), b.take(maxT*zv*strataRoundWordBytes), b.take(maxT*zv*strataRoundWordBytes)
		r.qFull, r.qCur = b.take(maxT*nh*strataRoundQGate*hd*strataRoundWordBytes), b.take(maxT*nh*hd*strataRoundWordBytes)
		r.kCur, r.vCur = b.take(maxT*nkv*hd*strataRoundWordBytes), b.take(maxT*nkv*hd*strataRoundWordBytes)
		r.idxRawL, r.qIdx = b.take(nQ*maxT*id*strataRoundWordBytes), b.take(maxT*iq*id*strataRoundWordBytes)
		r.scores, r.sel = b.take(maxT*uint64(r.maxBlocks)*strataRoundWordBytes), b.take(maxT*uint64(r.capacity)*strataRoundWordBytes)
		r.attn, r.attn32 = b.take(maxT*nh*hd*strataRoundWordBytes), b.take(maxT*nh*hd*strataRoundWordBytes)
		r.attnScratch = b.take(maxT * uint64(r.attnScratchFloats) * strataRoundWordBytes)
		r.tailSnap = b.take(nQ * ts * strataRoundWordBytes)
		r.logits, r.w, r.idsD = b.take(maxT*strataRoundExperts*strataRoundWordBytes), b.take(maxT*k*strataRoundWordBytes), b.take(maxT*k*strataRoundWordBytes)
		r.shared, r.parts, r.hitOut = b.take(maxT*n*strataRoundWordBytes), b.take(maxT*k*n*strataRoundWordBytes), b.take(maxT*k*n*strataRoundWordBytes)
		r.hitSlot, r.hitDst, r.hitCount = b.take(maxT*k*strataRoundWordBytes), b.take(maxT*k*strataRoundWordBytes), b.take(strataRoundHitCounts*strataRoundWordBytes)
		r.planD = b.take(strataRoundPlanGroups * uint64(r.planWords+strataRoundPlanPad) * strataRoundWordBytes)
		r.staging = b.take(strataRoundStagingBlobs * r.maxBlob)
		r.pleHistory = b.take(strataRoundHistoryRows * strataPLEWidth * strataRoundWordBytes)
		r.hitXQ, r.hitXS = b.take(maxT*(n/strataNativeQ8K)*strataRoundQ80Bytes), b.take(maxT*(n/strataNativeQ8K)*strataRoundWordBytes)
		r.natXQ = b.take(maxT * (n / strataNativeQ8K) * strataNativeQ81Bytes)
		// hit_scratch_: the native experts' scratch (the S2 grouped kernels' larger share is not ported: Flash-Next's
		// experts are native).
		r.hitScratch = b.take(strataNativeExpertScratchBytes(int64(maxT*k), spec.SharedFF))
		r.headMixed, r.headInject = b.take(maxT*n*strataRoundWordBytes), b.take(hc*strataRoundWordBytes)
		r.shBF16 = b.take(maxT * n * strataRoundBF16Bytes)
		r.shGate, r.shUp = b.take(maxT*uint64(spec.SharedFF)*strataRoundWordBytes), b.take(maxT*uint64(spec.SharedFF)*strataRoundWordBytes)
		r.shG = b.take((maxT + strataRoundShGWords) * strataRoundWordBytes)
		r.headLogits = b.take(maxT * uint64(spec.Vocab) * strataRoundWordBytes)
		r.argScratch = b.take(strataArgmaxScratchSize(int32(maxT)))
		r.one = b.take(strataRoundOneWords * strataRoundWordBytes)
		r.histSnap = b.take(maxT * hs * strataRoundWordBytes)
		r.pleKey, r.pleVal = b.take(maxT*strataPLEWidth*strataRoundWordBytes), b.take(maxT*n*strataRoundWordBytes)
		r.pleScratch = b.take((strataRoundPLEParts*strataPLEWidth + n + hc) * strataRoundWordBytes)
		r.pleNormalized = b.take(strataPLEWidth * strataRoundWordBytes)
		r.pleKeyQ81 = b.take(strataNativeQ81Size(strataGREmbedding, 1))
	}
	var count strataRoundBump
	carve(&count)
	if r.arena, err = r.resources.allocate(state, count.used); err != nil {
		return fmt.Errorf("strata round: the device arena (%d MiB) does not fit: %w", count.used>>strataRoundMiBShift, err)
	}
	r.arenaBytes = count.used
	// cudaMemset(arena_, 0) then cudaMemcpy(one_): the clear lands before one_ is set, the copy not ordered after work on
	// the executor's stream.
	if err := state.Driver.MemsetD32Async(r.arena, 0, count.used/strataRoundWordBytes, state.Stream); err != nil {
		return err
	}
	if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
		return err
	}
	carve(&strataRoundBump{base: r.arena})
	if err := state.Driver.MemcpyHtoD(r.one, int32Words([]int32{1})); err != nil {
		return fmt.Errorf("strata round: the arena could not be set: %w", err)
	}
	for _, stream := range []*driver.Stream{&r.copies, &r.cs, &r.shCS} {
		if *stream, err = state.Driver.StreamCreate(strataRoundStreamNonBlocking); err != nil {
			return fmt.Errorf("strata round: stream create failed: %w", err)
		}
	}
	return state.Driver.StreamSynchronize(state.Stream)
}

// strataRoundStreamNonBlocking is cudaStreamNonBlocking; strataRoundMiBShift a byte count's MiB.
const (
	strataRoundStreamNonBlocking = 1
	strataRoundMiBShift          = 20
)

// int32Words is values' little-endian bytes.
func int32Words(values []int32) []byte {
	data := make([]byte, 0, len(values)*strataRoundWordBytes)
	for _, value := range values {
		data = binary.LittleEndian.AppendUint32(data, uint32(value))
	}
	return data
}

// release frees what init made.
func (r *StrataRound) release(state *device.State) error {
	var errs []error
	for index, exec := range r.exec {
		if exec != 0 {
			errs = append(errs, state.Driver.GraphExecDestroy(exec))
			r.exec[index] = 0
		}
	}
	if r.commitExec != 0 {
		errs = append(errs, state.Driver.GraphExecDestroy(r.commitExec))
		r.commitExec = 0
	}
	for _, stream := range []*driver.Stream{&r.cs, &r.shCS, &r.copies} {
		if *stream != 0 {
			errs = append(errs, state.Driver.StreamDestroy(*stream))
			*stream = 0
		}
	}
	if r.arena != 0 {
		errs = append(errs, state.Driver.MemFree(r.arena))
		r.arena = 0
	}
	if r.samplerScratch != 0 {
		errs = append(errs, state.Driver.MemFree(r.samplerScratch))
		r.samplerScratch = 0
	}
	for _, part := range append(r.mappedParts(0, 0), struct {
		part  *strataRoundMapped
		bytes uint64
	}{&r.plan, 0}) {
		if part.part.host != nil {
			errs = append(errs, state.Driver.MemFreeHost(part.part.host))
			*part.part = strataRoundMapped{}
		}
	}
	return errors.Join(errs...)
}

// CloseStrataRound releases the round on the executor's worker once the card has finished with it.
func (e *Executor) CloseStrataRound(ctx context.Context, r *StrataRound) error {
	if r == nil {
		return nil
	}
	return e.worker.Do(ctx, func(state *device.State) error {
		return errors.Join(state.Driver.StreamSynchronize(r.cs), r.release(state))
	})
}
