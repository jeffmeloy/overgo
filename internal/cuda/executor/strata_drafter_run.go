package executor

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"time"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
)

// mtp.cpp's record_forward, record_front and record_rest, capture_prefill_dev, capture_round and capture_step, prefill
// (E-4: the records uploaded at once, each group's inputs copied on the card) and draft (min_p > 0: the round graph,
// then a step graph each while the last draft's probability reaches min_p, each output read from mapped memory as it
// lands).

// strataDrafterRecord is one recording's view: the drafter, its kernels and the device state.
type strataDrafterRecord struct {
	*StrataDrafter
	state     *device.State
	functions functionSet
}

// onStream runs launches on stream, as a captured launch onto the drafter's side stream.
func (w *strataDrafterRecord) onStream(stream driver.Stream, launches func() error) error {
	saved := w.state.Stream
	w.state.Stream = stream
	defer func() { w.state.Stream = saved }()
	return launches()
}

// stepAt and posAt are a step row's record and positions.
func (w *strataDrafterRecord) stepAt(row int64) driver.DevicePtr {
	return w.dStep + driver.DevicePtr(row*strataDrafterStepWords*strataRoundWordBytes)
}
func (w *strataDrafterRecord) posAt(row int64) driver.DevicePtr {
	return w.dPos + driver.DevicePtr(row*strataDrafterHeads*strataRoundWordBytes)
}

// hyper is a hyper-connection read's args for token t of the drafter's rows.
func (w *strataDrafterRecord) hyper(weights StrataRoundHyper, t int64, apply bool, blockOut, injPrev, injectOut, mixed driver.DevicePtr) strataFusedGRArgs {
	args := strataFusedGRArgs{
		R: at(w.residual, t*strataGRStreams*strataGREmbedding), WNorm: weights.Norm, WDown: weights.Down, WUp: weights.Up,
		WInject: weights.Inject, Eps: strataRoundEpsilon, Lo: at(w.lo, t*strataGRRank), Rs: at(w.rs, t*strataGRStreams),
		Mixed: at(mixed, t*strataGREmbedding),
	}
	args.ROut = args.R
	if weights.Inject != 0 {
		args.InjectOut = at(injectOut, t*strataGRStreams)
	}
	if apply {
		args.Apply, args.BoPrev, args.InjPrev = 1, at(blockOut, t*strataGREmbedding), at(injPrev, t*strataGRStreams)
	}
	return args
}

// front is record_front: the layer's front for t rows at step rows [row0, +t): the embedding, the fc projections, the
// attention hyper-connection read (R_, inj_, mixed_) and the K/V appended.
func (w *strataDrafterRecord) front(t, row0 int64) error {
	s := w.spec
	n, hc := int64(strataGREmbedding), int64(strataGRStreams)
	nkv, hd := int64(tensor.QSAKeyValueHeads), int64(tensor.QSAHeadWidth)
	step, pos := w.stepAt(row0), w.posAt(row0)
	// ---- the two input branches
	if err := launchStrataEmbedRowsQ80(w.state, w.functions, s.Embedding, s.EmbeddingRow, w.dTok, int32(t), n, w.emb); err != nil {
		return err
	}
	if err := launchStrataNativeQSANorm(w.state, w.functions, w.emb, s.EmbeddingNorm, w.en, int32(n), int32(t), strataRoundEpsilon); err != nil {
		return err
	}
	if err := launchStrataNativeQuantizeQ81(w.state, w.functions, w.en, w.xq, int32(n), int32(t)); err != nil {
		return err
	}
	if err := launchStrataNativeMMVQ(w.state, w.functions, s.FCEmbedding.Type, s.FCEmbedding.Pointer, w.xq, w.e2,
		int32(n), int32(n), int32(t)); err != nil {
		return err
	}
	if err := launchStrataNativeQSANorm(w.state, w.functions, w.rin, s.HiddenNorm, w.hn, int32(hc*n), int32(t), strataRoundEpsilon); err != nil {
		return err
	}
	for c0 := int64(0); c0 < t*hc; c0 += strataDrafterHiddenChunk {
		nc := min(strataDrafterHiddenChunk, t*hc-c0)
		if err := launchStrataNativeQuantizeQ81(w.state, w.functions, at(w.hn, c0*n), w.xq, int32(n), int32(nc)); err != nil {
			return err
		}
		if err := launchStrataNativeMMVQ(w.state, w.functions, s.FCHidden.Type, s.FCHidden.Pointer, w.xq, at(w.h2, c0*n),
			int32(n), int32(n), int32(nc)); err != nil {
			return err
		}
	}
	if err := launchStrataAddStreamsBroadcast(w.state, w.functions, w.h2, w.e2, w.residual, n, int32(hc), int32(t)); err != nil {
		return err
	}
	// ---- the attention hyper-connection
	args := make([]strataFusedGRArgs, t)
	for token := range t {
		args[token] = w.hyper(s.Attention, token, false, 0, 0, w.inject, w.mixed)
	}
	if err := launchStrataFusedGRRead(w.state, w.functions, args, w.xn); err != nil {
		return err
	}
	// ---- attention: K/V into the layer's own cache
	if err := launchStrataNativeQuantizeQ81(w.state, w.functions, w.mixed, w.xq, int32(n), int32(t)); err != nil {
		return err
	}
	if err := launchStrataNativeMMVQ(w.state, w.functions, s.Key.Type, s.Key.Pointer, w.xq, w.kCur, int32(n), int32(nkv*hd), int32(t)); err != nil {
		return err
	}
	if err := launchStrataNativeMMVQ(w.state, w.functions, s.Value.Type, s.Value.Pointer, w.xq, w.vCur, int32(n), int32(nkv*hd), int32(t)); err != nil {
		return err
	}
	// #783 PR-f: the K norm and rope fused for a one-token window (K rows of one token are consecutive in pos).
	if t == 1 {
		if err := launchStrataNativeNormRope(w.state, w.functions, w.kCur, int32(hd), s.KeyNorm, w.kCur, int32(nkv), int32(hd),
			strataNativeNormRopeRotation, strataRoundEpsilon, w.rope, pos, 0); err != nil {
			return err
		}
	} else {
		if err := launchStrataNativeQSANorm(w.state, w.functions, w.kCur, s.KeyNorm, w.kCur, int32(hd), int32(t*nkv), strataRoundEpsilon); err != nil {
			return err
		}
		for token := range t {
			keys := at(w.kCur, token*nkv*hd)
			if err := launchStrataNativeRopeApply(w.state, w.functions, keys, keys, int32(nkv), int32(hd),
				strataNativeNormRopeRotation, w.rope, pos+driver.DevicePtr(token*strataDrafterHeads*strataRoundWordBytes), 0); err != nil {
				return err
			}
		}
	}
	geometry := strataKVGeometry{kvHeads: tensor.QSAKeyValueHeads, headDim: tensor.QSAHeadWidth, pageSize: tensor.QSABlockCells}
	return launchStrataKVAppendSteps(w.state, w.functions, w.pools, w.host, w.table, step, w.kCur, w.vCur, geometry,
		int32(t), strataDrafterStepWords, int32(nkv*hd))
}

// rest is record_rest for row 0 at step row stepRow: the attention from the q projection, the MLP, the final mixer,
// the head, the draft and its probability. It needs row 0's R_, inj_ and mixed_ (its q8_1 rows in xq_).
func (w *strataDrafterRecord) rest(stepRow int64) error {
	s := w.spec
	n, nh, hd := int64(strataGREmbedding), int64(tensor.QSAQueryHeads), int64(tensor.QSAHeadWidth)
	step, pos := w.stepAt(stepRow), w.posAt(stepRow)
	// ---- dense attention over every cell of the window
	if err := launchStrataNativeMMVQ(w.state, w.functions, s.Query.Type, s.Query.Pointer, w.xq, w.qFull, int32(n),
		int32(nh*strataRoundQGate*hd), 1); err != nil {
		return err
	}
	if err := launchStrataNativeNormRope(w.state, w.functions, w.qFull, int32(strataRoundQGate*hd), s.QueryNorm, w.qCur,
		int32(nh), int32(hd), strataNativeNormRopeRotation, strataRoundEpsilon, w.rope, pos, 0); err != nil {
		return err
	}
	if w.window > 0 {
		if err := launchStrataWindowIDs(w.state, w.functions, step, 1, int32(w.window), w.ident, w.capacity); err != nil {
			return err
		}
	}
	attention := strataAttentionPools{kv: w.pools, table: w.table}
	if err := launchStrataDecodeAttention(w.state, w.functions, w.qCur, attention, w.ident, step, w.capacity,
		tensor.QSAKeyValueHeads, tensor.QSABlockCells, w.attnScratch, w.attn, 1, true); err != nil {
		return err
	}
	if err := launchStrataNativeQSAGate(w.state, w.functions, w.attn, w.qFull, w.attn32, int32(nh), int32(hd)); err != nil {
		return err
	}
	if err := launchStrataNativeQuantizeQ81(w.state, w.functions, w.attn32, w.xq, int32(nh*hd), 1); err != nil {
		return err
	}
	if err := launchStrataNativeMMVQ(w.state, w.functions, s.Output.Type, s.Output.Pointer, w.xq, w.blockOut, int32(nh*hd), int32(n), 1); err != nil {
		return err
	}
	// ---- the MLP hyper-connection (the attention write folded in)
	if err := launchStrataFusedGRRead(w.state, w.functions,
		[]strataFusedGRArgs{w.hyper(s.FFN, 0, true, w.blockOut, w.inject, w.inject2, w.mixed)}, w.xn); err != nil {
		return err
	}
	// ---- MoE: the shared expert on a branch beside the router and the 512 resident experts, joined before the
	// combine, then the write
	fork, err := w.state.Driver.RecordEvent(w.cs)
	if err != nil {
		return err
	}
	if err := w.state.Driver.WaitEvent(w.side, fork); err != nil {
		return err
	}
	if err := w.onStream(w.side, w.sharedExpert); err != nil {
		return err
	}
	join, err := w.state.Driver.RecordEvent(w.side)
	if err != nil {
		return err
	}
	if err := launchStrataBF16GEMV(w.state, w.functions, w.mixed, s.Router, w.logits, n, strataRoundExperts); err != nil {
		return errors.Join(err, w.state.Driver.ReleaseEvent(join))
	}
	if err := launchStrataNativeRoute(w.state, w.functions, w.logits, w.ids, w.w); err != nil {
		return errors.Join(err, w.state.Driver.ReleaseEvent(join))
	}
	if err := w.experts(); err != nil {
		return errors.Join(err, w.state.Driver.ReleaseEvent(join))
	}
	if err := w.state.Driver.WaitEvent(w.cs, join); err != nil {
		return err
	}
	if err := launchStrataNativeMoECombineMulti(w.state, w.functions, w.parts, w.w, w.shared, w.y, n, strataRoundRouted, 1); err != nil {
		return err
	}
	if err := launchStrataNativeGRPost(w.state, w.functions, w.residual, w.y, w.inject2, w.residual, int32(n), strataGRStreams, 1); err != nil {
		return err
	}
	// ---- the final mixer and the main model's head over the draft vocabulary
	if err := launchStrataFusedGRRead(w.state, w.functions,
		[]strataFusedGRArgs{w.hyper(s.Mixer, 0, false, 0, 0, 0, w.sample)}, w.xn); err != nil {
		return err
	}
	if err := launchStrataNativeQuantizeQ81(w.state, w.functions, w.sample, w.xq, int32(n), 1); err != nil {
		return err
	}
	vocab := int32(len(s.Vocabulary))
	if err := launchStrataNativeMMVQ(w.state, w.functions, s.MainHead.Type, w.head, w.xq, w.headLogits, int32(n), vocab, 1); err != nil {
		return err
	}
	if err := launchStrataArgmaxRows(w.state, w.functions, w.headLogits, 1, vocab, w.argScratch, w.outIDs); err != nil {
		return err
	}
	if err := launchStrataRowTopProbSplit(w.state, w.functions, w.headLogits, 1, vocab, w.outIDs, w.probs, w.topScratch); err != nil {
		return err
	}
	return launchStrataMapIDs(w.state, w.functions, w.outIDs, w.vocabulary, 1)
}

// sharedExpert is shared_expert's native path for row 0: its q8_1 (xq_), gate and up, SwiGLU with its q8_1, down,
// then the gate input's sigmoid scaling the row.
func (w *strataDrafterRecord) sharedExpert() error {
	s := w.spec
	n, ff := int32(strataGREmbedding), int32(s.SharedFF)
	gate, up, g := strataDrafterSharedParts(w.sharedScratch, uint64(ff))
	if err := launchStrataNativeQuantizeQ81(w.state, w.functions, w.mixed, w.xq, n, 1); err != nil {
		return err
	}
	if err := launchStrataNativeMMVQ(w.state, w.functions, s.SharedGate.Type, s.SharedGate.Pointer, w.xq, gate, n, ff, 1); err != nil {
		return err
	}
	if err := launchStrataNativeMMVQ(w.state, w.functions, s.SharedUp.Type, s.SharedUp.Pointer, w.xq, up, n, ff, 1); err != nil {
		return err
	}
	if err := launchStrataNativeSwigluQuantizeQ81(w.state, w.functions, gate, up, w.xq, ff, 1); err != nil {
		return err
	}
	if err := launchStrataNativeMMVQ(w.state, w.functions, s.SharedDown.Type, s.SharedDown.Pointer, w.xq, w.shared, ff, n, 1); err != nil {
		return err
	}
	if err := launchStrataBF16GEMV(w.state, w.functions, w.mixed, s.SharedGateInput, g, int64(n), 1); err != nil {
		return err
	}
	return launchStrataSigmoidScaleRows(w.state, w.functions, w.shared, g, n, 1)
}

// experts is the 512 resident experts through the grouped hit kernels: the window's routes grouped on the card, the
// row's scaled Q8_0, the groups' products into parts_ at each entry's slot.
func (w *strataDrafterRecord) experts() error {
	s := w.spec
	entries := int64(strataRoundRouted)
	if err := launchStrataGroupResident(w.state, w.functions, w.ids, int32(entries), strataRoundRouted, s.Experts, s.ExpertBlob,
		w.groupPointers, w.groupStart, w.groupCounts, w.hitDst, w.hitSlot); err != nil {
		return err
	}
	if err := launchStrataQuantizeQ80Scaled(w.state, w.functions, w.mixed, w.hitXQ, w.hitXS, strataGREmbedding); err != nil {
		return err
	}
	return launchStrataMoEGroupedS2(w.state, w.functions, w.groupPointers, w.groupStart, w.groupCounts, w.hitDst, w.hitSlot,
		entries, entries, w.hitXQ, w.hitXS, w.hitScratch, w.parts)
}

// selectDraft is mtp_select into output j: row *row_ of R_ to Rin_[0], its token to tok_[0], and the token and its
// probability to the mapped outputs.
func (w *strataDrafterRecord) selectDraft(j int32) error {
	return launchStrataMTPSelect(w.state, w.functions, w.residual, strataGRStreams*strataGREmbedding, w.outIDs,
		w.dRow+strataRoundWordBytes, w.rin, w.dTok, w.out.device, j, w.probs, w.prob.device)
}

// capture records launches on the drafter's stream into a graph, instantiated and uploaded into exec.
func (d *StrataDrafter) capture(state *device.State, functions functionSet, exec *driver.GraphExec, record func(w *strataDrafterRecord) error) error {
	if *exec != 0 {
		return nil
	}
	if err := d.resources.buffers.reserveGraph(state); err != nil {
		return err
	}
	if err := state.Driver.StreamBeginCapture(d.cs); err != nil {
		return fmt.Errorf("mtp: begin capture: %w", err)
	}
	w := &strataDrafterRecord{StrataDrafter: d, state: state, functions: functions}
	recorded := w.onStream(d.cs, func() error { return record(w) })
	graph, ended := state.Driver.StreamEndCapture(d.cs)
	if recorded != nil || ended != nil {
		return errors.Join(recorded, ended, state.Driver.GraphDestroy(graph))
	}
	instance, err := state.Driver.GraphInstantiate(graph)
	if err = errors.Join(err, state.Driver.GraphDestroy(graph)); err != nil {
		return fmt.Errorf("mtp: capture: %w", err)
	}
	*exec = instance
	// an explicit upload: the first launch's implicit one blocked behind a device-side spin (verify.cpp)
	return errors.Join(state.Driver.GraphUpload(instance, d.cs), state.Driver.StreamSynchronize(d.cs))
}

// recordPrefill is capture_prefill_dev's graph: the K/V only for rows [0, t), tok_, step_, pos_ and Rin_ filled before
// it launches.
func (w *strataDrafterRecord) recordPrefill(t int64) error { return w.front(t, 0) }

// recordRound is capture_round's graph: the catch-up (the layer's front for the window's t cells, their K/V), then its
// rest for row a only, on row a's intermediates copied to row 0, at the cell staged in step row 2 max_t - 1; then the
// selection of draft 0.
func (w *strataDrafterRecord) recordRound(t int64) error {
	hcn := int64(strataGRStreams * strataGREmbedding)
	ra := strataDrafterStepSets*int64(w.spec.MaxTokens) - 1
	nh := int64(strataDrafterHeads)
	// the draft round's seven inputs in one launch
	if err := launchStrataCopyFromMappedMulti(w.state, w.functions, []strataMappedCopy{
		{w.dTok, w.tok.device, t}, {w.dStep, w.step.device, strataDrafterStepSets * t * strataDrafterStepWords}, {w.dPos, w.pos.device, strataDrafterStepSets * t * nh},
		{w.dRow, w.row.device, strataDrafterRowRead}, {w.rin, w.spec.WindowResidual, t * hcn}, {w.stepAt(ra), w.step.device + driver.DevicePtr(ra*strataDrafterStepWords*strataRoundWordBytes), strataDrafterStepWords},
		{w.posAt(ra), w.pos.device + driver.DevicePtr(ra*nh*strataRoundWordBytes), nh},
	}); err != nil {
		return err
	}
	if err := w.front(t, 0); err != nil {
		return err
	}
	if t > 1 {
		if err := launchStrataCopyRowToFirst(w.state, w.functions, w.dRow, w.residual, hcn, w.inject, strataGRStreams,
			w.mixed, strataGREmbedding); err != nil {
			return err
		}
		if err := launchStrataNativeQuantizeQ81(w.state, w.functions, w.mixed, w.xq, strataGREmbedding, 1); err != nil {
			return err
		}
	}
	if err := w.rest(ra); err != nil {
		return err
	}
	return w.selectDraft(0)
}

// recordStep is capture_step's graph for chain step j: one row at the cell staged in step row max_t + j - 1, from the
// previous step's residual and token (left in Rin_[0] and tok_[0] by mtp_select), draft j to the mapped outputs.
func (w *strataDrafterRecord) recordStep(j int64) error {
	row := int64(w.spec.MaxTokens) + j - 1
	nh := int64(strataDrafterHeads)
	if err := launchStrataCopyFromMappedMulti(w.state, w.functions, []strataMappedCopy{
		{w.stepAt(row), w.step.device + driver.DevicePtr(row*strataDrafterStepWords*strataRoundWordBytes), strataDrafterStepWords},
		{w.posAt(row), w.pos.device + driver.DevicePtr(row*nh*strataRoundWordBytes), nh},
	}); err != nil {
		return err
	}
	if err := w.front(1, row); err != nil {
		return err
	}
	if err := w.rest(row); err != nil {
		return err
	}
	return w.selectDraft(int32(j))
}

// putStep is draft's put: step row's record and every head's position for cell.
func (d *StrataDrafter) putStep(row int, cell int64) {
	step, pos := d.step.words(), d.pos.words()
	record := step[row*strataDrafterStepWords:]
	record[0], record[1] = int32(cell), int32(cell+1)
	record[strataRoundStepBlocks], record[strataRoundStepWidth] = int32((cell+1)/tensor.QSABlockCells), int32(cell+1)
	for h := range strataDrafterHeads {
		pos[row*strataDrafterHeads+h] = int32(cell)
	}
}

// strataDrafterSpinQuery is how many spins pass between looks at the clock, which asks every strataRoundQueryInterval
// whether the stream ended without an output (draft's 1024 spins).
const strataDrafterSpinQuery = 1 << 10

// StrataDraft is a round's drafts: the tokens and their probabilities under the draft layer, Count of them made.
type StrataDraft struct {
	Tokens        []int32
	Probabilities []float32
	Count         int
}

// DraftStrata is MtpDrafter::draft with min_p > 0: the catch-up over the cells up to the accepted row a (t = a + 1,
// tokens the window's argmaxes: row i pairs R at p + i with the token at p + i + 1), then the chain from row a: the
// round graph's draft 0, then a step graph each while the last draft's probability reaches minP, at most max_t - 1
// (or max_drafts) of them and none past the K/V's end. drafted, where set, takes each draft as it is read, before the
// next step runs, as the drafter prefetches each draft's n-gram rows (prefetch_ple). commit, the window's commit
// readied, is launched first in the same run as the round graph, as Strata runs its commit and its first draft step
// as one: apart, each window waited on the device worker once more.
func (e *Executor) DraftStrata(ctx context.Context, d *StrataDrafter, tokens []int32, p int64, a int, minP float32, commit StrataCommit, drafted func(int32)) (StrataDraft, error) {
	if d.spec.WindowResidual == 0 {
		return StrataDraft{}, errors.New("mtp: the drafter reads no verify window")
	}
	maxT := d.spec.MaxTokens
	if len(tokens) < 1 || len(tokens) > maxT || a < 0 || a >= len(tokens) {
		return StrataDraft{}, errors.New("mtp: draft arguments out of range")
	}
	// #783 PR-i: the round runs for the cells up to the accepted row a; the rejected rows' K/V is not caught up (no
	// later read reaches a cell past the one being drafted, and the next round writes it first).
	t := a + 1
	steps := int(min(int64(min(maxT-1, d.spec.MaxDrafts)), d.spec.MaxCells-(p+int64(a))))
	draft := StrataDraft{Tokens: make([]int32, maxT-1), Probabilities: make([]float32, maxT-1)}
	if steps <= 0 {
		return draft, e.worker.Do(ctx, commit.launch)
	}
	tok := d.tok.words()
	for i := range t {
		tok[i] = tokens[i]
		d.putStep(i, p+int64(i))
	}
	d.putStep(strataDrafterStepSets*maxT-1, p+int64(a)) // draft 0's cell
	for j := 1; j < steps; j++ {
		d.putStep(maxT+j-1, p+int64(a)+int64(j))
	}
	out, prob := d.out.words(), d.prob.floats()
	for j := range steps {
		atomic.StoreInt32(&out[j], -1)
	}
	row := d.row.words()
	row[0], row[1] = int32(a), 0
	launch := func(exec driver.GraphExec) error {
		return e.worker.Do(ctx, func(state *device.State) error {
			if err := state.Driver.GraphLaunch(exec, d.cs); err != nil {
				return fmt.Errorf("mtp draft: %w", err)
			}
			_, err := state.Driver.StreamQuery(d.cs)
			return err
		})
	}
	// The host spins on each draft without yielding its processor, as draft's wait_step does (the round's await).
	wait := func(j int) error {
		queried := time.Now()
		for spins := 1; atomic.LoadInt32(&out[j]) < 0; spins++ {
			if spins%strataDrafterSpinQuery != 0 {
				continue
			}
			if now := time.Now(); now.Sub(queried) >= strataRoundQueryInterval {
				queried = now
				var done bool
				if err := e.worker.Do(ctx, func(state *device.State) error {
					var err error
					done, err = state.Driver.StreamQuery(d.cs)
					return err
				}); err != nil || done {
					return err
				}
			}
		}
		return nil
	}
	// the commit, the graphs captured where a window size or a step is new, and the round graph, in one run
	if err := e.worker.Do(ctx, func(state *device.State) error {
		if err := commit.launch(state); err != nil {
			return err
		}
		resources, err := e.ensureResources(state, false, 0, 0, 0)
		if err != nil {
			return err
		}
		if err := d.capture(state, resources.functions, &d.roundExec[t], func(w *strataDrafterRecord) error { return w.recordRound(int64(t)) }); err != nil {
			return err
		}
		for j := 1; j < steps; j++ {
			step := int64(j)
			if err := d.capture(state, resources.functions, &d.stepExec[j], func(w *strataDrafterRecord) error { return w.recordStep(step) }); err != nil {
				return err
			}
		}
		if err := state.Driver.GraphLaunch(d.roundExec[t], d.cs); err != nil {
			return fmt.Errorf("mtp draft: %w", err)
		}
		_, err = state.Driver.StreamQuery(d.cs)
		return err
	}); err != nil {
		return draft, err
	}
	if err := wait(0); err != nil {
		return draft, err
	}
	draft.Tokens[0], draft.Probabilities[0] = atomic.LoadInt32(&out[0]), loadFloat(&prob[0])
	draft.Count = 1
	for j := 1; j < steps && draft.Probabilities[j-1] >= minP; j++ {
		if err := launch(d.stepExec[j]); err != nil {
			return draft, err
		}
		if drafted != nil {
			drafted(draft.Tokens[j-1])
		}
		if err := wait(j); err != nil {
			return draft, err
		}
		draft.Tokens[j], draft.Probabilities[j] = atomic.LoadInt32(&out[j]), loadFloat(&prob[j])
		draft.Count++
	}
	if drafted != nil {
		drafted(draft.Tokens[draft.Count-1])
	}
	if err := e.worker.Do(context.WithoutCancel(ctx), func(state *device.State) error { return state.Driver.StreamSynchronize(d.cs) }); err != nil {
		return draft, err
	}
	d.rounds++
	return draft, nil
}

// loadFloat reads a mapped float the card writes, atomically as the volatile word it is.
func loadFloat(value *float32) float32 {
	bits := atomic.LoadUint32((*uint32)(unsafe.Pointer(value)))
	return *(*float32)(unsafe.Pointer(&bits))
}

// FirstDraftStrata is draft_first: the round's one cell from residual (the streams of the row before it, on the card)
// and token, staged into the window's residual rows, then draft(1, token, cell, 0).
func (e *Executor) FirstDraftStrata(ctx context.Context, d *StrataDrafter, residual driver.DevicePtr, token int32, cell int64, minP float32) (StrataDraft, error) {
	hcn := uint64(strataGRStreams * strataGREmbedding * strataRoundWordBytes)
	if err := e.worker.Do(ctx, func(state *device.State) error {
		if err := state.Driver.MemcpyDtoDAsync(d.spec.WindowResidual, residual, hcn, state.Stream); err != nil {
			return fmt.Errorf("mtp: staging the first residual failed: %w", err)
		}
		return state.Driver.StreamSynchronize(state.Stream)
	}); err != nil {
		return StrataDraft{}, err
	}
	return e.DraftStrata(ctx, d, []int32{token}, cell, 0, minP, StrataCommit{}, nil)
}

// PrefillStrataDrafter is prefill's E-4 path: the n prompt cells from cell0, rows the main model's final residual
// streams of each (on the card) and next the token after each; the records uploaded at once, each group of up to max_t
// rows its inputs copied on the card and its graph launched, the cells the window can never reach again skipped.
func (e *Executor) PrefillStrataDrafter(ctx context.Context, d *StrataDrafter, rows driver.DevicePtr, next []int32, cell0 int64) error {
	n := int64(len(next))
	if n == 0 {
		return nil
	}
	maxT := int64(d.spec.MaxTokens)
	firstNeeded := int64(0)
	if d.window > 0 && d.prompt > 0 {
		firstNeeded = d.prompt - d.window - strataDrafterRingLead
	}
	nh := int64(strataDrafterHeads)
	perRow := 1 + strataDrafterStepWords + nh
	records := make([]int32, n*perRow)
	tok, step, pos := records[:n], records[n:n+strataDrafterStepWords*n], records[n+strataDrafterStepWords*n:]
	for i := range n {
		cell := cell0 + i
		tok[i] = next[i]
		record := step[i*strataDrafterStepWords:]
		record[0], record[1] = int32(cell), int32(cell+1)
		record[strataRoundStepBlocks], record[strataRoundStepWidth] = int32((cell+1)/tensor.QSABlockCells), int32(cell+1)
		for h := range nh {
			pos[i*nh+h] = int32(cell)
		}
	}
	hcn := int64(strataGRStreams * strataGREmbedding)
	return e.worker.Do(ctx, func(state *device.State) error {
		resources, err := e.ensureResources(state, false, 0, 0, 0)
		if err != nil {
			return err
		}
		if d.prefillCapacity < n*perRow {
			if d.prefillRecords != 0 {
				if err := state.Driver.MemFree(d.prefillRecords); err != nil {
					return err
				}
				d.prefillRecords, d.prefillCapacity = 0, 0
			}
			if d.prefillRecords, err = d.resources.allocate(state, uint64(n*perRow)*strataRoundWordBytes); err != nil {
				return fmt.Errorf("mtp prefill: the input records do not fit: %w", err)
			}
			d.prefillCapacity = n * perRow
		}
		// The prompt's rows were written on the executor's stream.
		if err := state.Driver.StreamWait(d.cs, state.Stream); err != nil {
			return err
		}
		if err := state.Driver.MemcpyHtoD(d.prefillRecords, int32Words(records)); err != nil {
			return fmt.Errorf("mtp prefill: the input records upload failed: %w", err)
		}
		dTok := d.prefillRecords
		dStep := dTok + driver.DevicePtr(n*strataRoundWordBytes)
		dPos := dStep + driver.DevicePtr(strataDrafterStepWords*n*strataRoundWordBytes)
		for c := int64(0); c < n; c += maxT {
			t := min(maxT, n-c)
			if cell0+c+t <= firstNeeded {
				continue
			}
			if err := d.capture(state, resources.functions, &d.prefillExec[t], func(w *strataDrafterRecord) error { return w.recordPrefill(t) }); err != nil {
				return err
			}
			copies := []struct {
				dst, src driver.DevicePtr
				bytes    uint64
			}{
				{d.dTok, dTok + driver.DevicePtr(c*strataRoundWordBytes), uint64(t) * strataRoundWordBytes},
				{d.dStep, dStep + driver.DevicePtr(c*strataDrafterStepWords*strataRoundWordBytes), uint64(t*strataDrafterStepWords) * strataRoundWordBytes},
				{d.dPos, dPos + driver.DevicePtr(c*nh*strataRoundWordBytes), uint64(t*nh) * strataRoundWordBytes},
				{d.rin, rows + driver.DevicePtr(c*hcn*strataRoundWordBytes), uint64(t*hcn) * strataRoundWordBytes},
			}
			for _, copy := range copies {
				if err := state.Driver.MemcpyDtoDAsync(copy.dst, copy.src, copy.bytes, d.cs); err != nil {
					return fmt.Errorf("mtp prefill: %w", err)
				}
			}
			if err := state.Driver.GraphLaunch(d.prefillExec[t], d.cs); err != nil {
				return fmt.Errorf("mtp prefill: %w", err)
			}
		}
		return state.Driver.StreamSynchronize(d.cs)
	})
}
