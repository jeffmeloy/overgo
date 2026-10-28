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

// verify.cpp's capture, capture_commit, stage_inputs, run and commit at the bench's configuration: a graph a window
// size and one for the commit, the inputs written to mapped memory, the graph launched, each layer's doorbell served on
// the host while it runs, the outputs read from mapped memory; a commit after a window of several tokens, a one-token
// window advancing the state itself.

// StrataRoundPlan is GpuPlanSink over a window's plan in mapped memory: the VRAM groups' counts, starts, entries' rows
// and tokens and blob addresses, then the PCIe groups' (their blobs' device aliases) and starts; the staging their
// copy lands in and how many blobs it holds; Publish raises flag A, Fetch takes the PCIe share's copy (pcie_mode 2:
// none, the graph's copy kernel moves them) and raises flag B.
type StrataRoundPlan struct {
	Counts, Start, Dst, Tok, PCIeStart []int32
	Pointers, PCIe                     []uint64
	Cap, StagingCap                    int64
	Staging                            uint64
	Publish                            func()
	Fetch                              func(blobs int)
}

// StrataRoundPool serves the round's host side (drive_pool_multi and the PLE's gather): ServeLayer takes a layer's
// window - tokens rows of activations x and routes ids - publishes its plan and writes the CPU's rows of out, a row an
// entry; Lexical gathers the window's n-gram embedding rows.
type StrataRoundPool interface {
	ServeLayer(layer int, x []float32, ids []int32, tokens int, plan *StrataRoundPlan, out []float32) error
	Lexical(tokens []int32, rows []float32) error
}

// words is a mapped part's int32 words.
func (m strataRoundMapped) words() []int32 {
	return unsafe.Slice((*int32)(unsafe.Pointer(&m.host[0])), len(m.host)/strataRoundWordBytes)
}

// floats is a mapped part's floats.
func (m strataRoundMapped) floats() []float32 {
	return unsafe.Slice((*float32)(unsafe.Pointer(&m.host[0])), len(m.host)/strataRoundWordBytes)
}

// word is a mapped flag's word, read and written atomically as the volatile word it is.
func (m strataRoundMapped) word() *uint32 {
	return (*uint32)(unsafe.Pointer(&m.host[0]))
}

// raiseFlag is raise_flag: the flag only rises.
func raiseFlag(flag *uint32, value uint32) {
	for {
		current := atomic.LoadUint32(flag)
		if current >= value || atomic.CompareAndSwapUint32(flag, current, value) {
			return
		}
	}
}

// planSink is set_plan_slot(0): the plan's parts as the host writes them.
func (r *StrataRound) planSink() *StrataRoundPlan {
	capacity := int64(r.spec.MaxTokens) * strataRoundRouted
	words := r.plan.words()
	start := words[strataRoundPlanHeader:]
	dst := start[capacity+1:]
	tok := dst[capacity:]
	offset := (strataRoundPlanHeader + (capacity + 1) + strataRoundPlanEntryArrays*capacity + 1) &^ 1
	pointers := unsafe.Slice((*uint64)(unsafe.Pointer(&words[offset])), strataRoundPlanEntryArrays*capacity)
	plan := &StrataRoundPlan{
		Counts: words[:strataRoundPlanHeader], Start: start[:capacity+1], Dst: dst[:capacity], Tok: tok[:capacity],
		Pointers: pointers[:capacity], PCIe: pointers[capacity:], PCIeStart: words[offset+strataRoundPlanHeader*capacity:][:capacity+1],
		Cap: capacity, Staging: uint64(r.staging), StagingCap: strataRoundStagingBlobs,
	}
	return plan
}

// capture is Verifier::capture: the t-token window recorded into a graph once, instantiated and uploaded.
func (r *StrataRound) capture(state *device.State, functions functionSet, t int) error {
	if r.exec[t] != 0 {
		return nil
	}
	if err := r.resources.buffers.reserveGraph(state); err != nil {
		return err
	}
	if err := state.Driver.StreamBeginCapture(r.cs); err != nil {
		return fmt.Errorf("strata round: begin capture failed: %w", err)
	}
	recorded := r.recordWindow(state, functions, t)
	graph, ended := state.Driver.StreamEndCapture(r.cs)
	if recorded != nil || ended != nil {
		return errors.Join(recorded, ended, state.Driver.GraphDestroy(graph))
	}
	exec, err := state.Driver.GraphInstantiate(graph)
	if err = errors.Join(err, state.Driver.GraphDestroy(graph)); err != nil {
		return fmt.Errorf("strata round: instantiate: %w", err)
	}
	r.exec[t] = exec
	return errors.Join(state.Driver.GraphUpload(exec, r.cs), state.Driver.StreamSynchronize(r.cs))
}

// captureCommit is Verifier::capture_commit: the commit's words, every gated-delta layer's history and state over the
// kept count in one launch of each kernel, the QSA layers' tails restored and their kept rows' indexer keys appended in
// one launch, and the PLE history taken from the last kept row's snapshot.
func (r *StrataRound) captureCommit(state *device.State, functions functionSet) error {
	if r.commitExec != 0 {
		return nil
	}
	if err := r.resources.buffers.reserveGraph(state); err != nil {
		return err
	}
	if err := state.Driver.StreamBeginCapture(r.cs); err != nil {
		return fmt.Errorf("strata round: begin commit capture failed: %w", err)
	}
	saved := state.Stream
	state.Stream = r.cs
	recorded := r.recordCommit(state, functions)
	state.Stream = saved
	graph, ended := state.Driver.StreamEndCapture(r.cs)
	if recorded != nil || ended != nil {
		return errors.Join(recorded, ended, state.Driver.GraphDestroy(graph))
	}
	exec, err := state.Driver.GraphInstantiate(graph)
	if err = errors.Join(err, state.Driver.GraphDestroy(graph)); err != nil {
		return fmt.Errorf("strata round: commit capture: %w", err)
	}
	r.commitExec = exec
	return nil
}

func (r *StrataRound) recordCommit(state *device.State, functions functionSet) error {
	s := r.spec
	mt := int64(s.MaxTokens)
	c, hv := s.ConvChannels, s.ValueHeads
	if err := launchStrataCopyI32FromMapped(state, functions, r.dCommit, r.commit.device, strataRoundCommitHeader+mt); err != nil {
		return err
	}
	var histories, states []driver.DevicePtr
	var windows []strataGatedDeltaWindow
	var layers strataIndexerLayers
	var qsa int
	var maxCells int64
	var gi, qi int64
	ts := int64(strataRoundTailDeltas)
	for _, layer := range s.Layers {
		if layer.QSA {
			if qsa >= strataIndexerLayersMax {
				return errors.New("strata round: more QSA layers than one commit launch takes")
			}
			cells := layer.Cells
			layers.gamma[qsa], layers.tailSource[qsa] = layer.IndexerKNorm, at(r.tailSnap, qi*ts)
			layers.tail[qsa], layers.dead[qsa], layers.pooled[qsa] = cells.Tail, cells.Spare, cells.Pooled
			layers.blockPos[qsa], maxCells = cells.BlockPosition, cells.MaxCells
			qsa, qi = qsa+1, qi+1
			continue
		}
		histories, states = append(histories, layer.History), append(states, layer.State)
		windows = append(windows, strataGatedDeltaWindow{
			qkv: at(r.qkvL, gi*mt*c), rows: at(r.hL, gi*mt*c), gate: at(r.gateL, gi*mt*hv), beta: at(r.betaL, gi*mt*hv),
		})
		gi++
	}
	if err := launchStrataGatedDeltaCommitLayers(state, functions, histories, states, r.dCommit, windows, int32(c),
		int32(s.KeyHeads), int32(hv)); err != nil {
		return err
	}
	if err := launchStrataNativeIndexerStepsLayers(state, functions, r.idxRawL, mt*tensor.QSAIndexerWidth,
		at(r.dCommit, strataRoundCommitHeader), 1, int32(mt), 0, strataRoundEpsilon, layers, qsa, int32(maxCells), r.rope,
		s.Positions); err != nil {
		return err
	}
	hs := int64(strataRoundHistoryRows * strataPLEWidth)
	return launchStrataCopyIndexed(state, functions, r.pleHistory, r.histSnap, hs, at(r.dCommit, 1), hs)
}

// stageInputs is stage_inputs: the window's tokens, steps and positions to mapped staging, and the handshake words
// cleared.
func (r *StrataRound) stageInputs(tokens []int32, pos0 int64) {
	s := r.spec
	mt := int64(s.MaxTokens)
	tok, step, pos := r.tok.words(), r.step.words(), r.pos.words()
	keys := pos[mt*tensor.QSAQueryHeads:]
	indexer := keys[mt*tensor.QSAKeyValueHeads:]
	for t, token := range tokens {
		tok[t] = token
		cell := uint32(pos0) + uint32(t)
		held := cell + 1
		record := step[t*strataRoundStepCount:]
		record[0], record[1] = int32(cell), int32(held)
		record[strataRoundStepBlocks], record[strataRoundStepWidth] = int32(held/tensor.QSABlockCells), int32(tensor.QSASelectionWidth(held))
		for h := range tensor.QSAQueryHeads {
			pos[t*tensor.QSAQueryHeads+h] = int32(cell)
		}
		for h := range tensor.QSAKeyValueHeads {
			keys[t*tensor.QSAKeyValueHeads+h] = int32(cell)
		}
		for h := range tensor.QSAIndexerHeads {
			indexer[t*tensor.QSAIndexerHeads+h] = int32(cell)
		}
	}
	for _, flag := range []strataRoundMapped{r.seq, r.flag, r.flagA, r.flagB} {
		atomic.StoreUint32(flag.word(), 0)
	}
	r.lastT, r.lastPos0 = len(tokens), pos0
	copy(r.lastTokens[:], tokens)
}

// The step record's words past pos and n_kv: the blocks held and the selection's width (qsa_step_fill).
const (
	strataRoundStepBlocks = 2
	strataRoundStepWidth  = 3
)

// strataRoundLayerTimeout is how long the host waits for a layer's doorbell before it ends the window (run's 20 s);
// strataRoundQueryEvery how many spins pass between looks at the clock, and strataRoundQueryInterval how long between
// asks whether the graph has finished without ringing (run's 2 ms).
const (
	strataRoundLayerTimeout  = 20 * time.Second
	strataRoundQueryEvery    = 1 << 10
	strataRoundQueryInterval = 2 * time.Millisecond
)

// RunStrataRound is Verifier::run for the tokens of a window at positions from pos0: the graphs captured once, the
// inputs staged, the window's graph launched, each layer's doorbell served by pool as it rings (its plan, its CPU rows,
// the PLE rows at the first), then the outputs: each row's greedy token.
func (e *Executor) RunStrataRound(ctx context.Context, r *StrataRound, tokens []int32, pos0 int64, pool StrataRoundPool, out []int32) error {
	t := len(tokens)
	if t < 1 || t > r.spec.MaxTokens || len(out) < t {
		return errors.New("strata round: window size out of range")
	}
	r.stageInputs(tokens, pos0)
	// One run: the graphs captured where a window size is new, the window's graph launched, and a sampled request's
	// head sampling queued behind it - launched, not captured, so its parameters are this call's own (a captured kernel
	// would replay the same draws forever). Row t's draw is Philox(seed, pos0 + t): tied to the position it samples, not
	// to how the text was cut into windows, so a seed replays the same text whatever the drafts were. Apart, a window
	// waited on the device worker twice more, and a sampled one on the card once more.
	if err := e.worker.Do(ctx, func(state *device.State) error {
		resources, err := e.ensureResources(state, false, 0, 0, 0)
		if err != nil {
			return err
		}
		if err := r.capture(state, resources.functions, t); err != nil {
			return err
		}
		if err := r.captureCommit(state, resources.functions); err != nil {
			return err
		}
		// (the sampler's lists made before the launch: an allocation may move experts, which a running window reads)
		vocab := int32(r.spec.Vocab)
		if r.sampling != nil && r.samplerScratch == 0 {
			if r.samplerScratch, err = r.resources.allocate(state, strataSamplerScratchBytes(int32(r.spec.MaxTokens), vocab)); err != nil {
				return fmt.Errorf("strata round: the sampler's lists do not fit: %w", err)
			}
		}
		// The window follows the work the executor's stream holds: the expert residency publishes its tables there,
		// queued behind the runs before (writeOrdered), and a window must read the tables they leave.
		if err := state.Driver.StreamWait(r.cs, state.Stream); err != nil {
			return err
		}
		if err := state.Driver.GraphLaunch(r.exec[t], r.cs); err != nil {
			return fmt.Errorf("strata round: launch: %w", err)
		}
		if r.sampling != nil {
			saved := state.Stream
			state.Stream = r.cs
			err = launchStrataSampleTokens(state, resources.functions, r.headLogits, int32(t), vocab, 0, 0, *r.sampling,
				uint64(pos0), r.out.device, r.samplerScratch)
			state.Stream = saved
			if err != nil {
				return fmt.Errorf("strata round: the head sampling failed: %w", err)
			}
		}
		_, err = state.Driver.StreamQuery(r.cs)
		return err
	}); err != nil {
		return err
	}
	served := r.serve(ctx, e, tokens, pool)
	// out_ is the mapped h_out_: synced, it is readable
	synced := e.worker.Do(context.WithoutCancel(ctx), func(state *device.State) error { return state.Driver.StreamSynchronize(r.cs) })
	if err := errors.Join(served, synced); err != nil {
		return err
	}
	words := r.out.words()
	for row := range t {
		out[row] = int32(atomic.LoadUint32((*uint32)(unsafe.Pointer(&words[row]))))
	}
	return nil
}

// serve is run's host loop: for each layer, the doorbell awaited, the pool served, an unpublished plan published empty,
// the PLE rows gathered after the first layer's, then the flag raised.
func (r *StrataRound) serve(ctx context.Context, e *Executor, tokens []int32, pool StrataRoundPool) error {
	n := len(tokens)
	x, ids, ymiss := r.x.floats(), r.ids.words(), r.ymiss.floats()
	plan := r.planSink()
	for k := range len(r.spec.Layers) {
		want := uint32(k + 1)
		if err := r.await(ctx, e, k, want); err != nil {
			raiseFlag(r.flagA.word(), uint32(len(r.spec.Layers)))
			raiseFlag(r.flagB.word(), uint32(len(r.spec.Layers)))
			raiseFlag(r.flag.word(), uint32(len(r.spec.Layers)))
			return err
		}
		plan.Publish = func() { atomic.StoreUint32(r.flagA.word(), want) }
		plan.Fetch = func(int) { raiseFlag(r.flagB.word(), want) }
		served := pool.ServeLayer(k, x[:n*strataGREmbedding], ids[:n*strataRoundRouted], n, plan, ymiss)
		if atomic.LoadUint32(r.flagA.word()) != want {
			// the pool did not publish a plan: an empty one
			plan.Counts[0], plan.Counts[strataRoundHitCount], plan.Counts[strataRoundPCIeCount] = 0, 0, 0
			plan.Start[0], plan.PCIeStart[0] = 0, 0
			atomic.StoreUint32(r.flagA.word(), want)
			raiseFlag(r.flagB.word(), want)
		}
		if k == 0 && served == nil {
			served = pool.Lexical(tokens, r.ple.floats()[:n*strataGREmbedding])
		}
		atomic.StoreUint32(r.flag.word(), want)
		if served != nil {
			// a failed layer still releases the graph, which never waits on a flag that will not rise
			for _, flag := range []strataRoundMapped{r.flagA, r.flagB, r.flag} {
				raiseFlag(flag.word(), uint32(len(r.spec.Layers)))
			}
			return fmt.Errorf("strata round: layer %d's experts: %w", k, served)
		}
	}
	return nil
}

// await spins until the doorbell's ring reaches want, asking now and then whether the graph has ended without it.
func (r *StrataRound) await(ctx context.Context, e *Executor, layer int, want uint32) error {
	// The host spins on the doorbell without yielding its processor, as run's host thread does: yielding each spin let
	// the goroutine writing the tokens out run in its place, a layer's experts then served late while the card waited.
	started := time.Now()
	queried := started
	for spins := 1; atomic.LoadUint32(r.seq.word()) < want; spins++ {
		if spins%strataRoundQueryEvery != 0 {
			continue
		}
		if now := time.Now(); now.Sub(queried) < strataRoundQueryInterval {
			continue
		} else {
			queried = now
		}
		var done bool
		if err := e.worker.Do(ctx, func(state *device.State) error {
			var err error
			done, err = state.Driver.StreamQuery(r.cs)
			return err
		}); err != nil {
			return err
		}
		if done && atomic.LoadUint32(r.seq.word()) < want {
			return fmt.Errorf("strata round: layer %d never rang (graph finished)", layer)
		}
		if time.Since(started) > strataRoundLayerTimeout {
			return fmt.Errorf("strata round: timed out at layer %d", layer)
		}
	}
	return nil
}

// CommitStrataRound is Verifier::commit: the kept count and positions to the commit's words, then the commit graph on
// the window's stream, not waited for (set_commit_async); a one-token window advanced the state itself.
func (e *Executor) CommitStrataRound(ctx context.Context, r *StrataRound, kept int) error {
	commit, err := PrepareStrataRoundCommit(r, kept)
	if err != nil || commit.round == nil {
		return err
	}
	return e.worker.Do(ctx, commit.launch)
}

// StrataCommit is a round's commit readied (PrepareStrataRoundCommit), whose graph a draft launches before its own work
// (DraftStrata), as Strata runs its commit and its first draft step as one run; the zero value commits nothing.
type StrataCommit struct{ round *StrataRound }

// PrepareStrataRoundCommit writes the commit's words, the kept count and positions, and answers the commit to launch,
// none where the window held one token, which advanced the state itself.
func PrepareStrataRoundCommit(r *StrataRound, kept int) (StrataCommit, error) {
	if kept < 1 || kept > r.lastT {
		return StrataCommit{}, errors.New("strata round: commit count out of range")
	}
	words := r.commit.words()
	words[0], words[1] = int32(kept), int32(kept-1)
	for t := range r.spec.MaxTokens {
		words[strataRoundCommitHeader+t] = -1
		if t < kept {
			words[strataRoundCommitHeader+t] = int32(r.lastPos0) + int32(t)
		}
	}
	if r.lastT == 1 {
		return StrataCommit{}, nil
	}
	return StrataCommit{r}, nil
}

// launch puts the commit graph on the window's stream, not waited for.
func (c StrataCommit) launch(state *device.State) error {
	if c.round == nil {
		return nil
	}
	if err := state.Driver.GraphLaunch(c.round.commitExec, c.round.cs); err != nil {
		return fmt.Errorf("strata round: commit launch: %w", err)
	}
	_, err := state.Driver.StreamQuery(c.round.cs)
	return err
}

// LoadStrataRoundHistory writes the sequence's PLE history into the round from rows, token-major (each of the history's
// rows of every channel, the oldest first, as Overgo's prompt read leaves it): the round holds a channel's rows
// together.
func (e *Executor) LoadStrataRoundHistory(ctx context.Context, r *StrataRound, rows []float32) error {
	if len(rows) != strataRoundHistoryRows*strataPLEWidth {
		return errors.New("strata round: the PLE history's rows differ from the round's")
	}
	channels := make([]float32, len(rows))
	for row := range strataRoundHistoryRows {
		for channel := range strataPLEWidth {
			channels[channel*strataRoundHistoryRows+row] = rows[row*strataPLEWidth+channel]
		}
	}
	return e.worker.Do(ctx, func(state *device.State) error {
		return errors.Join(state.Driver.StreamSynchronize(r.cs), state.Driver.MemcpyHtoD(r.pleHistory, float32Words(channels)))
	})
}

// StrataRoundHistory reads the sequence's PLE history back token-major, as LoadStrataRoundHistory takes it.
func (e *Executor) StrataRoundHistory(ctx context.Context, r *StrataRound) ([]float32, error) {
	data := make([]byte, strataRoundHistoryRows*strataPLEWidth*strataRoundWordBytes)
	if err := e.worker.Do(ctx, func(state *device.State) error {
		return errors.Join(state.Driver.StreamSynchronize(r.cs), state.Driver.MemcpyDtoH(data, r.pleHistory))
	}); err != nil {
		return nil, err
	}
	channels := unsafe.Slice((*float32)(unsafe.Pointer(&data[0])), len(data)/strataRoundWordBytes)
	rows := make([]float32, len(channels))
	for row := range strataRoundHistoryRows {
		for channel := range strataPLEWidth {
			rows[row*strataPLEWidth+channel] = channels[channel*strataRoundHistoryRows+row]
		}
	}
	return rows, nil
}

// float32Words is values' bytes.
func float32Words(values []float32) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), len(values)*strataRoundWordBytes)
}

// SetStrataRoundSampling is set_sampling: the request's sampling, nil for greedy windows (the graph's argmax).
func (r *StrataRound) SetStrataRoundSampling(sampling *StrataSampling) { r.sampling = sampling }
