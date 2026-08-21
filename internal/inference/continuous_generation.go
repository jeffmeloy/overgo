package inference

import (
	"context"
	"errors"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"overgo/internal/tensor"
	"overgo/internal/tokenizer"
)

// ContinuousGeneratorOptions: fused scheduler bounds.
type ContinuousGeneratorOptions struct {
	MaxSequences  int
	ContextShift  bool
	KeepTokens    uint32
	DiscardTokens int
}

// ContinuousGenerator: dynamic token-step scheduler.
type ContinuousGenerator struct {
	runner  *Runner
	batch   continuousBatchAPI
	options ContinuousGeneratorOptions
	ctx     context.Context
	cancel  context.CancelFunc
	submit  chan continuousGenerateRequest
	done    chan struct{}
	close   sync.Once
}

type continuousBatchAPI interface {
	Step(context.Context, []SequenceBatchInput) ([]SequenceBatchOutput, error)
	Remove(context.Context, SequenceID) error
	Close(context.Context) error
}

type continuousGreedyBatchAPI interface {
	StepGreedy(context.Context, []SequenceBatchInput) ([]SequenceBatchOutput, error)
}

type continuousTopKBatchAPI interface {
	StepTopK(context.Context, []SequenceBatchInput, uint32) ([]SequenceBatchOutput, error)
}

type continuousGenerateRequest struct {
	ctx      context.Context
	prompt   string
	options  GenerateOptions
	response chan continuousGenerateResult
}

type continuousGenerateResult struct {
	ids  []tokenizer.TokenID
	text string
	err  error
}

type continuousGenerateState struct {
	id         SequenceID
	request    continuousGenerateRequest
	ids        []tokenizer.TokenID
	generated  strings.Builder
	topKIDs    []int
	topKLogits []float32
	index      int
	started    time.Time
	evaluated  bool
}

// NewContinuousGenerator: fused device scheduler.
func (r *Runner) NewContinuousGenerator(
	options ContinuousGeneratorOptions,
) (*ContinuousGenerator, error) {
	if r == nil {
		return nil, errRunnerNil
	}
	if options.MaxSequences <= 0 {
		return nil, errors.New("inference: maximum sequence count must be positive")
	}
	batch, err := r.NewContinuousBatch(ContinuousBatchOptions{
		MaxSequences:  options.MaxSequences,
		Device:        true,
		ContextShift:  options.ContextShift,
		KeepTokens:    options.KeepTokens,
		DiscardTokens: options.DiscardTokens,
	})
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(context.Background())
	generator := &ContinuousGenerator{
		runner: r, batch: batch, options: options,
		ctx: ctx, cancel: cancel,
		submit: make(chan continuousGenerateRequest, options.MaxSequences),
		done:   make(chan struct{}),
	}
	go generator.run()
	return generator, nil
}

// Generate: queued fused generation.
func (g *ContinuousGenerator) Generate(
	ctx context.Context,
	prompt string,
	options GenerateOptions,
) ([]tokenizer.TokenID, string, error) {
	if g == nil || g.runner == nil {
		return nil, "", errors.New("inference: continuous generator is nil")
	}
	if err := g.validateOptions(options); err != nil {
		return nil, "", err
	}
	request := continuousGenerateRequest{
		ctx: ctx, prompt: prompt, options: options,
		response: make(chan continuousGenerateResult, 1),
	}
	select {
	case g.submit <- request:
	case <-ctx.Done():
		return nil, "", ctx.Err()
	case <-g.done:
		return nil, "", errors.New("inference: continuous generator is closed")
	}
	select {
	case result := <-request.response:
		return result.ids, result.text, result.err
	case <-ctx.Done():
		return nil, "", ctx.Err()
	case <-g.done:
		return nil, "", errors.New("inference: continuous generator is closed")
	}
}

func (g *ContinuousGenerator) validateOptions(options GenerateOptions) error {
	if options.ProjectedInputs != nil || options.CachePrompt ||
		options.MinCacheReuse != 0 || options.LoRAConfigured || len(options.LoRA) > 0 ||
		g.runner.hasInvocationLoRA() ||
		options.KeepTokens != int(g.options.KeepTokens) ||
		options.DiscardTokens != g.options.DiscardTokens ||
		options.ContextShift != g.options.ContextShift {
		return errors.New("inference: request is incompatible with continuous generation plan")
	}
	return nil
}

func (r *Runner) hasInvocationLoRA() bool {
	for _, loaded := range r.loraAdapters {
		if loaded.adapter != nil && len(loaded.adapter.InvocationTokens) > 0 {
			return true
		}
	}
	return false
}

// Close: stop scheduler and release active caches.
func (g *ContinuousGenerator) Close(ctx context.Context) error {
	if g == nil || g.cancel == nil || g.done == nil {
		return nil
	}
	g.close.Do(g.cancel)
	select {
	case <-g.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (g *ContinuousGenerator) run() {
	defer close(g.done)
	defer g.batch.Close(context.Background())
	active := make(map[SequenceID]*continuousGenerateState)
	stateIDs := make([]SequenceID, 0, g.options.MaxSequences)
	inputs := make([]SequenceBatchInput, 0, g.options.MaxSequences)
	stepping := make([]*continuousGenerateState, 0, g.options.MaxSequences)
	var nextID SequenceID
	for {
		if len(active) == 0 {
			select {
			case <-g.ctx.Done():
				g.rejectQueued(g.ctx.Err())
				return
			case request := <-g.submit:
				nextID++
				g.admit(active, nextID, request)
			}
		}
		for len(active) < g.options.MaxSequences {
			select {
			case request := <-g.submit:
				nextID++
				g.admit(active, nextID, request)
			default:
				goto admitted
			}
		}
	admitted:
		if g.ctx.Err() != nil {
			g.failAll(active, g.ctx.Err())
			g.rejectQueued(g.ctx.Err())
			return
		}
		stateIDs = sortedContinuousStateIDs(active, stateIDs[:0])
		inputs = inputs[:0]
		stepping = stepping[:0]
		for _, id := range stateIDs {
			state := active[id]
			if err := state.request.ctx.Err(); err != nil {
				_ = g.batch.Remove(context.Background(), id)
				g.respond(state, continuousGenerateResult{err: err})
				delete(active, id)
				continue
			}
			tokens := state.ids
			if state.evaluated {
				tokens = state.ids[len(state.ids)-1:]
			}
			inputs = append(inputs, SequenceBatchInput{ID: id, Tokens: tokens})
			stepping = append(stepping, state)
		}
		if len(inputs) == 0 {
			continue
		}
		greedyBatch, greedy := g.batch.(continuousGreedyBatchAPI)
		greedy = greedy && g.runner.forwardProgram().DeviceBatchSelection() &&
			continuousStatesUseDeviceGreedy(stepping)
		topKBatch, bounded := g.batch.(continuousTopKBatchAPI)
		topK := 0
		if !greedy && bounded && !g.runner.program.Model.ProjectedInput().DiscreteTokens {
			topK, bounded = continuousStatesUseDeviceTopK(stepping, int(g.runner.spec.VocabularySize))
		} else if !greedy {
			bounded = false
		}
		var outputs []SequenceBatchOutput
		var err error
		if greedy {
			outputs, err = greedyBatch.StepGreedy(g.ctx, inputs)
		} else if bounded {
			outputs, err = topKBatch.StepTopK(g.ctx, inputs, uint32(topK))
		} else {
			outputs, err = g.batch.Step(g.ctx, inputs)
		}
		if err != nil {
			g.failStates(active, stepping, err)
			continue
		}
		for index, output := range outputs {
			state := stepping[index]
			if !state.evaluated {
				state.evaluated = true
				if callback := state.request.options.OnPromptEvaluated; callback != nil {
					callback(PromptEvaluation{
						Tokens: len(state.ids), Duration: time.Since(state.started),
					})
				}
			}
			var result continuousGenerateResult
			var complete bool
			if greedy {
				result, complete = g.acceptSampleEvent(state, TokenEvent{ID: output.Token})
			} else if bounded {
				result, complete = g.sampleTopK(state, output.Candidates)
			} else {
				result, complete = g.sample(state, output.Logits)
			}
			if !complete {
				continue
			}
			_ = g.batch.Remove(context.Background(), state.id)
			delete(active, state.id)
			g.respond(state, result)
		}
	}
}

func continuousStatesUseDeviceGreedy(states []*continuousGenerateState) bool {
	if len(states) == 0 {
		return false
	}
	for _, state := range states {
		options := state.request.options
		if !options.DeviceGreedy || options.PostSamplingProbabilities != 0 ||
			options.Sampler == nil || !options.Sampler.IsRawGreedy() {
			return false
		}
	}
	return true
}

func continuousStatesUseDeviceTopK(
	states []*continuousGenerateState,
	vocabulary int,
) (int, bool) {
	limit := 0
	for _, state := range states {
		options := state.request.options
		if !options.DeviceTopK || options.PostSamplingProbabilities != 0 || options.Sampler == nil {
			return 0, false
		}
		candidateLimit, ok := options.Sampler.BoundedTopK()
		if !ok || candidateLimit <= 0 || candidateLimit > vocabulary ||
			candidateLimit > int(tensor.MaxTopKPairs) ||
			(limit != 0 && candidateLimit != limit) {
			return 0, false
		}
		limit = candidateLimit
	}
	return limit, limit > 0
}

func (g *ContinuousGenerator) admit(
	active map[SequenceID]*continuousGenerateState,
	id SequenceID,
	request continuousGenerateRequest,
) {
	if err := request.ctx.Err(); err != nil {
		request.response <- continuousGenerateResult{err: err}
		return
	}
	options := request.options
	if err := normalizeGenerateOptions(&options); err != nil {
		request.response <- continuousGenerateResult{err: err}
		return
	}
	request.options = options
	ids, err := g.runner.promptTokenIDs(request.prompt, options)
	if err != nil {
		request.response <- continuousGenerateResult{err: err}
		return
	}
	if options.MaxNewTokens == 0 {
		text, decodeErr := g.runner.vocab.Decode(ids, false)
		request.response <- continuousGenerateResult{ids: ids, text: text, err: decodeErr}
		return
	}
	active[id] = &continuousGenerateState{
		id: id, request: request, ids: ids, started: time.Now(),
	}
}

func (g *ContinuousGenerator) sample(
	state *continuousGenerateState,
	logits []float32,
) (continuousGenerateResult, bool) {
	options := state.request.options
	event, err := sampleGenerationToken(logits, state.ids, options)
	if err != nil {
		return continuousGenerateResult{err: err}, true
	}
	return g.acceptSampleEvent(state, event)
}

func (g *ContinuousGenerator) sampleTopK(
	state *continuousGenerateState,
	candidates []LogitCandidate,
) (continuousGenerateResult, bool) {
	if cap(state.topKIDs) < len(candidates) {
		state.topKIDs = make([]int, len(candidates))
		state.topKLogits = make([]float32, len(candidates))
	} else {
		state.topKIDs = state.topKIDs[:len(candidates)]
		state.topKLogits = state.topKLogits[:len(candidates)]
	}
	for index, candidate := range candidates {
		state.topKIDs[index], state.topKLogits[index] = int(candidate.ID), candidate.Logit
	}
	next, err := state.request.options.Sampler.SampleTopK(
		state.topKIDs, state.topKLogits, int(g.runner.spec.VocabularySize),
	)
	if err != nil {
		return continuousGenerateResult{err: err}, true
	}
	return g.acceptSampleEvent(state, TokenEvent{ID: tokenizer.TokenID(next)})
}

func (g *ContinuousGenerator) acceptSampleEvent(
	state *continuousGenerateState,
	event TokenEvent,
) (continuousGenerateResult, bool) {
	if err := state.request.ctx.Err(); err != nil {
		return continuousGenerateResult{err: err}, true
	}
	options := state.request.options
	event.Index = state.index
	state.ids = append(state.ids, event.ID)
	stop, deliverErr := g.runner.deliverGenerationToken(&event, options, &state.generated)
	if deliverErr != nil {
		return continuousGenerateResult{err: deliverErr}, true
	}
	state.index++
	complete := state.index >= options.MaxNewTokens || stop
	if !complete {
		return continuousGenerateResult{}, false
	}
	text, decodeErr := g.runner.vocab.Decode(state.ids, false)
	return continuousGenerateResult{ids: slices.Clone(state.ids), text: text, err: decodeErr}, true
}

func sortedContinuousStateIDs(
	active map[SequenceID]*continuousGenerateState,
	ids []SequenceID,
) []SequenceID {
	for id := range active {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(left, right int) bool { return ids[left] < ids[right] })
	return ids
}

func (g *ContinuousGenerator) failStates(
	active map[SequenceID]*continuousGenerateState,
	states []*continuousGenerateState,
	err error,
) {
	for _, state := range states {
		_ = g.batch.Remove(context.Background(), state.id)
		delete(active, state.id)
		g.respond(state, continuousGenerateResult{err: err})
	}
}

func (g *ContinuousGenerator) failAll(
	active map[SequenceID]*continuousGenerateState,
	err error,
) {
	states := make([]*continuousGenerateState, 0, len(active))
	for _, state := range active {
		states = append(states, state)
	}
	g.failStates(active, states, err)
}

func (g *ContinuousGenerator) rejectQueued(err error) {
	for {
		select {
		case request := <-g.submit:
			request.response <- continuousGenerateResult{err: err}
		default:
			return
		}
	}
}

func (g *ContinuousGenerator) respond(
	state *continuousGenerateState,
	result continuousGenerateResult,
) {
	select {
	case state.request.response <- result:
	default:
	}
}
