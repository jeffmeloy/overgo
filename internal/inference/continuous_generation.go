package inference

import (
	"context"
	"errors"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"llamacpp2go/internal/tokenizer"
)

// ContinuousGeneratorOptions: fused scheduler bounds.
type ContinuousGeneratorOptions struct {
	MaxSequences  int
	PageTokens    uint32
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
	id        SequenceID
	request   continuousGenerateRequest
	ids       []tokenizer.TokenID
	generated strings.Builder
	index     int
	started   time.Time
	evaluated bool
}

// NewContinuousGenerator: fused device scheduler.
func (r *Runner) NewContinuousGenerator(
	options ContinuousGeneratorOptions,
) (*ContinuousGenerator, error) {
	if r == nil {
		return nil, errors.New("inference: runner is nil")
	}
	if options.MaxSequences <= 0 {
		return nil, errors.New("inference: maximum sequence count must be positive")
	}
	batch, err := r.NewContinuousBatch(ContinuousBatchOptions{
		MaxSequences:  options.MaxSequences,
		PageTokens:    options.PageTokens,
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

// Generate: queued fused generation; unsupported request modes fall back.
func (g *ContinuousGenerator) Generate(
	ctx context.Context,
	prompt string,
	options GenerateOptions,
) ([]tokenizer.TokenID, string, error) {
	if g == nil || g.runner == nil {
		return nil, "", errors.New("inference: continuous generator is nil")
	}
	if g.requiresFallback(options) {
		return g.runner.Generate(ctx, prompt, options)
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

func (g *ContinuousGenerator) requiresFallback(options GenerateOptions) bool {
	return options.ProjectedInputs != nil || options.CachePrompt ||
		options.MinCacheReuse != 0 || options.LoRAConfigured || len(options.LoRA) > 0 ||
		g.runner.hasInvocationLoRA() ||
		options.KeepTokens != int(g.options.KeepTokens) ||
		options.DiscardTokens != g.options.DiscardTokens ||
		options.ContextShift != g.options.ContextShift
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
		ids := sortedContinuousStateIDs(active)
		inputs := make([]SequenceBatchInput, 0, len(ids))
		stepping := make([]*continuousGenerateState, 0, len(ids))
		for _, id := range ids {
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
		outputs, err := g.batch.Step(g.ctx, inputs)
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
			result, complete := g.sample(state, output.Logits)
			if !complete {
				continue
			}
			_ = g.batch.Remove(context.Background(), state.id)
			delete(active, state.id)
			g.respond(state, result)
		}
	}
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
	if err := state.request.ctx.Err(); err != nil {
		return continuousGenerateResult{err: err}, true
	}
	options := state.request.options
	event, err := sampleGenerationToken(logits, state.ids, options)
	if err != nil {
		return continuousGenerateResult{err: err}, true
	}
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

func sortedContinuousStateIDs(active map[SequenceID]*continuousGenerateState) []SequenceID {
	ids := make([]SequenceID, 0, len(active))
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
