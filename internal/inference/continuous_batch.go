package inference

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sort"
	"sync"

	"overgo/internal/recipe"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// SequenceID: continuous-batch cache owner.
type SequenceID uint64

// ContinuousBatchOptions: cache and admission policy.
type ContinuousBatchOptions struct {
	MaxSequences  int
	Device        bool
	ContextShift  bool
	KeepTokens    uint32
	DiscardTokens int
}

// SequenceBatchInput: one sequence append.
type SequenceBatchInput struct {
	ID     SequenceID
	Tokens []tokenizer.TokenID
}

// SequenceBatchOutput: one committed append result.
type SequenceBatchOutput struct {
	ID         SequenceID
	Hidden     reference.Value
	Logits     []float32
	Candidates []LogitCandidate
	Token      tokenizer.TokenID
	Tokens     uint32
	Position   uint32
	Pages      []SequenceCachePage
}

// LogitCandidate: bounded device sampling pair.
type LogitCandidate struct {
	ID    tokenizer.TokenID
	Logit float32
}

// SequenceCachePage: logical half-open cache page.
type SequenceCachePage struct {
	Start  uint32
	Tokens uint32
}

// SequenceBatchState: observable sequence-cache state.
type SequenceBatchState struct {
	ID       SequenceID
	Tokens   uint32
	Position uint32
	Device   bool
	Pages    []SequenceCachePage
}

// ContinuousBatch: dynamic independent-sequence cache set.
type ContinuousBatch struct {
	runner     *Runner
	options    ContinuousBatchOptions
	operation  sync.Mutex
	mu         sync.Mutex
	sequences  map[SequenceID]*continuousSequence
	deferred   []*deviceKVCache
	cleanupErr error
	closed     bool
}

type continuousSequence struct {
	host   *KVCache
	device *deviceKVCache
}

// NewContinuousBatch: dynamic sequence scheduler.
func (r *Runner) NewContinuousBatch(
	options ContinuousBatchOptions,
) (*ContinuousBatch, error) {
	if r == nil {
		return nil, errRunnerNil
	}
	if options.MaxSequences <= 0 {
		return nil, errors.New("inference: maximum sequence count must be positive")
	}
	if err := r.lockOpen(); err != nil {
		return nil, err
	}
	defer r.mu.Unlock()
	if options.Device && !r.hasPreloadedWeights() {
		return nil, errors.New("inference: device batch requires preloaded weights")
	}
	forward := r.forwardProgram()
	if options.Device && !forward.PersistentDeviceCache() {
		return nil, errors.New("inference: architecture does not support a retained device cache")
	}
	if !forward.ContinuousBatch() {
		return nil, errors.New("inference: architecture does not support continuous KV batching")
	}
	return &ContinuousBatch{
		runner:    r,
		options:   options,
		sequences: make(map[SequenceID]*continuousSequence),
	}, nil
}

// Step: transactional multi-sequence append.
func (b *ContinuousBatch) Step(
	ctx context.Context,
	inputs []SequenceBatchInput,
) ([]SequenceBatchOutput, error) {
	return b.step(ctx, inputs, deviceOutputLogits, 0)
}

// StepGreedy: device argmax with retained token feedback.
func (b *ContinuousBatch) StepGreedy(
	ctx context.Context,
	inputs []SequenceBatchInput,
) ([]SequenceBatchOutput, error) {
	return b.step(ctx, inputs, deviceOutputGreedy, 0)
}

// StepTopK: bounded device candidate transfer.
func (b *ContinuousBatch) StepTopK(
	ctx context.Context,
	inputs []SequenceBatchInput,
	topK uint32,
) ([]SequenceBatchOutput, error) {
	return b.step(ctx, inputs, deviceOutputTopK, topK)
}

func (b *ContinuousBatch) step(
	ctx context.Context,
	inputs []SequenceBatchInput,
	mode deviceOutputMode,
	topK uint32,
) ([]SequenceBatchOutput, error) {
	if b == nil || b.runner == nil {
		return nil, errors.New("inference: continuous batch is nil")
	}
	if len(inputs) == 0 {
		return nil, errors.New("inference: continuous batch step is empty")
	}
	plan, err := compileDeviceOutputPlan(mode, topK, b.runner.spec.VocabularySize)
	if err != nil {
		return nil, err
	}
	b.operation.Lock()
	defer b.operation.Unlock()
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return nil, errors.New("inference: continuous batch is closed")
	}
	seen := make(map[SequenceID]struct{}, len(inputs))
	newCount := len(b.sequences)
	for index, input := range inputs {
		if len(input.Tokens) == 0 {
			b.mu.Unlock()
			return nil, fmt.Errorf("inference: sequence %d token append is empty", index)
		}
		if _, duplicate := seen[input.ID]; duplicate {
			b.mu.Unlock()
			return nil, fmt.Errorf("inference: sequence ID %d is duplicated", input.ID)
		}
		seen[input.ID] = struct{}{}
		if _, exists := b.sequences[input.ID]; !exists {
			newCount++
		}
	}
	if newCount > b.options.MaxSequences {
		b.mu.Unlock()
		return nil, fmt.Errorf(
			"inference: continuous batch would contain %d sequences, maximum %d",
			newCount,
			b.options.MaxSequences,
		)
	}
	b.mu.Unlock()
	r := b.runner
	if err := r.lockOpen(); err != nil {
		return nil, err
	}
	defer r.mu.Unlock()
	if b.options.Device {
		return b.stepDeviceLocked(ctx, inputs, plan)
	}
	if !plan.fullLogits() {
		return nil, errors.New("inference: reduced device output requires a device batch")
	}
	candidates := make(map[SequenceID]*continuousSequence, len(inputs))
	outputs := make([]SequenceBatchOutput, len(inputs))
	for index, input := range inputs {
		current := b.sequences[input.ID]
		if current == nil {
			current = &continuousSequence{}
		}
		cache := current.host
		var err error
		cache, err = r.cacheForAppendKeeping(
			cache,
			len(input.Tokens),
			b.options.ContextShift,
			b.options.KeepTokens,
			b.options.DiscardTokens,
		)
		if err != nil {
			return nil, fmt.Errorf("inference: sequence %d: %w", input.ID, err)
		}
		hidden, next, err := r.forwardCachedLocked(ctx, input.Tokens, cache)
		if err != nil {
			return nil, fmt.Errorf("inference: sequence %d: %w", input.ID, err)
		}
		last := hidden.LastRowView()
		if len(last.Data) == 0 {
			return nil, fmt.Errorf("inference: sequence %d hidden state is incompatible", input.ID)
		}
		outputInfo := r.outputTensor()
		logits, err := r.logits(ctx, outputInfo, last.Data)
		if err != nil {
			return nil, fmt.Errorf("inference: sequence %d logits: %w", input.ID, err)
		}
		candidates[input.ID] = &continuousSequence{host: next}
		outputs[index] = SequenceBatchOutput{
			ID: input.ID, Hidden: hidden, Logits: logits, Tokens: next.Tokens,
			Position: next.Position,
			Pages:    sequencePages(next.Tokens, b.runner.program.Decode.Session),
		}
	}
	b.mu.Lock()
	for id, candidate := range candidates {
		if current := b.sequences[id]; current != nil && current.device != nil {
			b.deferCleanup(current.device, current.device.Release(context.Background()))
		}
		b.sequences[id] = candidate
	}
	b.mu.Unlock()
	return outputs, nil
}

func (b *ContinuousBatch) stepDeviceLocked(
	ctx context.Context,
	inputs []SequenceBatchInput,
	plan deviceOutputPlan,
) ([]SequenceBatchOutput, error) {
	r := b.runner
	appends := make([]deviceBatchAppend, len(inputs))
	working := make([]*deviceKVCache, len(inputs))
	cleanupWorking := func() error {
		var errs []error
		for index, cache := range working {
			current := b.sequences[inputs[index].ID]
			if cache != nil && (current == nil || cache != current.device) {
				if err := cache.Release(context.Background()); err != nil {
					errs = append(errs, err)
					b.deferred = append(b.deferred, cache)
				}
			}
		}
		return errors.Join(errs...)
	}
	for index, input := range inputs {
		current := b.sequences[input.ID]
		if current != nil {
			working[index] = current.device
		}
		if working[index] != nil && b.options.ContextShift {
			shifted, err := r.compactDeviceCacheForAppend(
				ctx, working[index], len(input.Tokens), b.options.KeepTokens,
				b.options.DiscardTokens, true,
			)
			if err != nil {
				return nil, errors.Join(
					fmt.Errorf("inference: sequence %d: %w", input.ID, err),
					cleanupWorking(),
				)
			}
			working[index] = shifted
		}
		appends[index] = deviceBatchAppend{
			Tokens: input.Tokens, Past: working[index],
		}
	}
	next, err := r.forwardDeviceCachedBatchLocked(ctx, appends, plan)
	cleanupErr := cleanupWorking()
	if err != nil {
		return nil, errors.Join(err, cleanupErr)
	}
	outputs := make([]SequenceBatchOutput, len(inputs))
	for index, cache := range next {
		input := inputs[index]
		outputs[index] = SequenceBatchOutput{
			ID: input.ID, Logits: slices.Clone(cache.Logits), Candidates: slices.Clone(cache.Candidates), Token: cache.Selected,
			Tokens: cache.Tokens, Position: cache.Position,
			Pages: sequencePages(cache.Tokens, b.runner.program.Decode.Session),
		}
	}
	b.mu.Lock()
	b.cleanupErr = errors.Join(b.cleanupErr, cleanupErr)
	for index, input := range inputs {
		if current := b.sequences[input.ID]; current != nil && current.device != nil {
			b.deferCleanup(current.device, current.device.Release(context.Background()))
		}
		b.sequences[input.ID] = &continuousSequence{device: next[index]}
	}
	b.mu.Unlock()
	return outputs, nil
}

// Remove: releases one sequence cache.
func (b *ContinuousBatch) Remove(ctx context.Context, id SequenceID) error {
	if b == nil {
		return errors.New("inference: continuous batch is nil")
	}
	b.operation.Lock()
	b.mu.Lock()
	sequence, ok := b.sequences[id]
	if !ok {
		b.mu.Unlock()
		b.operation.Unlock()
		return fmt.Errorf("inference: sequence ID %d is not active", id)
	}
	delete(b.sequences, id)
	b.mu.Unlock()
	b.operation.Unlock()
	if sequence.device == nil {
		return nil
	}
	if err := sequence.device.Release(ctx); err != nil {
		b.mu.Lock()
		closed := b.closed
		if !closed {
			b.deferred = append(b.deferred, sequence.device)
		}
		b.mu.Unlock()
		if closed {
			err = errors.Join(err, sequence.device.Release(context.Background()))
		}
		return err
	}
	return nil
}

// Fork: clones one sequence cache.
func (b *ContinuousBatch) Fork(source, destination SequenceID) error {
	if b == nil {
		return errors.New("inference: continuous batch is nil")
	}
	b.operation.Lock()
	defer b.operation.Unlock()
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return errors.New("inference: continuous batch is closed")
	}
	if len(b.sequences) >= b.options.MaxSequences {
		return errors.New("inference: continuous batch sequence capacity reached")
	}
	if _, exists := b.sequences[destination]; exists {
		return fmt.Errorf("inference: sequence ID %d is already active", destination)
	}
	sequence, exists := b.sequences[source]
	if !exists {
		return fmt.Errorf("inference: sequence ID %d is not active", source)
	}
	if sequence.device != nil {
		fork, err := cloneDeviceCache(sequence.device)
		if err != nil {
			return err
		}
		b.sequences[destination] = &continuousSequence{device: fork}
		return nil
	}
	b.sequences[destination] = &continuousSequence{host: cloneCache(sequence.host)}
	return nil
}

func cloneDeviceCache(source *deviceKVCache) (*deviceKVCache, error) {
	if source == nil || source.owner == nil || !source.owner.retain() {
		return nil, errors.New("inference: retained device cache is unavailable")
	}
	if source.storage != nil && !source.storage.retain() {
		_ = source.owner.release(context.Background())
		return nil, errors.New("inference: retained device cache storage is unavailable")
	}
	result := *source
	result.session = nil
	result.sessionBranch = 0
	result.Keys = slices.Clone(source.Keys)
	result.Values = slices.Clone(source.Values)
	result.Logits = slices.Clone(source.Logits)
	result.Candidates = slices.Clone(source.Candidates)
	result.Pages = make([]deviceKVPage, len(source.Pages))
	for index, page := range source.Pages {
		result.Pages[index] = page
		result.Pages[index].Keys = slices.Clone(page.Keys)
		result.Pages[index].Values = slices.Clone(page.Values)
	}
	result.States = make([]deviceLayerStates, len(source.States))
	for layer, states := range source.States {
		if states == nil {
			continue
		}
		result.States[layer] = states.Clone()
	}
	return &result, nil
}

// Snapshot: sorted sequence-cache metadata.
func (b *ContinuousBatch) Snapshot() []SequenceBatchState {
	if b == nil {
		return nil
	}
	b.operation.Lock()
	defer b.operation.Unlock()
	b.mu.Lock()
	defer b.mu.Unlock()
	states := make([]SequenceBatchState, 0, len(b.sequences))
	for id, sequence := range b.sequences {
		state := SequenceBatchState{ID: id}
		if sequence.device != nil {
			state.Tokens = sequence.device.Tokens
			state.Position = sequence.device.Position
			state.Device = true
		} else if sequence.host != nil {
			state.Tokens = sequence.host.Tokens
			state.Position = sequence.host.Position
		}
		state.Pages = sequencePages(state.Tokens, b.runner.program.Decode.Session)
		states = append(states, state)
	}
	sort.Slice(states, func(left, right int) bool {
		return states[left].ID < states[right].ID
	})
	return states
}

// Close: releases all sequence caches.
func (b *ContinuousBatch) Close(ctx context.Context) error {
	if b == nil {
		return nil
	}
	b.operation.Lock()
	b.mu.Lock()
	if b.closed && len(b.sequences) == 0 && len(b.deferred) == 0 {
		b.mu.Unlock()
		b.operation.Unlock()
		return nil
	}
	b.closed = true
	errs := []error{b.cleanupErr}
	b.cleanupErr = nil
	deferred := b.deferred
	sequences := b.sequences
	b.deferred = nil
	b.sequences = make(map[SequenceID]*continuousSequence)
	b.mu.Unlock()
	b.operation.Unlock()
	remaining := deferred[:0]
	for _, cache := range deferred {
		if err := cache.Release(ctx); err != nil {
			errs = append(errs, err)
			remaining = append(remaining, cache)
		}
	}
	for _, sequence := range sequences {
		if sequence.device != nil {
			if err := sequence.device.Release(ctx); err != nil {
				errs = append(errs, err)
				remaining = append(remaining, sequence.device)
				continue
			}
		}
	}
	if len(remaining) > 0 {
		b.mu.Lock()
		b.deferred = append(b.deferred, remaining...)
		b.mu.Unlock()
	}
	return errors.Join(errs...)
}

func (b *ContinuousBatch) deferCleanup(cache *deviceKVCache, err error) {
	if err == nil {
		return
	}
	b.cleanupErr = errors.Join(b.cleanupErr, err)
	b.deferred = append(b.deferred, cache)
}

func sequencePages(tokens uint32, session recipe.SessionPolicy) []SequenceCachePage {
	pageTokens := cachePageTokens(tokens, session)
	if pageTokens == 0 {
		return nil
	}
	pageCount := (uint64(tokens) + uint64(pageTokens) - 1) / uint64(pageTokens)
	pages := make([]SequenceCachePage, 0, int(pageCount))
	for start := uint32(0); start < tokens; start += pageTokens {
		count := min(pageTokens, tokens-start)
		pages = append(pages, SequenceCachePage{Start: start, Tokens: count})
	}
	return pages
}
