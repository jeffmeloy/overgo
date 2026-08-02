package inference

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"

	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

// SequenceID: continuous-batch cache owner.
type SequenceID uint64

// ContinuousBatchOptions: cache and admission policy.
type ContinuousBatchOptions struct {
	MaxSequences  int
	PageTokens    uint32
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
	ID       SequenceID
	Hidden   reference.Value
	Logits   []float32
	Tokens   uint32
	Position uint32
	Pages    []SequenceCachePage
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
	runner    *Runner
	options   ContinuousBatchOptions
	mu        sync.Mutex
	sequences map[SequenceID]*continuousSequence
	closed    bool
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
		return nil, errors.New("inference: runner is nil")
	}
	if options.MaxSequences <= 0 {
		return nil, errors.New("inference: maximum sequence count must be positive")
	}
	if options.PageTokens == 0 {
		options.PageTokens = r.cachePageTokens
		if options.PageTokens == 0 {
			options.PageTokens = 256
		}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, errors.New("inference: runner is closed")
	}
	if options.Device && !r.hasPreloadedWeights() {
		return nil, errors.New("inference: device batch requires preloaded weights")
	}
	if options.Device && !supportsPersistentDeviceCache(r.spec) {
		return nil, errors.New("inference: architecture does not support a retained device cache")
	}
	if r.spec.NonCausalAttention || r.spec.Architecture == "t5" {
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
	if b == nil || b.runner == nil {
		return nil, errors.New("inference: continuous batch is nil")
	}
	if len(inputs) == 0 {
		return nil, errors.New("inference: continuous batch step is empty")
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return nil, errors.New("inference: continuous batch is closed")
	}
	seen := make(map[SequenceID]struct{}, len(inputs))
	newCount := len(b.sequences)
	for index, input := range inputs {
		if len(input.Tokens) == 0 {
			return nil, fmt.Errorf("inference: sequence %d token append is empty", index)
		}
		if _, duplicate := seen[input.ID]; duplicate {
			return nil, fmt.Errorf("inference: sequence ID %d is duplicated", input.ID)
		}
		seen[input.ID] = struct{}{}
		if _, exists := b.sequences[input.ID]; !exists {
			newCount++
		}
	}
	if newCount > b.options.MaxSequences {
		return nil, fmt.Errorf(
			"inference: continuous batch would contain %d sequences, maximum %d",
			newCount,
			b.options.MaxSequences,
		)
	}
	r := b.runner
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, errors.New("inference: runner is closed")
	}
	if b.options.Device {
		return b.stepDeviceLocked(ctx, inputs)
	}
	candidates := make(map[SequenceID]*continuousSequence, len(inputs))
	outputs := make([]SequenceBatchOutput, len(inputs))
	releaseCandidates := func() {
		for _, candidate := range candidates {
			if candidate.device != nil {
				_ = candidate.device.Release(context.Background())
			}
		}
	}
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
			releaseCandidates()
			return nil, fmt.Errorf("inference: sequence %d: %w", input.ID, err)
		}
		hidden, next, err := r.forwardCachedLocked(ctx, input.Tokens, cache)
		if err != nil {
			releaseCandidates()
			return nil, fmt.Errorf("inference: sequence %d: %w", input.ID, err)
		}
		width := int(r.spec.EmbeddingLength)
		if hidden.Shape.Rank != 2 || width <= 0 || len(hidden.Data) < width {
			releaseCandidates()
			return nil, fmt.Errorf("inference: sequence %d hidden state is incompatible", input.ID)
		}
		outputInfo := r.weights.TokenEmbedding
		if r.weights.Output != nil {
			outputInfo = *r.weights.Output
		}
		logits, err := r.logits(ctx, outputInfo, hidden.Data[len(hidden.Data)-width:])
		if err != nil {
			releaseCandidates()
			return nil, fmt.Errorf("inference: sequence %d logits: %w", input.ID, err)
		}
		candidates[input.ID] = &continuousSequence{host: next}
		outputs[index] = SequenceBatchOutput{
			ID: input.ID, Hidden: hidden, Logits: logits, Tokens: next.Tokens,
			Position: next.Position,
			Pages:    sequencePages(next.Tokens, b.options.PageTokens),
		}
	}
	for id, candidate := range candidates {
		if current := b.sequences[id]; current != nil && current.device != nil {
			_ = current.device.Release(context.Background())
		}
		b.sequences[id] = candidate
	}
	return outputs, nil
}

func (b *ContinuousBatch) stepDeviceLocked(
	ctx context.Context,
	inputs []SequenceBatchInput,
) ([]SequenceBatchOutput, error) {
	r := b.runner
	appends := make([]deviceBatchAppend, len(inputs))
	working := make([]*deviceKVCache, len(inputs))
	cleanupWorking := func() {
		for index, cache := range working {
			current := b.sequences[inputs[index].ID]
			if cache != nil && (current == nil || cache != current.device) {
				_ = cache.Release(context.Background())
			}
		}
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
				cleanupWorking()
				return nil, fmt.Errorf("inference: sequence %d: %w", input.ID, err)
			}
			working[index] = shifted
		}
		appends[index] = deviceBatchAppend{Tokens: input.Tokens, Past: working[index]}
	}
	next, err := r.forwardDeviceCachedBatchLocked(ctx, appends)
	cleanupWorking()
	if err != nil {
		return nil, err
	}
	releaseNext := func() {
		for _, cache := range next {
			_ = cache.Release(context.Background())
		}
	}
	outputs := make([]SequenceBatchOutput, len(inputs))
	for index, cache := range next {
		cache.PageTokens = b.options.PageTokens
		if err := rebuildDeviceCachePages(cache, b.options.PageTokens); err != nil {
			releaseNext()
			return nil, err
		}
		input := inputs[index]
		outputs[index] = SequenceBatchOutput{
			ID: input.ID, Logits: append([]float32(nil), cache.Logits...),
			Tokens: cache.Tokens, Position: cache.Position,
			Pages: sequencePages(cache.Tokens, b.options.PageTokens),
		}
	}
	for index, input := range inputs {
		if current := b.sequences[input.ID]; current != nil && current.device != nil {
			_ = current.device.Release(context.Background())
		}
		b.sequences[input.ID] = &continuousSequence{device: next[index]}
	}
	return outputs, nil
}

// Remove: releases one sequence cache.
func (b *ContinuousBatch) Remove(ctx context.Context, id SequenceID) error {
	if b == nil {
		return errors.New("inference: continuous batch is nil")
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	sequence, ok := b.sequences[id]
	if !ok {
		return fmt.Errorf("inference: sequence ID %d is not active", id)
	}
	delete(b.sequences, id)
	if sequence.device != nil {
		return sequence.device.Release(ctx)
	}
	return nil
}

// Fork: clones one host-cache sequence.
func (b *ContinuousBatch) Fork(source, destination SequenceID) error {
	if b == nil {
		return errors.New("inference: continuous batch is nil")
	}
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
		return errors.New("inference: retained device sequence cannot be forked")
	}
	b.sequences[destination] = &continuousSequence{host: cloneCache(sequence.host)}
	return nil
}

// Snapshot: sorted sequence-cache metadata.
func (b *ContinuousBatch) Snapshot() []SequenceBatchState {
	if b == nil {
		return nil
	}
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
		state.Pages = sequencePages(state.Tokens, b.options.PageTokens)
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
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return nil
	}
	b.closed = true
	var errs []error
	for id, sequence := range b.sequences {
		if sequence.device != nil {
			errs = append(errs, sequence.device.Release(ctx))
		}
		delete(b.sequences, id)
	}
	return errors.Join(errs...)
}

func sequencePages(tokens, pageTokens uint32) []SequenceCachePage {
	if pageTokens == 0 {
		pageTokens = 256
	}
	pageCount := (uint64(tokens) + uint64(pageTokens) - 1) / uint64(pageTokens)
	pages := make([]SequenceCachePage, 0, int(pageCount))
	for start := uint32(0); start < tokens; start += pageTokens {
		count := min(pageTokens, tokens-start)
		pages = append(pages, SequenceCachePage{Start: start, Tokens: count})
	}
	return pages
}

func supportsPersistentDeviceCache(spec model.Spec) bool {
	profile, ok := model.LookupArchitecture(spec.Architecture)
	if !ok {
		return false
	}
	return profile.Has(model.ArchitectureQwenGDN) ||
		!profile.Has(model.ArchitectureRecurrent) &&
			!profile.Has(model.ArchitectureMLA) &&
			spec.Architecture != "gemma3n" &&
			spec.Architecture != "lfm2" && spec.Architecture != "lfm2moe"
}
