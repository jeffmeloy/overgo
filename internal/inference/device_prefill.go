package inference

import (
	"context"
	"errors"
	"math"

	"overgo/internal/cuda/executor"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tokenizer"
)

// prefillDeviceCachedLocked appends prompt chunks and retains final-position outputs.
// Only a memory refusal before execution reduces a chunk and retries it.
// The caller retains ownership of the supplied prefix on every exit.
func (r *Runner) prefillDeviceCachedLocked(ctx context.Context, appendInput deviceBatchAppend, plan deviceOutputPlan, chunk int) (_ *deviceKVCache, err error) {
	if chunk <= 0 || len(appendInput.Tokens) == 0 || plan.span() {
		return nil, errors.New("inference: bounded prefill requires tokens, a positive chunk and final-position output")
	}
	past := appendInput.Past
	defer func() {
		if err != nil && past != appendInput.Past {
			err = errors.Join(err, past.Release(context.Background()))
		}
	}()
	// The prompt's cache is sized for the whole prompt at its first chunk: a later chunk that outgrew its page would
	// allocate a new storage beside the one it replaces, and near the context limit that pair outgrows the card at any
	// chunk size (Flash-Next at 129K tokens: 14.5 GB beside 7.3 GB, so the chunks fell to one token each).
	reserve := uint32(len(appendInput.Tokens))
	if past != nil {
		reserve += past.Tokens
	}
	reserve = max(reserve, appendInput.Reserve)
	for remaining := appendInput.Tokens; len(remaining) > 0; {
		chunk = min(chunk, len(remaining))
		source := past
		if past != nil && past.Selection.Pointer != 0 {
			// Prompt tokens override decode feedback, including a one-token tail.
			prefix := *past
			prefix.Selection = executor.DeviceValue{}
			source = &prefix
		}
		chunkInput := deviceBatchAppend{Tokens: remaining[:chunk], Past: source, Reserve: reserve}
		if appendInput.Projected != nil {
			chunkInput.Projected = appendInput.Projected.span(len(appendInput.Tokens)-len(remaining), chunk)
		}
		next, runErr := r.forwardDeviceCachedBranchedBatchLocked(ctx, []deviceBatchAppend{chunkInput}, plan)
		var refused *executor.MemoryCapacityError
		if errors.As(runErr, &refused) {
			// The prompt cache yields before the chunk shrinks: an entry the prompt does not extend holds a whole
			// storage, which no chunk size frees (14 GiB beside a new 240,000-token prompt).
			yielded, yieldErr := r.yieldPromptCaches(ctx, source)
			if yieldErr != nil {
				return nil, errors.Join(runErr, yieldErr)
			}
			if yielded {
				continue
			}
		}
		if errors.As(runErr, &refused) && chunk > 1 {
			// Bisection reduces the candidate without assuming linear memory growth.
			chunk = (chunk + 1) / tensor.PairedExtent
			continue
		}
		if runErr != nil {
			return nil, runErr
		}
		previous := past
		past = next[0]
		if previous != appendInput.Past {
			if err := previous.Release(context.Background()); err != nil {
				return nil, err
			}
		}
		remaining = remaining[chunk:]
	}
	return past, nil
}

// devicePromptChunkTokens bounds the tokens a prompt's read appends a chunk, as Strata's prompt path reads at most
// 8,192 (--prefill auto): a chunk's activations take card memory the expert residency gives up, and after a
// 16K-token Flash-Next prompt read whole 36,566 of 47,937 experts stayed on the card, read in chunks of 8,192 40,581,
// the chats after it decoding about 6% faster and the prompt read as fast (a 237K-token one 3% slower).
// It minimizes the read's per-chunk overhead (the experts of few slots the CPU computes, about 165, 90, 50 and 76 ms
// a chunk of 2,048 to 16,384 tokens) plus the decode's misses on the experts a chunk's memory displaced (0.16 s over
// 300 tokens at 8,192 against 2,048, no more over 440): 1.32, 0.56, 0.26 and 0.50 s at a 16K read, 8,192 for every
// answer's length (the derive-tuning chunk record of 2026-10-24). Provisional: the load-costs step computes it from
// the CPU's time a slot and the card's time a token.
const devicePromptChunkTokens = 8192

// promptReadPlan is the output plan of mode for a prompt read of tokens tokens: a request that drafts over its prompt
// reads its rows' streams into the draft's history, and a read of more than one token marks its routed nodes as a
// prompt's (tensor.Builder.SetPromptExperts), whose small groups then take the grouped products, the CPU computing the
// experts in host memory: a chat right after a 16K Flash-Next read ran its prompt through the decode products, which
// read the experts the long read had moved out over the link, its first token 0.78 s against 0.17.
func (r *Runner) promptReadPlan(mode deviceOutputMode, tokens int) (deviceOutputPlan, error) {
	plan, err := compileDeviceOutputPlan(mode, 0, r.spec.VocabularySize)
	if err != nil {
		return deviceOutputPlan{}, err
	}
	plan.streams = r.hyperDraft != nil && r.hyperDraft.prompt != nil
	plan.promptExperts = tokens > 1
	return plan, nil
}

// readDevicePromptLocked reads the prompt's tokens after cached onto past under the output plan of mode, sizing the
// cache for the prompt and the output tokens it may decode after it, within the context, so that no part of the read
// nor a decode step outgrows the cache into a copy. A recurrent prompt its entry keeps (keep), with at least
// promptBoundaryTokens before its last turn, is read in two parts, the second its last turn, and the cache before that
// turn is answered too, nil otherwise: a recurrent state cannot step back a token, so a later request that keeps the
// turns before it reads on from there.
func (r *Runner) readDevicePromptLocked(
	ctx context.Context,
	mode deviceOutputMode,
	ids []tokenizer.TokenID,
	cached, output int,
	past *deviceKVCache,
	projected *ProjectedInputs,
	keep bool,
) (*deviceKVCache, *deviceKVCache, error) {
	plan, err := r.promptReadPlan(mode, len(ids)-cached)
	if err != nil {
		return nil, nil, err
	}
	reserve := uint64(len(ids)) + uint64(max(output, 0))
	if limit := uint64(r.spec.ContextLength); limit > 0 {
		reserve = min(reserve, max(limit, uint64(len(ids))))
	}
	read := func(tokens []tokenizer.TokenID, from *deviceKVCache) (*deviceKVCache, error) {
		next, err := r.forwardDeviceCachedBatchLocked(ctx, []deviceBatchAppend{{
			Tokens: tokens, Past: from, Projected: projected, Reserve: uint32(min(reserve, math.MaxUint32)),
		}}, plan)
		if err != nil {
			return nil, err
		}
		return next[0], nil
	}
	turn := r.lastTurnStart(ids)
	if !keep || projected != nil || turn <= cached || turn < r.promptBoundaryTokens || !r.hasRecurrentCache() {
		end, err := read(ids[cached:], past)
		return end, nil, err
	}
	boundary, err := read(ids[cached:turn], past)
	if err != nil {
		return nil, nil, err
	}
	// The turn's header is marked as the read is (promptReadPlan), unless one token, a decode step's append, which a
	// kept decode session replays (a 16K Flash-Next prompt's header of 7 tokens: 68 ms against 183).
	plan.promptExperts = len(ids)-turn > 1
	end, err := read(ids[turn:], boundary)
	if err != nil {
		return nil, nil, errors.Join(err, boundary.Release(context.Background()))
	}
	return end, boundary, nil
}

// reusableDeviceCacheStorage is the storage an append from past writes in place: past's own where it fits the append
// and nothing holds it but past and prompt-cache entries that end at or before past's tokens, which an append after
// them leaves as they are (a copy beside a 240,000-token Flash-Next cache asked another 14 GiB); nil otherwise.
//   - past is one of those entries or the one cache beside them, since the runner's own steps hold a storage in turn
func (r *Runner) reusableDeviceCacheStorage(past *deviceKVCache, plan deviceCacheTargetPlan) *deviceCacheStorage {
	if past == nil || !past.storage.aliases(past) || !deviceCacheStorageFits(past.storage, plan) {
		return nil
	}
	entries := make(map[*deviceKVCache]struct{})
	for _, entry := range r.promptCaches {
		for _, cache := range entry.deviceCaches() {
			if cache.storage != past.storage {
				continue
			}
			if cache.Tokens > past.Tokens {
				return nil
			}
			entries[cache] = struct{}{}
		}
	}
	if holders := past.storage.holders(); holders == 0 || holders > len(entries)+1 {
		return nil
	}
	return past.storage
}

func (r *Runner) checkDevicePrefillMemory(ctx context.Context, execution *executor.IndexedGraph, plan deviceCacheTargetPlan, past *deviceKVCache) error {
	var slots []executor.OutputSlot
	add := func(value deviceCacheTargetValue) {
		if value.enabled {
			slots = append(slots, value.slot)
		}
	}
	for _, layer := range plan.layers {
		add(layer.key)
		add(layer.value)
		for _, state := range layer.states {
			add(state)
		}
	}
	var capacities []uint64
	fresh := r.reusableDeviceCacheStorage(past, plan) == nil
	if plan.elements > 0 && fresh {
		shape, err := tensor.NewShape(plan.elements)
		if err != nil {
			return err
		}
		bytes, err := shape.Bytes(dtype.F32)
		if err != nil {
			return err
		}
		capacities = []uint64{bytes}
	}
	// New QSA or attention cells, and the rotation table at the first QSA step, are buffers the read allocates beside
	// its graph.
	if plan.qsaCapacity > 0 && fresh {
		capacities = append(capacities, r.qsaCellBytes(plan.qsaCapacity))
	}
	if plan.cellCapacity > 0 && fresh {
		capacities = append(capacities, r.attentionCellBytes(plan.cellCapacity))
	}
	if plan.qsaCapacity > 0 && r.qsaRope == nil {
		capacities = append(capacities, qsaRopeBytes(r.qsaRopeRows()))
	}
	if streams := r.qsaSlots(plan.qsaCapacity) > 0; streams && (r.qsaStage == nil || r.qsaStage.capacity < plan.qsaCapacity) {
		capacities = append(capacities, qsaStageBytes(plan.qsaCapacity))
	}
	return r.cuda.CheckRetainedMemory(ctx, execution.Graph, execution.Inputs, slots, capacities)
}
