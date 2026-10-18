package inference

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"hash"
	"math"
	"slices"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/cuda/executor"
	"overgo/internal/model"
	"overgo/internal/recipe"
	"overgo/internal/statecodec"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// SupportsPromptCache reports whether an initialized runner supports the bounded,
// signature-checked pool used by Generate. Schedulers declare their own support.
func (r *Runner) SupportsPromptCache() bool { return r != nil && r.vocab != nil }

const (
	promptStateMagic = "L2GSLT01"
	promptStateLabel = "prompt"
)

// ErrPromptStateRefused marks a prompt state the runner refuses to save or restore.
//   - an empty pool has nothing to save
//   - an entry computed under other LoRA scales is refused because the payload binds the current ones
//   - a projected entry is refused because its state depends on inputs its tokens do not name
//   - an encoder-only entry and a context-shifted cache are refused because their tokens do not name their cache
//   - a malformed payload, or one bound to another model, recipe or LoRA scale set, is refused because its cache would decode wrong tokens.
var ErrPromptStateRefused = errors.New("inference: prompt state refused")

// SavePromptState encodes the most recent prompt-cache entry and answers the payload and its token count.
//   - payload = magic, model signature, model ID, recipe ID, token count, token IDs, then the SaveCache bytes
//   - the recipe ID binds placement, residency and the numeric path, because SaveCache checks shape alone
//   - a device entry is read back to the host first.
func (r *Runner) SavePromptState(ctx context.Context) ([]byte, int, error) {
	if err := r.lockOpen(); err != nil {
		return nil, 0, err
	}
	defer r.mu.Unlock()
	if len(r.promptCaches) == 0 || r.promptCaches[0] == nil {
		return nil, 0, fmt.Errorf("%w: the prompt cache is empty", ErrPromptStateRefused)
	}
	entry := r.promptCaches[0]
	if entry.HasProjection || (entry.Cache == nil && entry.Device == nil) {
		return nil, 0, fmt.Errorf("%w: the latest entry is projected or encoder-only", ErrPromptStateRefused)
	}
	if entry.LoRASignature != r.currentLoRASignature() {
		return nil, 0, fmt.Errorf("%w: the latest entry was computed under other LoRA scales", ErrPromptStateRefused)
	}
	cache := entry.Cache
	if entry.Device != nil {
		var err error
		if cache, err = r.downloadDeviceCache(ctx, entry.Device); err != nil {
			return nil, 0, err
		}
	}
	if cache.Tokens != uint32(len(entry.Tokens)) || effectiveCachePosition(cache) != cache.Tokens {
		return nil, 0, fmt.Errorf("%w: the cache holds %d tokens at position %d for %d prompt tokens",
			ErrPromptStateRefused, cache.Tokens, effectiveCachePosition(cache), len(entry.Tokens))
	}
	cacheData, err := r.SaveCache(cache)
	if err != nil {
		return nil, 0, err
	}
	signature, err := r.sessionModelSignature()
	if err != nil {
		return nil, 0, err
	}
	modelID, recipeID := r.ModelID().String(), r.program.Identity.Recipe.String()
	size, fits := checked.Add64(
		uint64(len(promptStateMagic)), uint64(len(signature)),
		binaryschema.Uint32Bytes, uint64(len(modelID)), binaryschema.Uint32Bytes, uint64(len(recipeID)),
		binaryschema.Uint32Bytes, uint64(len(entry.Tokens))*binaryschema.Uint32Bytes, uint64(len(cacheData)),
	)
	if !fits {
		return nil, 0, errors.New("inference: prompt state size overflows")
	}
	encoder := statecodec.NewEncoderCapacity(uint64(math.MaxInt), size)
	encoder.Raw([]byte(promptStateMagic))
	encoder.Raw(signature[:])
	encoder.String32(modelID)
	encoder.String32(recipeID)
	encoder.U32(uint32(len(entry.Tokens)))
	for _, token := range entry.Tokens {
		encoder.I32(int32(token))
	}
	encoder.Raw(cacheData)
	payload, err := encoder.Data()
	return payload, len(entry.Tokens), err
}

// RestorePromptState decodes a SavePromptState payload into the prompt-cache pool and answers its token count.
//   - a payload decodePromptState refuses is refused with ErrPromptStateRefused
//   - the entry keeps no hidden row and no logits, so a prompt equal to its tokens recomputes the last token
//   - the cache goes to the device when the runner keeps prompt caches there.
func (r *Runner) RestorePromptState(ctx context.Context, data []byte) (int, error) {
	if err := r.lockOpen(); err != nil {
		return 0, err
	}
	defer r.mu.Unlock()
	tokens, cache, err := r.decodePromptState(data)
	if err != nil {
		return 0, fmt.Errorf("%w: %w", ErrPromptStateRefused, err)
	}
	entry := &cachedPrompt{Tokens: tokens, Cache: cache, LoRASignature: r.currentLoRASignature(), Decoded: true}
	if r.persistentDeviceCache() {
		if entry.Device, err = r.uploadDeviceCache(ctx, cache); err != nil {
			return 0, err
		}
		entry.Cache = nil
	}
	return len(tokens), r.storePromptCache(ctx, entry)
}

// decodePromptState checks a SavePromptState payload against the runner and answers its tokens and cache.
func (r *Runner) decodePromptState(data []byte) ([]tokenizer.TokenID, *KVCache, error) {
	decoder, err := r.modelStateDecoder(data, promptStateMagic, promptStateLabel)
	if err != nil {
		return nil, nil, err
	}
	modelID, recipeID := decoder.String32(decoder.Remaining()), decoder.String32(decoder.Remaining())
	count := decoder.U32()
	if decoder.Err() != nil || uint64(count)*binaryschema.Uint32Bytes > decoder.Remaining() {
		return nil, nil, fmt.Errorf("inference: %s state is truncated", promptStateLabel)
	}
	if modelID != r.ModelID().String() || recipeID != r.program.Identity.Recipe.String() {
		return nil, nil, fmt.Errorf("inference: %s state binds model %s and recipe %s, the runner serves %s and %s",
			promptStateLabel, modelID, recipeID, r.ModelID(), r.program.Identity.Recipe)
	}
	tokens := make([]tokenizer.TokenID, count)
	for index := range tokens {
		tokens[index] = tokenizer.TokenID(decoder.I32())
		if _, known := r.vocab.Token(tokens[index]); !known {
			return nil, nil, fmt.Errorf("inference: %s state token %d has out-of-range ID %d", promptStateLabel, index, tokens[index])
		}
	}
	cache, err := r.LoadCache(decoder.Raw(decoder.Remaining()))
	if err != nil {
		return nil, nil, err
	}
	if cache.Tokens != count || effectiveCachePosition(cache) != count {
		return nil, nil, fmt.Errorf("inference: %s state cache holds %d tokens at position %d for %d prompt tokens",
			promptStateLabel, cache.Tokens, effectiveCachePosition(cache), count)
	}
	return tokens, cache, nil
}

// persistentDeviceCache reports whether prompt caches live on the device.
func (r *Runner) persistentDeviceCache() bool {
	return r.hasPreloadedWeights() && r.forwardProgram().PersistentDeviceCache()
}

func (r *Runner) promotePromptCache(index int) *cachedPrompt {
	if index < 0 || index >= len(r.promptCaches) {
		return nil
	}
	selected := r.promptCaches[index]
	if index > 0 {
		copy(r.promptCaches[1:index+1], r.promptCaches[:index])
		r.promptCaches[0] = selected
	}
	return selected
}

func (r *Runner) selectPromptCache(
	requested []tokenizer.TokenID,
	minimum int,
	device bool,
) (*cachedPrompt, int) {
	var selected *cachedPrompt
	selectedIndex := -1
	best := 0
	signature := r.currentLoRASignature()
	for index, candidate := range r.promptCaches {
		if candidate == nil ||
			candidate.HasProjection ||
			candidate.LoRASignature != signature ||
			(device && candidate.Device == nil) ||
			(!device && candidate.Cache == nil) {
			continue
		}
		common := reusablePromptPrefix(candidate.Tokens, requested, minimum)
		// On a tie the entry with fewer tokens past the prefix trims least.
		if common > best || (common == best && common > 0 && len(candidate.Tokens) < len(selected.Tokens)) {
			selected = candidate
			selectedIndex = index
			best = common
		}
	}
	if selectedIndex > 0 {
		r.promotePromptCache(selectedIndex)
	}
	return selected, best
}

func (r *Runner) selectProjectedPromptCache(
	requested []tokenizer.TokenID,
	projection [32]byte,
	minimum int,
) (*cachedPrompt, int) {
	if len(requested) < minimum {
		return nil, 0
	}
	lora := r.currentLoRASignature()
	for index, candidate := range r.promptCaches {
		if candidate == nil || !candidate.HasProjection || candidate.Cache == nil ||
			candidate.LoRASignature != lora || candidate.ProjectionSignature != projection ||
			!slices.Equal(candidate.Tokens, requested) {
			continue
		}
		return r.promotePromptCache(index), len(requested)
	}
	return nil, 0
}

func (r *Runner) selectEncoderSourceCache(
	requested []tokenizer.TokenID,
	minimum int,
) (*cachedPrompt, int) {
	if len(requested) < minimum {
		return nil, 0
	}
	signature := r.currentLoRASignature()
	for index, candidate := range r.promptCaches {
		if candidate == nil ||
			candidate.HasProjection ||
			candidate.LoRASignature != signature ||
			candidate.Cache != nil ||
			!slices.Equal(candidate.Tokens, requested) {
			continue
		}
		_, rows, valid := candidate.Hidden.MatrixExtents()
		if !valid || rows != len(requested) {
			continue
		}
		return r.promotePromptCache(index), len(requested)
	}
	return nil, 0
}

func (r *Runner) ownsDevicePromptCache(cache *deviceKVCache) bool {
	if cache == nil {
		return false
	}
	for _, candidate := range r.promptCaches {
		if candidate != nil && candidate.Device == cache {
			return true
		}
	}
	return false
}

func (r *Runner) storePromptCache(
	ctx context.Context,
	next *cachedPrompt,
) error {
	capacity := r.promptCacheCapacity
	if capacity <= 0 {
		capacity = 1
	}
	var release []*cachedPrompt
	filtered := make([]*cachedPrompt, 0, capacity)
	for _, candidate := range r.promptCaches {
		if candidate == nil {
			continue
		}
		if candidate.LoRASignature == next.LoRASignature &&
			candidate.HasProjection == next.HasProjection &&
			candidate.ProjectionSignature == next.ProjectionSignature &&
			slices.Equal(candidate.Tokens, next.Tokens) {
			release = append(release, candidate)
			continue
		}
		filtered = append(filtered, candidate)
	}
	r.promptCaches = append([]*cachedPrompt{next}, filtered...)
	if len(r.promptCaches) > capacity {
		release = append(release, r.promptCaches[capacity:]...)
		r.promptCaches = r.promptCaches[:capacity]
	}
	evicted := release[:0]
	for _, candidate := range release {
		if candidate.Device != nil && candidate.Device != next.Device {
			evicted = append(evicted, candidate)
		}
	}
	// The request that stores the prompt may be cancelled, but the evicted caches' memory is released
	// regardless; a release that still fails keeps its owner listed, so a later eviction or Close retries it.
	failed, err := releasePromptCaches(context.WithoutCancel(ctx), evicted)
	r.promptCaches = append(r.promptCaches, failed...)
	return err
}

func projectedInputsSignature(inputs ProjectedInputs) [32]byte {
	digest := sha256.New()
	hashUint64(digest, uint64(len(inputs.EmbeddingOverrides)))
	for _, override := range inputs.EmbeddingOverrides {
		hashUint64(digest, uint64(override.TokenIndex))
		hashFloat32s(digest, override.Embedding)
	}
	if inputs.MultiAxisPositions == nil {
		hashUint64(digest, 0)
	} else {
		hashUint64(digest, 1)
		for _, axis := range *inputs.MultiAxisPositions {
			hashUint64(digest, uint64(len(axis)))
			for _, value := range axis {
				hashUint64(digest, uint64(value))
			}
		}
	}
	hashUint64(digest, uint64(len(inputs.DeepstackEmbeddings)))
	for _, value := range inputs.DeepstackEmbeddings {
		hashUint64(digest, uint64(value.Shape.Rank))
		for dimension := range value.Shape.Rank {
			hashUint64(digest, value.Shape.Dims[dimension])
		}
		hashFloat32s(digest, value.Data)
	}
	hashUint64(digest, uint64(len(inputs.BidirectionalAttentionBlocks)))
	for _, block := range inputs.BidirectionalAttentionBlocks {
		hashUint64(digest, uint64(block.Start))
		hashUint64(digest, uint64(block.End))
	}
	hashUint64(digest, uint64(len(inputs.VisualExpertBlocks)))
	for _, block := range inputs.VisualExpertBlocks {
		hashUint64(digest, uint64(block.Start))
		hashUint64(digest, uint64(block.End))
	}
	var result [32]byte
	copy(result[:], digest.Sum(nil))
	return result
}

func hashFloat32s(digest hash.Hash, values []float32) {
	hashUint64(digest, uint64(len(values)))
	for _, value := range values {
		hashUint64(digest, uint64(math.Float32bits(value)))
	}
}

func hashUint64(digest hash.Hash, value uint64) {
	var encoded [8]byte
	binary.LittleEndian.PutUint64(encoded[:], value)
	_, _ = digest.Write(encoded[:])
}

func (r *Runner) trimHostPromptCache(
	hidden reference.Value,
	cache *KVCache,
	keep uint32,
) (reference.Value, *KVCache, error) {
	trimmed, err := r.trimHostPromptKV(cache, keep)
	if err != nil {
		return reference.Value{}, nil, err
	}
	_, rows, valid := hidden.MatrixExtents()
	if !valid || uint64(rows) != uint64(cache.Tokens) {
		return reference.Value{}, nil, fmt.Errorf(
			"inference: prompt hidden shape %v does not contain %d tokens",
			hidden.Shape.Slice(),
			cache.Tokens,
		)
	}
	result, err := hidden.Rows(tensor.FirstOffset, uint64(keep))
	if err != nil {
		return reference.Value{}, nil, err
	}
	return result, trimmed, nil
}

// trimHostPromptKV keeps the first keep tokens of a host cache.
func (r *Runner) trimHostPromptKV(cache *KVCache, keep uint32) (*KVCache, error) {
	if cache == nil || keep == 0 || keep >= cache.Tokens {
		return nil, errors.New("inference: invalid host prompt cache suffix trim")
	}
	trimmed, err := r.RemoveCacheRange(cache, keep, cache.Tokens-keep)
	if err != nil {
		return nil, err
	}
	trimmed.Position = keep
	return trimmed, nil
}

// retainDecodedPromptCache keeps what a generation decoded as a prompt-cache
// entry: every id but the pending last one, whose state is not computed yet,
// so a continuation forwards only that token. A cache holding other than
// those tokens (a context shift) is not kept.
func (r *Runner) retainDecodedPromptCache(
	ctx context.Context,
	ids []tokenizer.TokenID,
	hidden reference.Value,
	host *KVCache,
	device *deviceKVCache,
) error {
	decoded := len(ids) - 1
	switch {
	case decoded < 1:
		return nil
	case device != nil && device.Tokens != uint32(decoded):
		return nil
	case device == nil && (host == nil || host.Tokens != uint32(decoded)):
		return nil
	}
	return r.storePromptCache(ctx, &cachedPrompt{
		Tokens: slices.Clone(ids[:decoded]), Hidden: hidden, Cache: host, Device: device,
		LoRASignature: r.currentLoRASignature(), Decoded: true,
	})
}

func trimDeviceCacheSuffix(cache *deviceKVCache, keep uint32, session recipe.SessionPolicy) error {
	if cache == nil || keep == 0 || keep >= cache.Tokens {
		return errors.New("inference: invalid device prompt cache suffix trim")
	}
	for index := range cache.Keys {
		values := [...]struct {
			label string
			value *executor.DeviceValue
		}{{"key", &cache.Keys[index]}, {"value", &cache.Values[index]}}
		for _, binding := range values {
			shape, valid := tensor.WithTrailingExtent(binding.value.Shape, uint64(keep))
			if !model.CacheStateToken.MatchesTokenExtent(binding.value.Shape, cache.Tokens) || !valid {
				return fmt.Errorf(
					"inference: layer %d device cache %s shape %v does not contain %d tokens",
					index,
					binding.label,
					binding.value.Shape.Slice(),
					cache.Tokens,
				)
			}
			binding.value.Shape = shape
		}
	}
	cache.Tokens = keep
	cache.Position = keep
	cache.session = nil
	cache.Logits = nil
	cache.Candidates = nil
	return rebuildDeviceCachePages(cache, session)
}
