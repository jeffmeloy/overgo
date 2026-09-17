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

	"overgo/internal/cuda/executor"
	"overgo/internal/model"
	"overgo/internal/recipe"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// SupportsPromptCache reports whether an initialized runner supports the bounded,
// signature-checked pool used by Generate. Schedulers declare their own support.
func (r *Runner) SupportsPromptCache() bool { return r != nil && r.vocab != nil }

// ClearPromptCaches: releases retained request state; preserves prepared assets.
func (r *Runner) ClearPromptCaches(ctx context.Context) error {
	if r == nil {
		return errRunnerNil
	}
	if ctx == nil {
		return errors.New("inference: prompt-cache context is nil")
	}
	if err := r.lockOpen(); err != nil {
		return err
	}
	caches := r.detachPromptCaches()
	r.mu.Unlock()
	failed, err := releasePromptCaches(ctx, caches)
	if len(failed) == 0 {
		return err
	}
	r.mu.Lock()
	closed := r.closed
	if !closed {
		r.promptCaches = append(r.promptCaches, failed...)
	}
	r.mu.Unlock()
	if closed {
		_, retryErr := releasePromptCaches(context.Background(), failed)
		err = errors.Join(err, retryErr)
	}
	return err
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
		if common > best {
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
	var errs []error
	for _, candidate := range release {
		if candidate.Device != nil && candidate.Device != next.Device {
			errs = append(errs, candidate.Device.Release(ctx))
			candidate.Device = nil
		}
	}
	return errors.Join(errs...)
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
	if cache == nil || keep == 0 || keep >= cache.Tokens {
		return reference.Value{}, nil, errors.New(
			"inference: invalid host prompt cache suffix trim",
		)
	}
	trimmed, err := r.RemoveCacheRange(cache, keep, cache.Tokens-keep)
	if err != nil {
		return reference.Value{}, nil, err
	}
	trimmed.Position = keep
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
