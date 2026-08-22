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
	"overgo/internal/recipe"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

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
	if !checked.Nonzero(len(failed)) {
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
	if !checked.ValidIndex(index, len(r.promptCaches)) {
		return nil
	}
	selected := r.promptCaches[index]
	if checked.Nonzero(index) {
		copy(r.promptCaches[tensor.SingletonExtent:index+tensor.SingletonExtent], r.promptCaches[:index])
		r.promptCaches[tensor.FirstOffset] = selected
	}
	return selected
}

func (r *Runner) selectPromptCache(
	requested []tokenizer.TokenID,
	minimum int,
	device bool,
) (*cachedPrompt, int) {
	var selected *cachedPrompt
	var selectedIndex int
	var found bool
	var best int
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
			found = true
			best = common
		}
	}
	if found && checked.Nonzero(selectedIndex) {
		r.promotePromptCache(selectedIndex)
	}
	return selected, best
}

func (r *Runner) selectProjectedPromptCache(
	requested []tokenizer.TokenID,
	projection [sha256.Size]byte,
	minimum int,
) (*cachedPrompt, int) {
	if len(requested) < minimum {
		var reused int
		return nil, reused
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
	var reused int
	return nil, reused
}

func (r *Runner) selectEncoderSourceCache(
	requested []tokenizer.TokenID,
	minimum int,
) (*cachedPrompt, int) {
	if len(requested) < minimum {
		var reused int
		return nil, reused
	}
	signature := r.currentLoRASignature()
	for index, candidate := range r.promptCaches {
		if candidate == nil {
			continue
		}
		_, _, validHidden := tensor.MatrixExtents(candidate.Hidden.Shape)
		if candidate.HasProjection ||
			candidate.LoRASignature != signature ||
			candidate.Cache != nil ||
			!validHidden ||
			!slices.Equal(candidate.Tokens, requested) {
			continue
		}
		return r.promotePromptCache(index), len(requested)
	}
	var reused int
	return nil, reused
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
	if !checked.PositiveInts(capacity) {
		capacity = tensor.SingletonExtent
	}
	var release []*cachedPrompt
	var initialLength int
	filtered := make([]*cachedPrompt, initialLength, capacity)
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

func projectedInputsSignature(inputs ProjectedInputs) [sha256.Size]byte {
	digest := sha256.New()
	hashUint64(digest, uint64(len(inputs.EmbeddingOverrides)))
	for _, override := range inputs.EmbeddingOverrides {
		hashUint64(digest, uint64(override.TokenIndex))
		hashFloat32s(digest, override.Embedding)
	}
	if inputs.MultiAxisPositions == nil {
		var absent uint64
		hashUint64(digest, absent)
	} else {
		hashUint64(digest, tensor.SingletonExtent)
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
	var result [sha256.Size]byte
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
	var encoded [binaryschema.Uint64Bytes]byte
	binary.LittleEndian.PutUint64(encoded[:], value)
	_, _ = digest.Write(encoded[:])
}

func (r *Runner) trimHostPromptCache(
	hidden reference.Value,
	cache *KVCache,
	keep uint32,
) (reference.Value, *KVCache, error) {
	if cache == nil || !checked.Nonzero(keep) || keep >= cache.Tokens {
		return reference.Value{}, nil, errors.New(
			"inference: invalid host prompt cache suffix trim",
		)
	}
	trimmed, err := r.RemoveCacheRange(cache, keep, cache.Tokens-keep)
	if err != nil {
		return reference.Value{}, nil, err
	}
	trimmed.Position = keep
	_, hiddenTokens, validHidden := tensor.MatrixExtents(hidden.Shape)
	if !validHidden || !checked.Equal(hiddenTokens, uint64(cache.Tokens)) {
		return reference.Value{}, nil, fmt.Errorf(
			"inference: prompt hidden shape %v does not contain %d tokens",
			hidden.Shape.Slice(),
			cache.Tokens,
		)
	}
	result, err := reference.SelectRows(hidden, tensor.FirstOffset, uint64(keep))
	if err != nil {
		return reference.Value{}, nil, errors.New("inference: prompt hidden data is shorter than its shape")
	}
	return result, trimmed, nil
}

func trimDeviceCacheSuffix(cache *deviceKVCache, keep uint32, session recipe.SessionPolicy) error {
	if cache == nil || !checked.Nonzero(keep) || keep >= cache.Tokens {
		return errors.New("inference: invalid device prompt cache suffix trim")
	}
	for index := range cache.Keys {
		for label, value := range map[string]*executor.DeviceValue{
			"key":   &cache.Keys[index],
			"value": &cache.Values[index],
		} {
			_, _, tokens, validShape := tensor.Extents3(value.Shape)
			shape, validTrim := tensor.WithTrailingExtent(value.Shape, uint64(keep))
			if !validShape || !validTrim || !checked.Equal(tokens, uint64(cache.Tokens)) {
				return fmt.Errorf(
					"inference: layer %d device cache %s shape %v does not contain %d tokens",
					index,
					label,
					value.Shape.Slice(),
					cache.Tokens,
				)
			}
			value.Shape = shape
		}
	}
	cache.Tokens = keep
	cache.Position = keep
	cache.session = nil
	cache.Logits = nil
	cache.Candidates = nil
	return rebuildDeviceCachePages(cache, session)
}
