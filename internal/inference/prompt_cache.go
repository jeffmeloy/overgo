package inference

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"llamacpp2go/internal/cuda/executor"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

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
		copy(r.promptCaches[1:selectedIndex+1], r.promptCaches[:selectedIndex])
		r.promptCaches[0] = selected
	}
	return selected, best
}

func (r *Runner) selectT5SourceCache(
	requested []tokenizer.TokenID,
	minimum int,
) (*cachedPrompt, int) {
	if len(requested) < minimum {
		return nil, 0
	}
	signature := r.currentLoRASignature()
	for index, candidate := range r.promptCaches {
		if candidate == nil ||
			candidate.LoRASignature != signature ||
			candidate.Cache != nil ||
			candidate.Hidden.Shape.Rank != 2 ||
			!slices.Equal(candidate.Tokens, requested) {
			continue
		}
		if index > 0 {
			copy(r.promptCaches[1:index+1], r.promptCaches[:index])
			r.promptCaches[0] = candidate
		}
		return candidate, len(requested)
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
		if candidate.LoRASignature == next.LoRASignature && slices.Equal(candidate.Tokens, next.Tokens) {
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
	if hidden.Shape.Rank != 2 ||
		hidden.Shape.Dims[1] != uint64(cache.Tokens) {
		return reference.Value{}, nil, fmt.Errorf(
			"inference: prompt hidden shape %v does not contain %d tokens",
			hidden.Shape.Slice(),
			cache.Tokens,
		)
	}
	width := hidden.Shape.Dims[0]
	count := width * uint64(keep)
	if count > uint64(len(hidden.Data)) {
		return reference.Value{}, nil, errors.New(
			"inference: prompt hidden data is shorter than its shape",
		)
	}
	shape := hidden.Shape
	shape.Dims[1] = uint64(keep)
	result := reference.Value{
		Shape: shape,
		Data:  append([]float32(nil), hidden.Data[:int(count)]...),
	}
	return result, trimmed, nil
}

func trimDeviceCacheSuffix(cache *deviceKVCache, keep uint32) error {
	if cache == nil || keep == 0 || keep >= cache.Tokens {
		return errors.New("inference: invalid device prompt cache suffix trim")
	}
	for index := range cache.Keys {
		for label, value := range map[string]*executor.DeviceValue{
			"key":   &cache.Keys[index],
			"value": &cache.Values[index],
		} {
			if value.Shape.Rank != 3 ||
				value.Shape.Dims[2] != uint64(cache.Tokens) {
				return fmt.Errorf(
					"inference: layer %d device cache %s shape %v does not contain %d tokens",
					index,
					label,
					value.Shape.Slice(),
					cache.Tokens,
				)
			}
			value.Shape.Dims[2] = uint64(keep)
		}
	}
	cache.Tokens = keep
	cache.Position = keep
	cache.Logits = nil
	return nil
}
