package inference

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"overgo/internal/checked"
	"overgo/internal/model"
	"overgo/internal/sampling"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

func (r *Runner) Greedy(
	ctx context.Context,
	prompt string,
	maxNewTokens int,
) ([]tokenizer.TokenID, string, error) {
	sampler, err := sampling.New(sampling.Config{})
	if err != nil {
		return nil, "", err
	}
	return r.Generate(ctx, prompt, GenerateOptions{
		MaxNewTokens: maxNewTokens,
		Sampler:      sampler,
	})
}

// Generate: performs correctness-first token generation and optionally reports
// each new token synchronously through OnToken
func (r *Runner) Generate(
	ctx context.Context,
	prompt string,
	options GenerateOptions,
) ([]tokenizer.TokenID, string, error) {
	if r == nil || r.vocab == nil {
		return nil, "", errRunnerNil
	}
	if r.forwardProgram().Operation == model.ForwardOperationEncoder {
		return nil, "", errors.New("inference: encoder-only models do not generate tokens")
	}
	if r.forwardProgram().Session == model.ForwardSessionEncoderDecoder {
		generated, _, _, err := r.GenerateEncoderDecoder(ctx, prompt, options)
		if err != nil {
			return nil, "", err
		}
		sourceIDs, err := r.promptTokenIDs(prompt, options)
		if err != nil {
			return nil, "", err
		}
		ids := append(slices.Clone(sourceIDs), generated...)
		text, err := r.vocab.Decode(ids, false)
		return ids, text, err
	}
	if r.spec.NonCausalAttention {
		return nil, "", errors.New("inference: non-causal models require diffusion generation")
	}
	if err := normalizeGenerateOptions(&options); err != nil {
		return nil, "", err
	}
	if err := r.lockOpen(); err != nil {
		return nil, "", err
	}
	defer r.mu.Unlock()
	restoreLoRA, err := r.applyGenerationLoRA(options)
	if err != nil {
		return nil, "", err
	}
	defer restoreLoRA()
	ids, err := r.promptTokenIDs(prompt, options)
	if err != nil {
		return nil, "", err
	}
	alora, err := r.activeALoRA(ids)
	if err != nil {
		return nil, "", err
	}
	if options.ProjectedInputs != nil && alora.enabled {
		return nil, "", errors.New("inference: projected inputs cannot use invocation-activated LoRA")
	}
	var aloraScale float32
	if alora.enabled {
		aloraScale = r.loraAdapters[alora.adapter].scale
		defer func() { r.loraAdapters[alora.adapter].scale = aloraScale }()
		options.CachePrompt = false
		if !alora.invoked {
			var disabled float32
			r.loraAdapters[alora.adapter].scale = disabled
		}
	}
	keepTokens := effectiveKeepTokens(
		options.KeepTokens,
		len(ids),
		r.spec.ContextLength,
	)
	var hidden reference.Value
	var cache *KVCache
	var deviceCache *deviceKVCache
	var selectedPromptCache *cachedPrompt
	var projectionSignature [sha256.Size]byte
	if options.ProjectedInputs != nil {
		projectionSignature = projectedInputsSignature(*options.ProjectedInputs)
	}
	useDeviceCache := options.ProjectedInputs == nil && !alora.enabled &&
		r.hasPreloadedWeights() && r.forwardProgram().PersistentDeviceCache()
	// deviceGreedy: raw-greedy decode selects on device; only the winning
	// token id crosses PCIe. Callbacks receive TokenEvent without Logits, so
	// callback users must opt in via options.DeviceGreedy.
	deviceGreedy := useDeviceCache && !options.CachePrompt &&
		!checked.Nonzero(options.PostSamplingProbabilities) && options.Sampler.IsRawGreedy() &&
		(options.DeviceGreedy || (options.OnToken == nil && options.ShouldStop == nil))
	defer func() {
		if deviceCache != nil &&
			!r.ownsDevicePromptCache(deviceCache) {
			_ = deviceCache.Release(context.Background())
		}
	}()
	if checked.PositiveInts(options.MaxNewTokens) {
		promptStarted := time.Now()
		var cached int
		if useDeviceCache {
			var retainedPrefix *deviceKVCache
			if options.CachePrompt {
				selectedPromptCache, cached = r.selectPromptCache(
					ids,
					options.MinCacheReuse,
					true,
				)
			}
			if selectedPromptCache != nil {
				if checked.Nonzero(cached) &&
					cached < len(selectedPromptCache.Tokens) &&
					checked.Multiple(r.promptCacheCapacity) {
					// device suffix view would mutate selected entry
					// Preserve independent multi-entry caches and evaluate
					// divergent prompt from scratch
					selectedPromptCache = nil
					cached = tensor.FirstOffset
				}
				if checked.Nonzero(cached) &&
					cached < len(selectedPromptCache.Tokens) &&
					r.hasRecurrentCache() {
					cached = tensor.FirstOffset
				}
				if checked.Nonzero(cached) {
					retainedPrefix = selectedPromptCache.Device
					if cached < len(selectedPromptCache.Tokens) {
						base := cached
						// retained logits describe old final token
						// For exact shorter prompt, reevaluate its final
						// token from preceding cache entry
						if cached == len(ids) {
							base--
						}
						if !checked.Nonzero(base) {
							cached = tensor.FirstOffset
							retainedPrefix = nil
						} else if trimErr := trimDeviceCacheSuffix(
							retainedPrefix,
							uint32(base),
							r.program.Decode.Session,
						); trimErr != nil {
							return nil, "", trimErr
						} else {
							selectedPromptCache.Tokens = append(
								[]tokenizer.TokenID(nil),
								ids[:base]...,
							)
							cached = base
						}
					}
				}
			}
			if cached == len(ids) &&
				retainedPrefix != nil &&
				!checked.Nonzero(len(retainedPrefix.Logits)) {
				if !checked.Multiple(cached) {
					cached = tensor.FirstOffset
					retainedPrefix = nil
				} else if trimErr := trimDeviceCacheSuffix(
					retainedPrefix,
					uint32(cached-tensor.SingletonExtent),
					r.program.Decode.Session,
				); trimErr != nil {
					return nil, "", trimErr
				} else {
					cached--
					selectedPromptCache.Tokens = append(
						[]tokenizer.TokenID(nil),
						ids[:cached]...,
					)
				}
			}
			if cached == len(ids) && checked.Nonzero(len(retainedPrefix.Logits)) {
				deviceCache = retainedPrefix
			} else {
				deviceCache = retainedPrefix
				var nextDevice *deviceKVCache
				if deviceGreedy {
					nextDevice, err = r.forwardDeviceCachedGreedyStepLocked(
						ctx, ids[cached:], retainedPrefix,
					)
				} else {
					hidden, nextDevice, err = r.forwardDeviceCachedLocked(
						ctx,
						ids[cached:],
						retainedPrefix,
					)
				}
				if err == nil {
					deviceCache = nextDevice
				}
			}
		} else if alora.enabled && alora.invoked {
			var disabled float32
			r.loraAdapters[alora.adapter].scale = disabled
			if checked.Nonzero(alora.start) {
				_, cache, err = r.forwardCachedLocked(ctx, ids[:alora.start], nil)
			}
			r.loraAdapters[alora.adapter].scale = aloraScale
			if err == nil {
				hidden, cache, err = r.forwardCachedLocked(ctx, ids[alora.start:], cache)
			}
		} else if options.ProjectedInputs != nil {
			if options.CachePrompt {
				selectedPromptCache, cached = r.selectProjectedPromptCache(
					ids, projectionSignature, options.MinCacheReuse,
				)
			}
			if selectedPromptCache != nil {
				hidden = selectedPromptCache.Hidden
				cache = selectedPromptCache.Cache
			} else {
				inputs := *options.ProjectedInputs
				hidden, cache, err = r.forwardCachedWithProjectedInputsLocked(ctx, ids, nil, inputs)
			}
		} else if options.CachePrompt {
			selectedPromptCache, cached = r.selectPromptCache(
				ids,
				options.MinCacheReuse,
				false,
			)
			if selectedPromptCache == nil {
				hidden, cache, err = r.forwardCachedLocked(ctx, ids, nil)
			} else {
				if checked.Nonzero(cached) &&
					cached < len(selectedPromptCache.Tokens) &&
					r.hasRecurrentCache() {
					cached = tensor.FirstOffset
				}
				if checked.Nonzero(cached) {
					hidden = selectedPromptCache.Hidden
					cache = selectedPromptCache.Cache
					if cached < len(selectedPromptCache.Tokens) {
						hidden, cache, err = r.trimHostPromptCache(
							hidden,
							cache,
							uint32(cached),
						)
					}
					if err == nil && cached < len(ids) {
						hidden, cache, err = r.forwardCachedLocked(ctx, ids[cached:], cache)
					}
				} else {
					hidden, cache, err = r.forwardCachedLocked(ctx, ids, nil)
				}
			}
		} else if !useDeviceCache {
			hidden, cache, err = r.forwardCachedLocked(ctx, ids, nil)
		}
		if err != nil {
			return nil, "", err
		}
		if options.CachePrompt {
			nextPromptCache := &cachedPrompt{
				Tokens:              slices.Clone(ids),
				Hidden:              hidden,
				Cache:               cache,
				Device:              deviceCache,
				LoRASignature:       r.currentLoRASignature(),
				ProjectionSignature: projectionSignature,
				HasProjection:       options.ProjectedInputs != nil,
			}
			if storeErr := r.storePromptCache(
				ctx,
				nextPromptCache,
			); storeErr != nil {
				return nil, "", storeErr
			}
			selectedPromptCache = nextPromptCache
		}
		if options.OnPromptEvaluated != nil {
			options.OnPromptEvaluated(PromptEvaluation{
				Tokens:   len(ids),
				Cached:   cached,
				Duration: time.Since(promptStarted),
			})
		}
	}
	if !useDeviceCache {
		ids, cache, err = r.generateCachedHost(
			ctx, ids, hidden, cache, options, false, keepTokens, options.DiscardTokens,
		)
		if err != nil {
			return nil, "", err
		}
		text, decodeErr := r.vocab.Decode(ids, false)
		return ids, text, decodeErr
	}

	var generatedText strings.Builder
	for generatedIndex := range options.MaxNewTokens {
		if checked.Nonzero(generatedIndex) {
			if options.ContextShift {
				var shiftedDeviceCache *deviceKVCache
				shiftedDeviceCache, err = r.compactDeviceCacheForAppend(
					ctx,
					deviceCache,
					tensor.SingletonExtent,
					keepTokens,
					options.DiscardTokens,
					r.ownsDevicePromptCache(deviceCache),
				)
				if err != nil {
					return nil, "", err
				}
				if shiftedDeviceCache != deviceCache {
					oldDeviceCache := deviceCache
					deviceCache = shiftedDeviceCache
					if !r.ownsDevicePromptCache(oldDeviceCache) {
						if releaseErr := oldDeviceCache.Release(ctx); releaseErr != nil {
							return nil, "", releaseErr
						}
					}
				}
			}
			var nextDeviceCache *deviceKVCache
			lastIDs, _ := checked.LastSlice(ids)
			if deviceGreedy {
				nextDeviceCache, err = r.forwardDeviceCachedGreedyStepLocked(
					ctx,
					lastIDs,
					deviceCache,
				)
			} else {
				_, nextDeviceCache, err = r.forwardDeviceCachedLocked(
					ctx,
					lastIDs,
					deviceCache,
				)
			}
			if err == nil {
				oldDeviceCache := deviceCache
				deviceCache = nextDeviceCache
				if !r.ownsDevicePromptCache(oldDeviceCache) {
					err = oldDeviceCache.Release(ctx)
				}
			} else if nextDeviceCache != nil {
				_ = nextDeviceCache.Release(context.Background())
			}
			if err != nil {
				return nil, "", err
			}
		}
		var event TokenEvent
		if deviceGreedy {
			event = TokenEvent{ID: deviceCache.Selected}
		} else {
			var sampleErr error
			event, sampleErr = sampleGenerationToken(deviceCache.Logits, ids, options)
			if sampleErr != nil {
				return nil, "", sampleErr
			}
		}
		event.Index = generatedIndex
		ids = append(ids, event.ID)
		stop, deliverErr := r.deliverGenerationToken(&event, options, &generatedText)
		if deliverErr != nil {
			return nil, "", deliverErr
		}
		if stop {
			break
		}
	}
	text, err := r.vocab.Decode(ids, false)
	if err != nil {
		return nil, "", err
	}
	return ids, text, nil
}

func reusablePromptPrefix(cached, requested []tokenizer.TokenID, minimum int) int {
	if !checked.Nonzero(len(cached)) || !checked.Nonzero(len(requested)) {
		var unavailable int
		return unavailable
	}
	common := min(len(cached), len(requested))
	for index := range common {
		if cached[index] != requested[index] {
			common = index
			break
		}
	}
	if common < minimum {
		var unavailable int
		return unavailable
	}
	return common
}

func (r *Runner) hasRecurrentCache() bool {
	if r == nil {
		return false
	}
	for _, layer := range r.weights.Layers {
		if layer.Recurrent {
			return true
		}
	}
	return false
}

func (r *Runner) promptTokenIDs(prompt string, options GenerateOptions) ([]tokenizer.TokenID, error) {
	if options.PromptTokenIDs == nil {
		ids, err := r.vocab.Encode(prompt, tokenizer.EncodeOptions{
			AddSpecial:   true,
			ParseSpecial: options.ParseSpecial,
		})
		if err != nil {
			return nil, err
		}
		if !checked.Nonzero(len(ids)) {
			return nil, errors.New("inference: prompt produced no tokens")
		}
		return ids, nil
	}
	if !checked.Nonzero(len(options.PromptTokenIDs)) {
		return nil, errors.New("inference: exact prompt token list is empty")
	}
	ids := slices.Clone(options.PromptTokenIDs)
	for index, id := range ids {
		if _, ok := r.vocab.Token(id); !ok {
			return nil, fmt.Errorf(
				"inference: prompt token %d has out-of-range ID %d",
				index,
				id,
			)
		}
	}
	return ids, nil
}

func validateStopSequences(stops []string) error {
	for _, stop := range stops {
		if stop == "" {
			return errors.New("inference: stop sequence is empty")
		}
	}
	return nil
}

func matchesStopSequence(text string, stops []string) bool {
	for _, stop := range stops {
		if strings.Contains(text, stop) {
			return true
		}
	}
	return false
}
