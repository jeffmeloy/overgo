package inference

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"llamacpp2go/internal/model"
	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
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
		return nil, "", errors.New("inference: runner is nil")
	}
	if r.forwardPolicy() == model.ForwardT5Encoder {
		return nil, "", errors.New("inference: T5 encoder models do not generate tokens")
	}
	if r.forwardPolicy() == model.ForwardT5 {
		generated, _, _, err := r.GenerateT5(ctx, prompt, options)
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
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, "", errors.New("inference: runner is closed")
	}
	restoreLoRA, err := r.applyGenerationLoRA(options)
	if err != nil {
		return nil, "", err
	}
	defer restoreLoRA()
	ids, err := r.promptTokenIDs(prompt, options)
	if err != nil {
		return nil, "", err
	}
	aloraID, aloraStart, err := r.activeALoRA(ids)
	if err != nil {
		return nil, "", err
	}
	if options.ProjectedInputs != nil && aloraID >= 0 {
		return nil, "", errors.New("inference: projected inputs cannot use invocation-activated LoRA")
	}
	var aloraScale float32
	if aloraID >= 0 {
		aloraScale = r.loraAdapters[aloraID].scale
		defer func() { r.loraAdapters[aloraID].scale = aloraScale }()
		options.CachePrompt = false
		if aloraStart < 0 {
			r.loraAdapters[aloraID].scale = 0
		}
	}
	keepTokens := effectiveKeepTokens(
		options.KeepTokens,
		len(ids),
		r.spec.ContextLength,
	)
	outputTable := r.outputTensor()
	var hidden reference.Value
	var cache *KVCache
	var deviceCache *deviceKVCache
	var selectedPromptCache *cachedPrompt
	var projectionSignature [32]byte
	if options.ProjectedInputs != nil {
		projectionSignature = projectedInputsSignature(*options.ProjectedInputs)
	}
	useDeviceCache := options.ProjectedInputs == nil && aloraID < 0 &&
		r.hasPreloadedWeights() && supportsPersistentDeviceCache(r.spec)
	defer func() {
		if deviceCache != nil &&
			!r.ownsDevicePromptCache(deviceCache) {
			_ = deviceCache.Release(context.Background())
		}
	}()
	if options.MaxNewTokens > 0 {
		promptStarted := time.Now()
		cached := 0
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
				if cached > 0 &&
					cached < len(selectedPromptCache.Tokens) &&
					r.promptCacheCapacity > 1 {
					// device suffix view would mutate selected entry
					// Preserve independent multi-entry caches and evaluate
					// divergent prompt from scratch
					selectedPromptCache = nil
					cached = 0
				}
				if cached > 0 &&
					cached < len(selectedPromptCache.Tokens) &&
					r.hasRecurrentCache() {
					cached = 0
				}
				if cached > 0 {
					retainedPrefix = selectedPromptCache.Device
					if cached < len(selectedPromptCache.Tokens) {
						base := cached
						// retained logits describe old final token
						// For exact shorter prompt, reevaluate its final
						// token from preceding cache entry
						if cached == len(ids) {
							base--
						}
						if base == 0 {
							cached = 0
							retainedPrefix = nil
						} else if trimErr := trimDeviceCacheSuffix(
							retainedPrefix,
							uint32(base),
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
				len(retainedPrefix.Logits) == 0 {
				if cached <= 1 {
					cached = 0
					retainedPrefix = nil
				} else if trimErr := trimDeviceCacheSuffix(
					retainedPrefix,
					uint32(cached-1),
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
			if cached == len(ids) && len(retainedPrefix.Logits) > 0 {
				deviceCache = retainedPrefix
			} else {
				deviceCache = retainedPrefix
				var nextDevice *deviceKVCache
				hidden, nextDevice, err = r.forwardDeviceCachedLocked(
					ctx,
					ids[cached:],
					retainedPrefix,
				)
				if err == nil {
					deviceCache = nextDevice
				}
			}
		} else if aloraID >= 0 && aloraStart >= 0 {
			r.loraAdapters[aloraID].scale = 0
			if aloraStart > 0 {
				_, cache, err = r.forwardCachedLocked(ctx, ids[:aloraStart], nil)
			}
			r.loraAdapters[aloraID].scale = aloraScale
			if err == nil {
				hidden, cache, err = r.forwardCachedLocked(ctx, ids[aloraStart:], cache)
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
				if cached > 0 &&
					cached < len(selectedPromptCache.Tokens) &&
					r.hasRecurrentCache() {
					cached = 0
				}
				if cached > 0 {
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
	var generatedText strings.Builder
	for generatedIndex := range options.MaxNewTokens {
		if generatedIndex > 0 {
			if useDeviceCache {
				if options.ContextShift {
					var shiftedDeviceCache *deviceKVCache
					shiftedDeviceCache, err = r.compactDeviceCacheForAppend(
						ctx,
						deviceCache,
						1,
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
				hidden, nextDeviceCache, err = r.forwardDeviceCachedLocked(
					ctx,
					[]tokenizer.TokenID{ids[len(ids)-1]},
					deviceCache,
				)
				if err == nil {
					oldDeviceCache := deviceCache
					deviceCache = nextDeviceCache
					if !r.ownsDevicePromptCache(oldDeviceCache) {
						err = oldDeviceCache.Release(ctx)
					}
				} else if nextDeviceCache != nil {
					_ = nextDeviceCache.Release(context.Background())
				}
			} else {
				cache, err = r.cacheForAppendKeeping(
					cache,
					1,
					options.ContextShift,
					keepTokens,
					options.DiscardTokens,
				)
				if err != nil {
					return nil, "", err
				}
				hidden, cache, err = r.forwardCachedLocked(
					ctx,
					[]tokenizer.TokenID{ids[len(ids)-1]},
					cache,
				)
			}
			if err != nil {
				return nil, "", err
			}
		}
		var logits []float32
		var logitsErr error
		if useDeviceCache {
			logits = deviceCache.Logits
		} else {
			width := int(hidden.Shape.Dims[0])
			last := hidden.Data[len(hidden.Data)-width:]
			logits, logitsErr = r.logits(ctx, outputTable, last)
		}
		if logitsErr != nil {
			return nil, "", logitsErr
		}
		event, sampleErr := sampleGenerationToken(logits, ids, options)
		if sampleErr != nil {
			return nil, "", sampleErr
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
	if len(cached) == 0 || len(requested) == 0 {
		return 0
	}
	common := min(len(cached), len(requested))
	for index := 0; index < common; index++ {
		if cached[index] != requested[index] {
			common = index
			break
		}
	}
	if common < minimum {
		return 0
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
		if len(ids) == 0 {
			return nil, errors.New("inference: prompt produced no tokens")
		}
		return ids, nil
	}
	if len(options.PromptTokenIDs) == 0 {
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
	if len(stops) > 256 {
		return errors.New("inference: stop sequence count exceeds 256")
	}
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
