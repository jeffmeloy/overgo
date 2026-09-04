package inference

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/sequencescore"
	"overgo/internal/tokenizer"
)

// ScoringPrefix reports the model-declared sequence-scoring prefix:
// the DNA begin tag for hybrid DNA tokenizers, empty otherwise. The
// evaluation layer shapes pure-sequence input through it so scores
// measure the trained format.
func (r *Runner) ScoringPrefix() string {
	if r == nil || r.vocab == nil {
		return ""
	}
	prefix, _ := r.vocab.ScoringPrefix()
	return prefix
}

func (r *Runner) ScoreContinuations(
	ctx context.Context,
	prompt string,
	candidates []string,
) ([]sequencescore.Score, error) {
	return r.ScoreContinuationsParsed(ctx, prompt, candidates, false)
}

// ScoreContinuationsParsed scores candidates against a prompt whose
// special-token text is parsed into the declared tokens when
// parseSpecial is set -- a template-shaped prompt carries turn markers
// the model must see as single tokens, exactly as serving encodes them.
// Raw suite prompts keep parseSpecial false so benchmark text can never
// smuggle a control token.
func (r *Runner) ScoreContinuationsParsed(
	ctx context.Context,
	prompt string,
	candidates []string,
	parseSpecial bool,
) ([]sequencescore.Score, error) {
	if r == nil || r.vocab == nil {
		return nil, errRunnerNil
	}
	if ctx == nil {
		return nil, errors.New("inference: incomplete continuation scoring request")
	}
	promptIDs, continuations, err := r.continuationTokens(prompt, candidates, parseSpecial)
	if err != nil {
		return nil, err
	}
	if err := r.lockOpen(); err != nil {
		return nil, err
	}
	defer r.mu.Unlock()
	if r.hasPreloadedWeights() && r.forwardProgram().PersistentDeviceCache() {
		return r.scoreContinuationsDeviceLocked(ctx, promptIDs, continuations)
	}
	return r.scoreContinuationsHostLocked(ctx, promptIDs, continuations)
}

// continuationTokens encodes a scoring request into the scored context
// and one continuation per candidate.
func (r *Runner) continuationTokens(
	prompt string,
	candidates []string,
	parseSpecial bool,
) ([]tokenizer.TokenID, [][]tokenizer.TokenID, error) {
	// A choice suite scores two or more candidates against a prompt; a
	// sequence-scoring suite scores one candidate against the empty
	// prompt, whose encoding is the tokenizer's BOS context -- the
	// full-sequence likelihood. Both ride the same path below.
	if len(candidates) < 1 {
		return nil, nil, errors.New("inference: incomplete continuation scoring request")
	}
	encoding := tokenizer.EncodeOptions{AddSpecial: true, ParseSpecial: parseSpecial}
	promptIDs, err := r.vocab.Encode(prompt, encoding)
	if err != nil {
		return nil, nil, err
	}
	bosContext := false
	if len(promptIDs) == 0 && prompt == "" {
		// A tokenizer without auto-BOS encodes the empty prompt to
		// nothing. The BOS token stands in as the empty context; a
		// vocabulary without BOS at all conditions the single candidate
		// on its own first token instead -- the conditional likelihood a
		// sequence-scoring suite measures.
		bosContext = true
		if r.vocab.BOS >= 0 {
			promptIDs = []tokenizer.TokenID{r.vocab.BOS}
		} else if len(candidates) != 1 {
			return nil, nil, errors.New("inference: a vocabulary without BOS scores one sequence at a time")
		}
	}
	continuations := make([][]tokenizer.TokenID, len(candidates))
	shared := len(promptIDs)
	for index, candidate := range candidates {
		if candidate == "" {
			return nil, nil, errors.New("inference: continuation candidate is empty")
		}
		full, err := r.vocab.Encode(prompt+candidate, encoding)
		if err != nil {
			return nil, nil, err
		}
		if bosContext {
			if len(promptIDs) == 0 {
				if len(full) < 2 {
					return nil, nil, errors.New("inference: sequence is too short to score")
				}
				promptIDs, continuations[index] = full[:1], full[1:]
				continue
			}
			if len(full) == 0 {
				return nil, nil, errors.New("inference: continuation candidate produced no tokens")
			}
			continuations[index] = full
			continue
		}
		if len(full) <= len(promptIDs) {
			return nil, nil, fmt.Errorf("inference: candidate %d adds no token to the prompt", index)
		}
		// A candidate may merge its first characters into the prompt's
		// last token (":" + " (" -> ": ("), so its encoding need not
		// keep the prompt's tokens as a prefix. The scored context is
		// the longest token prefix shared by the prompt and every
		// candidate; the tokens past it, prompt tail included, are the
		// continuation, the same conditional likelihood lm-eval and
		// llama.cpp score across such a boundary.
		shared = min(shared, commonTokenPrefix(promptIDs, full))
		continuations[index] = full
	}
	if !bosContext {
		if shared == 0 {
			return nil, nil, errors.New("inference: the candidates share no prompt context")
		}
		promptIDs = promptIDs[:shared]
		for index := range continuations {
			continuations[index] = continuations[index][shared:]
		}
	}
	if len(promptIDs) == 0 {
		return nil, nil, errors.New("inference: continuation prompt produced no tokens")
	}
	return promptIDs, continuations, nil
}

// scoreContinuationsHostLocked scores through the host-cache forward: the
// prompt's hidden rows come back to the host, each longer candidate
// re-enters through a cloned host cache, and the logits of every scored
// position are read from the returned hidden rows. It serves runners
// without a resident device cache and is the reference the device path
// is measured against.
func (r *Runner) scoreContinuationsHostLocked(
	ctx context.Context,
	promptIDs []tokenizer.TokenID,
	continuations [][]tokenizer.TokenID,
) ([]sequencescore.Score, error) {
	hidden, cache, err := r.forwardCachedLocked(ctx, promptIDs, nil)
	if err != nil {
		return nil, err
	}
	last := hidden.LastRowView()
	if len(last.Data) == 0 {
		return nil, errors.New("inference: continuation hidden state is incompatible")
	}
	firstLogits, err := r.logitsBatch(ctx, r.outputTensor(), last)
	if err != nil {
		return nil, err
	}
	firstRow := firstLogits.LastRowView().Data
	firstNormalizer, err := logNormalizer(firstRow)
	if err != nil {
		return nil, err
	}
	result := make([]sequencescore.Score, len(continuations))
	for candidateIndex, continuation := range continuations {
		var score sequencescore.Accumulator
		value, err := negativeLogProbabilityNormalized(firstRow, firstNormalizer, int(continuation[0]))
		if err != nil {
			return nil, err
		}
		if err := score.Observe(-value); err != nil {
			return nil, err
		}
		if len(continuation) > 1 {
			branchHidden, _, err := r.forwardCachedLocked(ctx, continuation[:len(continuation)-1], cloneCache(cache))
			if err != nil {
				return nil, err
			}
			logits, err := r.logitsBatch(ctx, r.outputTensor(), branchHidden)
			if err != nil {
				return nil, err
			}
			for index, target := range continuation[1:] {
				value, err := negativeLogProbability(logits.RowView(uint64(index)).Data, int(target))
				if err != nil {
					return nil, err
				}
				if err := score.Observe(-value); err != nil {
					return nil, err
				}
			}
		}
		result[candidateIndex], err = score.Result()
		if err != nil {
			return nil, err
		}
	}
	return result, nil
}

// scoreContinuationsDeviceLocked scores through the resident device cache,
// the path the generator's prompt evaluation takes: one device forward
// per prompt whose last-position logits score every candidate's first
// token, and one device decode step per further candidate token, each
// step's logits scoring the token after it. Nothing but the final-row
// logits crosses to the host. The host-cache path (the reference for
// this one) brought every layer's state back through the host per case:
// the E4B BBH pass spent half its wall time in stream synchronization
// with the GPU idle, 2.3 s per case against 19 ms of kernel time.
func (r *Runner) scoreContinuationsDeviceLocked(
	ctx context.Context,
	promptIDs []tokenizer.TokenID,
	continuations [][]tokenizer.TokenID,
) ([]sequencescore.Score, error) {
	_, prompt, err := r.forwardDeviceCachedLocked(ctx, promptIDs, nil)
	if err != nil {
		return nil, err
	}
	defer func() { _ = prompt.Release(context.Background()) }()
	if len(prompt.Logits) == 0 {
		return nil, errors.New("inference: device continuation prompt returned no logits")
	}
	firstNormalizer, err := logNormalizer(prompt.Logits)
	if err != nil {
		return nil, err
	}
	result := make([]sequencescore.Score, len(continuations))
	for candidateIndex, continuation := range continuations {
		var score sequencescore.Accumulator
		value, err := negativeLogProbabilityNormalized(prompt.Logits, firstNormalizer, int(continuation[0]))
		if err != nil {
			return nil, err
		}
		if err := score.Observe(-value); err != nil {
			return nil, err
		}
		if len(continuation) > 1 {
			// Each candidate re-enters from the prompt cache; a step past a
			// shared past writes its own position before reading it, so the
			// candidates leave nothing for one another to read.
			cache := prompt
			for index, token := range continuation[:len(continuation)-1] {
				_, next, stepErr := r.forwardDeviceCachedLocked(ctx, []tokenizer.TokenID{token}, cache)
				if cache != prompt {
					_ = cache.Release(context.Background())
				}
				if stepErr != nil {
					return nil, stepErr
				}
				cache = next
				value, err := negativeLogProbability(cache.Logits, int(continuation[index+1]))
				if err != nil {
					_ = cache.Release(context.Background())
					return nil, err
				}
				if err := score.Observe(-value); err != nil {
					_ = cache.Release(context.Background())
					return nil, err
				}
			}
			if cache != prompt {
				_ = cache.Release(context.Background())
			}
		}
		result[candidateIndex], err = score.Result()
		if err != nil {
			return nil, err
		}
	}
	return result, nil
}
