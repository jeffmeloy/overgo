package inference

import (
	"context"
	"errors"
	"fmt"
	"slices"

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
	if r == nil || r.vocab == nil {
		return nil, errRunnerNil
	}
	// A choice suite scores two or more candidates against a prompt; a
	// sequence-scoring suite scores one candidate against the empty
	// prompt, whose encoding is the tokenizer's BOS context -- the
	// full-sequence likelihood. Both ride the same path below.
	if ctx == nil || len(candidates) < 1 {
		return nil, errors.New("inference: incomplete continuation scoring request")
	}
	promptIDs, err := r.vocab.Encode(prompt, tokenizer.EncodeOptions{AddSpecial: true})
	if err != nil {
		return nil, err
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
			return nil, errors.New("inference: a vocabulary without BOS scores one sequence at a time")
		}
	}
	continuations := make([][]tokenizer.TokenID, len(candidates))
	for index, candidate := range candidates {
		if candidate == "" {
			return nil, errors.New("inference: continuation candidate is empty")
		}
		full, err := r.vocab.Encode(prompt+candidate, tokenizer.EncodeOptions{AddSpecial: true})
		if err != nil {
			return nil, err
		}
		if bosContext {
			if len(promptIDs) == 0 {
				if len(full) < 2 {
					return nil, errors.New("inference: sequence is too short to score")
				}
				promptIDs, continuations[index] = full[:1], full[1:]
				continue
			}
			if len(full) == 0 {
				return nil, errors.New("inference: continuation candidate produced no tokens")
			}
			continuations[index] = full
			continue
		}
		if len(full) <= len(promptIDs) || !slices.Equal(full[:len(promptIDs)], promptIDs) {
			return nil, fmt.Errorf("inference: candidate %d changes the compiled prompt token prefix", index)
		}
		continuations[index] = full[len(promptIDs):]
	}
	if len(promptIDs) == 0 {
		return nil, errors.New("inference: continuation prompt produced no tokens")
	}
	if err := r.lockOpen(); err != nil {
		return nil, err
	}
	defer r.mu.Unlock()
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
	result := make([]sequencescore.Score, len(continuations))
	for candidateIndex, continuation := range continuations {
		var score sequencescore.Accumulator
		value, err := negativeLogProbability(firstLogits.LastRowView().Data, int(continuation[0]))
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
