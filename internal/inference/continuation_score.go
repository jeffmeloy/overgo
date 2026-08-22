package inference

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/sequencescore"
	"overgo/internal/tokenizer"
)

func (r *Runner) ScoreContinuations(
	ctx context.Context,
	prompt string,
	candidates []string,
) ([]sequencescore.Score, error) {
	if r == nil || r.vocab == nil {
		return nil, errRunnerNil
	}
	if ctx == nil || prompt == "" || !checked.Multiple(len(candidates)) {
		return nil, errors.New("inference: incomplete continuation scoring request")
	}
	promptIDs, err := r.vocab.Encode(prompt, tokenizer.EncodeOptions{AddSpecial: true})
	if err != nil {
		return nil, err
	}
	if !checked.Nonzero(len(promptIDs)) {
		return nil, errors.New("inference: continuation prompt produced no tokens")
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
		if len(full) <= len(promptIDs) || !slices.Equal(full[:len(promptIDs)], promptIDs) {
			return nil, fmt.Errorf("inference: candidate %d changes the compiled prompt token prefix", index)
		}
		continuations[index] = full[len(promptIDs):]
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
	if !checked.Nonzero(len(last.Data)) {
		return nil, errors.New("inference: continuation hidden state is incompatible")
	}
	firstLogits, err := r.logitsBatch(ctx, r.outputTensor(), last)
	if err != nil {
		return nil, err
	}
	vocabulary := r.vocab.Len()
	result := make([]sequencescore.Score, len(continuations))
	for candidateIndex, continuation := range continuations {
		var score sequencescore.Accumulator
		first, _ := checked.First(continuation)
		value, err := negativeLogProbability(firstLogits, int(first))
		if err != nil {
			return nil, err
		}
		if err := score.Observe(-value); err != nil {
			return nil, err
		}
		if checked.Multiple(len(continuation)) {
			prefix, _ := checked.Init(continuation)
			branchHidden, _, err := r.forwardCachedLocked(ctx, prefix, cloneCache(cache))
			if err != nil {
				return nil, err
			}
			logits, err := r.logitsBatch(ctx, r.outputTensor(), branchHidden)
			if err != nil {
				return nil, err
			}
			targets, _ := checked.Tail(continuation)
			for index, target := range targets {
				rowStart := index * vocabulary
				value, err := negativeLogProbability(logits[rowStart:rowStart+vocabulary], int(target))
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
