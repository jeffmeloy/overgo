package thoughtbank

import (
	"context"
	"errors"
	"fmt"
	"math"

	"overgo/internal/sequencescore"
)

// scoringWindowTokens is how many tokens the model reads at once when it
// scores a sequence: a longer sequence scores as disjoint windows, each read
// from the empty bank, the way llama.cpp's perplexity tool windows its text,
// so every window is a sequence the model's forward holds whole.
const scoringWindowTokens = 512

// minScoredWindowTokens is the least a window holds to score anything: a
// token to read, and a token scored after it.
const minScoredWindowTokens = 2

// ScoreContinuations scores each candidate as one sequence read from the
// empty bank: the sum of its tokens' log-probabilities given the tokens
// before them in their window, and the count of tokens it scored. A
// window's first token has no context, so it is read and not scored. The
// model's recurrent bank has no scoring prefix, so a prompt is refused:
// the held-out text a sequence-scoring suite scores is its own context.
func (g *Generator) ScoreContinuations(ctx context.Context, prompt string, candidates []string) ([]sequencescore.Score, error) {
	if g == nil || g.weights == nil || g.tokenizer == nil {
		return nil, errors.New("thoughtbank: generator is unavailable")
	}
	if prompt != "" {
		return nil, errors.New("thoughtbank: a sequence scores from the empty bank, so it takes no prompt")
	}
	scores := make([]sequencescore.Score, 0, len(candidates))
	for _, candidate := range candidates {
		score, err := g.scoreSequence(ctx, candidate)
		if err != nil {
			return nil, err
		}
		scores = append(scores, score)
	}
	return scores, nil
}

func (g *Generator) scoreSequence(ctx context.Context, text string) (sequencescore.Score, error) {
	encoded, err := g.tokenizer.Encode(text)
	if err != nil {
		return sequencescore.Score{}, err
	}
	var score sequencescore.Score
	slots := g.config.MemSeedSlots
	for start := 0; start < len(encoded); start += scoringWindowTokens {
		if err := ctx.Err(); err != nil {
			return sequencescore.Score{}, err
		}
		window := encoded[start:min(start+scoringWindowTokens, len(encoded))]
		if len(window) < minScoredWindowTokens {
			continue
		}
		ids := make([]int32, len(window))
		for index, token := range window {
			ids[index] = int32(token)
		}
		output, err := FastWeightBankLMForward(ids, make([]float32, slots*g.config.MemDim), slots, g.weights)
		if err != nil {
			return sequencescore.Score{}, err
		}
		vocabulary := g.weights.VocabSize
		for position := 1; position < len(ids); position++ {
			logits := output.Logits[(position-1)*vocabulary : position*vocabulary]
			score.LogProbability += logProbability(logits, int(ids[position]))
		}
		score.Tokens += uint64(len(ids) - 1)
	}
	if score.Tokens == 0 {
		return sequencescore.Score{}, fmt.Errorf("thoughtbank: a sequence of %d tokens has no token to score", len(encoded))
	}
	return score, nil
}

// logProbability is the log of the softmax probability logits give token.
func logProbability(logits []float32, token int) float64 {
	peak := float64(logits[argmaxLogit(logits)])
	var total float64
	for _, logit := range logits {
		total += math.Exp(float64(logit) - peak)
	}
	return float64(logits[token]) - peak - math.Log(total)
}
