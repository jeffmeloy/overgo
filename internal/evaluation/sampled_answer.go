package evaluation

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

// What ended a sampled answer.
const (
	sampledEndOfSequence = "end of sequence"
	sampledStopSequence  = "stop sequence"
	sampledOutputLimit   = "output limit"
)

// sampledGroup is everything the draws of one logical group are sampled
// from, and so names them: the group, the code that samples, the weights by
// an identity that is theirs alone, the tokenizer and the template, the
// prompt's own token ids, the ids that end a generation and the texts that
// stop one, the sampler's settings, the output limit and the group's seed.
// A group under another arm's weights, another checkpoint or update of them,
// or another setting of the protocol is another group: its draws are other
// records, and no draw of one is read as a draw of the other. Arms that
// share their weights, as all do before the first update, share their draws.
type sampledGroup struct {
	Group       string              `json:"group"`
	Code        string              `json:"code"`
	Weights     artifact.ID         `json:"weights"`
	Tokenizer   artifact.ID         `json:"tokenizer"`
	Template    string              `json:"template"`
	Prompt      []tokenizer.TokenID `json:"prompt"`
	Ends        []tokenizer.TokenID `json:"ends"`
	Stops       []string            `json:"stops,omitempty"`
	Temperature float32             `json:"temperature"`
	TopK        int                 `json:"top_k,omitzero"`
	TopP        float32             `json:"top_p,omitzero"`
	MaxTokens   int                 `json:"max_tokens"`
	Seed        int64               `json:"seed"`
}

// sampledAnswer is one draw of a group as the model sampled it: the draw's
// index and the seed its sampler was given, the ids it sampled in order, the
// logarithm of the probability the sampler drew each with, one an id, what
// ended it, and its text. Training reads the ids; the text is for a scorer
// and a reader, and is never tokenized to get them back: a tokenizer that
// merges pieces gives other ids for the text of ids sampled apart.
type sampledAnswer struct {
	Draw             uint64              `json:"draw"`
	Seed             int64               `json:"seed"`
	Completion       []tokenizer.TokenID `json:"completion"`
	LogProbabilities []float64           `json:"log_probabilities"`
	End              string              `json:"end"`
	Text             string              `json:"text"`
}

// updatedWeights names the weights a training specification has reached
// after updates updates from a checkpoint, without reading them: training is
// reproducible, so the three name one set of weights. No updates is the
// checkpoint itself, which was hashed once when it was published.
func updatedWeights(checkpoint, training artifact.ID, updates uint64) (artifact.ID, error) {
	if updates == 0 {
		return checkpoint, nil
	}
	return artifact.JSONID(artifact.KindProfile, struct {
		Checkpoint artifact.ID `json:"checkpoint"`
		Training   artifact.ID `json:"training"`
		Updates    uint64      `json:"updates"`
	}{checkpoint, training, updates})
}

// loadSampled reads the draws the store holds of a group, in the records an
// evaluation's answers are kept in (answers): a draw is an answer to the
// group at its index. A group that leaves its weights, its tokenizer, its
// prompt, its ending ids or its output limit unnamed names no draws.
func loadSampled(ctx context.Context, repository artifact.Repository, group sampledGroup) (*answers[sampledAnswer], error) {
	observed, err := artifact.JSONID(artifact.KindProfile, group)
	if err != nil || !group.Weights.Valid() || !group.Tokenizer.Valid() || len(group.Prompt) == 0 || len(group.Ends) == 0 || group.MaxTokens <= 0 {
		return nil, errors.Join(errors.New("evaluation: a sampled group names its weights, tokenizer, prompt ids, ending ids and output limit before it is drawn from"), err)
	}
	return loadObserved[sampledAnswer](ctx, repository, observed)
}

// tokens is a draw as training reads it: the group's prompt ids and the
// draw's, as they were sampled, and which of them are the completion.
func (group sampledGroup) tokens(answer sampledAnswer) (ids []tokenizer.TokenID, completion []bool) {
	ids = slices.Concat(group.Prompt, answer.Completion)
	completion = make([]bool, len(ids))
	for index := len(group.Prompt); index < len(ids); index++ {
		completion[index] = true
	}
	return ids, completion
}

// sample is the group's draw at index, which is one the store holds or the
// next. A draw the store holds is returned as it was recorded and nothing is
// sampled. The next is sampled from the group's prompt ids under a sampler
// seeded from the group's identity and the index, so a draw belongs to its
// group and its weights; each id is kept as it is sampled with the logarithm
// of the probability the sampler reports for it; and the draw is refused if
// the ids the generator returns are not the prompt's followed by those, or a
// sampled id came with no probability. It ended at an ending id, at the
// output limit, or at a stop text, in that order of asking.
func (group sampledGroup) sample(ctx context.Context, generator Generator, held *answers[sampledAnswer], index int) (sampledAnswer, error) {
	return held.answer(ctx, index, func() (sampledAnswer, error) {
		digest := sha256.Sum256(fmt.Appendf(nil, "%s/%d", held.observed, index))
		answer := sampledAnswer{Draw: uint64(index), Seed: int64(binary.BigEndian.Uint64(digest[:]))}
		sampler, err := sampling.New(sampling.Config{Temperature: group.Temperature, TopK: group.TopK, TopP: group.TopP, Seed: answer.Seed})
		if err != nil {
			return sampledAnswer{}, err
		}
		ids, text, err := generator.Generate(ctx, "", inference.GenerateOptions{
			MaxNewTokens: group.MaxTokens, Sampler: sampler, PromptTokenIDs: group.Prompt, StopSequences: group.Stops, PostSamplingProbabilities: 1,
			OnToken: func(event inference.TokenEvent) error {
				answer.Completion = append(answer.Completion, event.ID)
				answer.LogProbabilities = append(answer.LogProbabilities, math.Log(event.SelectedProbability))
				return nil
			},
		})
		if err != nil {
			return sampledAnswer{}, err
		}
		drawn := len(answer.Completion)
		if !slices.Equal(ids, slices.Concat(group.Prompt, answer.Completion)) || slices.ContainsFunc(answer.LogProbabilities, func(logarithm float64) bool {
			return math.IsNaN(logarithm) || math.IsInf(logarithm, 0) || logarithm > 0
		}) {
			return sampledAnswer{}, fmt.Errorf("evaluation: draw %d of group %q returned %d ids for a prompt of %d and %d sampled, or a sampled id without its probability", index, group.Group, len(ids), len(group.Prompt), drawn)
		}
		switch answer.Text = text; {
		case drawn > 0 && slices.Contains(group.Ends, answer.Completion[drawn-1]):
			answer.End = sampledEndOfSequence
		case drawn >= group.MaxTokens:
			answer.End = sampledOutputLimit
		case len(group.Stops) > 0:
			answer.End = sampledStopSequence
		default:
			return sampledAnswer{}, fmt.Errorf("evaluation: draw %d of group %q ended after %d ids at no ending id, short of its limit of %d, with no stop text named", index, group.Group, drawn, group.MaxTokens)
		}
		return answer, nil
	})
}
