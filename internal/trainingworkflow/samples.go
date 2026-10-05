package trainingworkflow

import (
	"context"
	"errors"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/densecausal"
)

const (
	trainingSamplesMediaType = "application/vnd.overgo.training-samples+json"
	trainingSamplesSchema    = "overgo/training-samples/v1"
	// sampleRecords is how many held-out conversations a run samples at each step.
	sampleRecords = 2
	// sampleTokens bounds how many tokens a sample generates, so sampling a
	// long reply costs a bounded number of forward passes on the host.
	sampleTokens = 48
)

// sampleAt is what the model generated after a prompt once a run had taken Step
// updates; step zero is the model before it trained.
type sampleAt struct {
	Step int    `json:"step"`
	Text string `json:"text"`
}

// sampleStrip is one held-out conversation's prompt, as far as its first reply,
// the reply the conversation holds, and what the model generated after the
// prompt at each step the run sampled, in step order.
type sampleStrip struct {
	Record   string     `json:"record"`
	Prompt   string     `json:"prompt"`
	Expected string     `json:"expected"`
	Samples  []sampleAt `json:"samples"`
}

// trainingSamples is a run's samples: the same prompts after the same greedy
// decoding, so the strips differ only by what training did to the model.
type trainingSamples struct {
	ID      artifact.ID   `json:"-"`
	Version uint16        `json:"version"`
	Trained artifact.ID   `json:"trained"`
	Strips  []sampleStrip `json:"strips"`
}

var trainingSamplesCodec = artifact.JSONDocumentCodec(
	"training samples", artifact.KindEvidence, trainingSamplesMediaType, trainingSamplesSchema,
	canonicalizeTrainingSamples,
	func(value trainingSamples) artifact.ID { return value.ID },
	func(value *trainingSamples, id artifact.ID) { value.ID = id },
	func(value trainingSamples) trainingSamples {
		value.Strips = slices.Clone(value.Strips)
		for index := range value.Strips {
			value.Strips[index].Samples = slices.Clone(value.Strips[index].Samples)
		}
		return value
	},
)

func canonicalizeTrainingSamples(value *trainingSamples) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Trained.Kind() != artifact.KindCheckpoint || len(value.Strips) == 0 {
		return errors.New("training workflow: incomplete training samples")
	}
	for _, strip := range value.Strips {
		if len(strip.Samples) == 0 {
			return errors.New("training workflow: a sample strip holds no sample")
		}
		for index := 1; index < len(strip.Samples); index++ {
			if strip.Samples[index].Step <= strip.Samples[index-1].Step {
				return errors.New("training workflow: samples are not in step order")
			}
		}
	}
	return nil
}

// sampler generates a run's samples from the model as it trains: the prompts
// are fixed when it is made, from the held-out conversations, and decoding is
// greedy, so each strip's samples answer the same question at each step.
type sampler struct {
	model   *densecausal.Model
	piece   func(int) string
	prompts []generationPrompt
	strips  []sampleStrip
}

// newSampler fixes the prompts of the first held-out records that learn a reply:
// the tokens before it are the prompt, its first reply is what the conversation
// holds, and a prompt too long for the window to generate after is passed over.
// A run whose held-out records offer none cannot be sampled.
func newSampler(model *densecausal.Model, piece func(int) string, heldout []string, records map[string]textRecord, sequence int) (*sampler, error) {
	result := &sampler{model: model, piece: piece}
	for _, id := range heldout {
		record := records[id]
		first := slices.Index(record.trained, true)
		if first < 1 {
			continue
		}
		end := first
		for end < len(record.tokens) && record.trained[end] {
			end++
		}
		length := min(end-first, sampleTokens)
		if first+length > sequence {
			continue
		}
		result.prompts = append(result.prompts, generationPrompt{tokens: record.tokens[:first], length: length})
		result.strips = append(result.strips, sampleStrip{
			Record: id, Prompt: spell(piece, record.tokens[:first]), Expected: spell(piece, record.tokens[first:first+length]),
		})
		if len(result.strips) == sampleRecords {
			break
		}
	}
	if len(result.strips) == 0 {
		return nil, errors.New("training workflow: no held-out conversation has a reply to sample after its prompt")
	}
	return result, nil
}

func spell(piece func(int) string, tokens []int) string {
	var text strings.Builder
	for _, token := range tokens {
		text.WriteString(piece(token))
	}
	return text.String()
}

// sample generates after each prompt with the model as it now stands and files
// the text under step.
func (sampler *sampler) sample(step int) error {
	for index, prompt := range sampler.prompts {
		answer, err := generateAnswer(sampler.model, sampler.piece, prompt)
		if err != nil {
			return err
		}
		sampler.strips[index].Samples = append(sampler.strips[index].Samples, sampleAt{Step: step, Text: answer.Text})
	}
	return nil
}

// publish stores the samples as a document about the trained model.
func (sampler *sampler) publish(ctx context.Context, store artifact.Repository, trained artifact.ID) (artifact.ID, error) {
	document, err := trainingSamplesCodec.NewInitial(trainingSamples{Trained: trained, Strips: sampler.strips})
	if err != nil {
		return artifact.ID{}, err
	}
	batch, err := trainingSamplesCodec.Batch("training/samples/"+document.ID.String(), document, artifact.DependencyLineage(document.ID, trained), nil)
	if err != nil {
		return artifact.ID{}, err
	}
	if _, err := artifact.Publish(ctx, store, batch); err != nil {
		return artifact.ID{}, err
	}
	return document.ID, nil
}
