package scorecard

import (
	_ "embed"
	"fmt"
	"strings"

	"overgo/internal/recipe"
	"overgo/internal/strictjson"
)

// protocolSamplesJSON holds, for a model whose held-out reading was scored
// under one prompting protocol, a sample of the same suite's questions asked
// another way, each question answered both ways. A reading says what a model
// scores as it was asked; a maker's published figure may have been asked
// otherwise, and the two then differ by the asking and not by the model.
// Qwen3.5-4B read 0.47 on MMLU-Pro against a published 79.1, and nothing on
// the scorecard said whether that was a fault or the protocol.
//
//go:embed protocol_samples.json
var protocolSamplesJSON []byte

// ProtocolSample is some of a suite's questions asked of one model two ways:
// under the protocol its held-out reading was scored under, and the way the
// sample names. It holds each question's outcome and no count, so that what
// is said of it is counted from what was answered. A sample is not a held-out
// reading: it is too small to stand for the suite, and it says how far the
// reading turns on the way the question is asked.
type ProtocolSample struct {
	// Model and Task are the scorecard row the sample stands beside, and
	// Protocol the prompting protocol of that row's reading.
	Model    string      `json:"model"`
	Task     recipe.Task `json:"task"`
	Protocol string      `json:"protocol"`
	// Way names the other way in a few words, and Asked says what it does.
	Way   string `json:"way"`
	Asked string `json:"asked"`
	// Limit is the tokens an answer may run to the other way. A question and
	// its answer that together stay under it were not cut short.
	Limit int `json:"limit"`
	// Published is the share the model's maker reports for the suite asked
	// that way, zero where none is published, and PublishedBy where it stands.
	Published   float64 `json:"published,omitzero"`
	PublishedBy string  `json:"published_by,omitzero"`
	// ProducedBy says what ran the sample and what holds its outcomes.
	ProducedBy string        `json:"produced_by"`
	Cases      []SampledCase `json:"cases"`
}

// SampledCase is one question's outcome: the option that is right, the option
// the model chose under the reading's protocol and the one it chose the other
// way, each unanswered where it named none, and the tokens the question and
// its answer ran to the other way.
type SampledCase struct {
	Name    string `json:"name"`
	Answer  int    `json:"answer"`
	Reading int    `json:"reading"`
	Sample  int    `json:"sample"`
	Tokens  int    `json:"tokens"`
}

// unanswered is the choice of an answer that named no option.
const unanswered = -1

// SampleTally is what a sample's outcomes count to.
type SampleTally struct {
	Cases int
	// Cut is the questions whose answer ran to the limit the other way.
	Cut int
	// Reading and Sample are the questions answered right each way, with the
	// standard error of each share.
	Reading, Sample           int
	ReadingNoise, SampleNoise float64
	// ReadingOnly and SampleOnly are the questions only one way answered right.
	ReadingOnly, SampleOnly int
	// MeanTokens is what a question and its answer ran to the other way.
	MeanTokens int
}

// Share is the share of a tally's questions that count answered right.
func (tally SampleTally) Share(right int) float64 {
	if tally.Cases == 0 {
		return 0
	}
	return float64(right) / float64(tally.Cases)
}

// Tally counts the sample's outcomes.
func (sample ProtocolSample) Tally() SampleTally {
	tally := SampleTally{Cases: len(sample.Cases)}
	tokens := 0
	for _, question := range sample.Cases {
		reading, other := question.Reading == question.Answer, question.Sample == question.Answer
		switch {
		case reading && other:
			tally.Reading, tally.Sample = tally.Reading+1, tally.Sample+1
		case reading:
			tally.Reading, tally.ReadingOnly = tally.Reading+1, tally.ReadingOnly+1
		case other:
			tally.Sample, tally.SampleOnly = tally.Sample+1, tally.SampleOnly+1
		}
		tokens += question.Tokens
		if question.Tokens >= sample.Limit {
			tally.Cut++
		}
	}
	if tally.Cases != 0 {
		tally.MeanTokens = tokens / tally.Cases
	}
	cases := uint64(tally.Cases)
	tally.ReadingNoise, tally.SampleNoise = shareError(tally.Share(tally.Reading), cases), shareError(tally.Share(tally.Sample), cases)
	return tally
}

// ProtocolSamples reads the embedded samples. Each stands beside one model
// and task once, says how it was asked and what produced it, and holds
// questions named once, each with a right option and a choice that is an
// option or unanswered.
func ProtocolSamples() ([]ProtocolSample, error) {
	var document struct {
		Samples []ProtocolSample `json:"samples"`
	}
	if err := strictjson.DecodeBytes(protocolSamplesJSON, &document); err != nil {
		return nil, fmt.Errorf("scorecard: protocol samples: %w", err)
	}
	seen := map[string]bool{}
	for _, sample := range document.Samples {
		key := sample.Model + "\n" + string(sample.Task)
		said := func(text string) bool { return strings.TrimSpace(text) != "" }
		if !said(sample.Model) || !sample.Task.Valid() || seen[key] || !said(sample.Protocol) || !said(sample.Way) || !said(sample.Asked) || !said(sample.ProducedBy) || sample.Limit <= 0 ||
			len(sample.Cases) == 0 || sample.Published < 0 || sample.Published > 1 || (sample.Published != 0) != said(sample.PublishedBy) {
			return nil, fmt.Errorf("scorecard: the protocol sample for %q %s is repeated or incomplete", sample.Model, sample.Task)
		}
		seen[key] = true
		questions := map[string]bool{}
		for _, question := range sample.Cases {
			if !said(question.Name) || questions[question.Name] || question.Answer < 0 || question.Reading < unanswered || question.Sample < unanswered || question.Tokens <= 0 {
				return nil, fmt.Errorf("scorecard: the protocol sample for %q %s holds question %q twice or without its outcome", sample.Model, sample.Task, question.Name)
			}
			questions[question.Name] = true
		}
	}
	return document.Samples, nil
}
