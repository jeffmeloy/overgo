package evaluation

import (
	"context"
	"encoding/json"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

// protocolSeparator joins the parts of a reading's protocol label: how the
// model was prompted, then each step its suite's texts were written by.
const protocolSeparator = ", "

// wordErrorMetrics are the names a word error rate is published under: a
// transcription report's, and a speech round trip's.
var wordErrorMetrics = []string{wordErrorRateMetric, roundTripErrorMetric}

// WordCounting says in words how a word error rate was counted, from the
// reading's metric and the protocol label it was scored under, so that a
// reader of two rates can see whether they are one measure. A rate counted
// with the recognizers' own normalizer charges a transcript nothing for its
// conventions; a rate counted over the words as each side wrote them charges
// "10" for "ten", and stands above what the same transcripts read the other
// way. It reports false for a metric that is no word error rate.
func WordCounting(metric, protocol string) (string, bool) {
	if !slices.Contains(wordErrorMetrics, metric) {
		return "", false
	}
	if slices.Contains(strings.Split(protocol, protocolSeparator), string(transcriptionWhisperEnglish)) {
		return "the reference and the transcript are both written by the English normalizer of Whisper's evaluation before their words are compared: numbers as digits, contractions and titles spelled out, and no hesitations, symbols or diacritics. It is how the Open ASR leaderboard and the recognizers' own model cards count, without that normalizer's table of British and American spellings, so a word spelled the other way is still an error", true
	}
	return "the reference and the transcript are lowercased, stripped of punctuation and compared word for word as each side wrote them, so a recognizer that writes \"10\" for \"ten\" or \"was not\" for \"wasn't\" is charged for it: the rate stands above what the same transcripts read counted the recognizers' way, and is not to be read against a rate counted that way", true
}

// namedCounting is the normalization a speech round-trip plan's suite names
// for its texts, which is part of how its reading was taken and so of its
// protocol label. A suite that names none, as every suite did before one
// could, adds nothing, and its label stays what it was.
func namedCounting(ctx context.Context, reader artifact.Reader, plan Plan) []string {
	profile, found, err := artifact.ReadContent(ctx, reader, plan.body.CaseProfile)
	if err != nil || !found {
		return nil
	}
	var suite SpeechRoundTripSuite
	if json.Unmarshal(profile.Data, &suite) != nil || suite.Kind != speechRoundTripKind {
		return nil
	}
	steps := make([]string, 0, len(suite.Normalization))
	for _, step := range suite.Normalization {
		steps = append(steps, string(step))
	}
	return steps
}
