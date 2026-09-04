package longform

import (
	"fmt"
	"strings"
)

// Floors is the declared bound a long-form run must clear before a
// suite pass may spend hours on the model. The rate floors are relative
// to the model's own short-prompt benchmark record, so a model is judged
// against itself rather than against a number that belongs to another
// size class; the output floor is the teacher-forced read of the text's
// own continuation, the measure that separates a runtime reading the
// long context wrongly from a model looping under greedy decoding.
type Floors struct {
	// PromptTokens is the long prompt's length in tokens.
	PromptTokens int `json:"prompt_tokens"`
	// OutputTokens is the greedy generation's token budget.
	OutputTokens int `json:"output_tokens"`
	// RateFraction bounds the long-form prompt and decode rates from
	// below as this fraction of the short-prompt benchmark's rates.
	RateFraction float64 `json:"rate_fraction"`
	// MinimumOutputTokens is the least output the decode rate is read
	// from: a model that stops before it produced too little to time,
	// and the run records the stop.
	MinimumOutputTokens int `json:"minimum_output_tokens"`
	// ScoreTokens is the length of the true continuation the run scores
	// under teacher forcing after the prompt.
	ScoreTokens int `json:"score_tokens"`
	// ShortContextTokens is the tail of the prompt the continuation is
	// scored against for comparison with the whole prompt.
	ShortContextTokens int `json:"short_context_tokens"`
	// ContextGain bounds from below the short-context NLL minus the
	// long-context NLL per token of the true continuation.
	ContextGain float64 `json:"context_gain"`
}

// Declared floors (owner rule 2026-09-04, verification before every
// suite pass). The prompt length is the long context the suites reach
// (BBH and MMLU-Pro cases run past 1500 tokens) and the output budget
// is long enough for a loop to show. Prefill on a 2000-token prompt runs
// the same GEMMs at better occupancy than the 181-token benchmark
// prompt and decode at 2000 tokens of context adds one attention read
// per layer, so half the short-prompt rates is a bound no healthy
// runtime crosses; 32 output tokens is the least a decode rate is read
// from. The teacher-forced read scores the text's next 128 tokens under
// the whole prompt and under its last 200 tokens: more context of the
// same document lowers a healthy model's NLL, so a long context that
// raises it by more than a quarter nat per token is a runtime reading
// the context wrongly. The distinct 4-gram ratio and the longest
// repeated span are recorded, not bounded: the first runs on the
// servable models showed the Qwen and MiniCPM checkpoints repeating a
// README bullet verbatim under greedy decoding while their context gain
// stayed positive, the model's own decoding behaviour on a list, not a
// runtime defect, and the suite passes score log-probabilities that a
// greedy loop does not touch.
const (
	declaredPromptTokens        = 2000
	declaredOutputTokens        = 256
	declaredRateFraction        = 0.5
	declaredMinimumOutputTokens = 32
	declaredScoreTokens         = 128
	declaredShortContextTokens  = 200
	declaredContextGain         = -0.25
)

// DeclaredFloors returns the floors every long-form run is judged by.
func DeclaredFloors() Floors {
	return Floors{
		PromptTokens: declaredPromptTokens, OutputTokens: declaredOutputTokens,
		RateFraction:        declaredRateFraction,
		MinimumOutputTokens: declaredMinimumOutputTokens,
		ScoreTokens:         declaredScoreTokens,
		ShortContextTokens:  declaredShortContextTokens,
		ContextGain:         declaredContextGain,
	}
}

// ShortRates are the model's short-prompt benchmark rates the long-form
// rates are bounded against.
type ShortRates struct {
	PromptTokensPerSecond float64 `json:"prompt_tokens_per_second"`
	DecodeTokensPerSecond float64 `json:"decode_tokens_per_second"`
}

// Verdict is the judged outcome of one run: every floor it failed is a
// reason, and a run with no reasons passed.
type Verdict struct {
	Passed  bool     `json:"passed"`
	Reasons []string `json:"reasons,omitempty"`
}

// Judge applies the floors to a measured run. A model without a
// short-prompt benchmark record has no rate bound and fails: the rate
// floors are the point of the run, and the benchmark is a minute's work.
// The decode rate is read only from an output long enough to time.
func Judge(measure Measure, short ShortRates, floors Floors) Verdict {
	var reasons []string
	if short.PromptTokensPerSecond <= 0 || short.DecodeTokensPerSecond <= 0 {
		reasons = append(reasons, "no short-prompt benchmark record bounds the rates; publish one with cmd/benchmark first")
	} else {
		if promptFloor := floors.RateFraction * short.PromptTokensPerSecond; measure.PromptTokensPerSecond < promptFloor {
			reasons = append(reasons, fmt.Sprintf("prompt %.1f tok/s is below the %.1f floor (%.0f%% of the %.1f short-prompt rate)",
				measure.PromptTokensPerSecond, promptFloor, 100*floors.RateFraction, short.PromptTokensPerSecond))
		}
		if decodeFloor := floors.RateFraction * short.DecodeTokensPerSecond; measure.OutputTokens >= floors.MinimumOutputTokens && measure.DecodeTokensPerSecond < decodeFloor {
			reasons = append(reasons, fmt.Sprintf("decode %.1f tok/s is below the %.1f floor (%.0f%% of the %.1f short-prompt rate)",
				measure.DecodeTokensPerSecond, decodeFloor, 100*floors.RateFraction, short.DecodeTokensPerSecond))
		}
	}
	if measure.OutputTokens < floors.MinimumOutputTokens {
		reasons = append(reasons, fmt.Sprintf("%d output token(s) is below the %d a decode rate is read from",
			measure.OutputTokens, floors.MinimumOutputTokens))
	}
	switch {
	case measure.Score.ScoreTokens == 0:
		reasons = append(reasons, "the true continuation was not scored")
	case measure.Score.ContextGain < floors.ContextGain:
		reasons = append(reasons, fmt.Sprintf("the long context raises the true continuation's NLL to %.3f from %.3f nat/token under the short context (gain %.3f, floor %.2f)",
			measure.Score.LongContextNLL, measure.Score.ShortContextNLL, measure.Score.ContextGain, floors.ContextGain))
	}
	return Verdict{Passed: len(reasons) == 0, Reasons: reasons}
}

// String renders the verdict as one line.
func (v Verdict) String() string {
	if v.Passed {
		return "PASS"
	}
	return "FAIL: " + strings.Join(v.Reasons, "; ")
}
