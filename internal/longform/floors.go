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
	// ShortPromptTokens and ShortOutputTokens are the SHORT fingerprint
	// shape: a short prompt with a greedy continuation whose token ids
	// and teacher-forced NLL a later run is compared against.
	ShortPromptTokens int `json:"short_prompt_tokens"`
	ShortOutputTokens int `json:"short_output_tokens"`
	// LadderStart is the first rung of the LONG ladder; rungs double
	// until the model's declared context length, the corpus, or the rung
	// budget ends the ladder.
	LadderStart int `json:"ladder_start"`
	// RungBudgetSeconds bounds one rung's prefill: the rung whose prefill
	// runs past it is the ladder's last.
	RungBudgetSeconds float64 `json:"rung_budget_seconds"`
	// CheckRungCeiling is the highest rung a regression check climbs to.
	CheckRungCeiling int `json:"check_rung_ceiling"`
	// IdenticalTokens is how many leading greedy tokens of every shape a
	// fresh run must reproduce from the record.
	IdenticalTokens int `json:"identical_tokens"`
	// NLLTolerance bounds the per-token NLL movement between a record
	// and a fresh run of the same shape.
	NLLTolerance float64 `json:"nll_tolerance"`
	// RateRegressionFraction bounds a fresh run's rates from below as
	// this fraction of the record's.
	RateRegressionFraction float64 `json:"rate_regression_fraction"`
}

// Declared floors (owner rule 2026-09-04, verification before every
// suite pass). The judged prompt length is the ladder's second rung,
// the long context the suites reach (BBH and MMLU-Pro cases run past
// 1500 tokens), and the output budget is long enough for a loop to
// show. Prefill on a 2048-token prompt runs
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
//
// The regression fingerprint (owner rule 2026-09-04, efficient model
// validation across time, performance and long context) adds the SHORT
// shape, 160 prompt tokens with a 64-token continuation, and the LONG
// ladder from 1024 tokens doubling, each rung bounded to a minute of
// prefill so the 27B's declared 256k context ends the ladder by budget
// rather than by hours; a check climbs to 8192. A fresh run must
// reproduce the record's first 48 greedy tokens per shape, the reference
// shape of the fp8 12B measurement of 2026-09-04, hold its NLL within
// 0.02 nat per token, and keep 80% of its rates: a code change that
// moves any of these moved the model.
const (
	declaredPromptTokens           = 2048
	declaredOutputTokens           = 256
	declaredRateFraction           = 0.5
	declaredMinimumOutputTokens    = 32
	declaredScoreTokens            = 128
	declaredShortContextTokens     = 200
	declaredContextGain            = -0.25
	declaredShortPromptTokens      = 160
	declaredShortOutputTokens      = 64
	declaredLadderStart            = 1024
	declaredRungBudgetSeconds      = 60
	declaredCheckRungCeiling       = 8192
	declaredIdenticalTokens        = 48
	declaredNLLTolerance           = 0.02
	declaredRateRegressionFraction = 0.8
)

// DeclaredFloors returns the floors every long-form run is judged by.
func DeclaredFloors() Floors {
	return Floors{
		PromptTokens: declaredPromptTokens, OutputTokens: declaredOutputTokens,
		RateFraction:           declaredRateFraction,
		MinimumOutputTokens:    declaredMinimumOutputTokens,
		ScoreTokens:            declaredScoreTokens,
		ShortContextTokens:     declaredShortContextTokens,
		ContextGain:            declaredContextGain,
		ShortPromptTokens:      declaredShortPromptTokens,
		ShortOutputTokens:      declaredShortOutputTokens,
		LadderStart:            declaredLadderStart,
		RungBudgetSeconds:      declaredRungBudgetSeconds,
		CheckRungCeiling:       declaredCheckRungCeiling,
		IdenticalTokens:        declaredIdenticalTokens,
		NLLTolerance:           declaredNLLTolerance,
		RateRegressionFraction: declaredRateRegressionFraction,
	}
}

// LadderRungs plans the ladder: rungs from LadderStart doubling while
// the rung and its scored continuation fit the corpus, the rung and
// its generation fit the model's declared context length, and the
// rung does not exceed the ceiling (zero for none). The rung budget is
// read while climbing, not here.
func LadderRungs(contextLength uint32, corpusTokens int, floors Floors, ceiling int) []int {
	var rungs []int
	room := max(floors.OutputTokens, floors.ScoreTokens)
	for rung := floors.LadderStart; rung > 0; rung *= 2 {
		if uint64(rung+room) > uint64(contextLength) || rung+floors.ScoreTokens > corpusTokens || (ceiling > 0 && rung > ceiling) {
			break
		}
		rungs = append(rungs, rung)
	}
	return rungs
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
