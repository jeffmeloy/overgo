package longform

import (
	"context"
	"errors"
	"fmt"
	"time"

	"overgo/internal/inference"
	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

// Cut takes the long prompt and its true continuation out of one text:
// the text is tokenized with the model's leading special token, the
// prompt is its first promptTokens tokens, and the continuation the
// scoreTokens that follow, so every model reads exactly those lengths
// and the continuation is the text's own next tokens, not a
// re-encoding. A text shorter than both cannot stand in for them.
func Cut(runner *inference.Runner, text string, promptTokens, scoreTokens int) (prompt, continuation []tokenizer.TokenID, err error) {
	ids, err := runner.TokenizeText(text, true, false)
	if err != nil {
		return nil, nil, err
	}
	if len(ids) < promptTokens+scoreTokens {
		return nil, nil, fmt.Errorf("longform: the text tokenizes to %d tokens, fewer than the %d prompt and %d continuation tokens the run needs",
			len(ids), promptTokens, scoreTokens)
	}
	return ids[:promptTokens], ids[promptTokens : promptTokens+scoreTokens], nil
}

// ContextScore is the teacher-forced score of the true continuation
// under the long context and under the short context cut from its
// tail: a runtime that reads the long context wrongly raises the long
// negative log-likelihood far above the short one, while a model that
// merely loops under greedy decoding still scores the true text.
type ContextScore struct {
	ScoreTokens     int     `json:"score_tokens"`
	LongContextNLL  float64 `json:"long_context_nll"`
	ShortContextNLL float64 `json:"short_context_nll"`
	// ContextGain is the short-context NLL minus the long-context NLL
	// per token: positive when the long context helps.
	ContextGain float64 `json:"context_gain"`
}

// Score reads the true continuation's negative log-likelihood per token
// under the whole prompt and under the prompt's leading token plus its
// last shortContext-1 tokens.
func Score(ctx context.Context, runner *inference.Runner, prompt, continuation []tokenizer.TokenID, shortContext int) (ContextScore, error) {
	if runner == nil {
		return ContextScore{}, errors.New("longform: runner is nil")
	}
	if len(continuation) == 0 || shortContext < 2 || shortContext > len(prompt) {
		return ContextScore{}, errors.New("longform: the score needs a continuation and a short context inside the prompt")
	}
	long, err := runner.ScoreContinuationTokens(ctx, prompt, [][]tokenizer.TokenID{continuation})
	if err != nil {
		return ContextScore{}, err
	}
	shortPrompt := append([]tokenizer.TokenID{prompt[0]}, prompt[len(prompt)-shortContext+1:]...)
	short, err := runner.ScoreContinuationTokens(ctx, shortPrompt, [][]tokenizer.TokenID{continuation})
	if err != nil {
		return ContextScore{}, err
	}
	if len(long) != 1 || len(short) != 1 || long[0].Tokens == 0 || short[0].Tokens == 0 {
		return ContextScore{}, errors.New("longform: the continuation scores are incomplete")
	}
	score := ContextScore{
		ScoreTokens:     len(continuation),
		LongContextNLL:  -long[0].LogProbability / float64(long[0].Tokens),
		ShortContextNLL: -short[0].LogProbability / float64(short[0].Tokens),
	}
	score.ContextGain = score.ShortContextNLL - score.LongContextNLL
	return score, nil
}

// Generation is one greedy run: the tokens it produced, their text, and
// the measure over them.
type Generation struct {
	Tokens  []tokenizer.TokenID
	Text    string
	Measure Measure
}

// warmupTokens is the decode budget of the unmeasured pass that
// precedes the measured run: kernels load and graphs instantiate on the
// first pass, and the benchmark measures after a warmup run too.
const warmupTokens = 4

// Warm runs the prompt once with a short decode so the measured run
// reads steady-state rates.
func Warm(ctx context.Context, runner *inference.Runner, prompt []tokenizer.TokenID) error {
	_, err := Run(ctx, runner, prompt, warmupTokens)
	return err
}

// Run prefills the prompt and decodes greedily up to the output budget,
// timing the prompt evaluation and the decode separately the way the
// benchmark does, and reads the degeneration measures off the tokens.
// End-of-generation is not banned: a model that stops early is recorded
// as stopped, and the verdict reads the stop.
func Run(ctx context.Context, runner *inference.Runner, prompt []tokenizer.TokenID, outputTokens int) (Generation, error) {
	if runner == nil {
		return Generation{}, errors.New("longform: runner is nil")
	}
	if len(prompt) == 0 || outputTokens <= 0 {
		return Generation{}, errors.New("longform: the run needs a prompt and an output budget")
	}
	sampler, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		return Generation{}, err
	}
	var (
		evaluation inference.PromptEvaluation
		tokens     = make([]tokenizer.TokenID, 0, outputTokens)
		firstToken time.Time
	)
	started := time.Now()
	_, _, err = runner.Generate(ctx, "", inference.GenerateOptions{
		MaxNewTokens:   outputTokens,
		Sampler:        sampler,
		DeviceGreedy:   true,
		PromptTokenIDs: prompt,
		OnPromptEvaluated: func(value inference.PromptEvaluation) {
			evaluation = value
		},
		OnToken: func(event inference.TokenEvent) error {
			if firstToken.IsZero() {
				firstToken = time.Now()
			}
			tokens = append(tokens, event.ID)
			return nil
		},
	})
	finished := time.Now()
	if err != nil {
		return Generation{}, err
	}
	total := finished.Sub(started)
	timeToFirst := total
	if !firstToken.IsZero() {
		timeToFirst = firstToken.Sub(started)
	}
	decode := total - timeToFirst
	measure := Measure{
		PromptTokens:       len(prompt),
		OutputTokens:       len(tokens),
		StoppedEarly:       len(tokens) < outputTokens,
		PromptMilliseconds: float64(evaluation.Duration) / float64(time.Millisecond),
		DecodeMilliseconds: float64(decode) / float64(time.Millisecond),
	}
	if uncached := evaluation.Tokens - evaluation.Cached; uncached > 0 && evaluation.Duration > 0 {
		measure.PromptTokensPerSecond = float64(uncached) / evaluation.Duration.Seconds()
	}
	if len(tokens) > 1 && decode > 0 {
		measure.DecodeTokensPerSecond = float64(len(tokens)-1) / decode.Seconds()
	}
	ids := make([]int32, len(tokens))
	for index, token := range tokens {
		ids[index] = int32(token)
	}
	measure.DistinctFourGramRatio = DistinctNGramRatio(ids, 4)
	measure.LongestRepeatedSpan = LongestRepeatedSpan(ids)
	// The text is rendered from the generated tokens alone: the
	// generation call's own text return carries the prompt in front.
	text, err := runner.Detokenize(tokens, inference.RenderText)
	if err != nil {
		return Generation{}, err
	}
	return Generation{Tokens: tokens, Text: text, Measure: measure}, nil
}
