package longform

import (
	"context"
	"errors"
	"fmt"
	"time"

	"overgo/internal/cuda/driver"
	"overgo/internal/inference"
	"overgo/internal/processmeasure"
	"overgo/internal/sampling"
	"overgo/internal/tensor/dtype"
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

// NLL reads the continuation's teacher-forced negative log-likelihood
// per token under the prompt.
func NLL(ctx context.Context, runner *inference.Runner, prompt, continuation []tokenizer.TokenID) (float64, error) {
	if runner == nil {
		return 0, errors.New("longform: runner is nil")
	}
	if len(prompt) == 0 || len(continuation) == 0 {
		return 0, errors.New("longform: the score needs a prompt and a continuation")
	}
	scores, err := runner.ScoreContinuationTokens(ctx, prompt, [][]tokenizer.TokenID{continuation})
	if err != nil {
		return 0, err
	}
	if len(scores) != 1 || scores[0].Tokens == 0 {
		return 0, errors.New("longform: the continuation score is incomplete")
	}
	return -scores[0].LogProbability / float64(scores[0].Tokens), nil
}

// Score reads the true continuation's negative log-likelihood per token
// under the whole prompt and under the prompt's leading token plus its
// last shortContext-1 tokens.
func Score(ctx context.Context, runner *inference.Runner, prompt, continuation []tokenizer.TokenID, shortContext int) (ContextScore, error) {
	if len(continuation) == 0 || shortContext < 2 || shortContext > len(prompt) {
		return ContextScore{}, errors.New("longform: the score needs a continuation and a short context inside the prompt")
	}
	long, err := NLL(ctx, runner, prompt, continuation)
	if err != nil {
		return ContextScore{}, err
	}
	shortPrompt := append([]tokenizer.TokenID{prompt[0]}, prompt[len(prompt)-shortContext+1:]...)
	short, err := NLL(ctx, runner, shortPrompt, continuation)
	if err != nil {
		return ContextScore{}, err
	}
	return ContextScore{
		ScoreTokens: len(continuation), LongContextNLL: long, ShortContextNLL: short, ContextGain: short - long,
	}, nil
}

// ShortShape is the SHORT fingerprint: the greedy continuation of a
// short prompt, by token id, and the teacher-forced NLL of the text's
// own next tokens under that prompt.
type ShortShape struct {
	PromptTokens int `json:"prompt_tokens"`
	// Zero preserves legacy records with unbound shape initialization.
	WarmupOutputTokens int     `json:"warmup_output_tokens,omitzero"`
	OutputIDs          []int32 `json:"output_ids"`
	NLL                float64 `json:"nll"`
	Measure            Measure `json:"measure"`
}

// Rung is one rung of the LONG ladder: the measure at that prompt
// length and the greedy tokens it produced.
type Rung struct {
	Measure   Measure `json:"measure"`
	OutputIDs []int32 `json:"output_ids"`
}

// Short runs the SHORT shape on the corpus tokens.
func Short(ctx context.Context, runner *inference.Runner, corpus []tokenizer.TokenID, floors Floors, protocol Protocol) (ShortShape, error) {
	if len(corpus) < floors.ShortPromptTokens+floors.ShortOutputTokens {
		return ShortShape{}, fmt.Errorf("longform: the corpus holds %d tokens, fewer than the short shape's %d", len(corpus), floors.ShortPromptTokens+floors.ShortOutputTokens)
	}
	prompt := corpus[:floors.ShortPromptTokens]
	continuation := corpus[floors.ShortPromptTokens : floors.ShortPromptTokens+floors.ShortOutputTokens]
	if err := Warm(ctx, runner, prompt, protocol); err != nil {
		return ShortShape{}, fmt.Errorf("short initialization: %w", err)
	}
	generation, err := Run(ctx, runner, prompt, floors.ShortOutputTokens, protocol)
	if err != nil {
		return ShortShape{}, err
	}
	nll, err := NLL(ctx, runner, prompt, continuation)
	if err != nil {
		return ShortShape{}, err
	}
	generation.Measure.Memory, err = runner.DeviceMemoryStats(ctx)
	if err != nil {
		return ShortShape{}, fmt.Errorf("short allocation accounting: %w", err)
	}
	return ShortShape{
		PromptTokens: floors.ShortPromptTokens, WarmupOutputTokens: WarmupOutputTokens,
		OutputIDs: tokenIDs(generation.Tokens), NLL: nll, Measure: generation.Measure,
	}, nil
}

// Ladder climbs the rungs in order, each a generation and a context
// score on the corpus prefix of the rung's length; it stops after the
// rung whose prefill ran past the rung budget, and reports why it
// stopped when it did not climb every planned rung.
func Ladder(ctx context.Context, runner *inference.Runner, corpus []tokenizer.TokenID, rungs []int, floors Floors, protocol Protocol, progress func(Rung) string) ([]Rung, string, error) {
	var climbed []Rung
	for index, length := range rungs {
		prompt := corpus[:length]
		continuation := corpus[length : length+floors.ScoreTokens]
		generation, err := Run(ctx, runner, prompt, floors.OutputTokens, protocol)
		if err != nil {
			if index == 0 {
				return nil, "", err
			}
			return climbed, fmt.Sprintf("rung %d: %v", length, err), nil
		}
		generation.Measure.Score, err = Score(ctx, runner, prompt, continuation, floors.ShortContextTokens)
		if err != nil {
			return climbed, fmt.Sprintf("rung %d score: %v", length, err), nil
		}
		generation.Measure.Memory, err = runner.DeviceMemoryStats(ctx)
		if err != nil {
			return climbed, "", fmt.Errorf("rung %d allocation accounting: %w", length, err)
		}
		climbed = append(climbed, Rung{Measure: generation.Measure, OutputIDs: tokenIDs(generation.Tokens)})
		// The caller reads each rung as it lands and may end the ladder
		// with a reason of its own, the device's memory for one.
		if progress != nil {
			if reason := progress(climbed[len(climbed)-1]); reason != "" && index+1 < len(rungs) {
				return climbed, fmt.Sprintf("after rung %d: %s", length, reason), nil
			}
		}
		if generation.Measure.PromptMilliseconds > floors.RungBudgetSeconds*1000 && index+1 < len(rungs) {
			return climbed, fmt.Sprintf("rung %d prefill took %.1fs, past the %.0fs rung budget", length,
				generation.Measure.PromptMilliseconds/1000, floors.RungBudgetSeconds), nil
		}
	}
	return climbed, "", nil
}

// executionCost splits the device counters read before the run, after
// the prompt, and after the run into the prompt's own work and the
// decode's work per output token.
func executionCost(before, afterPrompt, after driver.ExecutionStats, outputTokens int) Execution {
	cost := Execution{
		PromptKernelLaunches:         afterPrompt.KernelLaunches - before.KernelLaunches,
		PromptHostToDeviceBytes:      afterPrompt.HostToDeviceBytes - before.HostToDeviceBytes,
		PromptGraphInstantiations:    afterPrompt.GraphInstantiations - before.GraphInstantiations,
		PromptStreamSynchronizations: afterPrompt.StreamSynchronizations - before.StreamSynchronizations,
	}
	if outputTokens == 0 || afterPrompt.KernelLaunches == 0 {
		return cost
	}
	tokens := float64(outputTokens)
	cost.KernelLaunchesPerToken = float64(after.KernelLaunches-afterPrompt.KernelLaunches) / tokens
	cost.SynchronizationsPerToken = float64(after.StreamSynchronizations-afterPrompt.StreamSynchronizations) / tokens
	cost.HostToDeviceBytesPerToken = float64(after.HostToDeviceBytes-afterPrompt.HostToDeviceBytes) / tokens
	cost.DeviceToHostBytesPerToken = float64(after.DeviceToHostBytes-afterPrompt.DeviceToHostBytes) / tokens
	cost.GraphInstantiationsPerToken = float64(after.GraphInstantiations-afterPrompt.GraphInstantiations) / tokens
	cost.GraphLaunchesPerToken = float64(after.GraphLaunches-afterPrompt.GraphLaunches) / tokens
	return cost
}

// The evidence records token ids as int32 and the runner speaks
// tokenizer ids; one typed conversion serves both directions.
func tokenIDs(tokens []tokenizer.TokenID) []int32 { return dtype.ConvertSlice[int32](tokens) }

// Generation is one greedy run: the tokens it produced, their text, and
// the measure over them.
type Generation struct {
	Tokens  []tokenizer.TokenID
	Text    string
	Measure Measure
}

// WarmupOutputTokens is the decode budget of the unmeasured pass that
// precedes the measured run: kernels load and graphs instantiate on the
// first pass, and the benchmark measures after a warmup run too.
const WarmupOutputTokens = 4

// Warm runs the prompt once with a short decode so the measured run
// reads steady-state rates.
func Warm(ctx context.Context, runner *inference.Runner, prompt []tokenizer.TokenID, protocol Protocol) error {
	_, err := Run(ctx, runner, prompt, WarmupOutputTokens, protocol)
	return err
}

// Run prefills the prompt and decodes greedily up to the output budget,
// timing the prompt evaluation and the decode separately the way the
// benchmark does, and reads the degeneration measures off the tokens.
// RawContinuation retains natural stopping; GuardContinuation continues past
// sampled EOG tokens so the guard measures its full budget on the same path.
func Run(ctx context.Context, runner *inference.Runner, prompt []tokenizer.TokenID, outputTokens int, protocol Protocol) (Generation, error) {
	if runner == nil {
		return Generation{}, errors.New("longform: runner is nil")
	}
	if len(prompt) == 0 || outputTokens <= 0 {
		return Generation{}, errors.New("longform: the run needs a prompt and an output budget")
	}
	options, err := generationOptions(protocol, outputTokens)
	if err != nil {
		return Generation{}, err
	}
	var (
		evaluation  inference.PromptEvaluation
		tokens      = make([]tokenizer.TokenID, 0, outputTokens)
		firstToken  time.Duration
		afterPrompt driver.ExecutionStats
	)
	// The device counters are read around the run; a host-resident
	// runner has none and the counts stay zero.
	before, _ := runner.DeviceExecutionStats(ctx)
	started, err := processmeasure.Counter()
	if err != nil {
		return Generation{}, err
	}
	options.PromptTokenIDs = prompt
	options.OnPromptEvaluated = func(value inference.PromptEvaluation) {
		evaluation = value
		afterPrompt, _ = runner.DeviceExecutionStats(ctx)
	}
	options.OnToken = func(event inference.TokenEvent) error {
		if len(tokens) == 0 {
			var err error
			firstToken, err = processmeasure.Counter()
			if err != nil {
				return err
			}
		}
		tokens = append(tokens, event.ID)
		return nil
	}
	_, _, err = runner.Generate(ctx, "", options)
	if err != nil {
		return Generation{}, err
	}
	finished, err := processmeasure.Counter()
	if err != nil {
		return Generation{}, err
	}
	after, _ := runner.DeviceExecutionStats(ctx)
	total := finished - started
	timeToFirst := total
	if len(tokens) != 0 {
		timeToFirst = firstToken - started
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
	measure.Execution = executionCost(before, afterPrompt, after, len(tokens))
	ids := tokenIDs(tokens)
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

func generationOptions(protocol Protocol, outputTokens int) (inference.GenerateOptions, error) {
	if err := protocol.validate(); err != nil {
		return inference.GenerateOptions{}, err
	}
	sampler, err := sampling.New(sampling.Config{Temperature: 0})
	return inference.GenerateOptions{MaxNewTokens: outputTokens, Sampler: sampler, DeviceGreedy: true,
		ContinueAfterEOG: protocol == GuardContinuation}, err
}
