package inference

import (
	"context"
	"errors"
	"slices"
	"sync"
	"testing"
	"time"

	"overgo/internal/model"
	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

type fakeContinuousBatch struct {
	mu        sync.Mutex
	stepSizes []int
	greedy    int
	removed   []SequenceID
	stepOnce  sync.Once
	started   chan struct{}
	proceed   chan struct{}
}

func (f *fakeContinuousBatch) Step(
	_ context.Context,
	inputs []SequenceBatchInput,
) ([]SequenceBatchOutput, error) {
	return f.step(inputs, false), nil
}

func (f *fakeContinuousBatch) StepGreedy(
	_ context.Context,
	inputs []SequenceBatchInput,
) ([]SequenceBatchOutput, error) {
	return f.step(inputs, true), nil
}

func (f *fakeContinuousBatch) step(inputs []SequenceBatchInput, greedy bool) []SequenceBatchOutput {
	f.mu.Lock()
	f.stepSizes = append(f.stepSizes, len(inputs))
	if greedy {
		f.greedy++
	}
	f.mu.Unlock()
	if f.started != nil {
		f.stepOnce.Do(func() {
			close(f.started)
			<-f.proceed
		})
	}
	outputs := make([]SequenceBatchOutput, len(inputs))
	for index, input := range inputs {
		outputs[index] = SequenceBatchOutput{ID: input.ID, Logits: []float32{0, 4, 1}, Token: 1}
	}
	return outputs
}

func (f *fakeContinuousBatch) Remove(_ context.Context, id SequenceID) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.removed = append(f.removed, id)
	return nil
}

func (*fakeContinuousBatch) Close(context.Context) error { return nil }

func TestContinuousGeneratorFusesAndShrinksActiveSet(t *testing.T) {
	vocab := &tokenizer.Vocab{
		Tokens: []tokenizer.Token{
			{Text: "z", Type: tokenizer.TokenNormal},
			{Text: "a", Type: tokenizer.TokenNormal},
			{Text: "p", Type: tokenizer.TokenNormal},
		},
		EOS: tokenizer.NullToken, EOT: tokenizer.NullToken, EOM: tokenizer.NullToken,
	}
	batch := &fakeContinuousBatch{}
	ctx, cancel := context.WithCancel(t.Context())
	runner := fixtureRunner(model.Spec{CommonSpec: model.CommonSpec{Architecture: "qwen35"}}, model.Weights{})
	runner.vocab = vocab
	generator := &ContinuousGenerator{
		runner: runner, batch: batch,
		options: ContinuousGeneratorOptions{MaxSequences: 2},
		ctx:     ctx, cancel: cancel,
		submit: make(chan continuousGenerateRequest, 2), done: make(chan struct{}),
	}
	type outcome struct {
		ids []tokenizer.TokenID
		err error
	}
	results := make(chan outcome, 2)
	for _, count := range []int{1, 2} {
		go func() {
			sampler, err := sampling.New(sampling.Config{Temperature: 0})
			if err != nil {
				results <- outcome{err: err}
				return
			}
			options := GenerateOptions{
				MaxNewTokens: count, Sampler: sampler, DeviceGreedy: true,
				PromptTokenIDs: []tokenizer.TokenID{2},
			}
			if count == 1 {
				options.MaxNewTokens = 3
				options.ShouldStop = func(TokenEvent) bool { return true }
			}
			ids, _, err := generator.Generate(t.Context(), "", options)
			results <- outcome{ids: ids, err: err}
		}()
	}
	deadline := time.Now().Add(time.Second)
	for len(generator.submit) != 2 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if len(generator.submit) != 2 {
		t.Fatal("requests were not queued")
	}
	go generator.run()
	got := []int{}
	for range 2 {
		result := <-results
		if result.err != nil {
			t.Fatal(result.err)
		}
		got = append(got, len(result.ids))
	}
	slices.Sort(got)
	if !slices.Equal(got, []int{2, 3}) {
		t.Fatalf("result lengths = %v", got)
	}
	batch.mu.Lock()
	stepSizes := append([]int(nil), batch.stepSizes...)
	removed := append([]SequenceID(nil), batch.removed...)
	greedy := batch.greedy
	batch.mu.Unlock()
	if !slices.Equal(stepSizes, []int{2, 1}) {
		t.Fatalf("step sizes = %v", stepSizes)
	}
	if len(removed) != 2 {
		t.Fatalf("removed = %v", removed)
	}
	if greedy != 2 {
		t.Fatalf("device-greedy steps = %d, want 2", greedy)
	}
	if err := generator.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
}

func TestContinuousGeneratorRejectsUncompiledRequestPolicy(t *testing.T) {
	generator := &ContinuousGenerator{runner: &Runner{}}
	if _, _, err := generator.Generate(t.Context(), "fixture", GenerateOptions{CachePrompt: true}); err == nil {
		t.Fatal("uncompiled request policy accepted")
	}
}

func TestContinuousStatesUseDeviceTopKRequiresUniformSafePrefix(t *testing.T) {
	makeState := func(config sampling.Config) *continuousGenerateState {
		if config.Samplers == nil {
			config.Samplers = []sampling.SamplerStage{sampling.SamplerTopK, sampling.SamplerTemperature}
		}
		sampler, err := sampling.New(config)
		if err != nil {
			t.Fatal(err)
		}
		return &continuousGenerateState{request: continuousGenerateRequest{options: GenerateOptions{
			Sampler: sampler, DeviceTopK: true,
		}}}
	}
	states := []*continuousGenerateState{
		makeState(sampling.Config{Temperature: 0.8, TopK: 3}),
		makeState(sampling.Config{Temperature: 0.7, TopK: 3}),
	}
	if limit, ok := continuousStatesUseDeviceTopK(states, 5); !ok || limit != 3 {
		t.Fatalf("device top-K = %d, %v", limit, ok)
	}
	states[1] = makeState(sampling.Config{Temperature: 0.8, TopK: 2})
	if _, ok := continuousStatesUseDeviceTopK(states, 5); ok {
		t.Fatal("mixed top-K limits were accepted")
	}
	states[1] = makeState(sampling.Config{
		Temperature: 0.8, TopK: 3, RepeatLastN: -1, RepeatPenalty: 1.1,
		Samplers: []sampling.SamplerStage{
			sampling.SamplerPenalties, sampling.SamplerTopK, sampling.SamplerTemperature,
		},
	})
	if _, ok := continuousStatesUseDeviceTopK(states, 5); ok {
		t.Fatal("penalized sampler was accepted")
	}
}

func TestContinuousGeneratorCancelsOneFusedSequence(t *testing.T) {
	vocab := &tokenizer.Vocab{
		Tokens: []tokenizer.Token{
			{Text: "z", Type: tokenizer.TokenNormal},
			{Text: "a", Type: tokenizer.TokenNormal},
			{Text: "p", Type: tokenizer.TokenNormal},
		},
		EOS: tokenizer.NullToken, EOT: tokenizer.NullToken, EOM: tokenizer.NullToken,
	}
	batch := &fakeContinuousBatch{started: make(chan struct{}), proceed: make(chan struct{})}
	ctx, stop := context.WithCancel(t.Context())
	runner := fixtureRunner(model.Spec{CommonSpec: model.CommonSpec{Architecture: "qwen35"}}, model.Weights{})
	runner.vocab = vocab
	generator := &ContinuousGenerator{
		runner: runner, batch: batch,
		options: ContinuousGeneratorOptions{MaxSequences: 2},
		ctx:     ctx, cancel: stop,
		submit: make(chan continuousGenerateRequest, 2), done: make(chan struct{}),
	}
	cancelled, cancel := context.WithCancel(t.Context())
	results := make(chan error, 2)
	for index := range 2 {
		requestCtx := t.Context()
		if index == 0 {
			requestCtx = cancelled
		}
		go func() {
			sampler, err := sampling.New(sampling.Config{Temperature: 0})
			if err == nil {
				_, _, err = generator.Generate(requestCtx, "", GenerateOptions{
					MaxNewTokens: 2, Sampler: sampler,
					PromptTokenIDs: []tokenizer.TokenID{2},
				})
			}
			results <- err
		}()
	}
	deadline := time.Now().Add(time.Second)
	for len(generator.submit) != 2 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	go generator.run()
	<-batch.started
	cancel()
	close(batch.proceed)
	errs := []error{<-results, <-results}
	cancelledCount := 0
	for _, err := range errs {
		if errors.Is(err, context.Canceled) {
			cancelledCount++
		} else if err != nil {
			t.Fatal(err)
		}
	}
	if cancelledCount != 1 {
		t.Fatalf("cancelled results = %d", cancelledCount)
	}
	batch.mu.Lock()
	stepSizes := append([]int(nil), batch.stepSizes...)
	batch.mu.Unlock()
	if !slices.Equal(stepSizes, []int{2, 1}) {
		t.Fatalf("step sizes = %v", stepSizes)
	}
	if err := generator.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
}
