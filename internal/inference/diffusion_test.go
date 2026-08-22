package inference

import (
	"bytes"
	"context"
	"errors"
	"math"
	"math/rand"
	"reflect"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

func TestRunDiffusionTimestepConfidenceTransfer(t *testing.T) {
	const vocabularySize = 6
	mask := tokenizer.TokenID(5)
	steps := make([]DiffusionStep, 0, 4)
	options := DiffusionOptions{
		MaxLength: 5, Steps: 4, Algorithm: DiffusionConfidence,
		Schedule: DiffusionTimestep, Epsilon: 0.001, ShiftLogits: boolPointer(false),
		OnStep: func(step DiffusionStep) error {
			steps = append(steps, step)
			return nil
		},
	}
	evaluate := func(_ context.Context, tokens []tokenizer.TokenID) ([]float32, error) {
		logits := lowDiffusionLogits(len(tokens), vocabularySize)
		for position := range tokens {
			token := (position % 4) + 1
			logits[position*vocabularySize+token] = 100
		}
		return logits, nil
	}
	got, err := runDiffusion(
		context.Background(), []tokenizer.TokenID{0}, mask, vocabularySize, options, evaluate,
	)
	if err != nil {
		t.Fatal(err)
	}
	want := []tokenizer.TokenID{0, 2, 3, 4, 1}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("tokens = %v, want %v", got, want)
	}
	if len(steps) != 4 || steps[0].Step != 0 || steps[3].TotalSteps != 4 {
		t.Fatalf("steps = %+v", steps)
	}
	steps[0].Tokens[0] = 4
	if got[0] != 0 {
		t.Fatal("callback token snapshot aliases output")
	}
}

func TestRunDiffusionShiftedLogits(t *testing.T) {
	const vocabularySize = 4
	evaluate := func(_ context.Context, tokens []tokenizer.TokenID) ([]float32, error) {
		logits := lowDiffusionLogits(len(tokens), vocabularySize)
		logits[1] = 100
		logits[vocabularySize+2] = 100
		return logits, nil
	}
	base := DiffusionOptions{
		MaxLength: 2, Steps: 1, Algorithm: DiffusionConfidence,
		Schedule: DiffusionTimestep, Epsilon: 0.001,
	}
	for _, test := range []struct {
		shift bool
		want  tokenizer.TokenID
	}{{true, 1}, {false, 2}} {
		options := base
		options.ShiftLogits = boolPointer(test.shift)
		got, err := runDiffusion(
			context.Background(), []tokenizer.TokenID{0}, 3, vocabularySize, options, evaluate,
		)
		if err != nil {
			t.Fatal(err)
		}
		if got[1] != test.want {
			t.Fatalf("shift %v token = %d, want %d", test.shift, got[1], test.want)
		}
	}
}

func TestRunDiffusionRankingAlgorithms(t *testing.T) {
	evaluate := func(_ context.Context, tokens []tokenizer.TokenID) ([]float32, error) {
		logits := lowDiffusionLogits(len(tokens), 4)
		for position := range tokens {
			logits[position*4+1] = 100
		}
		return logits, nil
	}
	for algorithm := DiffusionOrigin; algorithm <= DiffusionConfidence; algorithm++ {
		options := DiffusionOptions{
			MaxLength: 2, Steps: 1, Algorithm: algorithm,
			Schedule: DiffusionTimestep, Epsilon: 0.001,
			ShiftLogits: boolPointer(false), Seed: 11,
		}
		got, err := runDiffusion(
			context.Background(), []tokenizer.TokenID{0}, 3, 4, options, evaluate,
		)
		if err != nil {
			t.Fatalf("algorithm %d: %v", algorithm, err)
		}
		if got[1] != 1 {
			t.Fatalf("algorithm %d token = %d", algorithm, got[1])
		}
	}
}

func TestRunDiffusionBlockCFG(t *testing.T) {
	const vocabularySize = 6
	mask := tokenizer.TokenID(5)
	calls := 0
	evaluate := func(_ context.Context, tokens []tokenizer.TokenID) ([]float32, error) {
		calls++
		logits := lowDiffusionLogits(len(tokens), vocabularySize)
		unconditional := tokens[0] == mask
		for position := range tokens {
			if unconditional {
				logits[position*vocabularySize+1] = 0
				logits[position*vocabularySize+2] = 200
			} else {
				logits[position*vocabularySize+1] = 100
				logits[position*vocabularySize+2] = 101
			}
		}
		return logits, nil
	}
	got, err := runDiffusion(
		context.Background(), []tokenizer.TokenID{0}, mask, vocabularySize,
		DiffusionOptions{
			MaxLength: 4, Steps: 4, Algorithm: DiffusionConfidence,
			Schedule: DiffusionBlock, BlockLength: 2, CFGScale: 1,
			ShiftLogits: boolPointer(false),
		},
		evaluate,
	)
	if err != nil {
		t.Fatal(err)
	}
	if want := []tokenizer.TokenID{0, 1, 1, 1}; !reflect.DeepEqual(got, want) {
		t.Fatalf("tokens = %v, want %v", got, want)
	}
	if calls != 8 {
		t.Fatalf("conditional/unconditional calls = %d, want 8", calls)
	}
}

func TestDiffusionTransferSchedules(t *testing.T) {
	remaining := 4
	want := []int{0, 1, 1, 2}
	for step := range want {
		got := diffusionTransferCount(step, 4, remaining, DiffusionTimestep, 0.001, nil)
		if got != want[step] {
			t.Fatalf("step %d transfer = %d, want %d", step, got, want[step])
		}
		remaining -= got
	}
	if got := diffusionBlockTransfers(10, 4); !reflect.DeepEqual(got, []int{3, 3, 2, 2}) {
		t.Fatalf("block transfers = %v", got)
	}
}

func TestDiffusionConfidenceAndSelection(t *testing.T) {
	result := sampling.SampleProbabilityResult{
		SelectedProbability: 0.25,
		Top: []sampling.TokenProbability{
			{ID: 1, Probability: 0.5},
			{ID: 2, Probability: 0.3},
			{ID: 3, Probability: 0.2},
		},
	}
	rng := rand.New(rand.NewSource(7))
	if got := diffusionConfidence(result, DiffusionConfidence, rng); got != 0.25 {
		t.Fatalf("confidence = %g", got)
	}
	if got := diffusionConfidence(result, DiffusionMargin, rng); math.Abs(got-0.2) > 1e-12 {
		t.Fatalf("margin = %g", got)
	}
	wantEntropy := -(0.5*math.Log(0.5+1e-10) + 0.3*math.Log(0.3+1e-10) + 0.2*math.Log(0.2+1e-10))
	if got := diffusionConfidence(result, DiffusionEntropy, rng); math.Abs(got-wantEntropy) > 1e-12 {
		t.Fatalf("entropy = %g, want %g", got, wantEntropy)
	}
	candidates := []diffusionCandidate{
		{position: 4, confidence: 0.8, order: 2},
		{position: 2, confidence: 0.8, order: 1},
		{position: 1, confidence: 0.2, order: 0},
	}
	selected := selectDiffusionCandidates(candidates, 2, 0, rng)
	if selected[0].position != 2 || selected[1].position != 4 {
		t.Fatalf("deterministic selection = %+v", selected)
	}
	selected = selectDiffusionCandidates(candidates, 3, 0.5, rand.New(rand.NewSource(9)))
	seen := map[int]bool{}
	for _, candidate := range selected {
		if seen[candidate.position] {
			t.Fatalf("probabilistic selection duplicated %+v", candidate)
		}
		seen[candidate.position] = true
	}
}

func TestDiffusionGumbelNoise(t *testing.T) {
	unchanged := []float32{-1, 0, 1}
	sampling.AddGumbelNoise(unchanged, 0, rand.New(rand.NewSource(3)))
	if !reflect.DeepEqual(unchanged, []float32{-1, 0, 1}) {
		t.Fatalf("zero-temperature logits = %v", unchanged)
	}
	first := []float32{-1, 0, 1}
	second := append([]float32(nil), first...)
	sampling.AddGumbelNoise(first, 0.7, rand.New(rand.NewSource(3)))
	sampling.AddGumbelNoise(second, 0.7, rand.New(rand.NewSource(3)))
	if !reflect.DeepEqual(first, second) || reflect.DeepEqual(first, []float32{-1, 0, 1}) {
		t.Fatalf("Gumbel outputs = %v/%v", first, second)
	}
	for _, value := range first {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || value <= 0 {
			t.Fatalf("invalid Gumbel output %g", value)
		}
	}
}

func TestRunDiffusionValidationAndCancellation(t *testing.T) {
	evaluator := func(_ context.Context, tokens []tokenizer.TokenID) ([]float32, error) {
		return make([]float32, len(tokens)*4), nil
	}
	valid := DiffusionOptions{
		MaxLength: 4, Steps: 2, Algorithm: DiffusionConfidence,
		Schedule: DiffusionTimestep, Epsilon: 0.001,
	}
	for name, mutate := range map[string]func(*DiffusionOptions){
		"length":  func(options *DiffusionOptions) { options.MaxLength = 1 },
		"steps":   func(options *DiffusionOptions) { options.Steps = 0 },
		"epsilon": func(options *DiffusionOptions) { options.Epsilon = 0 },
		"block-divisibility": func(options *DiffusionOptions) {
			options.Schedule, options.Epsilon, options.BlockLength = DiffusionBlock, 0, 3
		},
		"temperature": func(options *DiffusionOptions) { options.Temperature = -1 },
	} {
		t.Run(name, func(t *testing.T) {
			options := valid
			mutate(&options)
			if _, err := runDiffusion(context.Background(), []tokenizer.TokenID{0}, 3, 4, options, evaluator); err == nil {
				t.Fatal("invalid options accepted")
			}
		})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := runDiffusion(ctx, []tokenizer.TokenID{0}, 3, 4, valid, evaluator); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation error = %v", err)
	}
	badEvaluator := func(context.Context, []tokenizer.TokenID) ([]float32, error) {
		return []float32{1}, nil
	}
	if _, err := runDiffusion(context.Background(), []tokenizer.TokenID{0}, 3, 4, valid, badEvaluator); err == nil {
		t.Fatal("short logits accepted")
	}
}

func TestGenerateDiffusionRejectsInvalidRunnerBoundaries(t *testing.T) {
	options := DiffusionOptions{
		MaxLength: 2, Steps: 1, Schedule: DiffusionTimestep, Epsilon: 0.001,
		PromptTokenIDs: []tokenizer.TokenID{0},
	}
	var missing *Runner
	if _, _, err := missing.GenerateDiffusion(context.Background(), "", options); err == nil {
		t.Fatal("nil runner accepted")
	}
	causal := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "llama", ContextLength: 2}},
		vocab: &tokenizer.Vocab{Tokens: make([]tokenizer.Token, 2), Mask: 1}},
	}
	causal = attachFixtureProgram(causal)
	if _, _, err := causal.GenerateDiffusion(context.Background(), "", options); err == nil {
		t.Fatal("causal runner accepted")
	}
	noMask := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "dream", ContextLength: 2}},
		vocab: &tokenizer.Vocab{Tokens: make([]tokenizer.Token, 2), Mask: tokenizer.NullToken}},
	}
	noMask = attachFixtureProgram(noMask)
	if _, _, err := noMask.GenerateDiffusion(context.Background(), "", options); err == nil {
		t.Fatal("missing mask accepted")
	}
	tooLong := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "dream", ContextLength: 1}},
		vocab: &tokenizer.Vocab{Tokens: make([]tokenizer.Token, 2), Mask: 1}},
	}
	tooLong = attachFixtureProgram(tooLong)
	if _, _, err := tooLong.GenerateDiffusion(context.Background(), "", options); err == nil {
		t.Fatal("over-context diffusion accepted")
	}
}

func TestDiffusionShiftLogitsMetadata(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{file: &gguf.File{}}}
	if got, err := runner.diffusionShiftLogits(nil); err != nil || !got {
		t.Fatalf("default shift = %v, %v", got, err)
	}
	runner.file = diffusionMetadataFile(t, gguf.Value{Type: gguf.ValueTypeString, Data: "false"})
	if got, err := runner.diffusionShiftLogits(nil); err != nil || got {
		t.Fatalf("string shift = %v, %v", got, err)
	}
	runner.file = diffusionMetadataFile(t, gguf.Value{Type: gguf.ValueTypeBool, Data: true})
	if got, err := runner.diffusionShiftLogits(nil); err != nil || !got {
		t.Fatalf("bool shift = %v, %v", got, err)
	}
	runner.file = diffusionMetadataFile(t, gguf.Value{Type: gguf.ValueTypeString, Data: "invalid"})
	if _, err := runner.diffusionShiftLogits(nil); err == nil {
		t.Fatal("invalid shift metadata accepted")
	}
	override := false
	if got, err := runner.diffusionShiftLogits(&override); err != nil || got {
		t.Fatalf("override shift = %v, %v", got, err)
	}
}

func diffusionMetadataFile(t *testing.T, value gguf.Value) *gguf.File {
	t.Helper()
	var encoded bytes.Buffer
	if err := gguf.Write(&encoded, []gguf.Metadata{{
		Key: "diffusion.shift_logits", Value: value,
	}}, nil, gguf.WriteOptions{}); err != nil {
		t.Fatal(err)
	}
	file, err := gguf.Parse(bytes.NewReader(encoded.Bytes()), uint64(encoded.Len()), gguf.DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	return file
}

func lowDiffusionLogits(tokens, vocabulary int) []float32 {
	logits := make([]float32, tokens*vocabulary)
	for index := range logits {
		logits[index] = -100
	}
	return logits
}

func boolPointer(value bool) *bool {
	return &value
}
