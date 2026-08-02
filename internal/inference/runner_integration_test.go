package inference

import (
	"context"
	"fmt"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tokenizer"
)

func TestIncrementalCacheMatchesFullForward(t *testing.T) {
	modelPath := os.Getenv("LLAMACPP2GO_QWEN3_MODEL")
	if modelPath == "" {
		t.Skip("LLAMACPP2GO_QWEN3_MODEL is not set")
	}
	runner, err := Open(modelPath, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	ctx := context.Background()
	full, err := runner.Forward(ctx, []tokenizer.TokenID{9707, 27})
	if err != nil {
		t.Fatal(err)
	}
	_, cache, err := runner.ForwardCached(ctx, []tokenizer.TokenID{9707}, nil)
	if err != nil {
		t.Fatal(err)
	}
	state, err := runner.SaveCache(cache)
	if err != nil {
		t.Fatal(err)
	}
	cache, err = runner.LoadCache(state)
	if err != nil {
		t.Fatal(err)
	}
	incremental, cache, err := runner.ForwardCached(ctx, []tokenizer.TokenID{27}, cache)
	if err != nil {
		t.Fatal(err)
	}
	if cache.Tokens != 2 {
		t.Fatalf("cache token count = %d, want 2", cache.Tokens)
	}
	width := int(full.Shape.Dims[0])
	expected := full.Data[len(full.Data)-width:]
	if len(incremental.Data) != width {
		t.Fatalf("incremental output length = %d, want %d", len(incremental.Data), width)
	}
	var maximum float64
	for index, want := range expected {
		difference := math.Abs(float64(incremental.Data[index] - want))
		if difference > maximum {
			maximum = difference
		}
	}
	if maximum > 5e-4 {
		t.Fatalf("incremental/full max absolute difference = %g", maximum)
	}
	shifted, err := runner.ShiftCache(cache, 1)
	if err != nil {
		t.Fatal(err)
	}
	shiftedState, err := runner.SaveCache(shifted)
	if err != nil {
		t.Fatal(err)
	}
	shifted, err = runner.LoadCache(shiftedState)
	if err != nil {
		t.Fatal(err)
	}
	if shifted.Tokens != 1 || shifted.Position != 2 {
		t.Fatalf(
			"shifted cache count/position = %d/%d, want 1/2",
			shifted.Tokens,
			shifted.Position,
		)
	}
	_, shifted, err = runner.ForwardCached(
		ctx,
		[]tokenizer.TokenID{18},
		shifted,
	)
	if err != nil {
		t.Fatal(err)
	}
	if shifted.Tokens != 2 || shifted.Position != 3 {
		t.Fatalf(
			"appended shifted cache count/position = %d/%d, want 2/3",
			shifted.Tokens,
			shifted.Position,
		)
	}
	originalContext := runner.spec.ContextLength
	runner.spec.ContextLength = 2
	defer func() { runner.spec.ContextLength = originalContext }()
	shiftSampler, err := sampling.New(sampling.Config{})
	if err != nil {
		t.Fatal(err)
	}
	session, _, err := runner.StartSession(ctx, "Hello", GenerateOptions{
		MaxNewTokens: 1,
		Sampler:      shiftSampler,
		ContextShift: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	session, _, err = runner.ContinueSession(ctx, session, GenerateOptions{
		MaxNewTokens: 2,
		Sampler:      shiftSampler,
		ContextShift: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(session.TokenIDs) != 4 ||
		session.Cache.Tokens != 2 ||
		session.Cache.Position != 3 {
		t.Fatalf(
			"shifted session history/cache = %d/%d/%d, want 4/2/3",
			len(session.TokenIDs),
			session.Cache.Tokens,
			session.Cache.Position,
		)
	}
	firstPromptSampler, err := sampling.New(sampling.Config{})
	if err != nil {
		t.Fatal(err)
	}
	var firstEvaluation PromptEvaluation
	_, _, err = runner.Generate(ctx, "", GenerateOptions{
		MaxNewTokens:   1,
		Sampler:        firstPromptSampler,
		PromptTokenIDs: []tokenizer.TokenID{9707},
		CachePrompt:    true,
		OnPromptEvaluated: func(evaluation PromptEvaluation) {
			firstEvaluation = evaluation
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	reusedSampler, err := sampling.New(sampling.Config{})
	if err != nil {
		t.Fatal(err)
	}
	var reusedEvaluation PromptEvaluation
	reusedIDs, _, err := runner.Generate(ctx, "", GenerateOptions{
		MaxNewTokens:   1,
		Sampler:        reusedSampler,
		PromptTokenIDs: []tokenizer.TokenID{9707, 27},
		CachePrompt:    true,
		MinCacheReuse:  1,
		OnPromptEvaluated: func(evaluation PromptEvaluation) {
			reusedEvaluation = evaluation
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	fullSampler, err := sampling.New(sampling.Config{})
	if err != nil {
		t.Fatal(err)
	}
	fullIDs, _, err := runner.Generate(ctx, "", GenerateOptions{
		MaxNewTokens:   1,
		Sampler:        fullSampler,
		PromptTokenIDs: []tokenizer.TokenID{9707, 27},
	})
	if err != nil {
		t.Fatal(err)
	}
	if firstEvaluation.Cached != 0 ||
		reusedEvaluation.Cached != 1 ||
		reusedIDs[len(reusedIDs)-1] != fullIDs[len(fullIDs)-1] {
		t.Fatalf(
			"prompt reuse first/reused=%+v/%+v token=%d full=%d",
			firstEvaluation,
			reusedEvaluation,
			reusedIDs[len(reusedIDs)-1],
			fullIDs[len(fullIDs)-1],
		)
	}
	t.Logf("incremental/full max absolute difference = %g", maximum)
}

func TestPreloadedCachedLayerInputsMatchFullExtraction(t *testing.T) {
	modelPath := os.Getenv("LLAMACPP2GO_QWEN3_MODEL")
	if modelPath == "" {
		t.Skip("LLAMACPP2GO_QWEN3_MODEL is not set")
	}
	runner, err := OpenWithOptions(modelPath, OpenOptions{
		DeviceOrdinal:           0,
		PreloadQuantizedWeights: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	layers := []int32{0, int32(len(runner.weights.Layers) / 2), int32(len(runner.weights.Layers) - 1)}
	tokens := []tokenizer.TokenID{9707, 27}
	ctx := context.Background()
	full, err := runner.ExtractLayerInputs(ctx, tokens, layers)
	if err != nil {
		t.Fatal(err)
	}
	_, cache, first, err := runner.ForwardCachedExtractLayerInputs(ctx, tokens[:1], nil, layers)
	if err != nil {
		t.Fatal(err)
	}
	_, _, second, err := runner.ForwardCachedExtractLayerInputs(ctx, tokens[1:], cache, layers)
	if err != nil {
		t.Fatal(err)
	}
	width := int(full.Shape.Dims[0])
	assertMaximumDifference(t, "prefix layer inputs", first.Data, full.Data[:width], 1e-2)
	assertMaximumDifference(t, "incremental layer inputs", second.Data, full.Data[width:], 1e-2)
}

func assertMaximumDifference(t *testing.T, label string, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", label, len(got), len(want))
	}
	var maximum float64
	for index := range got {
		maximum = max(maximum, math.Abs(float64(got[index]-want[index])))
	}
	if maximum > tolerance {
		t.Fatalf("%s max absolute difference = %g, want <= %g", label, maximum, tolerance)
	}
	t.Logf("%s max absolute difference = %g", label, maximum)
}

func TestEmbeddingOverrideMatchesTokenLookupAndProducesUsableCache(t *testing.T) {
	modelPath := os.Getenv("LLAMACPP2GO_QWEN3_MODEL")
	if modelPath == "" {
		t.Skip("LLAMACPP2GO_QWEN3_MODEL is not set")
	}
	runner, err := Open(modelPath, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	ctx := context.Background()
	replacement, err := runner.loadEmbeddings(ctx, []uint32{27})
	if err != nil {
		t.Fatal(err)
	}
	want, err := runner.Forward(ctx, []tokenizer.TokenID{9707, 27})
	if err != nil {
		t.Fatal(err)
	}
	got, cache, err := runner.ForwardCachedWithEmbeddingOverrides(
		ctx,
		[]tokenizer.TokenID{9707, 9707},
		nil,
		[]EmbeddingOverride{{TokenIndex: 1, Embedding: append([]float32(nil), replacement.Data...)}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if cache.Tokens != 2 || cache.Position != 2 {
		t.Fatalf("override cache count/position = %d/%d, want 2/2", cache.Tokens, cache.Position)
	}
	if len(got.Data) != len(want.Data) {
		t.Fatalf("override output length = %d, want %d", len(got.Data), len(want.Data))
	}
	for index := range want.Data {
		if difference := math.Abs(float64(got.Data[index] - want.Data[index])); difference > 5e-4 {
			t.Fatalf("override output[%d] delta = %g", index, difference)
		}
	}
	_, cache, err = runner.ForwardCached(ctx, []tokenizer.TokenID{18}, cache)
	if err != nil {
		t.Fatal(err)
	}
	if cache.Tokens != 3 || cache.Position != 3 {
		t.Fatalf("continued cache count/position = %d/%d, want 3/3", cache.Tokens, cache.Position)
	}
}

func TestNativeQ8GreedyMatchesOracle(t *testing.T) {
	modelPath := os.Getenv("LLAMACPP2GO_QWEN3_MODEL")
	if modelPath == "" {
		t.Skip("LLAMACPP2GO_QWEN3_MODEL is not set")
	}
	assertNativeQuantGreedyOracle(t, modelPath)
}

func TestNativeQ8DeviceContextShift(t *testing.T) {
	modelPath := os.Getenv("LLAMACPP2GO_QWEN3_MODEL")
	if modelPath == "" {
		t.Skip("LLAMACPP2GO_QWEN3_MODEL is not set")
	}
	runner, err := OpenWithOptions(modelPath, OpenOptions{
		DeviceOrdinal:           0,
		PreloadQuantizedWeights: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	runner.spec.ContextLength = 2
	sampler, err := sampling.New(sampling.Config{})
	if err != nil {
		t.Fatal(err)
	}
	before, err := runner.DeviceExecutionStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	ids, text, err := runner.Generate(context.Background(), "Hello", GenerateOptions{
		MaxNewTokens: 3,
		Sampler:      sampler,
		ContextShift: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	after, err := runner.DeviceExecutionStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	hostSampler, err := sampling.New(sampling.Config{})
	if err != nil {
		t.Fatal(err)
	}
	hostIDs, hostText, err := runner.Generate(context.Background(), "Hello", GenerateOptions{
		MaxNewTokens: 3,
		Sampler:      hostSampler,
		ContextShift: true,
		CachePrompt:  true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(ids, hostIDs) ||
		text != hostText ||
		after.StreamSynchronizations-before.StreamSynchronizations != 3 {
		t.Fatalf(
			"shifted device/host generation = %v %q / %v %q stats=%+v/%+v",
			ids,
			text,
			hostIDs,
			hostText,
			before,
			after,
		)
	}
	cachedShiftSampler, err := sampling.New(sampling.Config{})
	if err != nil {
		t.Fatal(err)
	}
	beforeCachedShift, err := runner.DeviceExecutionStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = runner.Generate(context.Background(), "", GenerateOptions{
		MaxNewTokens:   2,
		Sampler:        cachedShiftSampler,
		PromptTokenIDs: []tokenizer.TokenID{9707, 27},
		ContextShift:   true,
		CachePrompt:    true,
	})
	if err != nil {
		t.Fatal(err)
	}
	afterCachedShift, err := runner.DeviceExecutionStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if afterCachedShift.DeviceToDeviceCopies <=
		beforeCachedShift.DeviceToDeviceCopies {
		t.Fatalf(
			"shared prompt context shift performed no preserving device copy: %+v/%+v",
			beforeCachedShift,
			afterCachedShift,
		)
	}
	reuseSampler, err := sampling.New(sampling.Config{})
	if err != nil {
		t.Fatal(err)
	}
	var cachedShiftEvaluation PromptEvaluation
	_, _, err = runner.Generate(context.Background(), "", GenerateOptions{
		MaxNewTokens:   1,
		Sampler:        reuseSampler,
		PromptTokenIDs: []tokenizer.TokenID{9707, 27},
		CachePrompt:    true,
		MinCacheReuse:  2,
		OnPromptEvaluated: func(evaluation PromptEvaluation) {
			cachedShiftEvaluation = evaluation
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if cachedShiftEvaluation.Cached != 2 {
		t.Fatalf("preserved device prompt cached %d tokens, want 2",
			cachedShiftEvaluation.Cached)
	}

	runner.spec.ContextLength = 5
	keepSampler, err := sampling.New(sampling.Config{})
	if err != nil {
		t.Fatal(err)
	}
	beforeKeep, err := runner.DeviceExecutionStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	keptIDs, _, err := runner.Generate(context.Background(), "", GenerateOptions{
		MaxNewTokens:   6,
		Sampler:        keepSampler,
		PromptTokenIDs: []tokenizer.TokenID{9707},
		ContextShift:   true,
		KeepTokens:     1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(keptIDs) != 7 {
		t.Fatalf("prefix-preserving context shift returned %d IDs, want 7",
			len(keptIDs))
	}
	afterKeep, err := runner.DeviceExecutionStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if afterKeep.DeviceToDeviceCopies <= beforeKeep.DeviceToDeviceCopies {
		t.Fatalf(
			"prefix-preserving context shift performed no device copies: %+v/%+v",
			beforeKeep,
			afterKeep,
		)
	}

	runner.spec.ContextLength = 16
	firstCacheSampler, err := sampling.New(sampling.Config{})
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = runner.Generate(context.Background(), "", GenerateOptions{
		MaxNewTokens:   1,
		Sampler:        firstCacheSampler,
		PromptTokenIDs: []tokenizer.TokenID{9707, 27, 358},
		CachePrompt:    true,
	})
	if err != nil {
		t.Fatal(err)
	}
	divergentSampler, err := sampling.New(sampling.Config{})
	if err != nil {
		t.Fatal(err)
	}
	var divergentEvaluation PromptEvaluation
	divergentIDs, _, err := runner.Generate(
		context.Background(),
		"",
		GenerateOptions{
			MaxNewTokens:   1,
			Sampler:        divergentSampler,
			PromptTokenIDs: []tokenizer.TokenID{9707, 27, 18},
			CachePrompt:    true,
			MinCacheReuse:  2,
			OnPromptEvaluated: func(evaluation PromptEvaluation) {
				divergentEvaluation = evaluation
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	fullSampler, err := sampling.New(sampling.Config{})
	if err != nil {
		t.Fatal(err)
	}
	fullIDs, _, err := runner.Generate(context.Background(), "", GenerateOptions{
		MaxNewTokens:   1,
		Sampler:        fullSampler,
		PromptTokenIDs: []tokenizer.TokenID{9707, 27, 18},
	})
	if err != nil {
		t.Fatal(err)
	}
	if divergentEvaluation.Cached != 2 ||
		divergentIDs[len(divergentIDs)-1] != fullIDs[len(fullIDs)-1] {
		t.Fatalf(
			"divergent device prompt reuse = %+v token %d, full token %d",
			divergentEvaluation,
			divergentIDs[len(divergentIDs)-1],
			fullIDs[len(fullIDs)-1],
		)
	}

	runner.promptCacheCapacity = 2
	for _, prompt := range [][]tokenizer.TokenID{
		{9707, 27, 358},
		{9707, 18, 358},
	} {
		entrySampler, samplerErr := sampling.New(sampling.Config{})
		if samplerErr != nil {
			t.Fatal(samplerErr)
		}
		if _, _, generationErr := runner.Generate(
			context.Background(),
			"",
			GenerateOptions{
				MaxNewTokens:   1,
				Sampler:        entrySampler,
				PromptTokenIDs: prompt,
				CachePrompt:    true,
				MinCacheReuse:  3,
			},
		); generationErr != nil {
			t.Fatal(generationErr)
		}
	}
	multiReuseSampler, err := sampling.New(sampling.Config{})
	if err != nil {
		t.Fatal(err)
	}
	var multiEvaluation PromptEvaluation
	_, _, err = runner.Generate(context.Background(), "", GenerateOptions{
		MaxNewTokens:   1,
		Sampler:        multiReuseSampler,
		PromptTokenIDs: []tokenizer.TokenID{9707, 27, 358},
		CachePrompt:    true,
		MinCacheReuse:  3,
		OnPromptEvaluated: func(evaluation PromptEvaluation) {
			multiEvaluation = evaluation
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if multiEvaluation.Cached != 3 || len(runner.promptCaches) != 2 {
		t.Fatalf("multi-entry prompt reuse = %+v entries=%d, want 3/2",
			multiEvaluation, len(runner.promptCaches))
	}
}

func TestNativeQ6KGreedyMatchesOracle(t *testing.T) {
	modelPath := os.Getenv("LLAMACPP2GO_QWEN3_Q6K_MODEL")
	if modelPath == "" {
		t.Skip("LLAMACPP2GO_QWEN3_Q6K_MODEL is not set")
	}
	assertNativeQuantGreedyOracle(t, modelPath)
}

func TestNativeQwen35HybridMatchesOracleAndResumes(t *testing.T) {
	modelPath := os.Getenv("LLAMACPP2GO_QWEN35_MODEL")
	if modelPath == "" {
		t.Skip("LLAMACPP2GO_QWEN35_MODEL is not set")
	}
	runner, err := OpenWithOptions(modelPath, OpenOptions{
		DeviceOrdinal:           0,
		PreloadQuantizedWeights: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()

	ctx := context.Background()
	firstSampler, err := sampling.New(sampling.Config{})
	if err != nil {
		t.Fatal(err)
	}
	session, _, err := runner.StartSession(ctx, "Hello", GenerateOptions{
		MaxNewTokens: 1,
		Sampler:      firstSampler,
	})
	if err != nil {
		t.Fatal(err)
	}
	cacheData, err := runner.SaveCache(session.Cache)
	if err != nil {
		t.Fatal(err)
	}
	session.Cache, err = runner.LoadCache(cacheData)
	if err != nil {
		t.Fatal(err)
	}
	sessionData, err := runner.SaveSession(session, firstSampler)
	if err != nil {
		t.Fatal(err)
	}
	resumedSampler, err := sampling.New(sampling.Config{})
	if err != nil {
		t.Fatal(err)
	}
	session, err = runner.LoadSession(sessionData, resumedSampler)
	if err != nil {
		t.Fatal(err)
	}
	session, text, err := runner.ContinueSession(ctx, session, GenerateOptions{
		MaxNewTokens: 2,
		Sampler:      resumedSampler,
	})
	if err != nil {
		t.Fatal(err)
	}
	wantIDs := []tokenizer.TokenID{9419, 11, 353, 1044}
	if !slices.Equal(session.TokenIDs, wantIDs) || text != "Hello, I am" {
		t.Fatalf(
			"Qwen3.5 resumed generation = %v %q, want %v %q",
			session.TokenIDs,
			text,
			wantIDs,
			"Hello, I am",
		)
	}
	gbnf, err := runner.CompileGBNF(`
root ::= "{" ws "\"ok\"" ws ":" ws boolean ws "}"
boolean ::= "true" | "false"
ws ::= [ \t\n\r]*
`, "root")
	if err != nil {
		t.Fatal(err)
	}
	gbnfSampler, err := sampling.New(sampling.Config{GBNF: gbnf})
	if err != nil {
		t.Fatal(err)
	}
	gbnfIDs, gbnfText, err := runner.Generate(
		ctx,
		"Return a boolean object:",
		GenerateOptions{MaxNewTokens: 32, Sampler: gbnfSampler},
	)
	if err != nil {
		t.Fatal(err)
	}
	wantGBNFIDs := []tokenizer.TokenID{
		5423, 264, 2623, 1576, 25, 4754, 547, 763, 1802, 92, 248046,
	}
	if !slices.Equal(gbnfIDs, wantGBNFIDs) ||
		gbnfText != `Return a boolean object:{"ok":true}` {
		t.Fatalf(
			"Qwen3.5 GBNF generation = %v %q, want %v %q",
			gbnfIDs,
			gbnfText,
			wantGBNFIDs,
			`Return a boolean object:{"ok":true}`,
		)
	}
	tokenGBNF, err := runner.CompileGBNF(
		`root ::= <|im_start|>`,
		"root",
	)
	if err != nil {
		t.Fatal(err)
	}
	tokenSampler, err := sampling.New(sampling.Config{GBNF: tokenGBNF})
	if err != nil {
		t.Fatal(err)
	}
	tokenIDs, tokenText, err := runner.Generate(
		ctx,
		"Hello",
		GenerateOptions{MaxNewTokens: 4, Sampler: tokenSampler},
	)
	if err != nil {
		t.Fatal(err)
	}
	wantTokenIDs := []tokenizer.TokenID{9419, 248045, 248046}
	if !slices.Equal(tokenIDs, wantTokenIDs) || tokenText != "Hello" {
		t.Fatalf(
			"Qwen3.5 named-token GBNF = %v %q, want %v %q",
			tokenIDs,
			tokenText,
			wantTokenIDs,
			"Hello",
		)
	}
	lazyGBNF, err := runner.CompileLazyGBNF(
		`root ::= "," " I am"`,
		"root",
		[]string{`(,)`},
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	lazySampler, err := sampling.New(sampling.Config{GBNF: lazyGBNF})
	if err != nil {
		t.Fatal(err)
	}
	lazyIDs, lazyText, err := runner.Generate(
		ctx,
		"Hello",
		GenerateOptions{MaxNewTokens: 8, Sampler: lazySampler},
	)
	if err != nil {
		t.Fatal(err)
	}
	wantLazyIDs := []tokenizer.TokenID{9419, 11, 353, 1044, 248046}
	if !slices.Equal(lazyIDs, wantLazyIDs) || lazyText != "Hello, I am" {
		t.Fatalf(
			"Qwen3.5 lazy GBNF = %v %q, want %v %q",
			lazyIDs,
			lazyText,
			wantLazyIDs,
			"Hello, I am",
		)
	}
	perplexity, err := runner.PerplexityWithOptions(
		ctx,
		strings.Repeat("Hello world. ", 40),
		PerplexityOptions{ContextSize: 32},
	)
	if err != nil {
		t.Fatal(err)
	}
	const oraclePerplexity = 1.0785
	if math.Abs(perplexity.Perplexity-oraclePerplexity) > 0.002 {
		t.Fatalf(
			"Qwen3.5 perplexity = %.7f, want oracle %.7f +/- 0.002",
			perplexity.Perplexity,
			oraclePerplexity,
		)
	}
}

func TestNativeQwen35FusedContinuousBatch(t *testing.T) {
	modelPath := os.Getenv("LLAMACPP2GO_QWEN35_MODEL")
	if modelPath == "" {
		t.Skip("LLAMACPP2GO_QWEN35_MODEL is not set")
	}
	runner, err := OpenWithOptions(modelPath, OpenOptions{
		DeviceOrdinal: 0, PreloadQuantizedWeights: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	ids, err := runner.TokenizeText("Hello", true, true)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := runner.NewContinuousBatch(ContinuousBatchOptions{
		MaxSequences: 2, Device: true, PageTokens: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer batch.Close(context.Background())
	outputs, err := batch.Step(context.Background(), []SequenceBatchInput{
		{ID: 10, Tokens: ids}, {ID: 20, Tokens: ids},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(outputs) != 2 || !slices.Equal(outputs[0].Logits, outputs[1].Logits) {
		t.Fatal("fused identical branches diverged")
	}
	next := tokenizer.TokenID(0)
	for index := 1; index < len(outputs[0].Logits); index++ {
		if outputs[0].Logits[index] > outputs[0].Logits[next] {
			next = tokenizer.TokenID(index)
		}
	}
	outputs, err = batch.Step(context.Background(), []SequenceBatchInput{
		{ID: 10, Tokens: []tokenizer.TokenID{next}},
		{ID: 20, Tokens: []tokenizer.TokenID{next}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(outputs[0].Logits, outputs[1].Logits) {
		t.Fatal("fused cached branches diverged")
	}
	if err := batch.Remove(context.Background(), 10); err != nil {
		t.Fatal(err)
	}
	states := batch.Snapshot()
	if len(states) != 1 || states[0].ID != 20 {
		t.Fatalf("post-remove states = %+v", states)
	}
}

func TestQwen35MTPAdvancesIndependentDraftState(t *testing.T) {
	modelPath := os.Getenv("LLAMACPP2GO_QWEN35_MTP_MODEL")
	if modelPath == "" {
		t.Skip("LLAMACPP2GO_QWEN35_MTP_MODEL is not set")
	}
	runner, err := OpenWithOptions(modelPath, OpenOptions{
		DeviceOrdinal: 0, PreloadQuantizedWeights: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	ctx := context.Background()
	session, err := runner.NewQwen35MTPSession(ctx, []tokenizer.TokenID{0})
	if err != nil {
		t.Fatal(err)
	}
	logits, next, err := runner.AdvanceQwen35MTP(ctx, 0, session)
	if err != nil {
		t.Fatal(err)
	}
	if logits.Shape.Dims[0] != uint64(runner.spec.VocabularySize) ||
		next.Position != session.Position+1 || next.Layer.Key.Shape.Dims[2] != 1 ||
		session.Layer.Key.Shape.Rank != 0 {
		t.Fatalf("unexpected Qwen3.5 MTP state: logits=%v before=%+v after=%+v", logits.Shape, session, next)
	}
	_, third, err := runner.AdvanceQwen35MTP(ctx, 0, next)
	if err != nil {
		t.Fatal(err)
	}
	if third.Position != next.Position+1 || third.Layer.Key.Shape.Dims[2] != 2 {
		t.Fatalf("Qwen3.5 MTP cache did not advance: %+v", third)
	}
	coordinatorSession, err := runner.NewQwen35MTPSession(ctx, []tokenizer.TokenID{0})
	if err != nil {
		t.Fatal(err)
	}
	draft, err := runner.DraftQwen35MTPGreedy(ctx, 0, coordinatorSession, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	verification, err := runner.VerifyQwen35MTPGreedy(ctx, runner, draft)
	if err != nil {
		t.Fatal(err)
	}
	if verification.Accepted < 0 || verification.Accepted > len(draft.Tokens) ||
		verification.Session.Position != coordinatorSession.Position+uint32(verification.Accepted)+1 ||
		verification.Session.TrunkCache.Position != verification.Session.Position {
		t.Fatalf("unexpected Qwen3.5 MTP verification: draft=%+v result=%+v", draft, verification)
	}
	draftSampler, err := sampling.New(sampling.Config{
		Temperature: 0.8, TopK: 32, TopP: 0.95, Seed: 17,
	})
	if err != nil {
		t.Fatal(err)
	}
	targetSampler, err := sampling.New(sampling.Config{
		Temperature: 0.8, TopK: 32, TopP: 0.95, Seed: 23,
	})
	if err != nil {
		t.Fatal(err)
	}
	draftSamplerBefore, err := draftSampler.SaveState()
	if err != nil {
		t.Fatal(err)
	}
	sampledDraft, err := runner.DraftQwen35MTPSampled(
		ctx, coordinatorSession, draftSampler, []tokenizer.TokenID{0}, 2, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	draftSamplerAfter, err := draftSampler.SaveState()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(draftSamplerBefore, draftSamplerAfter) {
		t.Fatal("Qwen3.5 sampled drafting changed caller sampler state")
	}
	sampledVerification, err := runner.VerifyQwen35MTPSampled(
		ctx, runner, sampledDraft, draftSampler, targetSampler,
	)
	if err != nil {
		t.Fatal(err)
	}
	if sampledVerification.Accepted < 0 ||
		sampledVerification.Accepted > len(sampledDraft.Tokens) ||
		sampledVerification.Session.Position !=
			coordinatorSession.Position+uint32(sampledVerification.Accepted)+1 ||
		sampledVerification.Session.TrunkCache.Position != sampledVerification.Session.Position {
		t.Fatalf(
			"unexpected sampled Qwen3.5 MTP verification: draft=%+v result=%+v",
			sampledDraft, sampledVerification,
		)
	}
}

func TestGemma4AssistantGreedyVerification(t *testing.T) {
	assistantPath := os.Getenv("LLAMACPP2GO_GEMMA4_ASSISTANT_MODEL")
	targetPath := os.Getenv("LLAMACPP2GO_GEMMA4_TARGET_MODEL")
	if assistantPath == "" || targetPath == "" {
		t.Skip("LLAMACPP2GO_GEMMA4_ASSISTANT_MODEL and LLAMACPP2GO_GEMMA4_TARGET_MODEL are not set")
	}
	assistant, err := OpenWithOptions(assistantPath, OpenOptions{DeviceOrdinal: 0, PreloadQuantizedWeights: true})
	if err != nil {
		t.Fatal(err)
	}
	defer assistant.Close()
	target, err := OpenWithOptions(targetPath, OpenOptions{DeviceOrdinal: 0, PreloadQuantizedWeights: true})
	if err != nil {
		t.Fatal(err)
	}
	defer target.Close()
	ctx := context.Background()
	session, err := assistant.NewGemma4AssistantSession(ctx, target, []tokenizer.TokenID{0})
	if err != nil {
		t.Fatal(err)
	}
	projectedSession, err := assistant.NewGemma4AssistantProjectedSession(
		ctx, target, []tokenizer.TokenID{0}, ProjectedInputs{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if projectedSession.Position != session.Position ||
		projectedSession.PendingHidden.Shape != session.PendingHidden.Shape {
		t.Fatalf("projected Gemma 4 assistant session mismatch: text=%+v projected=%+v", session, projectedSession)
	}
	draft, err := assistant.DraftGemma4AssistantGreedy(ctx, target, 0, session, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	verification, err := assistant.VerifyGemma4AssistantGreedy(ctx, target, draft)
	if err != nil {
		t.Fatal(err)
	}
	if verification.Accepted < 0 || verification.Accepted > len(draft.Tokens) ||
		verification.Session.Position != session.Position+uint32(verification.Accepted)+1 ||
		verification.Session.TargetCache.Position != verification.Session.Position {
		t.Fatalf("unexpected Gemma 4 assistant verification: draft=%+v result=%+v", draft, verification)
	}
	draftSampler, err := sampling.New(sampling.Config{Temperature: 0.8, TopK: 32, TopP: 0.95, Seed: 17})
	if err != nil {
		t.Fatal(err)
	}
	targetSampler, err := sampling.New(sampling.Config{Temperature: 0.8, TopK: 32, TopP: 0.95, Seed: 23})
	if err != nil {
		t.Fatal(err)
	}
	draftSamplerBefore, err := draftSampler.SaveState()
	if err != nil {
		t.Fatal(err)
	}
	sampledDraft, err := assistant.DraftGemma4AssistantSampled(
		ctx, target, session, draftSampler, []tokenizer.TokenID{0}, 2, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	draftSamplerAfter, err := draftSampler.SaveState()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(draftSamplerBefore, draftSamplerAfter) {
		t.Fatal("Gemma 4 assistant sampled drafting changed caller sampler state")
	}
	sampledVerification, err := assistant.VerifyGemma4AssistantSampled(
		ctx, target, sampledDraft, draftSampler, targetSampler,
	)
	if err != nil {
		t.Fatal(err)
	}
	if sampledVerification.Accepted < 0 || sampledVerification.Accepted > len(sampledDraft.Tokens) ||
		sampledVerification.Session.Position != session.Position+uint32(sampledVerification.Accepted)+1 ||
		sampledVerification.Session.TargetCache.Position != sampledVerification.Session.Position {
		t.Fatalf("unexpected sampled Gemma 4 assistant verification: draft=%+v result=%+v", sampledDraft, sampledVerification)
	}
}

func TestEagle3GreedyAndSampledVerification(t *testing.T) {
	draftPath := os.Getenv("LLAMACPP2GO_EAGLE3_MODEL")
	targetPath := os.Getenv("LLAMACPP2GO_EAGLE3_TARGET_MODEL")
	if draftPath == "" || targetPath == "" {
		t.Skip("LLAMACPP2GO_EAGLE3_MODEL and LLAMACPP2GO_EAGLE3_TARGET_MODEL are not set")
	}
	draftRunner, err := OpenWithOptions(draftPath, OpenOptions{DeviceOrdinal: 0, PreloadQuantizedWeights: true})
	if err != nil {
		t.Fatal(err)
	}
	defer draftRunner.Close()
	target, err := OpenWithOptions(targetPath, OpenOptions{DeviceOrdinal: 0, PreloadQuantizedWeights: true})
	if err != nil {
		t.Fatal(err)
	}
	defer target.Close()
	ctx := context.Background()
	session, err := draftRunner.NewEagle3Session(ctx, target, []tokenizer.TokenID{0})
	if err != nil {
		t.Fatal(err)
	}
	draft, err := draftRunner.DraftEagle3Greedy(ctx, target, 0, session, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	verification, err := draftRunner.VerifyEagle3Greedy(ctx, target, draft)
	if err != nil {
		t.Fatal(err)
	}
	if verification.Accepted < 0 || verification.Accepted > len(draft.Tokens) ||
		verification.Session.Position != session.Position+uint32(verification.Accepted)+1 ||
		verification.Session.TargetCache.Position != verification.Session.Position+1 {
		t.Fatalf("unexpected Eagle3 verification: draft=%+v result=%+v", draft, verification)
	}
	draftSampler, err := sampling.New(sampling.Config{Temperature: 0.8, TopK: 32, TopP: 0.95, Seed: 17})
	if err != nil {
		t.Fatal(err)
	}
	targetSampler, err := sampling.New(sampling.Config{Temperature: 0.8, TopK: 32, TopP: 0.95, Seed: 23})
	if err != nil {
		t.Fatal(err)
	}
	before, err := draftSampler.SaveState()
	if err != nil {
		t.Fatal(err)
	}
	sampledDraft, err := draftRunner.DraftEagle3Sampled(
		ctx, target, session, draftSampler, []tokenizer.TokenID{0}, 2, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	after, err := draftSampler.SaveState()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(before, after) {
		t.Fatal("Eagle3 sampled drafting changed caller sampler state")
	}
	sampledVerification, err := draftRunner.VerifyEagle3Sampled(
		ctx, target, sampledDraft, draftSampler, targetSampler,
	)
	if err != nil {
		t.Fatal(err)
	}
	if sampledVerification.Accepted < 0 || sampledVerification.Accepted > len(sampledDraft.Tokens) ||
		sampledVerification.Session.Position != session.Position+uint32(sampledVerification.Accepted)+1 ||
		sampledVerification.Session.TargetCache.Position != sampledVerification.Session.Position+1 {
		t.Fatalf("unexpected sampled Eagle3 verification: draft=%+v result=%+v", sampledDraft, sampledVerification)
	}
}

func TestDFlashGreedyAndSampledVerification(t *testing.T) {
	draftPath := os.Getenv("LLAMACPP2GO_DFLASH_MODEL")
	targetPath := os.Getenv("LLAMACPP2GO_DFLASH_TARGET_MODEL")
	if draftPath == "" || targetPath == "" {
		t.Skip("LLAMACPP2GO_DFLASH_MODEL and LLAMACPP2GO_DFLASH_TARGET_MODEL are not set")
	}
	draftRunner, err := OpenWithOptions(draftPath, OpenOptions{DeviceOrdinal: 0, PreloadQuantizedWeights: true})
	if err != nil {
		t.Fatal(err)
	}
	defer draftRunner.Close()
	target, err := OpenWithOptions(targetPath, OpenOptions{DeviceOrdinal: 0, PreloadQuantizedWeights: true})
	if err != nil {
		t.Fatal(err)
	}
	defer target.Close()
	maximum := min(2, int(draftRunner.spec.DFlashBlockSize)-1)
	if maximum <= 0 {
		t.Fatalf("invalid DFlash block size: %d", draftRunner.spec.DFlashBlockSize)
	}
	ctx := context.Background()
	session, err := draftRunner.NewDFlashSession(ctx, target, []tokenizer.TokenID{0})
	if err != nil {
		t.Fatal(err)
	}
	draft, err := draftRunner.DraftDFlashGreedy(ctx, target, 0, session, maximum, 0)
	if err != nil {
		t.Fatal(err)
	}
	verification, err := draftRunner.VerifyDFlashGreedy(ctx, target, draft)
	if err != nil {
		t.Fatal(err)
	}
	if verification.Accepted < 0 || verification.Accepted > len(draft.Tokens) ||
		verification.Session.Position != session.Position+uint32(verification.Accepted)+1 ||
		verification.Session.Cache.Position != verification.Session.Position ||
		verification.Session.TargetCache.Position != verification.Session.Position {
		t.Fatalf("unexpected DFlash verification: draft=%+v result=%+v", draft, verification)
	}
	draftSampler, err := sampling.New(sampling.Config{Temperature: 0.8, TopK: 32, TopP: 0.95, Seed: 17})
	if err != nil {
		t.Fatal(err)
	}
	targetSampler, err := sampling.New(sampling.Config{Temperature: 0.8, TopK: 32, TopP: 0.95, Seed: 23})
	if err != nil {
		t.Fatal(err)
	}
	before, err := draftSampler.SaveState()
	if err != nil {
		t.Fatal(err)
	}
	sampledDraft, err := draftRunner.DraftDFlashSampled(
		ctx, target, session, draftSampler, []tokenizer.TokenID{0}, maximum, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	after, err := draftSampler.SaveState()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(before, after) {
		t.Fatal("DFlash sampled drafting changed caller sampler state")
	}
	sampledVerification, err := draftRunner.VerifyDFlashSampled(
		ctx, target, sampledDraft, draftSampler, targetSampler,
	)
	if err != nil {
		t.Fatal(err)
	}
	if sampledVerification.Accepted < 0 || sampledVerification.Accepted > len(sampledDraft.Tokens) ||
		sampledVerification.Session.Position != session.Position+uint32(sampledVerification.Accepted)+1 ||
		sampledVerification.Session.Cache.Position != sampledVerification.Session.Position ||
		sampledVerification.Session.TargetCache.Position != sampledVerification.Session.Position {
		t.Fatalf("unexpected sampled DFlash verification: draft=%+v result=%+v", sampledDraft, sampledVerification)
	}
}

func TestWavTokenizerDecodeWaveform(t *testing.T) {
	modelPath := os.Getenv("LLAMACPP2GO_WAVTOKENIZER_MODEL")
	if modelPath == "" {
		t.Skip("LLAMACPP2GO_WAVTOKENIZER_MODEL is not set")
	}
	runner, err := OpenWithOptions(modelPath, OpenOptions{DeviceOrdinal: 0, PreloadQuantizedWeights: true})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	audio, err := runner.DecodeWavTokenizerWaveform(context.Background(), []tokenizer.TokenID{0})
	if err != nil {
		t.Fatal(err)
	}
	if len(audio) != 320 {
		t.Fatalf("WavTokenizer waveform length = %d, want 320", len(audio))
	}
	for index, sample := range audio {
		if math.IsNaN(float64(sample)) || math.IsInf(float64(sample), 0) {
			t.Fatalf("WavTokenizer waveform[%d] is not finite: %g", index, sample)
		}
	}
}

func TestCohere2MTPAdvancesIndependentDraftState(t *testing.T) {
	modelPath := os.Getenv("LLAMACPP2GO_COHERE2_MTP_MODEL")
	if modelPath == "" {
		t.Skip("LLAMACPP2GO_COHERE2_MTP_MODEL is not set")
	}
	runner, err := OpenWithOptions(modelPath, OpenOptions{
		DeviceOrdinal: 0, PreloadQuantizedWeights: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	ctx := context.Background()
	session, err := runner.NewCohere2MTPSession(ctx, []tokenizer.TokenID{0})
	if err != nil {
		t.Fatal(err)
	}
	logits, next, err := runner.AdvanceCohere2MTP(ctx, 0, session)
	if err != nil {
		t.Fatal(err)
	}
	if logits.Shape.Dims[0] != uint64(runner.spec.VocabularySize) ||
		next.Position != session.Position+1 || next.Layer.Key.Shape.Dims[2] != 1 ||
		session.Layer.Key.Shape.Rank != 0 {
		t.Fatalf("unexpected Cohere2-MoE MTP state: logits=%v before=%+v after=%+v", logits.Shape, session, next)
	}
	state, err := runner.SaveCohere2MTPSession(next)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := runner.LoadCohere2MTPSession(state)
	if err != nil {
		t.Fatal(err)
	}
	if restored.Position != next.Position || restored.Layer.Key.Shape.Dims[2] != 1 {
		t.Fatalf("unexpected restored Cohere2-MoE MTP state: %+v", restored)
	}
	draft, err := runner.DraftCohere2MTPGreedy(ctx, 0, session, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	verification, err := runner.VerifyCohere2MTPGreedy(ctx, runner, draft)
	if err != nil {
		t.Fatal(err)
	}
	if verification.Accepted < 0 || verification.Accepted > len(draft.Tokens) ||
		verification.Session.Position != session.Position+uint32(verification.Accepted)+1 ||
		verification.Session.TrunkCache.Position != verification.Session.Position {
		t.Fatalf("unexpected Cohere2-MoE MTP verification: draft=%+v result=%+v", draft, verification)
	}
	draftSampler, err := sampling.New(sampling.Config{
		Temperature: 0.8, TopK: 32, TopP: 0.95, Seed: 17,
	})
	if err != nil {
		t.Fatal(err)
	}
	targetSampler, err := sampling.New(sampling.Config{
		Temperature: 0.8, TopK: 32, TopP: 0.95, Seed: 23,
	})
	if err != nil {
		t.Fatal(err)
	}
	draftSamplerBefore, err := draftSampler.SaveState()
	if err != nil {
		t.Fatal(err)
	}
	sampledDraft, err := runner.DraftCohere2MTPSampled(
		ctx, session, draftSampler, []tokenizer.TokenID{0}, 2, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	draftSamplerAfter, err := draftSampler.SaveState()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(draftSamplerBefore, draftSamplerAfter) {
		t.Fatal("Cohere2-MoE sampled drafting changed caller sampler state")
	}
	sampledVerification, err := runner.VerifyCohere2MTPSampled(
		ctx, runner, sampledDraft, draftSampler, targetSampler,
	)
	if err != nil {
		t.Fatal(err)
	}
	if sampledVerification.Accepted < 0 || sampledVerification.Accepted > len(sampledDraft.Tokens) ||
		sampledVerification.Session.Position != session.Position+uint32(sampledVerification.Accepted)+1 ||
		sampledVerification.Session.TrunkCache.Position != sampledVerification.Session.Position {
		t.Fatalf("unexpected sampled Cohere2-MoE MTP verification: draft=%+v result=%+v", sampledDraft, sampledVerification)
	}
}

func TestNativeQ1BonsaiMatchesPinnedOracle(t *testing.T) {
	modelPath := os.Getenv("LLAMACPP2GO_BONSAI_MODEL")
	if modelPath == "" {
		t.Skip("LLAMACPP2GO_BONSAI_MODEL is not set")
	}
	runner, err := OpenWithOptions(modelPath, OpenOptions{
		DeviceOrdinal:           0,
		PreloadQuantizedWeights: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	if runner.rawWeights == nil {
		t.Fatal("Bonsai Q1_0 weights did not use native device storage")
	}
	ids, text, err := runner.Greedy(context.Background(), "Hello", 2)
	if err != nil {
		t.Fatal(err)
	}
	wantIDs := []tokenizer.TokenID{9419, 11, 353}
	if !slices.Equal(ids, wantIDs) || text != "Hello, I" {
		t.Fatalf(
			"Bonsai Q1_0 generation = %v %q, want %v %q",
			ids,
			text,
			wantIDs,
			"Hello, I",
		)
	}
}

func TestNativeGemma3PerplexityMatchesOracle(t *testing.T) {
	modelPath := os.Getenv("LLAMACPP2GO_GEMMA3_MODEL")
	if modelPath == "" {
		t.Skip("LLAMACPP2GO_GEMMA3_MODEL is not set")
	}
	runner, err := OpenWithOptions(modelPath, OpenOptions{
		DeviceOrdinal:           0,
		PreloadQuantizedWeights: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	chatPrompt, err := runner.FormatChat([]ChatMessage{
		{Role: "system", Content: "Be concise."},
		{Role: "user", Content: "Hello"},
	})
	if err != nil {
		t.Fatal(err)
	}
	const oracleChatPrompt = "<start_of_turn>user\nBe concise.\n\nHello" +
		"<end_of_turn>\n<start_of_turn>model\n"
	if chatPrompt != oracleChatPrompt {
		t.Fatalf("Gemma 3 chat prompt = %q, want %q", chatPrompt, oracleChatPrompt)
	}
	ids, text, err := runner.Greedy(context.Background(), "Hello", 3)
	if err != nil {
		t.Fatal(err)
	}
	wantIDs := []tokenizer.TokenID{2, 9259, 255999, 1018, 3689}
	if !slices.Equal(ids, wantIDs) || text != "Hello**What" {
		t.Fatalf(
			"Gemma 3 generation = %v %q, want %v %q",
			ids,
			text,
			wantIDs,
			"Hello**What",
		)
	}
	probe := strings.Repeat(
		"The quick brown fox jumps over the lazy dog. Numbers 1 2 3, punctuation, "+
			"and varied words make this a useful language model test. ",
		4,
	)
	result, err := runner.PerplexityWithOptions(
		context.Background(),
		probe,
		PerplexityOptions{ContextSize: 32},
	)
	if err != nil {
		t.Fatal(err)
	}
	const oracle = 33.1217
	relative := math.Abs(result.Perplexity-oracle) / oracle
	if relative > 0.005 {
		t.Fatalf(
			"Gemma 3 perplexity = %.7f, oracle %.7f, relative delta %.4f",
			result.Perplexity,
			oracle,
			relative,
		)
	}
}

func TestNativeUMT5EncoderMatchesOracle(t *testing.T) {
	modelPath := os.Getenv("LLAMACPP2GO_UMT5_MODEL")
	if modelPath == "" {
		t.Skip("LLAMACPP2GO_UMT5_MODEL is not set")
	}
	runner, err := OpenWithOptions(modelPath, OpenOptions{
		DeviceOrdinal:           0,
		PreloadQuantizedWeights: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	ids, err := runner.Vocab().Encode(
		"Hello world!",
		tokenizer.EncodeOptions{AddSpecial: true},
	)
	if err != nil {
		t.Fatal(err)
	}
	wantIDs := []tokenizer.TokenID{23231, 3914, 332, 1}
	if !slices.Equal(ids, wantIDs) {
		t.Fatalf("UMT5 token IDs = %v, want %v", ids, wantIDs)
	}
	hidden, err := runner.Forward(context.Background(), ids)
	if err != nil {
		t.Fatal(err)
	}
	width := int(hidden.Shape.Dims[0])
	tokens := int(hidden.Shape.Dims[1])
	mean := make([]float64, width)
	for token := range tokens {
		for channel := range width {
			mean[channel] += float64(hidden.Data[token*width+channel]) / float64(tokens)
		}
	}
	oraclePrefix := []float64{
		0.0009833, -0.0196109, -0.0288261, 0.0080970,
		0.0191293, 0.0153297, 0.0128889, 0.0485319,
		-0.0267938, -0.0074385, -0.0000551, 0.0276135,
		0.0758618, -0.0077735, 0.0066627, -0.0122798,
	}
	var prefixSquared float64
	for index, want := range oraclePrefix {
		difference := mean[index] - want
		prefixSquared += difference * difference
	}
	prefixRMSE := math.Sqrt(prefixSquared / float64(len(oraclePrefix)))
	if prefixRMSE > 0.0015 {
		t.Fatalf("UMT5 oracle-prefix RMSE = %g, want <= 0.0015", prefixRMSE)
	}
	var squaredNorm float64
	for _, value := range mean {
		squaredNorm += value * value
	}
	const oracleNorm = 2.41327351400548
	norm := math.Sqrt(squaredNorm)
	if relative := math.Abs(norm-oracleNorm) / oracleNorm; relative > 0.003 {
		t.Fatalf(
			"UMT5 pooled norm = %.9f, oracle %.9f, relative delta %.4f",
			norm,
			oracleNorm,
			relative,
		)
	}
	perToken, err := runner.EmbedTokensAdvanced(
		context.Background(),
		ids,
		EmbeddingOptions{Pooling: EmbeddingPoolingNone, Normalize: 2},
	)
	if err != nil {
		t.Fatal(err)
	}
	if perToken.Tokens != len(ids) ||
		len(perToken.Vectors) != len(ids) ||
		len(perToken.Vectors[0]) != width {
		t.Fatalf("UMT5 per-token embedding shape = tokens %d vectors %d x %d",
			perToken.Tokens,
			len(perToken.Vectors),
			len(perToken.Vectors[0]),
		)
	}
	for index := range min(16, width) {
		if difference := math.Abs(
			float64(perToken.Vectors[0][index] - hidden.Data[index]),
		); difference > 1e-6 {
			t.Fatalf("UMT5 per-token embedding[%d] delta = %g", index, difference)
		}
	}
	if _, _, err := runner.Greedy(context.Background(), "Hello", 1); err == nil {
		t.Fatal("UMT5 encoder unexpectedly accepted token generation")
	}
}

func assertNativeQuantGreedyOracle(t *testing.T, modelPath string) {
	t.Helper()
	runner, err := OpenWithOptions(modelPath, OpenOptions{
		DeviceOrdinal:           0,
		PreloadQuantizedWeights: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	beforeExecution, err := runner.DeviceExecutionStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	beforeMemory, err := runner.DeviceMemoryStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	ids, text, err := runner.Greedy(context.Background(), "Hello", 3)
	if err != nil {
		t.Fatal(err)
	}
	wantIDs := []tokenizer.TokenID{9707, 27, 18, 198}
	if !slices.Equal(ids, wantIDs) {
		t.Fatalf("token IDs = %v, want %v", ids, wantIDs)
	}
	if text != "Hello<3\n" {
		t.Fatalf("text = %q, want %q", text, "Hello<3\n")
	}
	afterExecution, err := runner.DeviceExecutionStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	afterMemory, err := runner.DeviceMemoryStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if afterExecution.StreamSynchronizations-beforeExecution.StreamSynchronizations != 3 ||
		afterExecution.HostToDeviceCopies-beforeExecution.HostToDeviceCopies != 6 ||
		afterExecution.DeviceToHostCopies-beforeExecution.DeviceToHostCopies != 3 {
		t.Fatalf(
			"dense device-cache execution delta before=%+v after=%+v",
			beforeExecution,
			afterExecution,
		)
	}
	if afterMemory.CurrentBytes != beforeMemory.CurrentBytes {
		t.Fatalf(
			"dense device-cache allocation leaked: before=%+v after=%+v",
			beforeMemory,
			afterMemory,
		)
	}
	tokenizedHello, err := runner.TokenizeText("Hello", false, true)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(tokenizedHello, []tokenizer.TokenID{9707}) {
		t.Fatalf("tokenized Hello = %v, want [9707]", tokenizedHello)
	}
	helloPiece, err := runner.TokenPiece(9707)
	if err != nil {
		t.Fatal(err)
	}
	if helloPiece != "Hello" {
		t.Fatalf("token piece 9707 = %q, want Hello", helloPiece)
	}
	exactSampler, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	probabilityEvents := 0
	exactIDs, exactText, err := runner.Generate(
		context.Background(),
		"ignored",
		GenerateOptions{
			MaxNewTokens:              3,
			Sampler:                   exactSampler,
			PromptTokenIDs:            []tokenizer.TokenID{9707},
			PostSamplingProbabilities: 3,
			OnToken: func(event TokenEvent) error {
				probabilityEvents++
				if event.SelectedProbability != 1 ||
					len(event.TopProbabilities) != 1 ||
					event.TopProbabilities[0].ID != int(event.ID) ||
					event.TopProbabilities[0].Probability != 1 {
					return fmt.Errorf(
						"post-sampling event = %+v",
						event,
					)
				}
				return nil
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(exactIDs, wantIDs) ||
		exactText != text ||
		probabilityEvents != 3 {
		t.Fatalf(
			"exact-token generation = %v %q, want %v %q",
			exactIDs,
			exactText,
			wantIDs,
			text,
		)
	}
	cacheSampler, err := sampling.New(sampling.Config{})
	if err != nil {
		t.Fatal(err)
	}
	var firstCacheEvaluation PromptEvaluation
	_, _, err = runner.Generate(context.Background(), "", GenerateOptions{
		MaxNewTokens:   1,
		Sampler:        cacheSampler,
		PromptTokenIDs: []tokenizer.TokenID{9707},
		CachePrompt:    true,
		OnPromptEvaluated: func(evaluation PromptEvaluation) {
			firstCacheEvaluation = evaluation
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	beforeReuse, err := runner.DeviceExecutionStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	reuseSampler, err := sampling.New(sampling.Config{})
	if err != nil {
		t.Fatal(err)
	}
	var reuseEvaluation PromptEvaluation
	reusedIDs, _, err := runner.Generate(context.Background(), "", GenerateOptions{
		MaxNewTokens:   1,
		Sampler:        reuseSampler,
		PromptTokenIDs: []tokenizer.TokenID{9707},
		CachePrompt:    true,
		MinCacheReuse:  1,
		OnPromptEvaluated: func(evaluation PromptEvaluation) {
			reuseEvaluation = evaluation
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	afterReuse, err := runner.DeviceExecutionStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if firstCacheEvaluation.Cached != 0 ||
		reuseEvaluation.Cached != 1 ||
		afterReuse.KernelLaunches != beforeReuse.KernelLaunches ||
		afterReuse.StreamSynchronizations != beforeReuse.StreamSynchronizations ||
		!slices.Equal(reusedIDs, []tokenizer.TokenID{9707, 27}) {
		t.Fatalf(
			"device prompt reuse first/reuse=%+v/%+v stats=%+v/%+v ids=%v",
			firstCacheEvaluation,
			reuseEvaluation,
			beforeReuse,
			afterReuse,
			reusedIDs,
		)
	}
	firstSampler, err := sampling.New(sampling.Config{})
	if err != nil {
		t.Fatal(err)
	}
	session, _, err := runner.StartSession(context.Background(), "Hello", GenerateOptions{
		MaxNewTokens: 1,
		Sampler:      firstSampler,
	})
	if err != nil {
		t.Fatal(err)
	}
	sessionData, err := runner.SaveSession(session, firstSampler)
	if err != nil {
		t.Fatal(err)
	}
	resumedSampler, err := sampling.New(sampling.Config{})
	if err != nil {
		t.Fatal(err)
	}
	session, err = runner.LoadSession(sessionData, resumedSampler)
	if err != nil {
		t.Fatal(err)
	}
	session, resumedText, err := runner.ContinueSession(
		context.Background(),
		session,
		GenerateOptions{MaxNewTokens: 2, Sampler: resumedSampler},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(session.TokenIDs, wantIDs) || resumedText != "Hello<3\n" {
		t.Fatalf("resumed generation = %v %q, want %v %q", session.TokenIDs, resumedText, wantIDs, "Hello<3\n")
	}
	dryBreakers, err := runner.TokenizeDryBreakers([]string{"\n", ":"})
	if err != nil {
		t.Fatal(err)
	}
	if len(dryBreakers) == 0 {
		t.Fatal("DRY breaker expansion produced no token sequences")
	}
	embedding, tokenCount, err := runner.Embed(context.Background(), "Hello")
	if err != nil {
		t.Fatal(err)
	}
	if len(embedding) != int(runner.Spec().EmbeddingLength) {
		t.Fatalf("embedding length = %d, want %d", len(embedding), runner.Spec().EmbeddingLength)
	}
	if tokenCount != 1 {
		t.Fatalf("embedding token count = %d, want 1", tokenCount)
	}
	exactEmbedding, exactTokenCount, err := runner.EmbedTokens(
		context.Background(),
		[]tokenizer.TokenID{9707},
	)
	if err != nil {
		t.Fatal(err)
	}
	if exactTokenCount != tokenCount || !slices.Equal(exactEmbedding, embedding) {
		t.Fatalf(
			"exact embedding count/equality = %d/%v, want %d/true",
			exactTokenCount,
			slices.Equal(exactEmbedding, embedding),
			tokenCount,
		)
	}
	var sumSquares float64
	for _, value := range embedding {
		sumSquares += float64(value) * float64(value)
	}
	if math.Abs(sumSquares-1) > 1e-5 {
		t.Fatalf("embedding squared norm = %g, want 1", sumSquares)
	}
	chatPrompt, err := runner.FormatChat([]ChatMessage{{Role: "user", Content: "Hello"}})
	if err != nil {
		t.Fatal(err)
	}
	chatPromptIDs, err := runner.Vocab().Encode(chatPrompt, tokenizer.EncodeOptions{
		AddSpecial:   true,
		ParseSpecial: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	countedChatIDs, err := runner.TokenizeText(chatPrompt, true, true)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(countedChatIDs, chatPromptIDs) {
		t.Fatalf("chat token count path IDs differ: %v vs %v", countedChatIDs, chatPromptIDs)
	}
	greedy, err := sampling.New(sampling.Config{})
	if err != nil {
		t.Fatal(err)
	}
	chatIDs, _, err := runner.Generate(context.Background(), chatPrompt, GenerateOptions{
		MaxNewTokens: 1,
		Sampler:      greedy,
		ParseSpecial: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(chatIDs) != len(chatPromptIDs)+1 {
		t.Fatalf("chat token count = %d, want prompt %d + 1", len(chatIDs), len(chatPromptIDs))
	}
}
