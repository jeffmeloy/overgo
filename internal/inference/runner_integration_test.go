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
