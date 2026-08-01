package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"

	"llamacpp2go/internal/inference"
	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tokenizer"
)

type stringListFlag []string

func (values *stringListFlag) String() string {
	return strings.Join(*values, ",")
}

func (values *stringListFlag) Set(value string) error {
	if value == "none" {
		*values = nil
		return nil
	}
	*values = append(*values, value)
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	maxNewTokens := flag.Int("n", 1, "maximum number of new tokens")
	contextShift := flag.Bool(
		"context-shift",
		false,
		"discard oldest attention KV entries to continue beyond model context",
	)
	keepTokens := flag.Int(
		"keep",
		0,
		"initial prompt tokens preserved by context shift; -1 keeps as many as possible",
	)
	discardTokens := flag.Int(
		"discard",
		0,
		"tokens removed per context shift; zero removes half of the discardable cache",
	)
	deviceOrdinal := flag.Int("device", 0, "CUDA device ordinal")
	preload := flag.Bool("preload", false, "dequantize all model weights once into CUDA memory")
	nativeQ8 := flag.Bool("native-q8", false, "preload Q8_0 weights without dequantizing them")
	nativeQuant := flag.Bool("native-quant", false, "preload supported quantized weights without dequantizing them")
	var loraPaths []string
	flag.Func("lora", "load GGUF LoRA adapter at scale 1; repeatable", func(value string) error {
		if strings.TrimSpace(value) == "" {
			return errors.New("LoRA path is empty")
		}
		loraPaths = append(loraPaths, value)
		return nil
	})
	temperature := flag.Float64("temp", 0, "sampling temperature; zero is greedy")
	dynatempRange := flag.Float64("dynatemp-range", 0, "dynamic temperature range; zero disables")
	dynatempExponent := flag.Float64("dynatemp-exp", 1, "entropy-to-temperature exponent")
	samplerNames := flag.String(
		"samplers",
		"penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature",
		"ordered sampler names separated by semicolons; use none for an empty chain",
	)
	topK := flag.Int("top-k", 40, "top-k candidates; zero disables")
	topP := flag.Float64("top-p", 0.95, "nucleus sampling probability")
	minP := flag.Float64("min-p", 0, "minimum probability relative to the most likely token; zero disables")
	typicalP := flag.Float64("typical-p", 1, "locally typical cumulative probability")
	topNSigma := flag.Float64("top-n-sigma", -1, "keep logits within N standard deviations of the maximum; non-positive disables")
	xtcProbability := flag.Float64("xtc-probability", 0, "chance of removing leading high-probability tokens")
	xtcThreshold := flag.Float64("xtc-threshold", 0.1, "XTC high-probability threshold; above 0.5 disables")
	minKeep := flag.Int("min-keep", 0, "minimum candidates retained by probability filters")
	adaptiveTarget := flag.Float64("adaptive-target", -1, "adaptive-p target probability; negative disables")
	adaptiveDecay := flag.Float64("adaptive-decay", 0.9, "adaptive-p EMA decay")
	logitBiasValues := stringListFlag{}
	flag.Var(&logitBiasValues, "logit-bias", "TOKEN=BIAS logit adjustment; use -inf to ban, repeatable")
	ignoreEOS := flag.Bool("ignore-eos", false, "ban all recognized end-of-generation tokens")
	repeatLastN := flag.Int("repeat-last-n", 0, "history tokens subject to penalties; -1 uses all, zero disables")
	repeatPenalty := flag.Float64("repeat-penalty", 1, "multiplicative repetition penalty")
	presencePenalty := flag.Float64("presence-penalty", 0, "penalty applied once to tokens in history")
	frequencyPenalty := flag.Float64("frequency-penalty", 0, "penalty applied per token occurrence in history")
	dryMultiplier := flag.Float64("dry-multiplier", 0, "DRY repetition penalty multiplier; zero disables")
	dryBase := flag.Float64("dry-base", 1.75, "DRY exponential penalty base")
	dryAllowedLength := flag.Int("dry-allowed-length", 2, "repetition length allowed before DRY penalties")
	dryPenaltyLastN := flag.Int("dry-penalty-last-n", -1, "history tokens scanned by DRY; -1 uses all")
	dryBreakers := stringListFlag{"\n", ":", "\"", "*"}
	flag.Var(&dryBreakers, "dry-sequence-breaker", "DRY restart string; repeat to add, or use 'none' first to clear defaults")
	grammarChoices := stringListFlag{}
	flag.Var(&grammarChoices, "grammar-choice", "exact allowed completion; repeat to add alternatives")
	grammarSource := flag.String("grammar", "", "GBNF source constraining generated text")
	grammarFile := flag.String("grammar-file", "", "read GBNF source from this file")
	grammarRoot := flag.String("grammar-root", "root", "GBNF root rule")
	grammarLazy := flag.Bool("grammar-lazy", false, "defer GBNF until a trigger matches")
	grammarTriggerPatterns := stringListFlag{}
	flag.Var(&grammarTriggerPatterns, "grammar-trigger-pattern", "lazy GBNF trigger regex; repeat to add")
	grammarTriggerTokenValues := stringListFlag{}
	flag.Var(&grammarTriggerTokenValues, "grammar-trigger-token", "lazy GBNF trigger token ID; repeat to add")
	mirostat := flag.Int("mirostat", 0, "Mirostat version; zero disables, 1 or 2 enables that version")
	mirostatTau := flag.Float64("mirostat-tau", 5, "Mirostat target surprise")
	mirostatEta := flag.Float64("mirostat-eta", 0.1, "Mirostat learning rate")
	seed := flag.Int64("seed", 0, "sampling RNG seed")
	projectedInputsFile := flag.String("projected-inputs", "", "projected multimodal input JSON")
	flag.Parse()
	if flag.NArg() != 2 {
		return errors.New("usage: generate [options] <model.gguf> <prompt>")
	}
	loraAdapters := make([]inference.LoRAConfig, len(loraPaths))
	for index, path := range loraPaths {
		loraAdapters[index] = inference.LoRAConfig{Path: path, Scale: 1}
	}
	runner, err := inference.OpenWithOptions(flag.Arg(0), inference.OpenOptions{
		DeviceOrdinal:           *deviceOrdinal,
		PreloadDeviceWeights:    *preload,
		PreloadQuantizedWeights: *nativeQ8 || *nativeQuant,
		LoRAAdapters:            loraAdapters,
	})
	if err != nil {
		return err
	}
	defer runner.Close()
	var tokenBreakers [][]int
	if *dryMultiplier != 0 && *dryPenaltyLastN != 0 {
		tokenBreakers, err = runner.TokenizeDryBreakers(dryBreakers)
		if err != nil {
			return err
		}
	}
	logitBiases := make([]sampling.LogitBias, 0, len(logitBiasValues))
	for _, raw := range logitBiasValues {
		tokenText, biasText, ok := strings.Cut(raw, "=")
		if !ok {
			return fmt.Errorf("generate: invalid logit bias %q; want TOKEN=BIAS", raw)
		}
		token, parseErr := strconv.ParseInt(tokenText, 10, 32)
		if parseErr != nil || token < 0 {
			return fmt.Errorf("generate: invalid logit-bias token %q", tokenText)
		}
		var bias float64
		if strings.EqualFold(biasText, "-inf") {
			bias = math.Inf(-1)
		} else {
			bias, parseErr = strconv.ParseFloat(biasText, 32)
			if parseErr != nil {
				return fmt.Errorf("generate: invalid logit bias %q", biasText)
			}
		}
		logitBiases = append(logitBiases, sampling.LogitBias{
			Token: int(token),
			Bias:  float32(bias),
		})
	}
	if *ignoreEOS {
		for _, token := range runner.SamplingEOGTokens() {
			logitBiases = append(logitBiases, sampling.LogitBias{
				Token: int(token),
				Bias:  float32(math.Inf(-1)),
			})
		}
	}
	var grammar *sampling.TokenGrammar
	if len(grammarChoices) > 0 {
		grammar, err = runner.TokenizeGrammarChoices(grammarChoices)
		if err != nil {
			return err
		}
	}
	if *grammarSource != "" && *grammarFile != "" {
		return errors.New("generate: -grammar and -grammar-file are mutually exclusive")
	}
	if len(grammarChoices) > 0 && (*grammarSource != "" || *grammarFile != "") {
		return errors.New("generate: exact grammar choices and GBNF are mutually exclusive")
	}
	source := *grammarSource
	if *grammarFile != "" {
		data, readErr := os.ReadFile(*grammarFile)
		if readErr != nil {
			return fmt.Errorf("generate: read grammar file: %w", readErr)
		}
		source = string(data)
	}
	var gbnf *sampling.GBNFGrammar
	if source != "" {
		triggerTokens := make([]tokenizer.TokenID, len(grammarTriggerTokenValues))
		for index, raw := range grammarTriggerTokenValues {
			value, parseErr := strconv.ParseInt(raw, 10, 32)
			if parseErr != nil || value < 0 {
				return fmt.Errorf("generate: invalid grammar trigger token %q", raw)
			}
			triggerTokens[index] = tokenizer.TokenID(value)
		}
		if *grammarLazy {
			gbnf, err = runner.CompileLazyGBNF(
				source,
				*grammarRoot,
				grammarTriggerPatterns,
				triggerTokens,
			)
		} else {
			if len(grammarTriggerPatterns) > 0 || len(triggerTokens) > 0 {
				return errors.New("generate: grammar triggers require -grammar-lazy")
			}
			gbnf, err = runner.CompileGBNF(source, *grammarRoot)
		}
		if err != nil {
			return err
		}
	} else if *grammarLazy ||
		len(grammarTriggerPatterns) > 0 ||
		len(grammarTriggerTokenValues) > 0 {
		return errors.New("generate: lazy grammar options require GBNF source")
	}
	samplerOrder, err := sampling.ParseSamplerOrder(*samplerNames)
	if err != nil {
		return fmt.Errorf("generate: parse samplers: %w", err)
	}
	var infillVocabulary *sampling.InfillVocabulary
	if slices.Contains(samplerOrder, sampling.SamplerInfill) {
		infillVocabulary, err = runner.SamplingInfillVocabulary()
		if err != nil {
			return fmt.Errorf("generate: load infill vocabulary: %w", err)
		}
	}
	sampler, err := sampling.New(sampling.Config{
		Temperature:      float32(*temperature),
		DynatempRange:    float32(*dynatempRange),
		DynatempExponent: float32(*dynatempExponent),
		TopK:             *topK,
		TopP:             float32(*topP),
		MinP:             float32(*minP),
		TypicalP:         float32(*typicalP),
		TopNSigma:        float32(*topNSigma),
		XTCProbability:   float32(*xtcProbability),
		XTCThreshold:     float32(*xtcThreshold),
		MinKeep:          *minKeep,
		AdaptiveTarget:   float32(*adaptiveTarget),
		AdaptiveDecay:    float32(*adaptiveDecay),
		RepeatLastN:      *repeatLastN,
		RepeatPenalty:    float32(*repeatPenalty),
		PresencePenalty:  float32(*presencePenalty),
		FrequencyPenalty: float32(*frequencyPenalty),
		DryMultiplier:    float32(*dryMultiplier),
		DryBase:          float32(*dryBase),
		DryAllowedLength: *dryAllowedLength,
		DryPenaltyLastN:  *dryPenaltyLastN,
		DryBreakers:      tokenBreakers,
		Mirostat:         *mirostat,
		MirostatTau:      float32(*mirostatTau),
		MirostatEta:      float32(*mirostatEta),
		Seed:             *seed,
		Grammar:          grammar,
		GBNF:             gbnf,
		Samplers:         samplerOrder,
		LogitBiases:      logitBiases,
		Infill:           infillVocabulary,
	})
	if err != nil {
		return err
	}
	options := inference.GenerateOptions{
		MaxNewTokens:  *maxNewTokens,
		Sampler:       sampler,
		ContextShift:  *contextShift,
		KeepTokens:    *keepTokens,
		DiscardTokens: *discardTokens,
	}
	if *projectedInputsFile != "" {
		projected, projectedErr := readProjectedInputs(*projectedInputsFile)
		if projectedErr != nil {
			return projectedErr
		}
		options.ProjectedInputs = &projected
	}
	var ids []tokenizer.TokenID
	var text string
	if runner.Spec().Architecture == "t5" {
		ids, text, _, err = runner.GenerateT5(context.Background(), flag.Arg(1), options)
	} else {
		ids, text, err = runner.Generate(context.Background(), flag.Arg(1), options)
	}
	if err != nil {
		return err
	}
	result := struct {
		IDs  []tokenizer.TokenID `json:"ids"`
		Text string              `json:"text"`
	}{
		IDs:  ids,
		Text: text,
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(result)
}
