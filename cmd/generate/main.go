package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/checked"
	"overgo/internal/clioptions"
	"overgo/internal/inference"
	"overgo/internal/model"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/projector"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
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
	clioptions.Main(run)
}

// topLogitIndices: descending top-n token ids by logit
func topLogitIndices(logits []float32, n int) []int {
	if n > len(logits) {
		n = len(logits)
	}
	top := make([]int, 0, n)
	for id, value := range logits {
		position := len(top)
		for position > 0 && value > logits[top[position-1]] {
			position--
		}
		if position >= n {
			continue
		}
		if len(top) < n {
			top = append(top, len(top))
		}
		copy(top[position+1:], top[position:])
		top[position] = id
	}
	return top
}

func run() error {
	maxNewTokens := clioptions.IntOverride(flag.CommandLine, "n", "maximum number of new tokens; unset uses recipe policy")
	contextShift := flag.Bool(
		"context-shift",
		false,
		"discard oldest attention KV entries to continue beyond model context",
	)
	keepTokens := clioptions.IntOverride(
		flag.CommandLine,
		"keep",
		"initial prompt tokens preserved by context shift; -1 keeps as many as possible",
	)
	discardTokens := clioptions.IntOverride(
		flag.CommandLine,
		"discard",
		"tokens removed per context shift; zero removes half of the discardable cache",
	)
	modelFlags := clioptions.AddModelFlags(flag.CommandLine, "load GGUF LoRA adapter at scale 1; repeatable")
	temperature := clioptions.Float64Override(flag.CommandLine, "temp", "sampling temperature; zero is greedy")
	dynatempRange := clioptions.Float64Override(flag.CommandLine, "dynatemp-range", "dynamic temperature range; zero disables")
	dynatempExponent := clioptions.Float64Override(flag.CommandLine, "dynatemp-exp", "entropy-to-temperature exponent")
	samplerNames := flag.String(
		"samplers",
		"",
		"ordered sampler names separated by semicolons; use none for an empty chain",
	)
	topK := clioptions.IntOverride(flag.CommandLine, "top-k", "top-k candidates; zero disables")
	topP := clioptions.Float64Override(flag.CommandLine, "top-p", "nucleus sampling probability")
	minP := clioptions.Float64Override(flag.CommandLine, "min-p", "minimum probability relative to the most likely token; zero disables")
	typicalP := clioptions.Float64Override(flag.CommandLine, "typical-p", "locally typical cumulative probability")
	topNSigma := clioptions.Float64Override(flag.CommandLine, "top-n-sigma", "keep logits within N standard deviations of the maximum; non-positive disables")
	xtcProbability := clioptions.Float64Override(flag.CommandLine, "xtc-probability", "chance of removing leading high-probability tokens")
	xtcThreshold := clioptions.Float64Override(flag.CommandLine, "xtc-threshold", "XTC high-probability threshold; above 0.5 disables")
	minKeep := clioptions.IntOverride(flag.CommandLine, "min-keep", "minimum candidates retained by probability filters")
	adaptiveTarget := clioptions.Float64Override(flag.CommandLine, "adaptive-target", "adaptive-p target probability; negative disables")
	adaptiveDecay := clioptions.Float64Override(flag.CommandLine, "adaptive-decay", "adaptive-p EMA decay")
	logitBiasValues := stringListFlag{}
	flag.Var(&logitBiasValues, "logit-bias", "TOKEN=BIAS logit adjustment; use -inf to ban, repeatable")
	ignoreEOS := flag.Bool("ignore-eos", false, "ban all recognized end-of-generation tokens")
	repeatLastN := clioptions.IntOverride(flag.CommandLine, "repeat-last-n", "history tokens subject to penalties; -1 uses all, zero disables")
	repeatPenalty := clioptions.Float64Override(flag.CommandLine, "repeat-penalty", "multiplicative repetition penalty")
	presencePenalty := clioptions.Float64Override(flag.CommandLine, "presence-penalty", "penalty applied once to tokens in history")
	frequencyPenalty := clioptions.Float64Override(flag.CommandLine, "frequency-penalty", "penalty applied per token occurrence in history")
	noRepeatNgramSize := clioptions.IntOverride(flag.CommandLine, "no-repeat-ngram-size", "block repeated n-grams; zero disables")
	ngramWindow := clioptions.IntOverride(flag.CommandLine, "ngram-window", "history window for no-repeat n-grams; zero uses all")
	dryMultiplier := clioptions.Float64Override(flag.CommandLine, "dry-multiplier", "DRY repetition penalty multiplier; zero disables")
	dryBase := clioptions.Float64Override(flag.CommandLine, "dry-base", "DRY exponential penalty base")
	dryAllowedLength := clioptions.IntOverride(flag.CommandLine, "dry-allowed-length", "repetition length allowed before DRY penalties")
	dryPenaltyLastN := clioptions.IntOverride(flag.CommandLine, "dry-penalty-last-n", "history tokens scanned by DRY; -1 uses all")
	dryBreakers := stringListFlag{}
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
	mirostat := clioptions.IntOverride(flag.CommandLine, "mirostat", "Mirostat version; zero disables, 1 or 2 enables that version")
	mirostatTau := clioptions.Float64Override(flag.CommandLine, "mirostat-tau", "Mirostat target surprise")
	mirostatEta := clioptions.Float64Override(flag.CommandLine, "mirostat-eta", "Mirostat learning rate")
	seed := clioptions.Int64Override(flag.CommandLine, "seed", "sampling RNG seed")
	projectedInputsFile := flag.String("projected-inputs", "", "projected multimodal input JSON")
	projectorPath := flag.String("mmproj", "", "multimodal projector GGUF")
	projectorCUDA := flag.Bool("mmproj-cuda", false, "offload supported multimodal projector operations to CUDA")
	imagePath := flag.String("image", "", "image input for multimodal generation")
	imageDynamicTiles := flag.Bool("image-dynamic-tiles", true, "enable dynamic local image tiles when supported")
	audioPath := flag.String("audio", "", "Gemma 4 mono 16 kHz WAV or raw float32-LE audio")
	videoFrames := stringListFlag{}
	flag.Var(&videoFrames, "video-frame", "ordered multimodal video frame; repeatable")
	videoPath := flag.String("video", "", "encoded multimodal video; GIF native, other formats through FFmpeg")
	videoMaxFrames := clioptions.IntOverride(flag.CommandLine, "video-max-frames", "maximum decoded video frames; unset uses recipe policy")
	videoFPS := clioptions.Float64Override(flag.CommandLine, "video-fps", "source FPS for multimodal video timestamps; unset uses recipe policy")
	ffmpegPath := flag.String("ffmpeg", os.Getenv("OVERGO_FFMPEG"), "FFmpeg executable for non-GIF video input")
	imageThinking := flag.Bool("image-thinking", true, "retain Qwen3.5 thinking preamble for image prompts")
	promptIDsFlag := flag.String("prompt-ids", "", "comma-separated prompt token IDs; bypasses tokenization (prompt argument optional)")
	debugTopLogits := clioptions.IntOverride(flag.CommandLine, "debug-top-logits", "print top-N (id, logit) pairs per generated step to stderr; zero disables")
	flag.Parse()
	explicit := clioptions.ExplicitOverrides(flag.CommandLine)
	if flag.NArg() != 2 && !(*promptIDsFlag != "" && flag.NArg() == 1) {
		return errors.New("usage: generate [options] <model.gguf> <prompt>  (prompt optional with -prompt-ids)")
	}
	args := flag.Args()
	prompt := ""
	if len(args) > 1 {
		prompt = args[1]
	}
	runner, err := modelFlags.OpenRunner(context.Background(), args[0])
	if err != nil {
		return err
	}
	defer runner.Close()
	repository, err := modelFlags.RepositoryPath()
	if err != nil {
		return err
	}
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return fmt.Errorf("generate: open model recipe repository: %w", err)
	}
	defer store.Close()
	execution, err := modelrecipe.ResolveActiveExecution(
		context.Background(), store, runner.ModelID(), recipe.TaskInference, modelrecipe.SessionWarm,
	)
	if err != nil {
		return fmt.Errorf("generate: resolve runtime policy: %w", err)
	}
	policy := execution.Policy.Interactive
	clioptions.ApplyDefault(explicit, "n", maxNewTokens, policy.OutputTokens)
	clioptions.ApplyDefault(explicit, "temp", temperature, float64(policy.Sampling.Temperature))
	clioptions.ApplyDefault(explicit, "dynatemp-range", dynatempRange, float64(policy.Sampling.DynatempRange))
	clioptions.ApplyDefault(explicit, "dynatemp-exp", dynatempExponent, float64(policy.Sampling.DynatempExponent))
	clioptions.ApplyDefault(explicit, "top-k", topK, policy.Sampling.TopK)
	clioptions.ApplyDefault(explicit, "top-p", topP, float64(policy.Sampling.TopP))
	clioptions.ApplyDefault(explicit, "min-p", minP, float64(policy.Sampling.MinP))
	clioptions.ApplyDefault(explicit, "typical-p", typicalP, float64(policy.Sampling.TypicalP))
	clioptions.ApplyDefault(explicit, "top-n-sigma", topNSigma, float64(policy.Sampling.TopNSigma))
	clioptions.ApplyDefault(explicit, "xtc-probability", xtcProbability, float64(policy.Sampling.XTCProbability))
	clioptions.ApplyDefault(explicit, "xtc-threshold", xtcThreshold, float64(policy.Sampling.XTCThreshold))
	clioptions.ApplyDefault(explicit, "min-keep", minKeep, policy.Sampling.MinKeep)
	clioptions.ApplyDefault(explicit, "adaptive-target", adaptiveTarget, float64(policy.Sampling.AdaptiveTarget))
	clioptions.ApplyDefault(explicit, "adaptive-decay", adaptiveDecay, float64(policy.Sampling.AdaptiveDecay))
	clioptions.ApplyDefault(explicit, "repeat-last-n", repeatLastN, policy.Sampling.RepeatLastN)
	clioptions.ApplyDefault(explicit, "repeat-penalty", repeatPenalty, float64(policy.Sampling.RepeatPenalty))
	clioptions.ApplyDefault(explicit, "presence-penalty", presencePenalty, float64(policy.Sampling.PresencePenalty))
	clioptions.ApplyDefault(explicit, "frequency-penalty", frequencyPenalty, float64(policy.Sampling.FrequencyPenalty))
	clioptions.ApplyDefault(explicit, "no-repeat-ngram-size", noRepeatNgramSize, policy.Sampling.NoRepeatNgramSize)
	clioptions.ApplyDefault(explicit, "ngram-window", ngramWindow, policy.Sampling.NgramWindow)
	clioptions.ApplyDefault(explicit, "dry-multiplier", dryMultiplier, float64(policy.Sampling.DryMultiplier))
	clioptions.ApplyDefault(explicit, "dry-base", dryBase, float64(policy.Sampling.DryBase))
	clioptions.ApplyDefault(explicit, "dry-allowed-length", dryAllowedLength, policy.Sampling.DryAllowedLength)
	clioptions.ApplyDefault(explicit, "dry-penalty-last-n", dryPenaltyLastN, policy.Sampling.DryPenaltyLastN)
	clioptions.ApplyDefault(explicit, "mirostat", mirostat, policy.Sampling.Mirostat)
	clioptions.ApplyDefault(explicit, "mirostat-tau", mirostatTau, float64(policy.Sampling.MirostatTau))
	clioptions.ApplyDefault(explicit, "mirostat-eta", mirostatEta, float64(policy.Sampling.MirostatEta))
	clioptions.ApplyDefault(explicit, "seed", seed, policy.Sampling.Seed)
	clioptions.ApplyDefault(explicit, "video-max-frames", videoMaxFrames, policy.Video.MaxFrames)
	clioptions.ApplyDefault(explicit, "video-fps", videoFPS, policy.Video.FPS)
	if _, set := explicit["samplers"]; !set {
		*samplerNames = sampling.FormatSamplerOrder(policy.Sampling.Samplers)
	}
	if _, set := explicit["dry-sequence-breaker"]; !set {
		dryBreakers = append(dryBreakers, policy.Sampling.DryBreakers...)
	}
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
		token, parseErr := tokenizer.ParseTokenID(tokenText)
		if parseErr != nil || token < 0 {
			return fmt.Errorf("generate: invalid logit-bias token %q", tokenText)
		}
		var bias float64
		if strings.EqualFold(biasText, "-inf") {
			bias = float64(sampling.BannedLogit())
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
				Bias:  sampling.BannedLogit(),
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
			value, parseErr := tokenizer.ParseTokenID(raw)
			if parseErr != nil || value < 0 {
				return fmt.Errorf("generate: invalid grammar trigger token %q", raw)
			}
			triggerTokens[index] = value
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
		Temperature:       float32(*temperature),
		DynatempRange:     float32(*dynatempRange),
		DynatempExponent:  float32(*dynatempExponent),
		TopK:              *topK,
		TopP:              float32(*topP),
		MinP:              float32(*minP),
		TypicalP:          float32(*typicalP),
		TopNSigma:         float32(*topNSigma),
		XTCProbability:    float32(*xtcProbability),
		XTCThreshold:      float32(*xtcThreshold),
		MinKeep:           *minKeep,
		AdaptiveTarget:    float32(*adaptiveTarget),
		AdaptiveDecay:     float32(*adaptiveDecay),
		RepeatLastN:       *repeatLastN,
		RepeatPenalty:     float32(*repeatPenalty),
		PresencePenalty:   float32(*presencePenalty),
		FrequencyPenalty:  float32(*frequencyPenalty),
		NoRepeatNgramSize: *noRepeatNgramSize,
		NgramWindow:       *ngramWindow,
		DryMultiplier:     float32(*dryMultiplier),
		DryBase:           float32(*dryBase),
		DryAllowedLength:  *dryAllowedLength,
		DryPenaltyLastN:   *dryPenaltyLastN,
		DryBreakers:       tokenBreakers,
		Mirostat:          *mirostat,
		MirostatTau:       float32(*mirostatTau),
		MirostatEta:       float32(*mirostatEta),
		Seed:              *seed,
		Grammar:           grammar,
		GBNF:              gbnf,
		Samplers:          samplerOrder,
		LogitBiases:       logitBiases,
		Infill:            infillVocabulary,
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
	if *debugTopLogits > 0 {
		topN := *debugTopLogits
		options.OnToken = func(event inference.TokenEvent) error {
			if len(event.Logits) == 0 {
				fmt.Fprintf(os.Stderr, "step %d: logits unavailable\n", event.Index)
				return nil
			}
			top := topLogitIndices(event.Logits, topN)
			fmt.Fprintf(os.Stderr, "step %d:", event.Index)
			for _, id := range top {
				fmt.Fprintf(os.Stderr, " %d=%.6f", id, event.Logits[id])
			}
			fmt.Fprintln(os.Stderr)
			return nil
		}
	}
	mediaInputs := 0
	if *imagePath != "" {
		mediaInputs++
	}
	if len(videoFrames) > 0 {
		mediaInputs++
	}
	if *videoPath != "" {
		mediaInputs++
	}
	if *audioPath != "" {
		mediaInputs++
	}
	if mediaInputs > 1 {
		return errors.New("generate: -image, -audio, -video, and -video-frame are mutually exclusive")
	}
	if (*projectorPath == "") != (mediaInputs == 0) {
		return errors.New("generate: -mmproj requires -image, -audio, -video, or at least one -video-frame")
	}
	if *videoPath != "" && *videoMaxFrames <= 0 {
		return errors.New("generate: -video-max-frames must be positive")
	}
	if (*videoPath != "" || len(videoFrames) > 0) && !checked.PositiveFinite64(*videoFPS) {
		return errors.New("generate: -video-fps must be finite and positive")
	}
	if *projectedInputsFile != "" && *projectorPath != "" {
		return errors.New("generate: -projected-inputs and -mmproj are mutually exclusive")
	}
	if *projectedInputsFile != "" {
		projected, projectedErr := readProjectedInputs(*projectedInputsFile)
		if projectedErr != nil {
			return projectedErr
		}
		options.ProjectedInputs = &projected
	}
	if mediaInputs > 0 {
		if runner.Spec().Profile().Forward.Session == model.ForwardSessionEncoderDecoder {
			return errors.New("generate: multimodal projection is unavailable for encoder-decoder programs")
		}
		var promptIDs []tokenizer.TokenID
		var projected inference.ProjectedInputs
		var projectedErr error
		projectorOptions := projector.OpenOptions{
			CUDA: *projectorCUDA, DeviceOrdinal: *modelFlags.DeviceOrdinal,
			DisableDynamicTiles: !*imageDynamicTiles,
		}
		if *imagePath != "" {
			promptIDs, projected, projectedErr = imageProjectedPrompt(
				context.Background(), store, runner, *projectorPath, *imagePath, prompt, *imageThinking,
				projectorOptions,
			)
		} else if *audioPath != "" {
			promptIDs, projected, projectedErr = audioProjectedPrompt(
				context.Background(), store, runner, *projectorPath, *audioPath, prompt,
				projectorOptions,
			)
		} else {
			promptIDs, projected, projectedErr = videoProjectedPrompt(
				context.Background(), store, runner, *projectorPath, videoFrames, *videoPath, *videoMaxFrames,
				*ffmpegPath, prompt, *videoFPS, *imageThinking,
				projectorOptions,
			)
		}
		if projectedErr != nil {
			return projectedErr
		}
		options.PromptTokenIDs = promptIDs
		options.ProjectedInputs = &projected
	}
	if *promptIDsFlag != "" {
		if mediaInputs > 0 {
			return errors.New("generate: -prompt-ids and media inputs are mutually exclusive")
		}
		for _, raw := range strings.Split(*promptIDsFlag, ",") {
			value, parseErr := tokenizer.ParseTokenID(strings.TrimSpace(raw))
			if parseErr != nil || value < 0 {
				return fmt.Errorf("generate: invalid prompt token id %q", raw)
			}
			options.PromptTokenIDs = append(options.PromptTokenIDs, value)
		}
	}
	var ids []tokenizer.TokenID
	var text string
	if runner.Spec().Profile().Forward.Session == model.ForwardSessionEncoderDecoder {
		ids, text, _, err = runner.GenerateEncoderDecoder(context.Background(), prompt, options)
	} else {
		ids, text, err = runner.Generate(context.Background(), prompt, options)
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
	return clioptions.WritePrettyJSON(os.Stdout, result)
}
