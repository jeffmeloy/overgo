//overgo:runtime-inputs caller

// Command generate runs local inference with text or projected media inputs.
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

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/clioptions"
	"overgo/internal/inference"
	"overgo/internal/model"
	"overgo/internal/modelcli"
	"overgo/internal/overgodb"
	"overgo/internal/projector"
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

// samplingOptions names the flag that overrides each field of the sampling policy.
//   - Samplers and DryBreakers have no row, because their flags parse lists
var samplingOptions = []struct {
	name, usage string
	field       func(*sampling.Policy) any
}{
	{"temp", "sampling temperature; zero is greedy", func(p *sampling.Policy) any { return &p.Temperature }},
	{"dynatemp-range", "dynamic temperature range; zero disables", func(p *sampling.Policy) any { return &p.DynatempRange }},
	{"dynatemp-exp", "entropy-to-temperature exponent", func(p *sampling.Policy) any { return &p.DynatempExponent }},
	{"top-k", "top-k candidates; zero disables", func(p *sampling.Policy) any { return &p.TopK }},
	{"top-p", "nucleus sampling probability", func(p *sampling.Policy) any { return &p.TopP }},
	{"min-p", "minimum probability relative to the most likely token; zero disables", func(p *sampling.Policy) any { return &p.MinP }},
	{"typical-p", "locally typical cumulative probability", func(p *sampling.Policy) any { return &p.TypicalP }},
	{"top-n-sigma", "keep logits within N standard deviations of the maximum; non-positive disables", func(p *sampling.Policy) any { return &p.TopNSigma }},
	{"xtc-probability", "chance of removing leading high-probability tokens", func(p *sampling.Policy) any { return &p.XTCProbability }},
	{"xtc-threshold", "XTC high-probability threshold; above 0.5 disables", func(p *sampling.Policy) any { return &p.XTCThreshold }},
	{"min-keep", "minimum candidates retained by probability filters", func(p *sampling.Policy) any { return &p.MinKeep }},
	{"adaptive-target", "adaptive-p target probability; negative disables", func(p *sampling.Policy) any { return &p.AdaptiveTarget }},
	{"adaptive-decay", "adaptive-p EMA decay", func(p *sampling.Policy) any { return &p.AdaptiveDecay }},
	{"repeat-last-n", "history tokens subject to penalties; -1 uses all, zero disables", func(p *sampling.Policy) any { return &p.RepeatLastN }},
	{"repeat-penalty", "multiplicative repetition penalty", func(p *sampling.Policy) any { return &p.RepeatPenalty }},
	{"presence-penalty", "penalty applied once to tokens in history", func(p *sampling.Policy) any { return &p.PresencePenalty }},
	{"frequency-penalty", "penalty applied per token occurrence in history", func(p *sampling.Policy) any { return &p.FrequencyPenalty }},
	{"no-repeat-ngram-size", "block repeated n-grams; zero disables", func(p *sampling.Policy) any { return &p.NoRepeatNgramSize }},
	{"ngram-window", "history window for no-repeat n-grams; zero uses all", func(p *sampling.Policy) any { return &p.NgramWindow }},
	{"dry-multiplier", "DRY repetition penalty multiplier; zero disables", func(p *sampling.Policy) any { return &p.DryMultiplier }},
	{"dry-base", "DRY exponential penalty base", func(p *sampling.Policy) any { return &p.DryBase }},
	{"dry-allowed-length", "repetition length allowed before DRY penalties", func(p *sampling.Policy) any { return &p.DryAllowedLength }},
	{"dry-penalty-last-n", "history tokens scanned by DRY; -1 uses all", func(p *sampling.Policy) any { return &p.DryPenaltyLastN }},
	{"mirostat", "Mirostat version; zero disables, 1 or 2 enables that version", func(p *sampling.Policy) any { return &p.Mirostat }},
	{"mirostat-tau", "Mirostat target surprise", func(p *sampling.Policy) any { return &p.MirostatTau }},
	{"mirostat-eta", "Mirostat learning rate", func(p *sampling.Policy) any { return &p.MirostatEta }},
	{"seed", "sampling RNG seed", func(p *sampling.Policy) any { return &p.Seed }},
}

// samplingOverrides defines one flag per row of samplingOptions and returns the function that writes each explicit flag's value into a policy.
//   - a float32 field's flag parses float64 and converts once, because package flag parses no float32
//   - an unset flag leaves its field at the policy's value
func samplingOverrides(flags *flag.FlagSet) func(clioptions.Overrides, *sampling.Policy) {
	writes := map[string]func(*sampling.Policy){}
	for _, option := range samplingOptions {
		switch field := option.field(&sampling.Policy{}).(type) {
		case *float32:
			value := clioptions.Float64Override(flags, option.name, option.usage)
			writes[option.name] = func(policy *sampling.Policy) { *option.field(policy).(*float32) = float32(*value) }
		case *int:
			flags.IntVar(field, option.name, *field, option.usage)
			writes[option.name] = func(policy *sampling.Policy) { *option.field(policy).(*int) = *field }
		case *int64:
			flags.Int64Var(field, option.name, *field, option.usage)
			writes[option.name] = func(policy *sampling.Policy) { *option.field(policy).(*int64) = *field }
		}
	}
	return func(explicit clioptions.Overrides, policy *sampling.Policy) {
		for name, write := range writes {
			if explicit.Has(name) {
				write(policy)
			}
		}
	}
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
	modelFlags := modelcli.AddModelFlags(flag.CommandLine, "load GGUF LoRA adapter at scale 1; repeatable")
	overrideSampling := samplingOverrides(flag.CommandLine)
	samplerNames := flag.String(
		"samplers",
		"",
		"ordered sampler names separated by semicolons; use none for an empty chain",
	)
	logitBiasValues := stringListFlag{}
	flag.Var(&logitBiasValues, "logit-bias", "TOKEN=BIAS logit adjustment; use -inf to ban, repeatable")
	ignoreEOS := flag.Bool("ignore-eos", false, "ban all recognized end-of-generation tokens")
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
	// The runner's policy carries the model's declared sampling, as the
	// server's does.
	policy := runner.RuntimePolicy().Interactive
	clioptions.ApplyDefault(explicit, "n", maxNewTokens, policy.OutputTokens)
	chosen := policy.Sampling
	overrideSampling(explicit, &chosen)
	clioptions.ApplyDefault(explicit, "video-max-frames", videoMaxFrames, policy.Video.MaxFrames)
	clioptions.ApplyDefault(explicit, "video-fps", videoFPS, policy.Video.FPS)
	if _, set := explicit["dry-sequence-breaker"]; !set {
		dryBreakers = append(dryBreakers, policy.Sampling.DryBreakers...)
	}
	var tokenBreakers [][]int
	if chosen.DryMultiplier != 0 && chosen.DryPenaltyLastN != 0 {
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
			bias, parseErr = strconv.ParseFloat(biasText, binaryschema.Width32Bits)
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
	// An unset -samplers runs the policy's chain as it is named, an omitted
	// one included; a set one, even none, is the chain.
	samplerOrder := policy.Sampling.Samplers
	if _, set := explicit["samplers"]; set {
		samplerOrder, err = sampling.ParseSamplerOrder(*samplerNames)
		if err != nil {
			return fmt.Errorf("generate: parse samplers: %w", err)
		}
	}
	var infillVocabulary *sampling.InfillVocabulary
	if slices.Contains(samplerOrder, sampling.SamplerInfill) {
		infillVocabulary, err = runner.SamplingInfillVocabulary()
		if err != nil {
			return fmt.Errorf("generate: load infill vocabulary: %w", err)
		}
	}
	config := chosen.Config()
	config.DryBreakers, config.Grammar, config.GBNF = tokenBreakers, grammar, gbnf
	config.Samplers, config.LogitBiases, config.Infill = samplerOrder, logitBiases, infillVocabulary
	sampler, err := sampling.New(config)
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
		repository, err := modelFlags.RepositoryPath()
		if err != nil {
			return err
		}
		store, err := overgodb.OpenReadOnly(repository)
		if err != nil {
			return fmt.Errorf("generate: open the projector's repository: %w", err)
		}
		defer store.Close()
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
		for raw := range strings.SplitSeq(*promptIDsFlag, ",") {
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
