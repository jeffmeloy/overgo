package inference

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"sort"
	"strings"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tokenizer"
)

// DiffusionAlgorithm: mask-position ranking mode.
type DiffusionAlgorithm uint8

const (
	DiffusionOrigin DiffusionAlgorithm = iota
	DiffusionEntropy
	DiffusionMargin
	DiffusionRandom
	DiffusionConfidence
)

// DiffusionSchedule: mask-transfer schedule.
type DiffusionSchedule uint8

const (
	DiffusionTimestep DiffusionSchedule = iota
	DiffusionBlock
)

// DiffusionStep: pre-step token snapshot.
type DiffusionStep struct {
	Step       int
	TotalSteps int
	Tokens     []tokenizer.TokenID
}

// DiffusionOptions: iterative generation controls.
type DiffusionOptions struct {
	MaxLength            int
	Steps                int
	Temperature          float32
	TopK                 int
	TopP                 float32
	Seed                 int64
	Algorithm            DiffusionAlgorithm
	Schedule             DiffusionSchedule
	Epsilon              float32
	BlockLength          int
	CFGScale             float32
	AlgorithmTemperature float32
	AddGumbelNoise       bool
	ShiftLogits          *bool
	PromptTokenIDs       []tokenizer.TokenID
	OnStep               func(DiffusionStep) error
}

type diffusionEvaluator func(context.Context, []tokenizer.TokenID) ([]float32, error)

// GenerateDiffusion: iterative non-causal mask transfer.
func (r *Runner) GenerateDiffusion(
	ctx context.Context,
	prompt string,
	options DiffusionOptions,
) ([]tokenizer.TokenID, string, error) {
	if r == nil || r.vocab == nil {
		return nil, "", errors.New("inference: runner is nil")
	}
	if !diffusionArchitecture(r.spec.Architecture) {
		return nil, "", fmt.Errorf("inference: architecture %q is not a diffusion model", r.spec.Architecture)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, "", errors.New("inference: runner is closed")
	}
	for _, adapter := range r.loraAdapters {
		if adapter.scale != 0 && adapter.adapter != nil && len(adapter.adapter.InvocationTokens) != 0 {
			return nil, "", errors.New("inference: aLoRA is unsupported for non-causal diffusion")
		}
	}
	ids, err := r.diffusionPromptTokens(prompt, options.PromptTokenIDs)
	if err != nil {
		return nil, "", err
	}
	if r.vocab.Mask == tokenizer.NullToken {
		return nil, "", errors.New("inference: diffusion vocabulary has no mask token")
	}
	if options.MaxLength > int(r.spec.ContextLength) {
		return nil, "", fmt.Errorf(
			"inference: diffusion maximum length %d exceeds context length %d",
			options.MaxLength, r.spec.ContextLength,
		)
	}
	shift, err := r.diffusionShiftLogits(options.ShiftLogits)
	if err != nil {
		return nil, "", err
	}
	options.ShiftLogits = &shift
	evaluate := func(ctx context.Context, tokens []tokenizer.TokenID) ([]float32, error) {
		hidden, evalErr := r.forwardNonCausalLocked(ctx, tokens)
		if evalErr != nil {
			return nil, evalErr
		}
		logits, evalErr := r.projectAllLogits(ctx, hidden)
		if evalErr != nil {
			return nil, evalErr
		}
		return logits.Data, nil
	}
	output, err := runDiffusion(
		ctx, ids, r.vocab.Mask, r.vocab.Len(), options, evaluate,
	)
	if err != nil {
		return nil, "", err
	}
	generated := slices.Clone(output[len(ids):])
	text, err := r.vocab.Decode(generated, false)
	if err != nil {
		return nil, "", err
	}
	return generated, text, nil
}

func diffusionArchitecture(architecture string) bool {
	profile, ok := model.LookupArchitecture(architecture)
	return ok && profile.Has(model.ArchitectureDiffusion)
}

func (r *Runner) diffusionPromptTokens(
	prompt string,
	exact []tokenizer.TokenID,
) ([]tokenizer.TokenID, error) {
	if exact == nil {
		ids, err := r.vocab.Encode(prompt, tokenizer.EncodeOptions{
			AddSpecial: true, ParseSpecial: true,
		})
		if err != nil {
			return nil, err
		}
		if len(ids) == 0 {
			return nil, errors.New("inference: diffusion prompt produced no tokens")
		}
		return ids, nil
	}
	if len(exact) == 0 {
		return nil, errors.New("inference: exact diffusion prompt is empty")
	}
	ids := slices.Clone(exact)
	for index, id := range ids {
		if _, ok := r.vocab.Token(id); !ok {
			return nil, fmt.Errorf("inference: diffusion prompt token %d has out-of-range ID %d", index, id)
		}
	}
	return ids, nil
}

func (r *Runner) diffusionShiftLogits(override *bool) (bool, error) {
	if override != nil {
		return *override, nil
	}
	if r.file == nil {
		return true, nil
	}
	value, ok := r.file.MetadataValue("diffusion.shift_logits")
	if !ok {
		return true, nil
	}
	switch value.Type {
	case gguf.ValueTypeBool:
		shift, stored := value.Data.(bool)
		if !stored {
			return false, errors.New("inference: diffusion.shift_logits storage is invalid")
		}
		return shift, nil
	case gguf.ValueTypeString:
		raw, stored := value.Data.(string)
		if !stored {
			return false, errors.New("inference: diffusion.shift_logits storage is invalid")
		}
		switch strings.ToLower(strings.TrimSpace(raw)) {
		case "true":
			return true, nil
		case "false":
			return false, nil
		default:
			return false, fmt.Errorf("inference: diffusion.shift_logits value %q is invalid", raw)
		}
	default:
		return false, errors.New("inference: diffusion.shift_logits must be bool or string")
	}
}

func runDiffusion(
	ctx context.Context,
	input []tokenizer.TokenID,
	mask tokenizer.TokenID,
	vocabularySize int,
	options DiffusionOptions,
	evaluate diffusionEvaluator,
) ([]tokenizer.TokenID, error) {
	if err := validateDiffusion(input, mask, vocabularySize, options, evaluate); err != nil {
		return nil, err
	}
	output := make([]tokenizer.TokenID, options.MaxLength)
	copy(output, input)
	for index := len(input); index < len(output); index++ {
		output[index] = mask
	}
	sampler, err := newDiffusionSampler(options)
	if err != nil {
		return nil, err
	}
	rng := rand.New(rand.NewSource(options.Seed))
	selectionRNG := rand.New(rand.NewSource(options.Seed))
	numBlocks, stepsPerBlock := 1, options.Steps
	if options.Schedule == DiffusionBlock {
		numBlocks = options.MaxLength / options.BlockLength
		stepsPerBlock = options.Steps / numBlocks
	}
	shift := options.ShiftLogits != nil && *options.ShiftLogits
	for block := 0; block < numBlocks; block++ {
		blockStart, blockEnd := 0, options.MaxLength
		var transfers []int
		if options.Schedule == DiffusionBlock {
			blockStart = len(input) + block*options.BlockLength
			blockEnd = min(len(input)+(block+1)*options.BlockLength, options.MaxLength)
			masked := countMasks(output, mask, blockStart, blockEnd)
			transfers = diffusionBlockTransfers(masked, stepsPerBlock)
		}
		for step := 0; step < stepsPerBlock; step++ {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			globalStep := block*stepsPerBlock + step
			if options.OnStep != nil {
				snapshot := slices.Clone(output)
				if err := options.OnStep(DiffusionStep{
					Step: globalStep, TotalSteps: options.Steps, Tokens: snapshot,
				}); err != nil {
					return nil, err
				}
			}
			logits, err := diffusionLogits(ctx, output, input, mask, options, vocabularySize, evaluate)
			if err != nil {
				return nil, err
			}
			positions := maskedPositions(output, mask, options.Schedule, blockStart, blockEnd)
			if len(positions) == 0 {
				break
			}
			if options.AddGumbelNoise && options.Temperature > 0 {
				addDiffusionGumbelNoise(logits, options.Temperature, rng)
			}
			transferCount := diffusionTransferCount(
				step, stepsPerBlock, len(positions), options.Schedule, options.Epsilon, transfers,
			)
			if options.Algorithm == DiffusionOrigin {
				probability := float32(transferCount) / float32(len(positions))
				for _, position := range positions {
					if rng.Float32() >= probability {
						continue
					}
					row := diffusionLogitRow(logits, vocabularySize, position, shift)
					token, sampleErr := sampler.Sample(row)
					if sampleErr != nil {
						return nil, fmt.Errorf("inference: diffusion sample position %d: %w", position, sampleErr)
					}
					output[position] = tokenizer.TokenID(token)
				}
				continue
			}
			ranked := make([]diffusionCandidate, len(positions))
			for index, position := range positions {
				row := diffusionLogitRow(logits, vocabularySize, position, shift)
				limit := 2
				if options.Algorithm == DiffusionEntropy {
					limit = vocabularySize
				}
				result, sampleErr := sampler.SampleWithHistoryProbabilities(row, nil, limit)
				if sampleErr != nil {
					return nil, fmt.Errorf("inference: diffusion sample position %d: %w", position, sampleErr)
				}
				ranked[index] = diffusionCandidate{
					position:   position,
					token:      tokenizer.TokenID(result.Token),
					confidence: diffusionConfidence(result, options.Algorithm, rng),
					order:      index,
				}
			}
			selected := selectDiffusionCandidates(
				ranked, transferCount, options.AlgorithmTemperature, selectionRNG,
			)
			for _, candidate := range selected {
				output[candidate.position] = candidate.token
			}
		}
	}
	return output, nil
}

func validateDiffusion(
	input []tokenizer.TokenID,
	mask tokenizer.TokenID,
	vocabularySize int,
	options DiffusionOptions,
	evaluate diffusionEvaluator,
) error {
	if evaluate == nil {
		return errors.New("inference: diffusion evaluator is nil")
	}
	if len(input) == 0 {
		return errors.New("inference: diffusion input is empty")
	}
	if vocabularySize < 1 || mask < 0 || int(mask) >= vocabularySize {
		return errors.New("inference: diffusion mask or vocabulary is invalid")
	}
	for index, token := range input {
		if token < 0 || int(token) >= vocabularySize {
			return fmt.Errorf("inference: diffusion input token %d is out of range", index)
		}
	}
	if options.MaxLength <= len(input) {
		return errors.New("inference: diffusion maximum length must exceed prompt length")
	}
	if options.Steps <= 0 {
		return errors.New("inference: diffusion step count must be positive")
	}
	if options.Algorithm > DiffusionConfidence {
		return errors.New("inference: diffusion algorithm is invalid")
	}
	if options.Schedule > DiffusionBlock {
		return errors.New("inference: diffusion schedule is invalid")
	}
	if options.Temperature < 0 || !finiteDiffusion(options.Temperature) ||
		options.AlgorithmTemperature < 0 || !finiteDiffusion(options.AlgorithmTemperature) ||
		options.CFGScale < 0 || !finiteDiffusion(options.CFGScale) {
		return errors.New("inference: diffusion temperatures and CFG scale must be finite and non-negative")
	}
	if options.TopK < 0 {
		return errors.New("inference: diffusion top-k is negative")
	}
	if options.TopP < 0 || options.TopP > 1 || !finiteDiffusion(options.TopP) {
		return errors.New("inference: diffusion top-p must be in [0,1]")
	}
	if options.Schedule == DiffusionTimestep {
		if options.Epsilon <= 0 || options.Epsilon > 1 || !finiteDiffusion(options.Epsilon) {
			return errors.New("inference: timestep diffusion epsilon must be in (0,1]")
		}
		if options.BlockLength != 0 {
			return errors.New("inference: diffusion epsilon and block length are mutually exclusive")
		}
	} else {
		if options.Epsilon != 0 {
			return errors.New("inference: diffusion epsilon and block length are mutually exclusive")
		}
		if options.BlockLength <= 0 || options.MaxLength%options.BlockLength != 0 {
			return errors.New("inference: diffusion maximum length must be divisible by block length")
		}
		numBlocks := options.MaxLength / options.BlockLength
		if options.Steps%numBlocks != 0 {
			return errors.New("inference: diffusion steps must be divisible by block count")
		}
	}
	if uint64(options.MaxLength) > uint64(^uint(0)>>1)/uint64(vocabularySize) {
		return errors.New("inference: diffusion logits size overflows int")
	}
	return nil
}

func newDiffusionSampler(options DiffusionOptions) (*sampling.Sampler, error) {
	stages := make([]sampling.SamplerStage, 0, 3)
	if options.TopK > 0 {
		stages = append(stages, sampling.SamplerTopK)
	}
	if options.TopP > 0 && options.TopP < 1 {
		stages = append(stages, sampling.SamplerTopP)
	}
	if options.Temperature > 0 {
		stages = append(stages, sampling.SamplerTemperature)
	}
	topP := options.TopP
	if topP == 0 {
		topP = 1
	}
	return sampling.New(sampling.Config{
		Temperature: options.Temperature,
		TopK:        options.TopK,
		TopP:        topP,
		Seed:        options.Seed,
		Samplers:    stages,
	})
}

func diffusionLogits(
	ctx context.Context,
	conditional, prompt []tokenizer.TokenID,
	mask tokenizer.TokenID,
	options DiffusionOptions,
	vocabularySize int,
	evaluate diffusionEvaluator,
) ([]float32, error) {
	conditionalLogits, err := evaluate(ctx, conditional)
	if err != nil {
		return nil, fmt.Errorf("inference: diffusion conditional evaluation: %w", err)
	}
	expected := len(conditional) * vocabularySize
	if len(conditionalLogits) != expected {
		return nil, fmt.Errorf("inference: diffusion logits count %d differs from expected %d", len(conditionalLogits), expected)
	}
	logits := slices.Clone(conditionalLogits)
	if options.CFGScale == 0 {
		return logits, nil
	}
	unconditional := slices.Clone(conditional)
	for index := range prompt {
		unconditional[index] = mask
	}
	unconditionalLogits, err := evaluate(ctx, unconditional)
	if err != nil {
		return nil, fmt.Errorf("inference: diffusion unconditional evaluation: %w", err)
	}
	if len(unconditionalLogits) != expected {
		return nil, fmt.Errorf("inference: diffusion unconditional logits count %d differs from expected %d", len(unconditionalLogits), expected)
	}
	scale := options.CFGScale + 1
	for index := range logits {
		logits[index] = unconditionalLogits[index] + scale*(logits[index]-unconditionalLogits[index])
	}
	return logits, nil
}

func diffusionLogitRow(logits []float32, vocabularySize, position int, shift bool) []float32 {
	row := position
	if shift && row > 0 {
		row--
	}
	return logits[row*vocabularySize : (row+1)*vocabularySize]
}

func countMasks(tokens []tokenizer.TokenID, mask tokenizer.TokenID, start, end int) int {
	start = max(0, min(start, len(tokens)))
	end = max(start, min(end, len(tokens)))
	count := 0
	for _, token := range tokens[start:end] {
		if token == mask {
			count++
		}
	}
	return count
}

func maskedPositions(
	tokens []tokenizer.TokenID,
	mask tokenizer.TokenID,
	schedule DiffusionSchedule,
	blockStart, blockEnd int,
) []int {
	result := make([]int, 0, len(tokens))
	for index, token := range tokens {
		if token == mask && (schedule != DiffusionBlock || index >= blockStart && index < blockEnd) {
			result = append(result, index)
		}
	}
	return result
}

func diffusionTransferCount(
	step, totalSteps, remaining int,
	schedule DiffusionSchedule,
	epsilon float32,
	blockTransfers []int,
) int {
	if remaining <= 0 {
		return 0
	}
	var count int
	if schedule == DiffusionTimestep {
		t := 1 - float32(step)/float32(totalSteps)*(1-epsilon)
		s := 1 - float32(step+1)/float32(totalSteps)*(1-epsilon)
		probability := float32(1)
		if step < totalSteps-1 {
			probability = 1 - s/t
		}
		count = int(float32(remaining) * probability)
	} else if step < len(blockTransfers) {
		count = blockTransfers[step]
	} else {
		count = remaining / (totalSteps - step)
	}
	return max(0, min(count, remaining))
}

func diffusionBlockTransfers(maskCount, steps int) []int {
	result := make([]int, steps)
	base, remainder := maskCount/steps, maskCount%steps
	for index := range result {
		result[index] = base
		if index < remainder {
			result[index]++
		}
	}
	return result
}

func addDiffusionGumbelNoise(logits []float32, temperature float32, rng *rand.Rand) {
	if temperature == 0 {
		return
	}
	for index, logit := range logits {
		uniform := max(rng.Float64(), 1e-20)
		noise := math.Pow(-math.Log(uniform), float64(temperature))
		exponential := float32(math.Exp(float64(logit)))
		logits[index] = float32(float64(exponential) / noise)
	}
}

type diffusionCandidate struct {
	position   int
	token      tokenizer.TokenID
	confidence float64
	order      int
}

func diffusionConfidence(
	result sampling.SampleProbabilityResult,
	algorithm DiffusionAlgorithm,
	rng *rand.Rand,
) float64 {
	switch algorithm {
	case DiffusionEntropy:
		entropy := float64(0)
		for _, candidate := range result.Top {
			entropy -= candidate.Probability * math.Log(candidate.Probability+1e-10)
		}
		return entropy
	case DiffusionMargin:
		if len(result.Top) < 2 {
			if len(result.Top) == 1 {
				return result.Top[0].Probability
			}
			return 0
		}
		return result.Top[0].Probability - result.Top[1].Probability
	case DiffusionRandom:
		return rng.Float64()
	default:
		return result.SelectedProbability
	}
}

func selectDiffusionCandidates(
	candidates []diffusionCandidate,
	count int,
	temperature float32,
	rng *rand.Rand,
) []diffusionCandidate {
	count = min(max(count, 0), len(candidates))
	if count == 0 {
		return nil
	}
	remaining := slices.Clone(candidates)
	if temperature == 0 {
		sort.SliceStable(remaining, func(left, right int) bool {
			if remaining[left].confidence == remaining[right].confidence {
				return remaining[left].order < remaining[right].order
			}
			return remaining[left].confidence > remaining[right].confidence
		})
		return remaining[:count]
	}
	selected := make([]diffusionCandidate, 0, count)
	for len(selected) < count {
		maximum := remaining[0].confidence
		for _, candidate := range remaining[1:] {
			maximum = max(maximum, candidate.confidence)
		}
		weights := make([]float64, len(remaining))
		total := float64(0)
		for index, candidate := range remaining {
			weights[index] = math.Exp((candidate.confidence - maximum) / float64(temperature))
			total += weights[index]
		}
		threshold := rng.Float64() * total
		chosen := len(remaining) - 1
		for index, weight := range weights {
			threshold -= weight
			if threshold < 0 {
				chosen = index
				break
			}
		}
		selected = append(selected, remaining[chosen])
		remaining = append(remaining[:chosen], remaining[chosen+1:]...)
	}
	return selected
}

func finiteDiffusion(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}
