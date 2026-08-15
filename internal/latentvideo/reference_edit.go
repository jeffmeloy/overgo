package latentvideo

import (
	"context"
	"errors"
	"fmt"
	"math"

	"overgo/internal/tensor/dtype"
)

type EditFlowConfig struct {
	InferenceSteps int
	TrainTimesteps int
	Shift          float32
	SigmaMin       float32
	SigmaMax       float32
	ExtraStep      bool
}

// CompileEditFlowSigmas: source-order LiveEdit FP32 schedule.
func CompileEditFlowSigmas(config EditFlowConfig, requested []int64) ([]float32, error) {
	if config.InferenceSteps <= 0 || config.TrainTimesteps <= 0 || config.Shift <= 0 ||
		config.SigmaMin < 0 || config.SigmaMax < config.SigmaMin || len(requested) == 0 {
		return nil, errors.New("reference edit flow: invalid config")
	}
	linspace := config.InferenceSteps
	if config.ExtraStep {
		linspace++
	}
	if linspace < 2 {
		return nil, errors.New("reference edit flow: insufficient schedule")
	}
	count := linspace
	if config.ExtraStep {
		count--
	}
	sigmas := make([]float32, count)
	timesteps := make([]float32, count)
	for index := range count {
		raw := float32(float64(config.SigmaMax) + float64(index)*float64(config.SigmaMin-config.SigmaMax)/float64(linspace-1))
		sigmas[index] = config.Shift * raw / (1 + (config.Shift-1)*raw)
		timesteps[index] = sigmas[index] * float32(config.TrainTimesteps)
	}
	selected := make([]float32, len(requested))
	for requestIndex, target := range requested {
		best := 0
		distance := math.Abs(float64(timesteps[0]) - float64(target))
		for index := 1; index < len(timesteps); index++ {
			candidate := math.Abs(float64(timesteps[index]) - float64(target))
			if candidate < distance {
				best, distance = index, candidate
			}
		}
		selected[requestIndex] = sigmas[best]
	}
	return selected, nil
}

type EditModelConfig struct {
	PatchSize     [3]int
	InputChannels int
	Dim           int
	TextLength    int
	Layers        int
	TrainSteps    int
}

type EditPlan struct {
	Latent               LatentGeometry
	SourceChannels       int
	InputChannels        int
	TokensPerFrame       int
	ChunkFrames          []int
	Timesteps            []int64
	SelfAttentionTokens  int
	CrossAttentionTokens int
	DenoiserCalls        int
	ContextRefreshCalls  int
}

// CompileEditPlan: shape-derived chunk and bounded-cache authority.
func CompileEditPlan(config EditModelConfig, latent LatentGeometry, sourceChannels, framesPerChunk, localAttentionFrames int, timesteps []int64) (EditPlan, error) {
	plan := EditPlan{Latent: latent, SourceChannels: sourceChannels}
	if latent.Channels <= 0 || latent.LatentFrames <= 0 || latent.LatentHeight <= 0 || latent.LatentWidth <= 0 ||
		sourceChannels <= 0 || config.InputChannels != latent.Channels+sourceChannels || framesPerChunk <= 0 ||
		config.PatchSize[0] != 1 || config.PatchSize[1] <= 0 || config.PatchSize[2] <= 0 ||
		latent.LatentHeight%config.PatchSize[1] != 0 || latent.LatentWidth%config.PatchSize[2] != 0 ||
		config.Dim <= 0 || config.TextLength <= 0 || config.Layers <= 0 || len(timesteps) == 0 {
		return plan, errors.New("reference edit plan: incomplete geometry")
	}
	for _, timestep := range timesteps {
		if timestep < 0 || config.TrainSteps > 0 && timestep > int64(config.TrainSteps) {
			return plan, fmt.Errorf("reference edit plan: timestep %d outside schedule", timestep)
		}
	}
	tokens, err := checkedProduct(latent.LatentHeight/config.PatchSize[1], latent.LatentWidth/config.PatchSize[2])
	if err != nil {
		return plan, fmt.Errorf("reference edit plan tokens: %w", err)
	}
	plan.InputChannels, plan.TokensPerFrame = config.InputChannels, tokens
	for remaining := latent.LatentFrames; remaining > 0; {
		frames := min(framesPerChunk, remaining)
		plan.ChunkFrames = append(plan.ChunkFrames, frames)
		remaining -= frames
	}
	cacheFrames := latent.LatentFrames
	if localAttentionFrames > 0 {
		cacheFrames = min(cacheFrames, localAttentionFrames)
	}
	if cacheFrames < min(framesPerChunk, latent.LatentFrames) {
		return plan, errors.New("reference edit plan: cache smaller than chunk")
	}
	plan.SelfAttentionTokens, err = checkedProduct(cacheFrames, tokens)
	if err != nil {
		return plan, err
	}
	plan.CrossAttentionTokens = config.TextLength
	plan.Timesteps = append([]int64(nil), timesteps...)
	plan.ContextRefreshCalls = len(plan.ChunkFrames)
	plan.DenoiserCalls, err = checkedProduct(len(plan.ChunkFrames), len(timesteps)+1)
	return plan, err
}

type EditPass string

const (
	EditDenoise        EditPass = "denoise"
	EditContextRefresh EditPass = "context_refresh"
)

type EditStep struct {
	Pass           EditPass
	Chunk          int
	Step           int
	StartFrame     int
	Frames         int
	StartToken     int
	EndToken       int
	Timestep       int64
	AdvanceHistory bool
}

// RunEditSchedule: chunk-outer, timestep-inner LiveEdit order.
func RunEditSchedule(ctx context.Context, plan EditPlan, contextTimestep int64, visit func(context.Context, EditStep) error) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if visit == nil || contextTimestep < 0 || plan.TokensPerFrame <= 0 || len(plan.ChunkFrames) == 0 || len(plan.Timesteps) == 0 {
		return errors.New("reference edit schedule: incomplete contract")
	}
	startFrame, calls := 0, 0
	for chunk, frames := range plan.ChunkFrames {
		startToken := startFrame * plan.TokensPerFrame
		endToken := (startFrame + frames) * plan.TokensPerFrame
		for step, timestep := range plan.Timesteps {
			if err := ctx.Err(); err != nil {
				return err
			}
			if err := visit(ctx, EditStep{Pass: EditDenoise, Chunk: chunk, Step: step, StartFrame: startFrame, Frames: frames, StartToken: startToken, EndToken: endToken, Timestep: timestep, AdvanceHistory: step == 0}); err != nil {
				return err
			}
			calls++
		}
		if err := visit(ctx, EditStep{Pass: EditContextRefresh, Chunk: chunk, Step: len(plan.Timesteps), StartFrame: startFrame, Frames: frames, StartToken: startToken, EndToken: endToken, Timestep: contextTimestep}); err != nil {
			return err
		}
		calls++
		startFrame += frames
	}
	if calls != plan.DenoiserCalls || startFrame != plan.Latent.LatentFrames {
		return errors.New("reference edit schedule: execution count mismatch")
	}
	return nil
}

type EditArithmetic string

const (
	EditFP32 EditArithmetic = "fp32"
	EditBF16 EditArithmetic = "bf16"
)

type EditSamplerRequest struct {
	Plan            EditPlan
	InitialNoise    []float32
	Source          []float32
	Sigmas          []float32
	ContextTimestep int64
	Arithmetic      EditArithmetic
	Denoise         func(context.Context, EditStep, []float32, []float32) error
	Noise           func(context.Context, EditStep, []float32) error
	CompleteChunk   func(context.Context, EditStep, []float32, []float32) error
}

type EditSamplerStats struct {
	DenoiseCalls, ContextRefreshCalls, NoiseCalls, PeakScratchElements int
}

// RunEditSampler: bounded no-pruning reference trajectory.
func RunEditSampler(ctx context.Context, request EditSamplerRequest) ([]float32, EditSamplerStats, error) {
	var stats EditSamplerStats
	plan, latent := request.Plan, request.Plan.Latent
	fullElements, err := checkedProduct(latent.Channels, latent.LatentFrames, latent.LatentHeight, latent.LatentWidth)
	if err != nil {
		return nil, stats, err
	}
	sourceElements, err := checkedProduct(plan.SourceChannels, latent.LatentFrames, latent.LatentHeight, latent.LatentWidth)
	if err != nil || len(request.InitialNoise) != fullElements || len(request.Source) != sourceElements || len(request.Sigmas) != len(plan.Timesteps) || request.Denoise == nil || len(plan.Timesteps) > 1 && request.Noise == nil {
		return nil, stats, errors.New("reference edit sampler: incomplete request")
	}
	if err := quantizeEdit(nil, request.Arithmetic); err != nil {
		return nil, stats, err
	}
	maxFrames := 0
	for _, frames := range plan.ChunkFrames {
		maxFrames = max(maxFrames, frames)
	}
	spatial := latent.LatentHeight * latent.LatentWidth
	chunkElements := latent.Channels * maxFrames * spatial
	sourceChunkElements := plan.SourceChannels * maxFrames * spatial
	current := make([]float32, chunkElements)
	clean := make([]float32, chunkElements)
	work := make([]float32, chunkElements)
	combined := make([]float32, chunkElements+sourceChunkElements)
	stats.PeakScratchElements = len(current) + len(clean) + len(work) + len(combined)
	output := make([]float32, fullElements)
	activeChunk := -1
	err = RunEditSchedule(ctx, plan, request.ContextTimestep, func(ctx context.Context, step EditStep) error {
		elements := latent.Channels * step.Frames * spatial
		sourceElements := plan.SourceChannels * step.Frames * spatial
		currentChunk, cleanChunk, workChunk := current[:elements], clean[:elements], work[:elements]
		combinedChunk := combined[:elements+sourceElements]
		if step.Chunk != activeChunk {
			if step.Pass != EditDenoise || step.Step != 0 {
				return errors.New("reference edit sampler: invalid chunk start")
			}
			if err := copyChannelFrames(currentChunk, step.Frames, 0, request.InitialNoise, latent.LatentFrames, step.StartFrame, latent.Channels, step.Frames, spatial); err != nil {
				return err
			}
			activeChunk = step.Chunk
		}
		copy(combinedChunk, currentChunk)
		sourceChunk := combinedChunk[elements:]
		if err := copyChannelFrames(sourceChunk, step.Frames, 0, request.Source, latent.LatentFrames, step.StartFrame, plan.SourceChannels, step.Frames, spatial); err != nil {
			return err
		}
		if err := request.Denoise(ctx, step, combinedChunk, workChunk); err != nil {
			return err
		}
		if step.Pass == EditContextRefresh {
			stats.ContextRefreshCalls++
			if request.CompleteChunk != nil {
				if err := request.CompleteChunk(ctx, step, currentChunk, sourceChunk); err != nil {
					return err
				}
			}
			return copyChannelFrames(output, latent.LatentFrames, step.StartFrame, currentChunk, step.Frames, 0, latent.Channels, step.Frames, spatial)
		}
		stats.DenoiseCalls++
		if err := quantizeEdit(workChunk, request.Arithmetic); err != nil {
			return err
		}
		for index := range cleanChunk {
			cleanChunk[index] = float32(float64(currentChunk[index]) - float64(request.Sigmas[step.Step])*float64(workChunk[index]))
		}
		if step.Step+1 == len(plan.Timesteps) {
			copy(currentChunk, cleanChunk)
			return nil
		}
		if err := request.Noise(ctx, step, workChunk); err != nil {
			return err
		}
		stats.NoiseCalls++
		nextSigma := request.Sigmas[step.Step+1]
		for index := range currentChunk {
			currentChunk[index] = (1-nextSigma)*cleanChunk[index] + nextSigma*workChunk[index]
		}
		return quantizeEdit(currentChunk, request.Arithmetic)
	})
	return output, stats, err
}

func quantizeEdit(values []float32, arithmetic EditArithmetic) error {
	switch arithmetic {
	case EditFP32:
		return nil
	case EditBF16:
		dtype.RoundBF16Slice(values)
		return nil
	default:
		return fmt.Errorf("reference edit sampler: unsupported arithmetic %q", arithmetic)
	}
}

func copyChannelFrames(destination []float32, destinationFrames, destinationStart int, source []float32, sourceFrames, sourceStart, channels, frames, spatial int) error {
	if destinationStart < 0 || sourceStart < 0 || frames < 0 || destinationStart+frames > destinationFrames || sourceStart+frames > sourceFrames || len(destination) < channels*destinationFrames*spatial || len(source) < channels*sourceFrames*spatial {
		return errors.New("reference edit sampler: frame copy outside tensor")
	}
	for channel := range channels {
		destinationOffset := (channel*destinationFrames + destinationStart) * spatial
		sourceOffset := (channel*sourceFrames + sourceStart) * spatial
		copy(destination[destinationOffset:destinationOffset+frames*spatial], source[sourceOffset:sourceOffset+frames*spatial])
	}
	return nil
}

func checkedProduct(values ...int) (int, error) {
	product := 1
	for _, value := range values {
		if value < 0 || value != 0 && product > int(^uint(0)>>1)/value {
			return 0, errors.New("integer overflow")
		}
		product *= value
	}
	return product, nil
}
