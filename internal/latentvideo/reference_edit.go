package latentvideo

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/checked"
	"overgo/internal/media"
	"overgo/internal/sampling"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

type EditFlowConfig = sampling.EditFlowConfig

// CompileEditFlowSigmas exposes the shared edit-flow schedule compiler.
var CompileEditFlowSigmas = sampling.CompileEditFlowSigmas

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
	if !checked.PositiveInts(latent.Channels, latent.LatentFrames, latent.LatentHeight, latent.LatentWidth, sourceChannels, framesPerChunk) {
		return plan, errors.New("reference edit plan: incomplete geometry")
	}
	inputChannels, ok := checked.AddInt(latent.Channels, sourceChannels)
	if !ok {
		return plan, errors.New("reference edit plan: incomplete geometry")
	}
	if !checked.Equal(config.InputChannels, inputChannels) {
		return plan, errors.New("reference edit plan: incomplete geometry")
	}
	if !checked.Equal(config.PatchSize[0], tensor.SingletonExtent) {
		return plan, errors.New("reference edit plan: incomplete geometry")
	}
	volume := media.VolumeGeometry{Frames: latent.LatentFrames, Height: latent.LatentHeight, Width: latent.LatentWidth}
	grid, err := media.VolumePatchGrid(volume, config.PatchSize)
	if err != nil {
		return plan, fmt.Errorf("reference edit plan: %w", err)
	}
	if !checked.PositiveInts(config.Dim, config.TextLength, config.Layers) {
		return plan, errors.New("reference edit plan: incomplete geometry")
	}
	if _, ok := checked.First(timesteps); !ok {
		return plan, errors.New("reference edit plan: incomplete geometry")
	}
	for _, timestep := range timesteps {
		if !checked.NonNegativeInt64(timestep) || checked.PositiveInts(config.TrainSteps) && !checked.AtMostInt64(timestep, int64(config.TrainSteps)) {
			return plan, fmt.Errorf("reference edit plan: timestep %d outside schedule", timestep)
		}
	}
	tokens, err := checkedProduct(grid[tensor.SingletonExtent], grid[tensor.PairedExtent])
	if err != nil {
		return plan, fmt.Errorf("reference edit plan tokens: %w", err)
	}
	plan.InputChannels, plan.TokensPerFrame = config.InputChannels, tokens
	for remaining := latent.LatentFrames; checked.PositiveInts(remaining); {
		frames := min(framesPerChunk, remaining)
		plan.ChunkFrames = append(plan.ChunkFrames, frames)
		remaining -= frames
	}
	cacheFrames := latent.LatentFrames
	if checked.PositiveInts(localAttentionFrames) {
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
	if visit == nil || !checked.NonNegativeInt64(contextTimestep) || !checked.PositiveInts(plan.TokensPerFrame) ||
		!checked.Nonempty(plan.ChunkFrames) || !checked.Nonempty(plan.Timesteps) {
		return errors.New("reference edit schedule: incomplete contract")
	}
	startFrame, calls := tensor.FirstOffset, tensor.FirstOffset
	for chunk, frames := range plan.ChunkFrames {
		startToken := startFrame * plan.TokensPerFrame
		endToken := (startFrame + frames) * plan.TokensPerFrame
		for step, timestep := range plan.Timesteps {
			if err := ctx.Err(); err != nil {
				return err
			}
			if err := visit(ctx, EditStep{Pass: EditDenoise, Chunk: chunk, Step: step, StartFrame: startFrame, Frames: frames, StartToken: startToken, EndToken: endToken, Timestep: timestep, AdvanceHistory: checked.Equal(step, tensor.FirstOffset)}); err != nil {
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
	if !checked.Equal(calls, plan.DenoiserCalls) || !checked.Equal(startFrame, plan.Latent.LatentFrames) {
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
	if err != nil || !checked.Equal(len(request.InitialNoise), fullElements) || !checked.Equal(len(request.Source), sourceElements) ||
		!checked.Equal(len(request.Sigmas), len(plan.Timesteps)) || request.Denoise == nil || checked.Multiple(len(plan.Timesteps)) && request.Noise == nil {
		return nil, stats, errors.New("reference edit sampler: incomplete request")
	}
	if err := quantizeEdit(nil, request.Arithmetic); err != nil {
		return nil, stats, err
	}
	maxFrames := tensor.FirstOffset
	for _, frames := range plan.ChunkFrames {
		maxFrames = max(maxFrames, frames)
	}
	spatial, err := checkedProduct(latent.LatentHeight, latent.LatentWidth)
	if err != nil {
		return nil, stats, err
	}
	chunkElements, err := checkedProduct(latent.Channels, maxFrames, spatial)
	if err != nil {
		return nil, stats, err
	}
	sourceChunkElements, err := checkedProduct(plan.SourceChannels, maxFrames, spatial)
	if err != nil {
		return nil, stats, err
	}
	current := make([]float32, chunkElements)
	clean := make([]float32, chunkElements)
	work := make([]float32, chunkElements)
	combined := make([]float32, chunkElements+sourceChunkElements)
	stats.PeakScratchElements = len(current) + len(clean) + len(work) + len(combined)
	output := make([]float32, fullElements)
	activeChunk := checked.UnknownCount()
	lastStep := checked.ReverseIndex(tensor.FirstOffset, len(plan.Timesteps))
	err = RunEditSchedule(ctx, plan, request.ContextTimestep, func(ctx context.Context, step EditStep) error {
		elements, productErr := checkedProduct(latent.Channels, step.Frames, spatial)
		if productErr != nil {
			return productErr
		}
		sourceElements, productErr := checkedProduct(plan.SourceChannels, step.Frames, spatial)
		if productErr != nil {
			return productErr
		}
		currentChunk, cleanChunk, workChunk := current[:elements], clean[:elements], work[:elements]
		combinedChunk := combined[:elements+sourceElements]
		if step.Chunk != activeChunk {
			if step.Pass != EditDenoise || !checked.Equal(step.Step, tensor.FirstOffset) {
				return errors.New("reference edit sampler: invalid chunk start")
			}
			if err := media.CopyPlanarFrames(currentChunk, step.Frames, tensor.FirstOffset, request.InitialNoise, latent.LatentFrames, step.StartFrame, latent.Channels, step.Frames, spatial); err != nil {
				return err
			}
			activeChunk = step.Chunk
		}
		copy(combinedChunk, currentChunk)
		sourceChunk := combinedChunk[elements:]
		if err := media.CopyPlanarFrames(sourceChunk, step.Frames, tensor.FirstOffset, request.Source, latent.LatentFrames, step.StartFrame, plan.SourceChannels, step.Frames, spatial); err != nil {
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
			return media.CopyPlanarFrames(output, latent.LatentFrames, step.StartFrame, currentChunk, step.Frames, tensor.FirstOffset, latent.Channels, step.Frames, spatial)
		}
		stats.DenoiseCalls++
		if err := quantizeEdit(workChunk, request.Arithmetic); err != nil {
			return err
		}
		for index := range cleanChunk {
			cleanChunk[index] = float32(float64(currentChunk[index]) - float64(request.Sigmas[step.Step])*float64(workChunk[index]))
		}
		if checked.Equal(step.Step, lastStep) {
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

func checkedProduct(values ...int) (int, error) {
	product, ok := checked.ProductInt(values...)
	if !ok {
		return 0, errors.New("integer overflow")
	}
	return product, nil
}
