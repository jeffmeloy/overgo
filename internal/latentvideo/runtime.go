package latentvideo

import (
	"context"
	"crypto/sha256"
	"errors"
	"path/filepath"
	"sync"

	"overgo/internal/cuda/driver"
	"overgo/internal/pytorchzip"
)

// GeneratorConfig: fixed artifact, geometry, precision, and device binding.
type GeneratorConfig struct {
	ModelDirectory string
	Policy         DenoiserPolicy
	LatentStats    VAELatentStats
	Frames         int
	Width          int
	Height         int
	Precision      DenoiserPrecision
	DeviceOrdinal  int
}

// GenerateRequest: one conditioned trajectory and streamed clip publication.
type GenerateRequest struct {
	Steps         int
	Shift         float64
	GuideScale    float64
	CondContext   []float32
	UncondContext []float32
	InitialSample []float32
	Noise         NoisePlan
	Sink          VideoFrameSink
	StepHook      func(step int, timestep int64)
}

// GenerationResult: output plus cumulative resident-session evidence.
type GenerationResult struct {
	Denoise       DenoiseResult
	Decode        VAEDecodeStats
	DenoiseMemory driver.MemoryStats
	DecodeMemory  driver.MemoryStats
	Execution     driver.ExecutionStats
	WeightBytes   uint64
}

// Generator: one resident Wan denoiser and causal VAE pair.
type Generator struct {
	mu sync.Mutex

	geometry LatentGeometry
	stats    VAELatentStats
	denoiser *DenoiserCUDASession
	decoder  *VAEDecoderCUDASession
	branches map[[sha256.Size]byte]any
}

// NewGenerator: compile and upload both production stages once.
func NewGenerator(config GeneratorConfig) (generator *Generator, err error) {
	if config.ModelDirectory == "" {
		return nil, errors.New("latent video generator: model directory is empty")
	}
	denoiserConfig, err := LoadDenoiserConfig(config.ModelDirectory, config.Policy)
	if err != nil {
		return nil, err
	}
	weights, err := LoadDenoiserWeights(config.ModelDirectory, denoiserConfig)
	if err != nil {
		return nil, err
	}
	geometry, err := denoiserConfig.CompileLatentGeometry(config.Frames, config.Width, config.Height)
	if err != nil {
		return nil, err
	}
	program, err := CompileDenoiserProgramPrecision(denoiserConfig, weights, geometry, config.Precision)
	if err != nil {
		return nil, err
	}
	denoiser, err := NewDenoiserCUDASession(program, config.DeviceOrdinal)
	if err != nil {
		return nil, err
	}
	generator = &Generator{
		geometry: geometry, stats: config.LatentStats, denoiser: denoiser,
		branches: make(map[[sha256.Size]byte]any, 2),
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, generator.Close())
			generator = nil
		}
	}()
	checkpoint := filepath.Join(config.ModelDirectory, "Wan2.1_VAE.pth")
	metadata, err := pytorchzip.ReadTensorMetadata(checkpoint)
	if err != nil {
		return generator, err
	}
	plan, err := CompileVAEDecoderPlan(metadata)
	if err != nil {
		return generator, err
	}
	generator.decoder, err = NewVAEDecoderCUDASession(checkpoint, plan, config.DeviceOrdinal)
	return generator, err
}

// Geometry: immutable compiled latent geometry.
func (g *Generator) Geometry() LatentGeometry { return g.geometry }

// Generate: denoise then stream decoded frames; sessions remain resident.
func (g *Generator) Generate(ctx context.Context, request GenerateRequest) (result GenerationResult, err error) {
	if g == nil {
		return result, errors.New("latent video generator: closed")
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.denoiser == nil || g.decoder == nil {
		return result, errors.New("latent video generator: closed")
	}
	if request.Sink == nil {
		return result, errors.New("latent video generator: frame sink is nil")
	}

	if ctx != nil {
		g.denoiser.ctx = ctx
	}
	result.Denoise, err = g.denoiser.Program.DenoiseWithBackend(g, DenoiseRequest{
		Steps: request.Steps, Shift: request.Shift, GuideScale: request.GuideScale,
		CondContext: request.CondContext, UncondContext: request.UncondContext,
		InitialSample: request.InitialSample, Noise: request.Noise, StepHook: request.StepHook,
	})
	if err != nil {
		return result, err
	}
	result.DenoiseMemory, err = g.denoiser.MemoryStats()
	if err != nil {
		return result, err
	}
	result.Execution, err = g.denoiser.ExecutionStats()
	if err != nil {
		return result, err
	}
	result.Decode, err = g.decoder.Decode(
		g.stats, result.Denoise.Latent,
		g.geometry.LatentFrames, g.geometry.LatentHeight, g.geometry.LatentWidth,
		request.Sink,
	)
	if err != nil {
		return result, err
	}
	result.DecodeMemory, err = g.decoder.MemoryStats()
	result.WeightBytes = g.denoiser.WeightBytes + g.decoder.WeightBytes
	return result, err
}

// ProjectBranchContext: retain the current prompt pair across requests.
func (g *Generator) ProjectBranchContext(values []float32) (any, error) {
	key := sha256.Sum256(driver.Bytes(values))
	if branch, ok := g.branches[key]; ok {
		return branch, nil
	}
	if len(g.branches) == 2 {
		if err := g.denoiser.ReleaseRequestResources(); err != nil {
			return nil, err
		}
		clear(g.branches)
	}
	branch, err := g.denoiser.ProjectBranchContext(values)
	if err != nil {
		return nil, err
	}
	g.branches[key] = branch
	return branch, nil
}

// ForwardHead: delegate into the retained denoiser graph.
func (g *Generator) ForwardHead(patchTokens, blockE, headE []float32, branch any) ([]float32, error) {
	return g.denoiser.ForwardHead(patchTokens, blockE, headE, branch)
}

// Close: release both resident stages.
func (g *Generator) Close() error {
	if g == nil {
		return nil
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	var errs []error
	if g.decoder != nil {
		errs = append(errs, g.decoder.Close())
		g.decoder = nil
	}
	if g.denoiser != nil {
		errs = append(errs, g.denoiser.Close())
		g.denoiser = nil
	}
	clear(g.branches)
	return errors.Join(errs...)
}
