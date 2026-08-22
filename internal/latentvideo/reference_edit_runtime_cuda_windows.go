//go:build windows

package latentvideo

import (
	"context"
	"crypto/sha256"
	"errors"
	"path/filepath"
	"sync"
	"time"

	"overgo/internal/checked"
	"overgo/internal/cuda/driver"
	"overgo/internal/pytorchzip"
	"overgo/internal/sampling"
	"overgo/internal/tensor"
)

// ReferenceEditRuntimeConfig binds one real LiveEdit artifact and geometry.
type ReferenceEditRuntimeConfig struct {
	WanDirectory, EditCheckpoint string
	Policy                       DenoiserPolicy
	LatentStats                  VAELatentStats
	Source                       SourceVideoShape
	FramesPerChunk               int
	LocalAttentionFrames         int
	Timesteps                    []int64
	Sigmas                       []float32
	ContextTimestep              int64
	Layers, DeviceOrdinal        int
	Seed                         int64
	TextContext                  []float32
}

type ReferenceEditRuntimeResult struct {
	Latent         []float32
	Sampler        EditSamplerStats
	Source         SourceEncodeStats
	Decode         VAEDecodeStats
	Denoiser       ReferenceEditDenoiserStats
	DenoiseWallSec float64
	DenoiseMemory  driver.MemoryStats
	DecoderMemory  driver.MemoryStats
}

// ReferenceEditRuntime composes shared codec, transformer, sampler, and decode
// owners. Host crossings are typed latent/patch publication boundaries.
type ReferenceEditRuntime struct {
	mu sync.Mutex

	config            ReferenceEditRuntimeConfig
	base              DenoiserConfig
	checkpoint        ReferenceEditCheckpoint
	sourcePlan        SourceCodecPlan
	editPlan          EditPlan
	encoder           *VAEEncoderCUDASession
	denoiser          *ReferenceEditDenoiserCUDASession
	decoder           *VAEDecoderCUDASession
	sourceFingerprint [sha256.Size]byte
	sourceLatent      []float32
	sourceStats       SourceEncodeStats
}

func NewReferenceEditRuntime(config ReferenceEditRuntimeConfig) (runtime *ReferenceEditRuntime, err error) {
	if config.WanDirectory == "" || config.EditCheckpoint == "" || len(config.TextContext) == 0 || len(config.Timesteps) == 0 || len(config.Sigmas) != len(config.Timesteps) {
		return nil, errors.New("reference edit runtime: incomplete config")
	}
	base, err := LoadDenoiserConfig(config.WanDirectory, config.Policy)
	if err != nil {
		return nil, err
	}
	checkpoint, err := CompileReferenceEditCheckpoint(config.EditCheckpoint, base)
	if err != nil {
		return nil, err
	}
	vaeCheckpoint := filepath.Join(config.WanDirectory, "Wan2.1_VAE.pth")
	catalog, err := pytorchzip.ReadCatalog(vaeCheckpoint)
	if err != nil {
		return nil, err
	}
	encoderGraph, err := CompileVAEEncoderPlan(catalog.Tensors)
	if err != nil {
		return nil, err
	}
	sourcePlan, err := CompileSourceCodecBoundary(SourceCodecProfile{
		InputChannels: encoderGraph.InputChannels, LatentChannels: encoderGraph.LatentChannels,
		Stride: encoderGraph.Stride, LatentStats: config.LatentStats,
	}, config.Source)
	if err != nil {
		return nil, err
	}
	latent := LatentGeometry{
		Channels: base.InDim, LatentFrames: sourcePlan.Latent.Frames,
		LatentHeight: sourcePlan.Latent.Height, LatentWidth: sourcePlan.Latent.Width,
	}
	latent.Grid = [3]int{
		latent.LatentFrames / checkpoint.Config.PatchSize[0],
		latent.LatentHeight / checkpoint.Config.PatchSize[1],
		latent.LatentWidth / checkpoint.Config.PatchSize[2],
	}
	latent.Seq = latent.Grid[0] * latent.Grid[1] * latent.Grid[2]
	if latent.LatentFrames > config.LocalAttentionFrames+config.FramesPerChunk {
		return nil, errors.New("reference edit runtime: sliding history window is not compiled")
	}
	editPlan, err := CompileEditPlan(EditModelConfig{
		PatchSize: checkpoint.Config.PatchSize, InputChannels: checkpoint.Config.InDim,
		Dim: checkpoint.Config.Dim, TextLength: checkpoint.Config.TextLen,
		Layers: checkpoint.Config.NumLayers, TrainSteps: checkpoint.Config.Policy.NumTrainTimesteps,
	}, latent, checkpoint.SourceChannels, config.FramesPerChunk, config.LocalAttentionFrames, config.Timesteps)
	if err != nil {
		return nil, err
	}
	for _, frames := range editPlan.ChunkFrames {
		if frames != config.FramesPerChunk {
			return nil, errors.New("reference edit runtime: ragged final chunk is not compiled")
		}
	}
	chunkGeometry := LatentGeometry{
		Channels: checkpoint.Config.InDim, LatentFrames: config.FramesPerChunk,
		LatentHeight: latent.LatentHeight, LatentWidth: latent.LatentWidth,
		Grid: [3]int{config.FramesPerChunk / checkpoint.Config.PatchSize[0], latent.Grid[1], latent.Grid[2]},
	}
	chunkGeometry.Seq = chunkGeometry.Grid[0] * chunkGeometry.Grid[1] * chunkGeometry.Grid[2]
	runtime = &ReferenceEditRuntime{
		config: config, base: base, checkpoint: checkpoint, sourcePlan: sourcePlan, editPlan: editPlan,
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, runtime.Close())
			runtime = nil
		}
	}()
	runtime.encoder, err = NewVAEEncoderCUDASession(vaeCheckpoint, encoderGraph, config.DeviceOrdinal)
	if err == nil {
		runtime.denoiser, err = NewReferenceEditDenoiserCUDASession(
			context.Background(), checkpoint, chunkGeometry, config.Layers,
			config.FramesPerChunk, config.LocalAttentionFrames, latent.LatentFrames, config.DeviceOrdinal, config.TextContext,
		)
	}
	if err != nil {
		return runtime, err
	}
	decoderPlan, err := CompileVAEDecoderPlan(catalog.Tensors)
	if err != nil {
		return runtime, err
	}
	runtime.decoder, err = NewVAEDecoderCUDASession(vaeCheckpoint, decoderPlan, config.DeviceOrdinal)
	return runtime, err
}

func (r *ReferenceEditRuntime) Run(ctx context.Context, source, initialNoise []float32, sink VideoFrameSink) (result ReferenceEditRuntimeResult, err error) {
	if r == nil {
		return result, errors.New("reference edit runtime: closed")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.encoder == nil || r.denoiser == nil || r.decoder == nil || sink == nil {
		return result, errors.New("reference edit runtime: closed or missing sink")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := r.denoiser.ResetHistory(); err != nil {
		return result, err
	}
	noise, err := newEditNoiseStream(r.config.Seed, r.config.DeviceOrdinal)
	if err != nil {
		return result, err
	}
	if checked.Empty(initialNoise) {
		initialElements, ok := checked.ProductInt(r.editPlan.Latent.Channels, r.editPlan.Latent.LatentFrames, r.editPlan.Latent.LatentHeight, r.editPlan.Latent.LatentWidth)
		spatial, spatialOK := checked.MulInt(r.editPlan.Latent.LatentHeight, r.editPlan.Latent.LatentWidth)
		if !ok || !spatialOK {
			return result, errors.New("reference edit runtime: initial noise geometry overflows")
		}
		initialNoise = make([]float32, initialElements)
		if err := noise.FillChannelMajor(initialNoise, r.editPlan.Latent.Channels, r.editPlan.Latent.LatentFrames, spatial); err != nil {
			return result, err
		}
	} else if err := noise.Advance(len(initialNoise)); err != nil {
		return result, err
	}
	fingerprint := sha256.Sum256(driver.Bytes(source))
	sourceLatent, sourceStats := r.sourceLatent, r.sourceStats
	if sourceLatent == nil || fingerprint != r.sourceFingerprint {
		sourceLatent, sourceStats, err = r.encoder.Encode(ctx, r.sourcePlan, source)
		if err != nil {
			return result, err
		}
		r.sourceFingerprint, r.sourceLatent, r.sourceStats = fingerprint, sourceLatent, sourceStats
	} else {
		sourceStats.CacheHit = true
		sourceStats.WallSeconds = float64(tensor.FirstOffset)
	}
	result.Source = sourceStats
	denoiseStarted := time.Now()
	result.Latent, result.Sampler, err = RunEditSampler(ctx, EditSamplerRequest{
		Plan: r.editPlan, InitialNoise: initialNoise, Source: sourceLatent,
		Sigmas: r.config.Sigmas, ContextTimestep: r.config.ContextTimestep, Arithmetic: EditBF16,
		Denoise: func(_ context.Context, step EditStep, combined, flow []float32) error {
			headE, blockE, err := CompileTimestepConditioning([]float64{float64(step.Timestep)}, r.denoiser.cold.timestepWeights)
			if err != nil {
				return err
			}
			patches, err := r.denoiser.cold.PatchifyLatent(combined)
			if err != nil {
				return err
			}
			head, err := r.denoiser.RunChunk(patches, blockE, headE, step.StartFrame, step.Pass == EditContextRefresh)
			if err != nil {
				return err
			}
			decoded, err := r.denoiser.cold.UnpatchifyLatent(head)
			if err != nil {
				return err
			}
			copy(flow, decoded)
			return nil
		},
		Noise: func(_ context.Context, step EditStep, destination []float32) error {
			return noise.FillChannelMajor(destination, r.editPlan.Latent.Channels, step.Frames, r.editPlan.Latent.LatentHeight*r.editPlan.Latent.LatentWidth)
		},
	})
	result.DenoiseWallSec = time.Since(denoiseStarted).Seconds()
	if err != nil {
		return result, err
	}
	result.Denoiser = r.denoiser.Stats()
	result.DenoiseMemory, err = r.denoiser.MemoryStats()
	if err != nil {
		return result, err
	}
	result.Decode, err = r.decoder.Decode(
		r.config.LatentStats, result.Latent,
		r.editPlan.Latent.LatentFrames, r.editPlan.Latent.LatentHeight, r.editPlan.Latent.LatentWidth, sink,
	)
	if err == nil {
		result.DecoderMemory, err = r.decoder.MemoryStats()
	}
	return result, err
}

type editNoiseStream struct {
	seed, offset          uint64
	smCount, threadsPerSM int
}

func newEditNoiseStream(seed int64, deviceOrdinal int) (editNoiseStream, error) {
	library, err := driver.Open()
	if err != nil {
		return editNoiseStream{}, err
	}
	defer library.Close()
	if err := library.Init(); err != nil {
		return editNoiseStream{}, err
	}
	smCount, threadsPerSM, err := library.DeviceProfile(driver.Device(deviceOrdinal))
	return editNoiseStream{seed: uint64(seed), smCount: smCount, threadsPerSM: threadsPerSM}, err
}

func (s *editNoiseStream) Advance(elements int) error {
	_, advance, err := s.plan(elements)
	if err == nil {
		s.offset += advance
	}
	return err
}

func (s *editNoiseStream) FillChannelMajor(destination []float32, channels, frames, spatial int) error {
	if err := checked.Length(destination, channels, frames, spatial); err != nil {
		return errors.New("reference edit noise: channel-major storage differs")
	}
	plan, advance, err := s.plan(len(destination))
	if err != nil {
		return err
	}
	frameMajor := make([]float32, len(destination))
	if err := sampling.FillCounterNormalNoise(frameMajor, plan); err != nil {
		return err
	}
	for frame := range frames {
		for channel := range channels {
			from := (frame*channels + channel) * spatial
			to := (channel*frames + frame) * spatial
			copy(destination[to:to+spatial], frameMajor[from:from+spatial])
		}
	}
	s.offset += advance
	return nil
}

func (s *editNoiseStream) plan(elements int) (sampling.CounterNoisePlan, uint64, error) {
	return sampling.CompileCounterNoisePlan(s.seed, s.offset, elements, s.smCount, s.threadsPerSM)
}

func (r *ReferenceEditRuntime) Close() error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	var errs []error
	if r.decoder != nil {
		errs = append(errs, r.decoder.Close())
		r.decoder = nil
	}
	if r.denoiser != nil {
		errs = append(errs, r.denoiser.Close())
		r.denoiser = nil
	}
	if r.encoder != nil {
		errs = append(errs, r.encoder.Close())
		r.encoder = nil
	}
	return errors.Join(errs...)
}

func (r *ReferenceEditRuntime) Plan() EditPlan {
	if r == nil {
		return EditPlan{}
	}
	return r.editPlan
}
