//go:build windows

package latentvideo

import (
	"context"
	"errors"
	"fmt"
	"time"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/media"
	"overgo/internal/pytorchzip"
	"overgo/internal/tensor"
)

type SourceEncodeStats struct {
	WallSeconds     float64
	WeightBytes     uint64
	PeakDeviceBytes uint64
	SourceChunks    int
	LatentFrames    int
	CacheHit        bool
}

// VAEEncoderCUDASession shares the resident causal-codec executor with decode.
type VAEEncoderCUDASession struct {
	Plan  VAEEncoderPlan
	codec *VAEDecoderCUDASession
}

func NewVAEEncoderCUDASession(checkpoint string, plan VAEEncoderPlan, ordinal int) (*VAEEncoderCUDASession, error) {
	codec, err := newVAECUDASession(checkpoint, plan.vaePlanCore, ordinal)
	if err != nil {
		return nil, err
	}
	return &VAEEncoderCUDASession{Plan: plan, codec: codec}, nil
}

func (s *VAEEncoderCUDASession) Encode(ctx context.Context, plan SourceCodecPlan, source []float32) ([]float32, SourceEncodeStats, error) {
	var stats SourceEncodeStats
	if s == nil || s.codec == nil {
		return nil, stats, errors.New("source codec CUDA session: closed")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	sourceElements, err := plan.SourceElements()
	if err != nil || !checked.Equal(len(source), sourceElements) || !checked.Equal(s.Plan.InputChannels, plan.Source.Channels) ||
		!checked.Equal(s.Plan.LatentChannels, plan.Latent.Channels) || !checked.Equal(s.Plan.Stride, plan.Profile.Stride) {
		return nil, stats, errors.New("source codec CUDA session: contract mismatch")
	}
	started := time.Now()
	s.codec.ctx = ctx
	states := make([]vaeDeviceOpState, len(s.Plan.Operations))
	latentElements, err := plan.LatentElements()
	if err != nil {
		return nil, stats, err
	}
	means := make([]float32, latentElements)
	sourceFrame, latentFrame := tensor.FirstOffset, tensor.FirstOffset
	sourceSpatial, ok := checked.MulInt(plan.Source.Height, plan.Source.Width)
	if !ok {
		return nil, stats, errors.New("source codec CUDA session: source geometry overflows")
	}
	latentSpatial, ok := checked.MulInt(plan.Latent.Height, plan.Latent.Width)
	if !ok {
		return nil, stats, errors.New("source codec CUDA session: latent geometry overflows")
	}
	for chunkIndex, chunkFrames := range plan.SourceChunks {
		staging := make([]float32, plan.Source.Channels*chunkFrames*sourceSpatial)
		if err := media.CopyPlanarFrames(staging, chunkFrames, tensor.FirstOffset, source, plan.Source.Frames, sourceFrame, plan.Source.Channels, chunkFrames, sourceSpatial); err != nil {
			return nil, stats, err
		}
		producedFrames := tensor.FirstOffset
		err = s.codec.worker.Do(ctx, func(state *device.State) error {
			x, allocErr := s.codec.buffer(state, media.ProgramCodecWorkspace(media.CodecWorkspaceActivation, tensor.FirstOffset), len(staging))
			if allocErr != nil {
				return allocErr
			}
			if copyErr := state.Driver.MemcpyHtoD(x, driver.Bytes(staging)); copyErr != nil {
				return copyErr
			}
			actIndex := tensor.FirstOffset
			volume, runErr := media.ExecuteCodecProgram("source codec CUDA", s.Plan.CodecProgram, states, media.CodecVolume[driver.DevicePtr]{
				Storage: x, Channels: plan.Source.Channels, Frames: chunkFrames, Height: plan.Source.Height, Width: plan.Source.Width,
			}, func(index int, operation media.CodecOperation[pytorchzip.TensorBinding], opState *vaeDeviceOpState, current media.CodecVolume[driver.DevicePtr]) (media.CodecVolume[driver.DevicePtr], error) {
				actIndex = tensor.SingletonExtent - actIndex
				next, frames, height, width, stepErr := s.codec.runOp(state, index, chunkIndex, operation, s.codec.weights[index], opState, current.Storage, media.ProgramCodecWorkspace(media.CodecWorkspaceActivation, actIndex), current.Frames, current.Height, current.Width)
				return media.CodecVolume[driver.DevicePtr]{Storage: next, Channels: operation.OutputChannels, Frames: frames, Height: height, Width: width}, stepErr
			})
			if runErr != nil {
				return fmt.Errorf("source codec CUDA chunk %d: %w", chunkIndex, runErr)
			}
			x, frames, height, width := volume.Storage, volume.Frames, volume.Height, volume.Width
			if !checked.Equal(height, plan.Latent.Height) || !checked.Equal(width, plan.Latent.Width) || !checked.PositiveInts(frames) {
				return fmt.Errorf("source codec CUDA chunk %d output=%dx%dx%d", chunkIndex, frames, height, width)
			}
			for channel := range plan.Latent.Channels {
				destination := means[(channel*plan.Latent.Frames+latentFrame)*latentSpatial:]
				sourcePointer := x + driver.DevicePtr(channel*frames*latentSpatial*binaryschema.Uint32Bytes)
				if copyErr := state.Driver.MemcpyDtoH(driver.Bytes(destination[:frames*latentSpatial]), sourcePointer); copyErr != nil {
					return copyErr
				}
			}
			producedFrames = frames
			return nil
		})
		if err != nil {
			return nil, stats, err
		}
		sourceFrame += chunkFrames
		latentFrame += producedFrames
	}
	if !checked.Equal(sourceFrame, plan.Source.Frames) || !checked.Equal(latentFrame, plan.Latent.Frames) {
		return nil, stats, fmt.Errorf("source codec CUDA consumed source=%d/%d latent=%d/%d", sourceFrame, plan.Source.Frames, latentFrame, plan.Latent.Frames)
	}
	latent := make([]float32, len(means))
	if err := NormalizeSourceLatent(latent, means, plan); err != nil {
		return nil, stats, err
	}
	stats = SourceEncodeStats{WallSeconds: time.Since(started).Seconds(), WeightBytes: s.codec.WeightBytes, SourceChunks: len(plan.SourceChunks), LatentFrames: latentFrame}
	if memory, memoryErr := s.codec.MemoryStats(); memoryErr == nil {
		stats.PeakDeviceBytes = memory.PeakBytes
	}
	return latent, stats, nil
}

func (s *VAEEncoderCUDASession) Close() error {
	if s == nil || s.codec == nil {
		return nil
	}
	err := s.codec.Close()
	s.codec = nil
	return err
}
