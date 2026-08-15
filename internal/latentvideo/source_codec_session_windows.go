//go:build windows

package latentvideo

import (
	"context"
	"errors"
	"fmt"
	"time"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

type SourceEncodeStats struct {
	WallSeconds     float64
	WeightBytes     uint64
	PeakDeviceBytes uint64
	SourceChunks    int
	LatentFrames    int
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
	if err != nil || len(source) != sourceElements || s.Plan.InputChannels != plan.Source.Channels ||
		s.Plan.LatentChannels != plan.Latent.Channels || s.Plan.Stride != plan.Profile.Stride {
		return nil, stats, errors.New("source codec CUDA session: contract mismatch")
	}
	started := time.Now()
	s.codec.ctx = ctx
	states := make([]vaeDeviceOpState, len(s.codec.ops))
	latentElements, err := plan.LatentElements()
	if err != nil {
		return nil, stats, err
	}
	means := make([]float32, latentElements)
	sourceFrame, latentFrame := 0, 0
	sourceSpatial := plan.Source.Height * plan.Source.Width
	latentSpatial := plan.Latent.Height * plan.Latent.Width
	for chunkIndex, chunkFrames := range plan.SourceChunks {
		staging := make([]float32, plan.Source.Channels*chunkFrames*sourceSpatial)
		if err := copyChannelFrames(staging, chunkFrames, 0, source, plan.Source.Frames, sourceFrame, plan.Source.Channels, chunkFrames, sourceSpatial); err != nil {
			return nil, stats, err
		}
		producedFrames := 0
		err = s.codec.worker.Do(ctx, func(state *device.State) error {
			x, allocErr := s.codec.buffer(state, "act_0", len(staging))
			if allocErr != nil {
				return allocErr
			}
			if copyErr := state.Driver.MemcpyHtoD(x, driver.Bytes(staging)); copyErr != nil {
				return copyErr
			}
			frames, height, width, actIndex := chunkFrames, plan.Source.Height, plan.Source.Width, 0
			for opIndex := range s.codec.ops {
				actIndex = 1 - actIndex
				x, frames, height, width, allocErr = s.codec.runOp(state, opIndex, chunkIndex, &states[opIndex], x, fmt.Sprintf("act_%d", actIndex), frames, height, width)
				if allocErr != nil {
					return fmt.Errorf("source codec CUDA %s chunk %d: %w", s.codec.ops[opIndex].prefix, chunkIndex, allocErr)
				}
			}
			if height != plan.Latent.Height || width != plan.Latent.Width || frames <= 0 {
				return fmt.Errorf("source codec CUDA chunk %d output=%dx%dx%d", chunkIndex, frames, height, width)
			}
			for channel := range plan.Latent.Channels {
				destination := means[(channel*plan.Latent.Frames+latentFrame)*latentSpatial:]
				sourcePointer := x + driver.DevicePtr(channel*frames*latentSpatial*4)
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
	if sourceFrame != plan.Source.Frames || latentFrame != plan.Latent.Frames {
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
