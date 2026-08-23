package latentvideo

import (
	"errors"
	"fmt"

	"overgo/internal/checked"
	"overgo/internal/media"
	"overgo/internal/pytorchzip"
	"overgo/internal/tensor"
)

// SourceCodecProfile: recipe-owned causal VAE boundary facts.
type SourceCodecProfile struct {
	InputChannels  int
	LatentChannels int
	Stride         [3]int
	LatentStats    VAELatentStats
}

// SourceVideoShape: planar [channel][frame][height][width] source extent.
type SourceVideoShape struct {
	Channels int
	Frames   int
	Height   int
	Width    int
}

type SourceCodecPlan struct {
	Source       SourceVideoShape
	Latent       SourceVideoShape
	SourceChunks []int
	Profile      SourceCodecProfile
}

// CompileSourceCodecBoundary derives the encoder's exact temporal and spatial boundary.
func CompileSourceCodecBoundary(profile SourceCodecProfile, source SourceVideoShape) (SourceCodecPlan, error) {
	if !checked.PositiveInts(profile.InputChannels, profile.LatentChannels) {
		return SourceCodecPlan{}, errors.New("source codec: invalid channel or source extent")
	}
	if !checked.Equal(source.Channels, profile.InputChannels) {
		return SourceCodecPlan{}, errors.New("source codec: invalid channel or source extent")
	}
	volume, err := media.DownsampledVolumeExactSpatial(source.Frames, source.Height, source.Width, profile.Stride)
	if err != nil {
		return SourceCodecPlan{}, fmt.Errorf("source codec: %w", err)
	}
	if err := media.ValidateChannelMoments(profile.LatentStats.Mean, profile.LatentStats.Std, profile.LatentChannels); err != nil {
		return SourceCodecPlan{}, fmt.Errorf("source codec: latent statistics: %w", err)
	}
	chunks := media.CausalChunkSchedule(source.Frames, profile.Stride[tensor.FirstOffset])
	return SourceCodecPlan{
		Source: source,
		Latent: SourceVideoShape{
			Channels: profile.LatentChannels,
			Frames:   volume.Frames,
			Height:   volume.Height,
			Width:    volume.Width,
		},
		SourceChunks: chunks,
		Profile:      profile,
	}, nil
}

// SourceElements returns the validated planar source size.
func (p SourceCodecPlan) SourceElements() (int, error) {
	return checkedProduct(p.Source.Channels, p.Source.Frames, p.Source.Height, p.Source.Width)
}

// LatentElements returns the validated channel-major latent size.
func (p SourceCodecPlan) LatentElements() (int, error) {
	return checkedProduct(p.Latent.Channels, p.Latent.Frames, p.Latent.Height, p.Latent.Width)
}

// NormalizeSourceLatent applies the adaptive encoder's channel-wise affine.
func NormalizeSourceLatent(out, meanOutput []float32, plan SourceCodecPlan) error {
	plane := plan.Latent.Frames * plan.Latent.Height * plan.Latent.Width
	return media.NormalizePlanarChannelsInto(out, meanOutput, plan.Profile.LatentStats.Mean, plan.Profile.LatentStats.Std, plan.Latent.Channels, plane)
}

// EncodeSourceVideo runs the checkpoint-derived causal encoder chunk stream.
func EncodeSourceVideo(checkpoint string, graph VAEEncoderPlan, plan SourceCodecPlan, source []float32) ([]float32, error) {
	sourceElements, err := plan.SourceElements()
	if err != nil {
		return nil, err
	}
	if !checked.Equal(len(source), sourceElements) {
		return nil, errors.New("source codec: encoder contract mismatch")
	}
	if _, ok := checked.First(graph.Operations); !ok {
		return nil, errors.New("source codec: encoder contract mismatch")
	}
	if !checked.Equal(graph.InputChannels, plan.Source.Channels) {
		return nil, errors.New("source codec: encoder contract mismatch")
	}
	if !checked.Equal(graph.LatentChannels, plan.Latent.Channels) {
		return nil, errors.New("source codec: encoder contract mismatch")
	}
	if !checked.Equal(graph.Stride, plan.Profile.Stride) {
		return nil, errors.New("source codec: encoder contract mismatch")
	}
	reader, err := pytorchzip.Open(checkpoint)
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	weights, err := loadVAEOps(reader, graph.vaePlanCore, "encoder")
	if err != nil {
		return nil, err
	}
	states := make([]vaeOpState, len(graph.Operations))
	latentElements, err := plan.LatentElements()
	if err != nil {
		return nil, err
	}
	means := make([]float32, latentElements)
	sourceFrame, latentFrame := tensor.FirstOffset, tensor.FirstOffset
	sourceSpatial, ok := checked.MulInt(plan.Source.Height, plan.Source.Width)
	if !ok {
		return nil, errors.New("source codec: source geometry overflows")
	}
	latentSpatial, ok := checked.MulInt(plan.Latent.Height, plan.Latent.Width)
	if !ok {
		return nil, errors.New("source codec: latent geometry overflows")
	}
	for chunkIndex, frames := range plan.SourceChunks {
		current := make([]float32, plan.Source.Channels*frames*sourceSpatial)
		if err := media.CopyPlanarFrames(current, frames, tensor.FirstOffset, source, plan.Source.Frames, sourceFrame, plan.Source.Channels, frames, sourceSpatial); err != nil {
			return nil, err
		}
		volume, err := media.ExecuteCodecProgram("source codec", graph.CodecProgram, states, media.CodecVolume[[]float32]{
			Storage: current, Channels: plan.Source.Channels, Frames: frames, Height: plan.Source.Height, Width: plan.Source.Width,
		}, func(index int, operation media.CodecOperation[[]pytorchzip.TensorBinding], state *vaeOpState, input media.CodecVolume[[]float32]) (media.CodecVolume[[]float32], error) {
			next, nextFrames, height, width, runErr := runVAEOp(operation, weights[index], state, chunkIndex, input.Storage, input.Frames, input.Height, input.Width)
			return media.CodecVolume[[]float32]{Storage: next, Channels: operation.OutputChannels, Frames: nextFrames, Height: height, Width: width}, runErr
		})
		if err != nil {
			return nil, fmt.Errorf("source codec chunk %d: %w", chunkIndex, err)
		}
		current, frames, height, width := volume.Storage, volume.Frames, volume.Height, volume.Width
		if err := checked.Length(current, graph.MomentChannels, frames, latentSpatial); err != nil {
			return nil, fmt.Errorf("source codec: chunk %d output storage: %w", chunkIndex, err)
		}
		if !checked.Equal(height, plan.Latent.Height) {
			return nil, fmt.Errorf("source codec: chunk %d output mismatch", chunkIndex)
		}
		if !checked.Equal(width, plan.Latent.Width) {
			return nil, fmt.Errorf("source codec: chunk %d output mismatch", chunkIndex)
		}
		if err := media.CopyPlanarFrames(means, plan.Latent.Frames, latentFrame, current, frames, tensor.FirstOffset, plan.Latent.Channels, frames, latentSpatial); err != nil {
			return nil, err
		}
		sourceFrame += plan.SourceChunks[chunkIndex]
		latentFrame += frames
	}
	if !checked.Equal(sourceFrame, plan.Source.Frames) {
		return nil, fmt.Errorf("source codec: consumed source=%d/%d latent=%d/%d", sourceFrame, plan.Source.Frames, latentFrame, plan.Latent.Frames)
	}
	if !checked.Equal(latentFrame, plan.Latent.Frames) {
		return nil, fmt.Errorf("source codec: consumed source=%d/%d latent=%d/%d", sourceFrame, plan.Source.Frames, latentFrame, plan.Latent.Frames)
	}
	latent := make([]float32, len(means))
	return latent, NormalizeSourceLatent(latent, means, plan)
}
