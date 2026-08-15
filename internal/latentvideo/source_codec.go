package latentvideo

import (
	"errors"
	"fmt"

	"overgo/internal/pytorchzip"
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
	if profile.InputChannels <= 0 || profile.LatentChannels <= 0 || source.Channels != profile.InputChannels ||
		source.Frames <= 0 || source.Height <= 0 || source.Width <= 0 {
		return SourceCodecPlan{}, errors.New("source codec: invalid channel or source extent")
	}
	for axis, stride := range profile.Stride {
		if stride <= 0 {
			return SourceCodecPlan{}, fmt.Errorf("source codec: stride[%d]=%d", axis, stride)
		}
	}
	if source.Height%profile.Stride[1] != 0 || source.Width%profile.Stride[2] != 0 {
		return SourceCodecPlan{}, errors.New("source codec: spatial extent is not stride-aligned")
	}
	if len(profile.LatentStats.Mean) != profile.LatentChannels || len(profile.LatentStats.Std) != profile.LatentChannels {
		return SourceCodecPlan{}, errors.New("source codec: latent statistics differ from channel count")
	}
	for channel, std := range profile.LatentStats.Std {
		if std == 0 {
			return SourceCodecPlan{}, fmt.Errorf("source codec: zero latent std at channel %d", channel)
		}
	}
	chunks := sourceChunkSchedule(source.Frames, profile.Stride[0])
	return SourceCodecPlan{
		Source: source,
		Latent: SourceVideoShape{
			Channels: profile.LatentChannels,
			Frames:   (source.Frames-1)/profile.Stride[0] + 1,
			Height:   source.Height / profile.Stride[1],
			Width:    source.Width / profile.Stride[2],
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
	elements, err := plan.LatentElements()
	if err != nil || len(out) != elements || len(meanOutput) != elements {
		return fmt.Errorf("source codec: latent extent mismatch out=%d mean=%d want=%d", len(out), len(meanOutput), elements)
	}
	plane := plan.Latent.Frames * plan.Latent.Height * plan.Latent.Width
	for channel := range plan.Latent.Channels {
		mean, std := plan.Profile.LatentStats.Mean[channel], plan.Profile.LatentStats.Std[channel]
		for index := range plane {
			offset := channel*plane + index
			out[offset] = (meanOutput[offset] - mean) / std
		}
	}
	return nil
}

// EncodeSourceVideo runs the checkpoint-derived causal encoder chunk stream.
func EncodeSourceVideo(checkpoint string, graph VAEEncoderPlan, plan SourceCodecPlan, source []float32) ([]float32, error) {
	sourceElements, err := plan.SourceElements()
	if err != nil || len(source) != sourceElements || len(graph.ops) == 0 ||
		graph.InputChannels != plan.Source.Channels || graph.LatentChannels != plan.Latent.Channels || graph.Stride != plan.Profile.Stride {
		return nil, errors.New("source codec: encoder contract mismatch")
	}
	reader, err := pytorchzip.Open(checkpoint)
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	ops, err := loadVAEOps(reader, graph.vaePlanCore, "encoder")
	if err != nil {
		return nil, err
	}
	states := make([]vaeOpState, len(ops))
	latentElements, err := plan.LatentElements()
	if err != nil {
		return nil, err
	}
	means := make([]float32, latentElements)
	sourceFrame, latentFrame := 0, 0
	sourceSpatial := plan.Source.Height * plan.Source.Width
	latentSpatial := plan.Latent.Height * plan.Latent.Width
	for chunkIndex, frames := range plan.SourceChunks {
		current := make([]float32, plan.Source.Channels*frames*sourceSpatial)
		if err := copyChannelFrames(current, frames, 0, source, plan.Source.Frames, sourceFrame, plan.Source.Channels, frames, sourceSpatial); err != nil {
			return nil, err
		}
		height, width := plan.Source.Height, plan.Source.Width
		for opIndex := range ops {
			current, frames, height, width, err = runVAEOp(ops[opIndex], &states[opIndex], chunkIndex, current, frames, height, width)
			if err != nil {
				return nil, fmt.Errorf("source codec: %s chunk %d: %w", ops[opIndex].prefix, chunkIndex, err)
			}
		}
		if len(current) != graph.MomentChannels*frames*latentSpatial || height != plan.Latent.Height || width != plan.Latent.Width {
			return nil, fmt.Errorf("source codec: chunk %d output mismatch", chunkIndex)
		}
		if err := copyChannelFrames(means, plan.Latent.Frames, latentFrame, current, frames, 0, plan.Latent.Channels, frames, latentSpatial); err != nil {
			return nil, err
		}
		sourceFrame += plan.SourceChunks[chunkIndex]
		latentFrame += frames
	}
	if sourceFrame != plan.Source.Frames || latentFrame != plan.Latent.Frames {
		return nil, fmt.Errorf("source codec: consumed source=%d/%d latent=%d/%d", sourceFrame, plan.Source.Frames, latentFrame, plan.Latent.Frames)
	}
	latent := make([]float32, len(means))
	return latent, NormalizeSourceLatent(latent, means, plan)
}

func sourceChunkSchedule(frames, temporalStride int) []int {
	chunks := make([]int, 0, 1+(frames-1+temporalStride-1)/temporalStride)
	for start := 0; start < frames; {
		count := 1
		if start > 0 {
			count = min(temporalStride, frames-start)
		}
		chunks = append(chunks, count)
		start += count
	}
	return chunks
}
