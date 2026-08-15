package latentvideo

import (
	"errors"
	"fmt"
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
