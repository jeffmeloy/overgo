package projector

import (
	"context"
	"errors"
	"fmt"
	"math"

	"overgo/internal/checked"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

const (
	htkMelScale     = 2595.0
	htkMelBreakHz   = 700.0
	htkMelLogBase10 = 10.0
)

func (r *Gemma4TowerRunner) EncodeAudio(
	ctx context.Context,
	samples []float32,
	sampleRate int,
	profile AudioProjectionProfile,
) (Gemma4AudioTowerOutput, error) {
	if r == nil || r.file == nil {
		return Gemma4AudioTowerOutput{}, errRunnerClosed
	}
	features, frames, err := r.audioPlan.preprocess(samples, sampleRate)
	if err != nil {
		return Gemma4AudioTowerOutput{}, err
	}
	return r.EncodeAudioFeatures(ctx, features, frames, profile)
}

func (r *Gemma4TowerRunner) EncodeAudioTrace(
	ctx context.Context,
	samples []float32,
	sampleRate int,
	profile AudioProjectionProfile,
) (Gemma4AudioTowerOutput, Gemma4AudioTowerTrace, error) {
	if r == nil || r.file == nil {
		return Gemma4AudioTowerOutput{}, Gemma4AudioTowerTrace{}, errRunnerClosed
	}
	features, frames, err := r.audioPlan.preprocess(samples, sampleRate)
	if err != nil {
		return Gemma4AudioTowerOutput{}, Gemma4AudioTowerTrace{}, err
	}
	output, trace, err := r.EncodeAudioFeaturesTrace(ctx, features, frames, profile)
	if err != nil {
		return Gemma4AudioTowerOutput{}, Gemma4AudioTowerTrace{}, err
	}
	trace.Stages["frontend"] = reference.Value{
		Shape: tensor.MustShape(uint64(r.spec.Audio.MelBins), uint64(frames)),
		Data:  features,
	}
	return output, trace, nil
}

type audioFrontendPlan struct {
	spec         Gemma4AudioTowerSpec
	window       []float64
	cosine, sine []float64
	filterbank   []float64
	bins         int
}

func newAudioFrontendPlan(spec Gemma4AudioTowerSpec) *audioFrontendPlan {
	window := make([]float64, spec.FrameLength)
	for index := range window {
		half := float64(tensor.SingletonExtent) / float64(tensor.PairedExtent)
		window[index] = half - half*math.Cos(float64(tensor.PairedExtent)*math.Pi*float64(index)/float64(spec.FrameLength))
	}
	bins := spec.FFTLength/tensor.PairedExtent + tensor.SingletonExtent
	cosine, sine := audioDFTBasis(bins, spec.FrameLength, spec.FFTLength)
	return &audioFrontendPlan{
		spec: spec, window: window, cosine: cosine, sine: sine,
		filterbank: audioMelFilterbank(spec, bins), bins: bins,
	}
}

func (p *audioFrontendPlan) preprocess(samples []float32, sampleRate int) ([]float32, int, error) {
	if p == nil {
		return nil, tensor.FirstOffset, errors.New("projector: audio frontend plan is unavailable")
	}
	spec := p.spec
	if sampleRate != spec.SampleRate {
		return nil, tensor.FirstOffset, fmt.Errorf("projector: audio sample rate=%d, want %d", sampleRate, spec.SampleRate)
	}
	if len(samples) == tensor.FirstOffset {
		return nil, tensor.FirstOffset, errors.New("projector: audio is empty")
	}
	for index, sample := range samples {
		if !checked.Finite32(sample) {
			return nil, tensor.FirstOffset, fmt.Errorf("projector: audio sample %d is not finite", index)
		}
	}
	padding := spec.FrameLength / tensor.PairedExtent
	frameSpan := spec.FrameLength + tensor.SingletonExtent
	if padding+len(samples) < frameSpan {
		return nil, tensor.FirstOffset, errors.New("projector: audio is shorter than one frame")
	}
	frames := (padding+len(samples)-frameSpan)/spec.HopLength + tensor.SingletonExtent
	padded := make([]float64, padding+len(samples))
	for index, sample := range samples {
		padded[padding+index] = float64(sample)
	}
	features := make([]float32, frames*spec.MelBins)
	windowed := make([]float64, spec.FrameLength)
	magnitudes := make([]float64, p.bins)
	for frame := range frames {
		base := frame * spec.HopLength
		for index := range windowed {
			windowed[index] = padded[base+index] * p.window[index]
		}
		for bin := range p.bins {
			var real, imaginary float64
			basis := bin * spec.FrameLength
			for index, value := range windowed {
				real += value * p.cosine[basis+index]
				imaginary += value * p.sine[basis+index]
			}
			magnitudes[bin] = math.Sqrt(real*real + imaginary*imaginary)
		}
		for mel := range spec.MelBins {
			var sum float64
			for bin, magnitude := range magnitudes {
				sum += magnitude * p.filterbank[bin*spec.MelBins+mel]
			}
			features[frame*spec.MelBins+mel] = float32(math.Log(sum + float64(spec.MelFloor)))
		}
	}
	return features, frames, nil
}

func audioDFTBasis(bins, frameLength, fftLength int) ([]float64, []float64) {
	cosine := make([]float64, bins*frameLength)
	sine := make([]float64, bins*frameLength)
	for bin := range bins {
		angle := -float64(tensor.PairedExtent) * math.Pi * float64(bin) / float64(fftLength)
		for index := range frameLength {
			cosine[bin*frameLength+index] = math.Cos(angle * float64(index))
			sine[bin*frameLength+index] = math.Sin(angle * float64(index))
		}
	}
	return cosine, sine
}

func audioMelFilterbank(spec Gemma4AudioTowerSpec, bins int) []float64 {
	mel := func(frequency float64) float64 {
		return htkMelScale * math.Log10(float64(tensor.SingletonExtent)+frequency/htkMelBreakHz)
	}
	frequency := func(value float64) float64 {
		return htkMelBreakHz * (math.Pow(htkMelLogBase10, value/htkMelScale) - float64(tensor.SingletonExtent))
	}
	minimum, maximum := mel(float64(spec.MinFrequency)), mel(float64(spec.MaxFrequency))
	points := make([]float64, spec.MelBins+tensor.PairedExtent)
	for index := range points {
		points[index] = frequency(minimum + (maximum-minimum)*float64(index)/float64(spec.MelBins+tensor.SingletonExtent))
	}
	filterbank := make([]float64, bins*spec.MelBins)
	for bin := range bins {
		fftFrequency := float64(spec.SampleRate) / float64(tensor.PairedExtent) * float64(bin) /
			float64(bins-tensor.SingletonExtent)
		for band := range spec.MelBins {
			rising := (fftFrequency - points[band]) / (points[band+tensor.SingletonExtent] - points[band])
			falling := (points[band+tensor.PairedExtent] - fftFrequency) /
				(points[band+tensor.PairedExtent] - points[band+tensor.SingletonExtent])
			filterbank[bin*spec.MelBins+band] = max(float64(tensor.FirstOffset), min(rising, falling))
		}
	}
	return filterbank
}
