package projector

import (
	"context"
	"errors"
	"fmt"
	"math"
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
		return Gemma4AudioTowerOutput{}, errors.New("projector: runner is closed")
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
		return Gemma4AudioTowerOutput{}, Gemma4AudioTowerTrace{}, errors.New("projector: runner is closed")
	}
	features, frames, err := r.audioPlan.preprocess(samples, sampleRate)
	if err != nil {
		return Gemma4AudioTowerOutput{}, Gemma4AudioTowerTrace{}, err
	}
	return r.EncodeAudioFeaturesTrace(ctx, features, frames, profile)
}

// PreprocessGemma4AudioTower: semicausal Hann/RFFT/HTK-log-mel frontend.
func PreprocessGemma4AudioTower(
	samples []float32,
	sampleRate int,
	spec Gemma4AudioTowerSpec,
) ([]float32, int, error) {
	if err := validateGemma4AudioFrontend(spec); err != nil {
		return nil, 0, err
	}
	return newGemma4AudioFrontendPlan(spec).preprocess(samples, sampleRate)
}

type gemma4AudioFrontendPlan struct {
	spec         Gemma4AudioTowerSpec
	window       []float64
	cosine, sine []float64
	filterbank   []float64
	bins         int
}

func newGemma4AudioFrontendPlan(spec Gemma4AudioTowerSpec) *gemma4AudioFrontendPlan {
	window := make([]float64, spec.FrameLength)
	for index := range window {
		window[index] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(index)/float64(spec.FrameLength))
	}
	bins := spec.FFTLength/2 + 1
	cosine, sine := gemma4AudioDFTBasis(bins, spec.FrameLength, spec.FFTLength)
	return &gemma4AudioFrontendPlan{
		spec: spec, window: window, cosine: cosine, sine: sine,
		filterbank: gemma4AudioMelFilterbank(spec, bins), bins: bins,
	}
}

func (p *gemma4AudioFrontendPlan) preprocess(samples []float32, sampleRate int) ([]float32, int, error) {
	if p == nil {
		return nil, 0, errors.New("projector: Gemma 4 audio frontend plan is unavailable")
	}
	spec := p.spec
	if sampleRate != spec.SampleRate {
		return nil, 0, fmt.Errorf("projector: Gemma 4 audio sample rate=%d, want %d", sampleRate, spec.SampleRate)
	}
	if len(samples) == 0 {
		return nil, 0, errors.New("projector: Gemma 4 audio is empty")
	}
	for index, sample := range samples {
		if !finite32(sample) {
			return nil, 0, fmt.Errorf("projector: Gemma 4 audio sample %d is not finite", index)
		}
	}
	padding := spec.FrameLength / 2
	frameSpan := spec.FrameLength + 1
	if padding+len(samples) < frameSpan {
		return nil, 0, errors.New("projector: Gemma 4 audio is shorter than one frame")
	}
	frames := (padding+len(samples)-frameSpan)/spec.HopLength + 1
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

func validateGemma4AudioFrontend(spec Gemma4AudioTowerSpec) error {
	if spec.MelBins <= 0 || spec.FFTLength <= 0 || spec.FFTLength < spec.FrameLength ||
		spec.FrameLength <= 0 || spec.HopLength <= 0 || spec.SampleRate <= 0 ||
		spec.MinFrequency < 0 || spec.MaxFrequency <= spec.MinFrequency ||
		spec.MaxFrequency > float32(spec.SampleRate)/2 || spec.MelFloor <= 0 {
		return fmt.Errorf("projector: invalid Gemma 4 audio frontend metadata: %+v", spec)
	}
	return nil
}

func gemma4AudioDFTBasis(bins, frameLength, fftLength int) ([]float64, []float64) {
	cosine := make([]float64, bins*frameLength)
	sine := make([]float64, bins*frameLength)
	for bin := range bins {
		angle := -2 * math.Pi * float64(bin) / float64(fftLength)
		for index := range frameLength {
			cosine[bin*frameLength+index] = math.Cos(angle * float64(index))
			sine[bin*frameLength+index] = math.Sin(angle * float64(index))
		}
	}
	return cosine, sine
}

func gemma4AudioMelFilterbank(spec Gemma4AudioTowerSpec, bins int) []float64 {
	mel := func(frequency float64) float64 {
		return htkMelScale * math.Log10(1+frequency/htkMelBreakHz)
	}
	frequency := func(value float64) float64 {
		return htkMelBreakHz * (math.Pow(htkMelLogBase10, value/htkMelScale) - 1)
	}
	minimum, maximum := mel(float64(spec.MinFrequency)), mel(float64(spec.MaxFrequency))
	points := make([]float64, spec.MelBins+2)
	for index := range points {
		points[index] = frequency(minimum + (maximum-minimum)*float64(index)/float64(spec.MelBins+1))
	}
	filterbank := make([]float64, bins*spec.MelBins)
	for bin := range bins {
		fftFrequency := float64(spec.SampleRate) / 2 * float64(bin) / float64(bins-1)
		for band := range spec.MelBins {
			rising := (fftFrequency - points[band]) / (points[band+1] - points[band])
			falling := (points[band+2] - fftFrequency) / (points[band+2] - points[band+1])
			filterbank[bin*spec.MelBins+band] = max(0, min(rising, falling))
		}
	}
	return filterbank
}
