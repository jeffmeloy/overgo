package audiodsp

import (
	"context"
	"errors"
	"math"

	"overgo/internal/checked"
)

// HTK and Slaney are named definitions of a frequency scale, not model facts.
const (
	htkMelScale        = 2595.0
	htkMelBreakHz      = 700.0
	htkMelLogBase      = 10.0
	slaneyLinearStepHz = 200.0 / 3.0
	slaneyLogStartHz   = 1000.0
	slaneyLogSpan      = 27.0
	slaneyLogRatio     = 6.4
)

func (p *Frontend) melFilterbank() ([]float64, error) {
	toMel := func(hz float64) float64 { return htkMelScale * math.Log10(1+hz/htkMelBreakHz) }
	toHz := func(mel float64) float64 { return htkMelBreakHz * (math.Pow(htkMelLogBase, mel/htkMelScale) - 1) }
	if p.config.Mel.Scale == "slaney" {
		logStep := math.Log(slaneyLogRatio) / slaneyLogSpan
		logStart := slaneyLogStartHz / slaneyLinearStepHz
		toMel = func(hz float64) float64 {
			if hz < slaneyLogStartHz {
				return hz / slaneyLinearStepHz
			}
			return logStart + math.Log(hz/slaneyLogStartHz)/logStep
		}
		toHz = func(mel float64) float64 {
			if mel < logStart {
				return mel * slaneyLinearStepHz
			}
			return slaneyLogStartHz * math.Exp((mel-logStart)*logStep)
		}
	}
	minimum, maximum := toMel(p.config.Mel.MinFrequency), toMel(p.config.Mel.MaxFrequency)
	points := make([]float64, p.bands+2)
	for index := range points {
		point := minimum + (maximum-minimum)*float64(index)/float64(p.bands+1)
		points[index] = toHz(point)
		if p.config.Mel.LinearInMel {
			points[index] = point
		}
		if !checked.Finite64(points[index]) || index > 0 && points[index] <= points[index-1] {
			return nil, errors.New("audio frontend: mel edges collapse at declared precision")
		}
	}
	bank := make([]float64, p.bins*p.bands)
	for bin := range p.bins {
		frequency := float64(p.config.SampleRate) / 2 * float64(bin) / (float64(p.config.FFTLength) / 2)
		if p.config.Mel.LinearInMel {
			frequency = toMel(frequency)
		}
		for band := range p.bands {
			rising := (frequency - points[band]) / (points[band+1] - points[band])
			falling := (points[band+2] - frequency) / (points[band+2] - points[band+1])
			value := max(0, min(rising, falling))
			if p.config.Mel.AreaNormalize {
				span := points[band+2] - points[band]
				if p.config.Mel.LinearInMel {
					span = toHz(points[band+2]) - toHz(points[band])
				}
				value *= 2 / span
			}
			if !checked.Finite64(value) {
				return nil, errors.New("audio frontend: non-finite filterbank")
			}
			bank[bin*p.bands+band] = value
		}
	}
	return bank, nil
}

func (p *Frontend) transform(ctx context.Context, features []float32, frames int) error {
	maximum := float32(math.Inf(-1))
	for _, value := range features {
		if !checked.Finite32(value) {
			return errors.New("audio frontend: non-finite logarithm")
		}
		maximum = max(maximum, value)
	}
	for index, scalar := range features {
		value := float64(scalar)
		if p.config.Log.DynamicRange != nil {
			value = max(value, float64(maximum)-*p.config.Log.DynamicRange)
		}
		features[index] = float32(value*p.config.Log.Scale + p.config.Log.Bias)
		if !checked.Finite32(features[index]) {
			return errors.New("audio frontend: non-finite scaled feature")
		}
	}
	if norm := p.config.Normalize; norm != nil {
		if norm.Mode == "fixed" {
			for index, value := range features {
				band := index % p.bands
				features[index] = float32((float64(value) - norm.Mean[band]) * norm.InverseStd[band])
				if !checked.Finite32(features[index]) {
					return errors.New("audio frontend: non-finite fixed normalization")
				}
			}
			return ctx.Err()
		}
		groups, width, stride := p.bands, frames, p.bands
		if norm.Mode == "all" {
			groups, width, stride = 1, len(features), 1
		}
		if width <= norm.Correction {
			return errors.New("audio frontend: too few observations for variance correction")
		}
		if norm.Float32 {
			return normalizeFeatures(ctx, features, groups, width, stride, norm.Correction, float32(norm.Epsilon))
		}
		return normalizeFeatures(ctx, features, groups, width, stride, norm.Correction, norm.Epsilon)
	}
	return ctx.Err()
}

func normalizeFeatures[T float32 | float64](ctx context.Context, features []float32, groups, width, stride, correction int, epsilon T) error {
	for group := range groups {
		if err := ctx.Err(); err != nil {
			return err
		}
		var mean T
		for index := range width {
			mean += T(features[group+index*stride])
		}
		mean /= T(width)
		var variance T
		for index := range width {
			delta := T(features[group+index*stride]) - mean
			variance += T(delta * delta)
		}
		denominator := T(math.Sqrt(float64(variance/T(width-correction)))) + epsilon
		for index := range width {
			at := group + index*stride
			features[at] = float32((T(features[at]) - mean) / denominator)
			if !checked.Finite32(features[at]) {
				return errors.New("audio frontend: non-finite normalization")
			}
		}
	}
	return ctx.Err()
}

func (norm NormalizeConfig) validate(bands int) error {
	if norm.Mode == "fixed" {
		if norm.Float32 || norm.Correction != 0 || norm.Epsilon != 0 || len(norm.Mean) != bands || len(norm.InverseStd) != bands {
			return errors.New("audio frontend: invalid fixed normalization geometry")
		}
		for band, mean := range norm.Mean {
			if !checked.Finite64(mean) || !checked.PositiveFinite64(norm.InverseStd[band]) {
				return errors.New("audio frontend: invalid fixed normalization statistics")
			}
		}
		return nil
	}
	if norm.Mode != "per-feature" && norm.Mode != "all" || norm.Correction < 0 || norm.Correction > 1 || !checked.PositiveFinite64(norm.Epsilon) || len(norm.Mean) != 0 || len(norm.InverseStd) != 0 {
		return errors.New("audio frontend: invalid normalization declaration")
	}
	if norm.Float32 && !checked.PositiveFinite64(float64(float32(norm.Epsilon))) {
		return errors.New("audio frontend: normalization epsilon is not positive finite float32")
	}
	return nil
}
