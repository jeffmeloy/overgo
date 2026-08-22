package inference

import (
	"context"
	"errors"
	"math"
	"runtime"

	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// audioWaveformTables: inverse-DFT basis and Hann window for one plan.
// Runner-owned: built once per Runner, never cached at package scope.
type audioWaveformTables struct {
	cosine, sine, hann []float32
}

// DecodeAudioWaveform: semantic tokens to 24 kHz mono samples.
func (r *Runner) DecodeAudioWaveform(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
) ([]float32, error) {
	features, err := r.DecodeAudioTokens(ctx, tokenIDs)
	if err != nil {
		return nil, err
	}
	plan := r.program.Model.AudioWaveform()
	if !plan.Valid() {
		return nil, errors.New("inference: audio waveform plan is invalid")
	}
	r.audioTablesOnce.Do(func() { r.audioTables = newAudioWaveformTables(plan) })
	return audioFeaturesToWaveform(plan, r.audioTables, features)
}

// audioFeaturesToWaveform reconstructs audio from log-magnitude/phase frames
// in ordered slabs: workers synthesize a worker-chunk of frame windows in
// parallel, then the slab is overlap-added sequentially in frame order, so
// per-sample float addition order matches the full-buffer computation exactly.
func audioFeaturesToWaveform(
	plan model.AudioWaveformPlan, tables *audioWaveformTables, features reference.Value,
) ([]float32, error) {
	frameExtent, validShape := tensor.MatrixRows(features.Shape, uint64(plan.FrameWidth()))
	frames, validFrames := checked.Int(frameExtent)
	featureElements, validFeatures := checked.MulInt(frames, plan.FrameWidth())
	if !plan.Valid() || !validShape || !validFrames || !validFeatures {
		return nil, errors.New("inference: audio feature shape is incompatible")
	}
	if len(features.Data) != featureElements {
		return nil, errors.New("inference: audio feature data is incompatible")
	}
	if tables == nil {
		return nil, errors.New("inference: audio waveform tables are missing")
	}
	overlapSize, validOverlap := checked.MulInt(frames-tensor.SingletonExtent, plan.HopSize)
	outputSize, validOutput := checked.AddInt(overlapSize, plan.FFTSize)
	if !validOverlap || !validOutput {
		return nil, errors.New("inference: audio output extent overflows")
	}
	trimmedSize := outputSize - 2*plan.PadSize
	audio := make([]float32, outputSize)
	envelope := make([]float32, outputSize)
	slabFrames := min(runtime.GOMAXPROCS(0), frames)
	windowElements, validWindows := checked.MulInt(slabFrames, plan.FFTSize)
	frameWork, validWork := checked.MulInt(plan.FFTSize, plan.Bins())
	if !validWindows || !validWork {
		return nil, errors.New("inference: audio workspace extent overflows")
	}
	windows := make([]float32, windowElements)
	for slabStart := 0; slabStart < frames; slabStart += slabFrames {
		slab := min(slabFrames, frames-slabStart)
		hostmath.ParallelRangeF64(slab, frameWork, func(start, end int) {
			realPart := make([]float32, plan.Bins())
			imaginaryPart := make([]float32, plan.Bins())
			for offset := start; offset < end; offset++ {
				base := (slabStart + offset) * plan.FrameWidth()
				for bin := 0; bin < plan.Bins(); bin++ {
					magnitude := float32(math.Exp(float64(features.Data[base+bin])))
					if magnitude > plan.MagnitudeLimit {
						magnitude = plan.MagnitudeLimit
					}
					phase := features.Data[base+plan.Bins()+bin]
					realPart[bin] = magnitude * float32(math.Cos(float64(phase)))
					imaginaryPart[bin] = magnitude * float32(math.Sin(float64(phase)))
				}
				output := windows[offset*plan.FFTSize : (offset+1)*plan.FFTSize]
				for sample := 0; sample < plan.FFTSize; sample++ {
					table := sample * plan.Bins()
					var sum float32
					for bin := 0; bin < plan.Bins(); bin++ {
						sum += realPart[bin]*tables.cosine[table+bin] -
							imaginaryPart[bin]*tables.sine[table+bin]
					}
					output[sample] = sum / float32(plan.Bins()) * tables.hann[sample]
				}
			}
		})
		for offset := 0; offset < slab; offset++ {
			start := (slabStart+offset)*plan.HopSize - plan.PadSize
			window := windows[offset*plan.FFTSize : (offset+1)*plan.FFTSize]
			for sample, value := range window {
				position := start + sample
				if position < 0 || position >= outputSize {
					continue
				}
				hann := tables.hann[sample]
				audio[position] += value
				envelope[position] += hann * hann
			}
		}
	}
	audio = audio[:trimmedSize]
	for index := range audio {
		if envelope[index] == 0 {
			return nil, errors.New("inference: audio overlap envelope is zero")
		}
		audio[index] /= envelope[index]
		if math.IsNaN(float64(audio[index])) || math.IsInf(float64(audio[index]), 0) {
			return nil, errors.New("inference: audio waveform is not finite")
		}
	}
	return audio, nil
}

func newAudioWaveformTables(plan model.AudioWaveformPlan) *audioWaveformTables {
	tables := &audioWaveformTables{
		cosine: make([]float32, plan.FFTSize*plan.Bins()),
		sine:   make([]float32, plan.FFTSize*plan.Bins()),
		hann:   make([]float32, plan.FFTSize),
	}
	for sample := 0; sample < plan.FFTSize; sample++ {
		tables.hann[sample] = float32(0.5 * (1 - math.Cos(2*math.Pi*float64(sample)/float64(plan.FFTSize))))
		for bin := 0; bin < plan.Bins(); bin++ {
			sine, cosine := math.Sincos(2 * math.Pi * float64(sample*bin) / float64(plan.FFTSize))
			index := sample*plan.Bins() + bin
			tables.cosine[index] = float32(cosine)
			tables.sine[index] = float32(sine)
		}
	}
	return tables
}
