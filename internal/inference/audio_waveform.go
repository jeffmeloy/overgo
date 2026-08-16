package inference

import (
	"context"
	"errors"
	"math"
	"runtime"
	"sync"

	"overgo/internal/model"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// audioWaveformSlabWorkerFrames: frames staged per worker per slab; bounds
// the window staging buffer at worker-chunk scale instead of frame scale.
const audioWaveformSlabWorkerFrames = 8

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
	if !plan.Valid() || features.Shape.Rank != 2 || features.Shape.Dims[0] != uint64(plan.FrameWidth()) ||
		features.Shape.Dims[1] == 0 || features.Shape.Dims[1] > uint64(math.MaxInt/plan.FrameWidth()) {
		return nil, errors.New("inference: audio feature shape is incompatible")
	}
	frames := int(features.Shape.Dims[1])
	if len(features.Data) != frames*plan.FrameWidth() || frames > math.MaxInt/plan.FFTSize {
		return nil, errors.New("inference: audio feature data is incompatible")
	}
	if tables == nil {
		return nil, errors.New("inference: audio waveform tables are missing")
	}
	outputSize := (frames-1)*plan.HopSize + plan.FFTSize
	trimmedSize := outputSize - 2*plan.PadSize
	audio := make([]float32, outputSize)
	envelope := make([]float32, outputSize)
	workers := min(runtime.GOMAXPROCS(0), frames)
	slabFrames := min(workers*audioWaveformSlabWorkerFrames, frames)
	windows := make([]float32, slabFrames*plan.FFTSize)
	for slabStart := 0; slabStart < frames; slabStart += slabFrames {
		slab := min(slabFrames, frames-slabStart)
		slabWorkers := min(workers, slab)
		var group sync.WaitGroup
		group.Add(slabWorkers)
		for worker := 0; worker < slabWorkers; worker++ {
			go func(worker int) {
				defer group.Done()
				realPart := make([]float32, plan.Bins())
				imaginaryPart := make([]float32, plan.Bins())
				for offset := worker; offset < slab; offset += slabWorkers {
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
			}(worker)
		}
		group.Wait()
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
