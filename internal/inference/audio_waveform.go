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

type audioWaveformTables struct {
	cosine, sine, hann []float32
}

var audioTableCache = struct {
	sync.Mutex
	byPlan map[model.AudioWaveformPlan]audioWaveformTables
}{byPlan: make(map[model.AudioWaveformPlan]audioWaveformTables)}

// DecodeAudioWaveform: semantic tokens to 24 kHz mono samples.
func (r *Runner) DecodeAudioWaveform(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
) ([]float32, error) {
	features, err := r.DecodeAudioTokens(ctx, tokenIDs)
	if err != nil {
		return nil, err
	}
	return audioFeaturesToWaveform(r.program.Model.AudioWaveform(), features)
}

func audioFeaturesToWaveform(plan model.AudioWaveformPlan, features reference.Value) ([]float32, error) {
	if !plan.Valid() || features.Shape.Rank != 2 || features.Shape.Dims[0] != uint64(plan.FrameWidth()) ||
		features.Shape.Dims[1] == 0 || features.Shape.Dims[1] > uint64(math.MaxInt/plan.FrameWidth()) {
		return nil, errors.New("inference: audio feature shape is incompatible")
	}
	frames := int(features.Shape.Dims[1])
	if len(features.Data) != frames*plan.FrameWidth() || frames > math.MaxInt/plan.FFTSize {
		return nil, errors.New("inference: audio feature data is incompatible")
	}
	tables := waveformTables(plan)
	windows := make([]float32, frames*plan.FFTSize)
	workers := min(runtime.GOMAXPROCS(0), frames)
	var group sync.WaitGroup
	group.Add(workers)
	for worker := 0; worker < workers; worker++ {
		go func(worker int) {
			defer group.Done()
			realPart := make([]float32, plan.Bins())
			imaginaryPart := make([]float32, plan.Bins())
			for frame := worker; frame < frames; frame += workers {
				base := frame * plan.FrameWidth()
				for bin := 0; bin < plan.Bins(); bin++ {
					magnitude := float32(math.Exp(float64(features.Data[base+bin])))
					if magnitude > 100 {
						magnitude = 100
					}
					phase := features.Data[base+plan.Bins()+bin]
					realPart[bin] = magnitude * float32(math.Cos(float64(phase)))
					imaginaryPart[bin] = magnitude * float32(math.Sin(float64(phase)))
				}
				output := windows[frame*plan.FFTSize : (frame+1)*plan.FFTSize]
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
	outputSize := (frames-1)*plan.HopSize + plan.FFTSize
	trimmedSize := outputSize - 2*plan.PadSize
	audio := make([]float32, outputSize)
	envelope := make([]float32, outputSize)
	for frame := 0; frame < frames; frame++ {
		start := frame*plan.HopSize - plan.PadSize
		window := windows[frame*plan.FFTSize : (frame+1)*plan.FFTSize]
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

func waveformTables(plan model.AudioWaveformPlan) audioWaveformTables {
	audioTableCache.Lock()
	defer audioTableCache.Unlock()
	if tables, ok := audioTableCache.byPlan[plan]; ok {
		return tables
	}
	tables := audioWaveformTables{
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
	audioTableCache.byPlan[plan] = tables
	return tables
}
