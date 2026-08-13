package inference

import (
	"context"
	"errors"
	"math"
	"runtime"
	"sync"

	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

const (
	AudioSampleRate = 24000
	audioFFTSize    = 1280
	audioHopSize    = 320
	audioPadSize    = 480
	audioBins       = audioFFTSize/2 + 1
	audioFrameWidth = audioBins * 2
)

var (
	audioTablesOnce sync.Once
	audioCosTable   []float32
	audioSinTable   []float32
	audioHann       []float32
)

// DecodeAudioWaveform: semantic tokens to 24 kHz mono samples.
func (r *Runner) DecodeAudioWaveform(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
) ([]float32, error) {
	features, err := r.DecodeAudioTokens(ctx, tokenIDs)
	if err != nil {
		return nil, err
	}
	return AudioFeaturesToWaveform(features)
}

// AudioFeaturesToWaveform: pinned ISTFT postprocessor.
func AudioFeaturesToWaveform(features reference.Value) ([]float32, error) {
	if features.Shape.Rank != 2 || features.Shape.Dims[0] != audioFrameWidth ||
		features.Shape.Dims[1] == 0 || features.Shape.Dims[1] > uint64(math.MaxInt/audioFrameWidth) {
		return nil, errors.New("inference: AudioDecoder feature shape is incompatible")
	}
	frames := int(features.Shape.Dims[1])
	if len(features.Data) != frames*audioFrameWidth || frames > math.MaxInt/audioFFTSize {
		return nil, errors.New("inference: AudioDecoder feature data is incompatible")
	}
	audioTablesOnce.Do(initAudioTables)
	windows := make([]float32, frames*audioFFTSize)
	workers := min(runtime.GOMAXPROCS(0), frames)
	var group sync.WaitGroup
	group.Add(workers)
	for worker := 0; worker < workers; worker++ {
		go func(worker int) {
			defer group.Done()
			realPart := make([]float32, audioBins)
			imaginaryPart := make([]float32, audioBins)
			for frame := worker; frame < frames; frame += workers {
				base := frame * audioFrameWidth
				for bin := 0; bin < audioBins; bin++ {
					magnitude := float32(math.Exp(float64(features.Data[base+bin])))
					if magnitude > 100 {
						magnitude = 100
					}
					phase := features.Data[base+audioBins+bin]
					realPart[bin] = magnitude * float32(math.Cos(float64(phase)))
					imaginaryPart[bin] = magnitude * float32(math.Sin(float64(phase)))
				}
				output := windows[frame*audioFFTSize : (frame+1)*audioFFTSize]
				for sample := 0; sample < audioFFTSize; sample++ {
					table := sample * audioBins
					var sum float32
					for bin := 0; bin < audioBins; bin++ {
						sum += realPart[bin]*audioCosTable[table+bin] -
							imaginaryPart[bin]*audioSinTable[table+bin]
					}
					output[sample] = sum / audioBins * audioHann[sample]
				}
			}
		}(worker)
	}
	group.Wait()
	outputSize := (frames-1)*audioHopSize + audioFFTSize
	trimmedSize := outputSize - 2*audioPadSize
	audio := make([]float32, outputSize)
	envelope := make([]float32, outputSize)
	for frame := 0; frame < frames; frame++ {
		start := frame*audioHopSize - audioPadSize
		window := windows[frame*audioFFTSize : (frame+1)*audioFFTSize]
		for sample, value := range window {
			position := start + sample
			if position < 0 || position >= outputSize {
				continue
			}
			hann := audioHann[sample]
			audio[position] += value
			envelope[position] += hann * hann
		}
	}
	audio = audio[:trimmedSize]
	for index := range audio {
		if envelope[index] == 0 {
			return nil, errors.New("inference: AudioDecoder overlap envelope is zero")
		}
		audio[index] /= envelope[index]
		if math.IsNaN(float64(audio[index])) || math.IsInf(float64(audio[index]), 0) {
			return nil, errors.New("inference: AudioDecoder waveform is not finite")
		}
	}
	return audio, nil
}

func initAudioTables() {
	audioCosTable = make([]float32, audioFFTSize*audioBins)
	audioSinTable = make([]float32, audioFFTSize*audioBins)
	audioHann = make([]float32, audioFFTSize)
	for sample := 0; sample < audioFFTSize; sample++ {
		audioHann[sample] = float32(0.5 * (1 - math.Cos(2*math.Pi*float64(sample)/audioFFTSize)))
		for bin := 0; bin < audioBins; bin++ {
			sine, cosine := math.Sincos(2 * math.Pi * float64(sample*bin) / audioFFTSize)
			index := sample*audioBins + bin
			audioCosTable[index] = float32(cosine)
			audioSinTable[index] = float32(sine)
		}
	}
}
