package inference

import (
	"context"
	"errors"
	"math"
	"runtime"
	"sync"

	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

const (
	WavTokenizerSampleRate = 24000
	wavTokenizerFFTSize    = 1280
	wavTokenizerHopSize    = 320
	wavTokenizerPadSize    = 480
	wavTokenizerBins       = wavTokenizerFFTSize/2 + 1
	wavTokenizerFrameWidth = wavTokenizerBins * 2
)

var (
	wavTokenizerTablesOnce sync.Once
	wavTokenizerCosTable   []float32
	wavTokenizerSinTable   []float32
	wavTokenizerHann       []float32
)

// DecodeWavTokenizerWaveform: semantic tokens to 24 kHz mono samples.
func (r *Runner) DecodeWavTokenizerWaveform(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
) ([]float32, error) {
	features, err := r.DecodeWavTokenizer(ctx, tokenIDs)
	if err != nil {
		return nil, err
	}
	return WavTokenizerFeaturesToWaveform(features)
}

// WavTokenizerFeaturesToWaveform: pinned ISTFT postprocessor.
func WavTokenizerFeaturesToWaveform(features reference.Value) ([]float32, error) {
	if features.Shape.Rank != 2 || features.Shape.Dims[0] != wavTokenizerFrameWidth ||
		features.Shape.Dims[1] == 0 || features.Shape.Dims[1] > uint64(math.MaxInt/wavTokenizerFrameWidth) {
		return nil, errors.New("inference: WavTokenizer feature shape is incompatible")
	}
	frames := int(features.Shape.Dims[1])
	if len(features.Data) != frames*wavTokenizerFrameWidth || frames > math.MaxInt/wavTokenizerFFTSize {
		return nil, errors.New("inference: WavTokenizer feature data is incompatible")
	}
	wavTokenizerTablesOnce.Do(initWavTokenizerTables)
	windows := make([]float32, frames*wavTokenizerFFTSize)
	workers := min(runtime.GOMAXPROCS(0), frames)
	var group sync.WaitGroup
	group.Add(workers)
	for worker := 0; worker < workers; worker++ {
		go func(worker int) {
			defer group.Done()
			realPart := make([]float32, wavTokenizerBins)
			imaginaryPart := make([]float32, wavTokenizerBins)
			for frame := worker; frame < frames; frame += workers {
				base := frame * wavTokenizerFrameWidth
				for bin := 0; bin < wavTokenizerBins; bin++ {
					magnitude := float32(math.Exp(float64(features.Data[base+bin])))
					if magnitude > 100 {
						magnitude = 100
					}
					phase := features.Data[base+wavTokenizerBins+bin]
					realPart[bin] = magnitude * float32(math.Cos(float64(phase)))
					imaginaryPart[bin] = magnitude * float32(math.Sin(float64(phase)))
				}
				output := windows[frame*wavTokenizerFFTSize : (frame+1)*wavTokenizerFFTSize]
				for sample := 0; sample < wavTokenizerFFTSize; sample++ {
					table := sample * wavTokenizerBins
					var sum float32
					for bin := 0; bin < wavTokenizerBins; bin++ {
						sum += realPart[bin]*wavTokenizerCosTable[table+bin] -
							imaginaryPart[bin]*wavTokenizerSinTable[table+bin]
					}
					output[sample] = sum / wavTokenizerBins * wavTokenizerHann[sample]
				}
			}
		}(worker)
	}
	group.Wait()
	outputSize := (frames-1)*wavTokenizerHopSize + wavTokenizerFFTSize
	trimmedSize := outputSize - 2*wavTokenizerPadSize
	audio := make([]float32, outputSize)
	envelope := make([]float32, outputSize)
	for frame := 0; frame < frames; frame++ {
		start := frame*wavTokenizerHopSize - wavTokenizerPadSize
		window := windows[frame*wavTokenizerFFTSize : (frame+1)*wavTokenizerFFTSize]
		for sample, value := range window {
			position := start + sample
			if position < 0 || position >= outputSize {
				continue
			}
			hann := wavTokenizerHann[sample]
			audio[position] += value
			envelope[position] += hann * hann
		}
	}
	audio = audio[:trimmedSize]
	for index := range audio {
		if envelope[index] == 0 {
			return nil, errors.New("inference: WavTokenizer overlap envelope is zero")
		}
		audio[index] /= envelope[index]
		if math.IsNaN(float64(audio[index])) || math.IsInf(float64(audio[index]), 0) {
			return nil, errors.New("inference: WavTokenizer waveform is not finite")
		}
	}
	return audio, nil
}

func initWavTokenizerTables() {
	wavTokenizerCosTable = make([]float32, wavTokenizerFFTSize*wavTokenizerBins)
	wavTokenizerSinTable = make([]float32, wavTokenizerFFTSize*wavTokenizerBins)
	wavTokenizerHann = make([]float32, wavTokenizerFFTSize)
	for sample := 0; sample < wavTokenizerFFTSize; sample++ {
		wavTokenizerHann[sample] = float32(0.5 * (1 - math.Cos(2*math.Pi*float64(sample)/wavTokenizerFFTSize)))
		for bin := 0; bin < wavTokenizerBins; bin++ {
			sine, cosine := math.Sincos(2 * math.Pi * float64(sample*bin) / wavTokenizerFFTSize)
			index := sample*wavTokenizerBins + bin
			wavTokenizerCosTable[index] = float32(cosine)
			wavTokenizerSinTable[index] = float32(sine)
		}
	}
}
