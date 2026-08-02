package inference

import (
	"math"
	"testing"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
)

func TestWavTokenizerFeaturesToWaveformDCFrame(t *testing.T) {
	data := make([]float32, wavTokenizerFrameWidth)
	for bin := 0; bin < wavTokenizerBins; bin++ {
		data[bin] = float32(math.Inf(-1))
	}
	data[0] = 0
	audio, err := WavTokenizerFeaturesToWaveform(reference.Value{
		Shape: tensor.MustShape(wavTokenizerFrameWidth, 1), Data: data,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(audio) != wavTokenizerHopSize {
		t.Fatalf("waveform length = %d, want %d", len(audio), wavTokenizerHopSize)
	}
	for index, value := range audio {
		hann := float32(0.5 * (1 - math.Cos(2*math.Pi*float64(index+wavTokenizerPadSize)/wavTokenizerFFTSize)))
		want := float32(1.0/wavTokenizerBins) / hann
		if delta := math.Abs(float64(value - want)); delta > 1e-6 {
			t.Fatalf("waveform[%d] = %g, want %g (delta %g)", index, value, want, delta)
		}
	}
}

func TestWavTokenizerFeaturesToWaveformRejectsShape(t *testing.T) {
	_, err := WavTokenizerFeaturesToWaveform(reference.Value{
		Shape: tensor.MustShape(wavTokenizerFrameWidth-1, 1),
		Data:  make([]float32, wavTokenizerFrameWidth-1),
	})
	if err == nil {
		t.Fatal("accepted incompatible WavTokenizer feature shape")
	}
}
