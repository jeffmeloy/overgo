package inference

import (
	"math"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

func TestAudioFeaturesToWaveformDCFrame(t *testing.T) {
	data := make([]float32, audioFrameWidth)
	for bin := 0; bin < audioBins; bin++ {
		data[bin] = float32(math.Inf(-1))
	}
	data[0] = 0
	audio, err := AudioFeaturesToWaveform(reference.Value{
		Shape: tensor.MustShape(audioFrameWidth, 1), Data: data,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(audio) != audioHopSize {
		t.Fatalf("waveform length = %d, want %d", len(audio), audioHopSize)
	}
	for index, value := range audio {
		hann := float32(0.5 * (1 - math.Cos(2*math.Pi*float64(index+audioPadSize)/audioFFTSize)))
		want := float32(1.0/audioBins) / hann
		if delta := math.Abs(float64(value - want)); delta > 1e-6 {
			t.Fatalf("waveform[%d] = %g, want %g (delta %g)", index, value, want, delta)
		}
	}
}

func TestAudioFeaturesToWaveformRejectsShape(t *testing.T) {
	_, err := AudioFeaturesToWaveform(reference.Value{
		Shape: tensor.MustShape(audioFrameWidth-1, 1),
		Data:  make([]float32, audioFrameWidth-1),
	})
	if err == nil {
		t.Fatal("accepted incompatible AudioDecoder feature shape")
	}
}
