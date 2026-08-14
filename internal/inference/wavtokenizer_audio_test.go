package inference

import (
	"math"
	"testing"

	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

func TestAudioFeaturesToWaveformDCFrame(t *testing.T) {
	plan := testAudioWaveformPlan(t)
	data := make([]float32, plan.FrameWidth())
	for bin := 0; bin < plan.Bins(); bin++ {
		data[bin] = float32(math.Inf(-1))
	}
	data[0] = 0
	audio, err := audioFeaturesToWaveform(plan, reference.Value{
		Shape: tensor.MustShape(uint64(plan.FrameWidth()), 1), Data: data,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(audio) != plan.HopSize {
		t.Fatalf("waveform length = %d, want %d", len(audio), plan.HopSize)
	}
	for index, value := range audio {
		hann := float32(0.5 * (1 - math.Cos(2*math.Pi*float64(index+plan.PadSize)/float64(plan.FFTSize))))
		want := float32(1.0/float64(plan.Bins())) / hann
		if delta := math.Abs(float64(value - want)); delta > 1e-6 {
			t.Fatalf("waveform[%d] = %g, want %g (delta %g)", index, value, want, delta)
		}
	}
}

func TestAudioFeaturesToWaveformRejectsShape(t *testing.T) {
	plan := testAudioWaveformPlan(t)
	_, err := audioFeaturesToWaveform(plan, reference.Value{
		Shape: tensor.MustShape(uint64(plan.FrameWidth()-1), 1),
		Data:  make([]float32, plan.FrameWidth()-1),
	})
	if err == nil {
		t.Fatal("accepted incompatible AudioDecoder feature shape")
	}
}

func testAudioWaveformPlan(t *testing.T) model.AudioWaveformPlan {
	t.Helper()
	profile, ok := model.LookupArchitecture("wavtokenizer-dec")
	if !ok {
		t.Fatal("audio-token profile is missing")
	}
	plan, err := model.CompileModelPlanWithProfile(
		model.Spec{CommonSpec: model.CommonSpec{Architecture: profile.Name}}, model.Weights{}, profile,
	)
	if err != nil {
		t.Fatal(err)
	}
	return plan.AudioWaveform()
}
