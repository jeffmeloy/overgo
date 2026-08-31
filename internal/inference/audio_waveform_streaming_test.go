package inference

import (
	"math"
	"runtime"
	"testing"

	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

// TestStreamedWaveformReconstruction proves the slab-streamed overlap-add is
// bit-identical to the direct full-buffer computation, that waveform tables
// are built per call site instead of a package-level cache, and that the
// magnitude clamp arrives through the typed AudioWaveformPlan.
func TestStreamedWaveformReconstruction(t *testing.T) {
	compiled := testAudioWaveformPlan(t)
	if compiled.MagnitudeLimit != 100 {
		t.Fatalf("compiled magnitude limit = %g, want 100", compiled.MagnitudeLimit)
	}
	small := model.AudioWaveformPlan{
		SampleRate: 24_000, FFTSize: 64, HopSize: 16, PadSize: 24, MagnitudeLimit: 100,
	}
	if !small.Valid() {
		t.Fatal("small waveform plan is invalid")
	}
	first, second := newAudioWaveformTables(small), newAudioWaveformTables(small)
	if &first.cosine[0] == &second.cosine[0] || &first.sine[0] == &second.sine[0] ||
		&first.hann[0] == &second.hann[0] {
		t.Fatal("waveform tables share backing arrays across builds")
	}

	slabFrames := runtime.GOMAXPROCS(0)
	for _, frames := range []int{
		tensor.SingletonExtent,
		tensor.TripleExtent,
		slabFrames,
		slabFrames + tensor.SingletonExtent,
		tensor.TripleExtent*slabFrames + tensor.SingletonExtent,
	} {
		features := syntheticAudioFeatures(small, frames)
		got, err := audioFeaturesToWaveform(small, first, features)
		if err != nil {
			t.Fatalf("frames %d: %v", frames, err)
		}
		requireBitIdentical(t, got, fullBufferWaveform(small, first, features), frames)
	}

	tables := newAudioWaveformTables(compiled)
	features := syntheticAudioFeatures(compiled, 3)
	got, err := audioFeaturesToWaveform(compiled, tables, features)
	if err != nil {
		t.Fatal(err)
	}
	requireBitIdentical(t, got, fullBufferWaveform(compiled, tables, features), 3)

	tight := small
	tight.MagnitudeLimit = 50
	features = syntheticAudioFeatures(small, 4)
	loose, err := audioFeaturesToWaveform(small, first, features)
	if err != nil {
		t.Fatal(err)
	}
	clamped, err := audioFeaturesToWaveform(tight, first, features)
	if err != nil {
		t.Fatal(err)
	}
	requireBitIdentical(t, clamped, fullBufferWaveform(tight, first, features), 4)
	differs := false
	for index := range clamped {
		if clamped[index] != loose[index] {
			differs = true
			break
		}
	}
	if !differs {
		t.Fatal("magnitude limit from the plan did not change the waveform")
	}
}

func requireBitIdentical(t *testing.T, got, want []float32, frames int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("frames %d: length %d, want %d", frames, len(got), len(want))
	}
	for index := range got {
		if math.Float32bits(got[index]) != math.Float32bits(want[index]) {
			t.Fatalf("frames %d: sample %d = %x, want %x",
				frames, index, math.Float32bits(got[index]), math.Float32bits(want[index]))
		}
	}
}

// syntheticAudioFeatures: deterministic log-magnitude/phase frames whose
// magnitudes straddle exp values on both sides of the 50 and 100 clamps.
func syntheticAudioFeatures(plan model.AudioWaveformPlan, frames int) reference.Value {
	data := make([]float32, frames*plan.FrameWidth())
	state := uint32(0x9E3779B9)
	for index := range data {
		state = state*1664525 + 1013904223
		uniform := float64(state>>8) / float64(1<<24)
		if index%plan.FrameWidth() < plan.Bins() {
			data[index] = float32(-3 + 8.5*uniform)
		} else {
			data[index] = float32(-math.Pi + 2*math.Pi*uniform)
		}
	}
	return reference.Value{
		Shape: tensor.MustShape(uint64(plan.FrameWidth()), uint64(frames)),
		Data:  data,
	}
}

// fullBufferWaveform: the direct reference — every frame window synthesized
// into one frames-scale buffer, then overlap-added sequentially in frame
// order. The streamed path must match this bit for bit.
func fullBufferWaveform(
	plan model.AudioWaveformPlan, tables *audioWaveformTables, features reference.Value,
) []float32 {
	frames := int(features.Shape.Dims[1])
	windows := make([]float32, frames*plan.FFTSize)
	realPart := make([]float32, plan.Bins())
	imaginaryPart := make([]float32, plan.Bins())
	for frame := range frames {
		base := frame * plan.FrameWidth()
		for bin := 0; bin < plan.Bins(); bin++ {
			magnitude := float32(math.Exp(float64(features.Data[base+bin])))
			if magnitude > plan.MagnitudeLimit {
				magnitude = plan.MagnitudeLimit
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
	outputSize := (frames-1)*plan.HopSize + plan.FFTSize
	audio := make([]float32, outputSize)
	envelope := make([]float32, outputSize)
	for frame := range frames {
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
	audio = audio[:outputSize-2*plan.PadSize]
	for index := range audio {
		audio[index] /= envelope[index]
	}
	return audio
}
