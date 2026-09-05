package audiodsp

import (
	"context"
	"errors"
	"math"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/media"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
)

func TestAudioSignalIntegrity(t *testing.T) {
	source := recipecontract.AudioReference{Audio: testutil.ArtifactID(t, artifact.KindFile, "wave"), Profile: testutil.ArtifactID(t, artifact.KindProfile, "format")}
	format := recipecontract.AudioFormat{SampleRate: 16000, Channels: 2, Encoding: "pcm-f32le"}
	for _, test := range []struct {
		name                       string
		samples                    []float32
		finite, nonfinite, clipped uint64
		peak, rms, dc              float64
	}{
		{"silent", []float32{0, 0}, 2, 0, 0, 0, 0, 0},
		{"near-silent", []float32{1.0 / 65536, -1.0 / 65536}, 2, 0, 0, 1.0 / 65536, 1.0 / 65536, 0},
		{"speech-range", []float32{.5, -.5}, 2, 0, 0, .5, .5, 0},
		{"clipped", []float32{1, -1}, 2, 0, 2, 1, 1, 0},
		{"non-finite", []float32{float32(math.NaN()), float32(math.Inf(1)), .5, -.5}, 2, 2, 0, .5, .5, 0},
		{"all-non-finite", []float32{float32(math.NaN()), float32(math.Inf(-1))}, 0, 2, 0, 0, 0, 0},
		{"constant", []float32{.25, .25}, 2, 0, 0, .25, .25, .25},
		{"finite-extreme", []float32{math.MaxFloat32, -math.MaxFloat32}, 2, 0, 2, math.MaxFloat32, math.MaxFloat32, 0},
	} {
		t.Run(test.name, func(t *testing.T) {
			p, err := MeasureSignal(t.Context(), source, media.DecodedAudio{Format: format, Samples: test.samples}, 1)
			if err != nil {
				t.Fatal(err)
			}
			if p.SampleCount != uint64(len(test.samples)) || p.FrameCount != uint64(len(test.samples)/2) || p.FiniteSampleCount != test.finite || p.NonFiniteSampleCount != test.nonfinite || p.ClippedSampleCount != test.clipped || p.PeakAbsolute != test.peak || p.RootMeanSquare != test.rms || p.DCOffset != test.dc {
				t.Fatalf("profile=%+v", p)
			}
		})
	}
	if _, err := MeasureSignal(t.Context(), source, media.DecodedAudio{Format: format, Samples: []float32{1}}, 1); err == nil {
		t.Fatal("partial channel frame accepted")
	}
	for _, threshold := range []float64{0, -1, math.NaN(), math.Inf(1)} {
		if _, err := MeasureSignal(t.Context(), source, media.DecodedAudio{Format: format}, threshold); err == nil {
			t.Fatal("invalid threshold accepted")
		}
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(errors.New("stop"))
	if _, err := MeasureSignal(ctx, source, media.DecodedAudio{Format: format}, 1); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}
