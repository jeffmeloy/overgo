package recipecontract

import (
	"math"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestAudioArtifactContractsUseSampleExactGeometry(t *testing.T) {
	profile := testutil.ArtifactID(t, artifact.KindProfile, "audio-format")
	audio := testutil.ArtifactID(t, artifact.KindFile, "audio")
	transcript := testutil.ArtifactID(t, artifact.KindOutput, "transcript")
	checkpoint := testutil.ArtifactID(t, artifact.KindCheckpoint, "audio-state")
	reference := AudioReference{Audio: audio, Profile: profile}

	geometry := AudioFrameGeometry{WindowSamples: 400, HopSamples: 160, FeatureBins: 80}
	frames, err := geometry.FrameCount(1_600)
	if err != nil || frames != 8 {
		t.Fatalf("frame count = %d, %v", frames, err)
	}
	if frames, err := geometry.FrameCount(399); err != nil || frames != 0 {
		t.Fatalf("short frame count = %d, %v", frames, err)
	}

	contracts := []interface{ Validate() error }{
		AudioFormat{SampleRate: 16_000, Channels: 1, Encoding: "pcm-f32le"},
		Transcription{Source: reference, Text: "sample transcript", Language: "en"},
		TimestampedAlignment{
			Source: reference, Transcription: transcript,
			Items: []AlignedText{
				{Span: SampleSpan{Start: 0, End: 800}, Text: "sample", Confidence: 0.9},
				{Span: SampleSpan{Start: 800, End: 1_600}, Text: "transcript", Confidence: 0.8},
			},
		},
		SpeechTurns{
			Source: reference, State: checkpoint,
			Turns: []SpeechTurn{
				{Span: SampleSpan{Start: 0, End: 1_000}, Speaker: "speaker-1", Confidence: 0.7},
				{Span: SampleSpan{Start: 500, End: 1_500}, Speaker: "speaker-2", Confidence: 0.6},
			},
		},
		ActivitySegments{
			Source: reference, State: checkpoint,
			Segments: []ActivitySegment{
				{Span: SampleSpan{Start: 0, End: 400}, Confidence: 0.75},
				{Span: SampleSpan{Start: 800, End: 1_200}, Confidence: 0.85},
			},
		},
		ConvertedAudio{Source: reference, Output: AudioReference{
			Audio: testutil.ArtifactID(t, artifact.KindOutput, "converted"), Profile: profile,
		}},
		GeneratedAudio{
			Context: testutil.ArtifactID(t, artifact.KindProfile, "generation-context"),
			State:   checkpoint,
			Output:  AudioReference{Audio: testutil.ArtifactID(t, artifact.KindOutput, "generated"), Profile: profile},
		},
	}
	for index, contract := range contracts {
		if err := contract.Validate(); err != nil {
			t.Fatalf("contract[%d]: %v", index, err)
		}
	}
}

func TestAudioArtifactContractsRejectAmbiguousTimingAndState(t *testing.T) {
	profile := testutil.ArtifactID(t, artifact.KindProfile, "audio-format")
	reference := AudioReference{
		Audio: testutil.ArtifactID(t, artifact.KindFile, "audio"), Profile: profile,
	}
	transcript := testutil.ArtifactID(t, artifact.KindOutput, "transcript")
	tests := []struct {
		name     string
		contract interface{ Validate() error }
	}{
		{"format", AudioFormat{SampleRate: 16_000, Channels: 1, Encoding: "PCM"}},
		{"frame", AudioFrameGeometry{WindowSamples: 400}},
		{"span", SampleSpan{Start: 20, End: 20}},
		{"alignment overlap", TimestampedAlignment{
			Source: reference, Transcription: transcript,
			Items: []AlignedText{
				{Span: SampleSpan{Start: 0, End: 20}, Text: "a", Confidence: 1},
				{Span: SampleSpan{Start: 10, End: 30}, Text: "b", Confidence: 1},
			},
		}},
		{"turn state", SpeechTurns{
			Source: reference, State: testutil.ArtifactID(t, artifact.KindProfile, "not-state"),
			Turns: []SpeechTurn{{Span: SampleSpan{Start: 0, End: 1}, Speaker: "speaker", Confidence: 1}},
		}},
		{"activity confidence", ActivitySegments{
			Source:   reference,
			Segments: []ActivitySegment{{Span: SampleSpan{Start: 0, End: 1}, Confidence: math.NaN()}},
		}},
		{"converted input as output", ConvertedAudio{Source: reference, Output: reference}},
		{"generated input as output", GeneratedAudio{
			Context: testutil.ArtifactID(t, artifact.KindProfile, "generation-context"), Output: reference,
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := test.contract.Validate(); err == nil {
				t.Fatal("invalid audio contract accepted")
			}
		})
	}
}
