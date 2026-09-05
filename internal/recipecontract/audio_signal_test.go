package recipecontract

import (
	"math"
	"slices"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestDecodedAudioSignalProfileDerivesDurationAndAdmission(t *testing.T) {
	profile := decodedAudioSignalFixture(t)
	duration, ok := profile.DurationNanoseconds()
	if !ok || duration != uint64(time.Second) {
		t.Fatalf("duration = %d, %t", duration, ok)
	}
	policy := AudioAdmissionPolicy{
		MinimumDurationNanoseconds: uint64(time.Second / 2), MaximumDurationNanoseconds: uint64(2 * time.Second),
		MinimumChannels: 1, MaximumChannels: 2, SilenceRMSThreshold: 0.001,
		MaximumClippedFraction: 0.01, MaximumAbsoluteDCOffset: 0.1,
	}
	decision, err := policy.Evaluate(profile)
	if err != nil {
		t.Fatal(err)
	}
	if decision.Outcome != AudioAdmissionAccepted || len(decision.Violations) != 0 {
		t.Fatalf("decision = %+v", decision)
	}
}

func TestAudioAdmissionQuarantinesEveryMeasuredFailureClass(t *testing.T) {
	profile := decodedAudioSignalFixture(t)
	profile.SchemaValid = false
	profile.DecodeStatus = AudioDecodeCorrupt
	profile.Format = AudioFormat{}
	profile.FrameCount, profile.SampleCount, profile.FiniteSampleCount = 0, 0, 0
	profile.ClipThreshold, profile.PeakAbsolute, profile.RootMeanSquare = 0, 0, 0
	profile.DCOffset = 0
	policy := AudioAdmissionPolicy{
		MinimumChannels: 1, MaximumChannels: 1, MaximumClippedFraction: 0,
		SilenceRMSThreshold: 0.01, MaximumAbsoluteDCOffset: 0.01,
	}
	decision, err := policy.Evaluate(profile)
	if err != nil {
		t.Fatal(err)
	}
	want := []AudioSignalViolation{AudioViolationDecodeIncomplete, AudioViolationSchemaInvalid}
	slices.Sort(want)
	if decision.Outcome != AudioAdmissionQuarantined || !slices.Equal(decision.Violations, want) {
		t.Fatalf("decision = %+v, want %v", decision, want)
	}

	profile = decodedAudioSignalFixture(t)
	profile.FrameCount = 800
	profile.SampleCount = 800
	profile.FiniteSampleCount = 798
	profile.NonFiniteSampleCount = 2
	profile.ClippedSampleCount = 80
	profile.PeakAbsolute = 1
	profile.RootMeanSquare = 0.001
	profile.DCOffset = 0.0008
	policy.MinimumDurationNanoseconds = uint64(time.Second)
	policy.MaximumAbsoluteDCOffset = 0.0005
	decision, err = policy.Evaluate(profile)
	if err != nil {
		t.Fatal(err)
	}
	want = []AudioSignalViolation{
		AudioViolationClipped, AudioViolationDCOffset, AudioViolationDurationBelowMinimum,
		AudioViolationNonFiniteSamples, AudioViolationSilent,
	}
	slices.Sort(want)
	if !slices.Equal(decision.Violations, want) {
		t.Fatalf("violations = %v, want %v", decision.Violations, want)
	}
}

func TestDecodedAudioSignalProfileRejectsInconsistentEvidence(t *testing.T) {
	tests := []DecodedAudioSignalProfile{
		{Source: decodedAudioSignalFixture(t).Source, SchemaValid: false, DecodeStatus: AudioDecodeComplete},
		func() DecodedAudioSignalProfile {
			value := decodedAudioSignalFixture(t)
			value.SampleCount++
			return value
		}(),
		func() DecodedAudioSignalProfile {
			value := decodedAudioSignalFixture(t)
			value.PeakAbsolute = math.NaN()
			return value
		}(),
		func() DecodedAudioSignalProfile {
			value := decodedAudioSignalFixture(t)
			value.ClippedSampleCount = 1
			return value
		}(),
	}
	for index, profile := range tests {
		if err := profile.Validate(); err == nil {
			t.Fatalf("profile[%d] accepted", index)
		}
	}
}

func decodedAudioSignalFixture(t *testing.T) DecodedAudioSignalProfile {
	t.Helper()
	return DecodedAudioSignalProfile{
		Source: AudioReference{
			Audio:   testutil.ArtifactID(t, artifact.KindFile, "signal.wav"),
			Profile: testutil.ArtifactID(t, artifact.KindProfile, "audio-format"),
		},
		SchemaValid: true, DecodeStatus: AudioDecodeComplete,
		Format:     AudioFormat{SampleRate: 16_000, Channels: 1, Encoding: "pcm-f32le"},
		FrameCount: 16_000, SampleCount: 16_000, FiniteSampleCount: 16_000,
		ClipThreshold: 1, PeakAbsolute: 0.5, RootMeanSquare: 0.125, DCOffset: 0.01,
	}
}
