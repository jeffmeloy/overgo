package dataset

import (
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipecontract"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestAudioSignalProfileAndAdmissionPublishExactLineage(t *testing.T) {
	store := overgodbtest.Open(t, t.TempDir())

	audioID := testutil.ArtifactID(t, artifact.KindFile, "audio")
	formatID := testutil.ArtifactID(t, artifact.KindProfile, "format")
	testutil.PublishArtifact(t, store, audioID)
	testutil.PublishArtifact(t, store, formatID)
	profile := testcheck.Must(audioSignalProfileCodec.NewInitial(AudioSignalProfileDocument{Profile: recipecontract.DecodedAudioSignalProfile{
		Source:      recipecontract.AudioReference{Audio: audioID, Profile: formatID},
		SchemaValid: true, DecodeStatus: recipecontract.AudioDecodeComplete,
		Format:     recipecontract.AudioFormat{SampleRate: 16_000, Channels: 1, Encoding: "pcm-f32le"},
		FrameCount: 16_000, SampleCount: 16_000, FiniteSampleCount: 16_000,
		ClipThreshold: 1, PeakAbsolute: 0.5, RootMeanSquare: 0.125, DCOffset: 0.01,
	}}))(t)
	policy := testcheck.Must(audioAdmissionPolicyCodec.NewInitial(AudioAdmissionPolicyDocument{Policy: recipecontract.AudioAdmissionPolicy{
		MinimumDurationNanoseconds: uint64(time.Second / 2), MaximumDurationNanoseconds: uint64(2 * time.Second),
		MinimumChannels: 1, MaximumChannels: 1, SilenceRMSThreshold: 0.001,
		MaximumClippedFraction: 0.01, MaximumAbsoluteDCOffset: 0.1,
	}}))(t)
	profileLineage := artifact.DependencyLineage(profile.ID, audioID, formatID)
	profileBatch := testcheck.Must(audioSignalProfileCodec.Batch(
		"fixture/audio-signal-profile", profile, profileLineage, nil,
	))(t)
	if _, err := artifact.CommitBatch(t.Context(), store, profileBatch); err != nil {
		t.Fatal(err)
	}
	policyBatch := testcheck.Must(audioAdmissionPolicyCodec.Batch("fixture/audio-admission-policy", policy, nil, nil))(t)
	if _, err := artifact.CommitBatch(t.Context(), store, policyBatch); err != nil {
		t.Fatal(err)
	}
	result := testcheck.Must(policy.Policy.Evaluate(profile.Profile))(t)
	decision := testcheck.Must(audioAdmissionDecisionCodec.NewInitial(AudioAdmissionDecisionDocument{
		Signal: profile.ID, Policy: policy.ID, Decision: result,
	}))(t)
	decisionLineage := artifact.DependencyLineage(decision.ID, decision.Signal, decision.Policy)
	batch := testcheck.Must(audioAdmissionDecisionCodec.Batch(
		"fixture/audio-admission-decision", decision, decisionLineage, nil,
	))(t)
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	loaded, err := audioAdmissionDecisionCodec.RequireExactLineage(
		t.Context(), store, decision.ID, func(value AudioAdmissionDecisionDocument) []artifact.Lineage {
			return artifact.DependencyLineage(value.ID, value.Signal, value.Policy)
		},
	)
	if err != nil || loaded.Decision.Outcome != recipecontract.AudioAdmissionAccepted {
		t.Fatalf("loaded decision = %+v, %v", loaded, err)
	}
}

func TestAudioAdmissionDecisionRejectsUnprovedLineage(t *testing.T) {
	store := overgodbtest.Open(t, t.TempDir())
	decision := testcheck.Must(audioAdmissionDecisionCodec.NewInitial(AudioAdmissionDecisionDocument{
		Signal:   testutil.ArtifactID(t, artifact.KindEvidence, "missing-signal"),
		Policy:   testutil.ArtifactID(t, artifact.KindProfile, "missing-policy"),
		Decision: recipecontract.AudioAdmissionDecision{Outcome: recipecontract.AudioAdmissionAccepted},
	}))(t)
	content := testcheck.Must(audioAdmissionDecisionCodec.Content(decision))(t)
	unbound := testcheck.Must(artifact.NewDocumentBatch("fixture/unbound-audio-admission", []artifact.Content{content}, nil, nil))(t)
	if _, err := artifact.CommitBatch(t.Context(), store, unbound); err != nil {
		t.Fatal(err)
	}
	if _, err := audioAdmissionDecisionCodec.RequireExactLineage(
		t.Context(), store, decision.ID, func(value AudioAdmissionDecisionDocument) []artifact.Lineage {
			return artifact.DependencyLineage(value.ID, value.Signal, value.Policy)
		},
	); err == nil {
		t.Fatal("audio admission decision accepted without stored lineage")
	}
}
