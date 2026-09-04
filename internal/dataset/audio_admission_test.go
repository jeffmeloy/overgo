package dataset

import (
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
)

func TestAudioSignalProfileAndAdmissionPublishExactLineage(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	audioID := testutil.ArtifactID(t, artifact.KindFile, "audio")
	formatID := testutil.ArtifactID(t, artifact.KindProfile, "format")
	testutil.PublishArtifact(t, store, audioID)
	testutil.PublishArtifact(t, store, formatID)
	profile, err := audioSignalProfileCodec.NewInitial(AudioSignalProfileDocument{Profile: recipecontract.DecodedAudioSignalProfile{
		Source:      recipecontract.AudioReference{Audio: audioID, Profile: formatID},
		SchemaValid: true, DecodeStatus: recipecontract.AudioDecodeComplete,
		Format:     recipecontract.AudioFormat{SampleRate: 16_000, Channels: 1, Encoding: "pcm-f32le"},
		FrameCount: 16_000, SampleCount: 16_000, FiniteSampleCount: 16_000,
		ClipThreshold: 1, PeakAbsolute: 0.5, RootMeanSquare: 0.125, DCOffset: 0.01,
	}})
	if err != nil {
		t.Fatal(err)
	}
	policy, err := audioAdmissionPolicyCodec.NewInitial(AudioAdmissionPolicyDocument{Policy: recipecontract.AudioAdmissionPolicy{
		MinimumDurationNanoseconds: uint64(time.Second / 2), MaximumDurationNanoseconds: uint64(2 * time.Second),
		MinimumChannels: 1, MaximumChannels: 1, SilenceRMSThreshold: 0.001,
		MaximumClippedFraction: 0.01, MaximumAbsoluteDCOffset: 0.1,
	}})
	if err != nil {
		t.Fatal(err)
	}
	profileLineage := artifact.DependencyLineage(profile.ID, audioID, formatID)
	profileBatch, err := audioSignalProfileCodec.Batch(
		"fixture/audio-signal-profile", profile, profileLineage, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, profileBatch); err != nil {
		t.Fatal(err)
	}
	policyBatch, err := audioAdmissionPolicyCodec.Batch("fixture/audio-admission-policy", policy, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, policyBatch); err != nil {
		t.Fatal(err)
	}
	result, err := policy.Policy.Evaluate(profile.Profile)
	if err != nil {
		t.Fatal(err)
	}
	decision, err := audioAdmissionDecisionCodec.NewInitial(AudioAdmissionDecisionDocument{
		Signal: profile.ID, Policy: policy.ID, Decision: result,
	})
	if err != nil {
		t.Fatal(err)
	}
	decisionLineage := artifact.DependencyLineage(decision.ID, decision.Signal, decision.Policy)
	batch, err := audioAdmissionDecisionCodec.Batch(
		"fixture/audio-admission-decision", decision, decisionLineage, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
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
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	decision, err := audioAdmissionDecisionCodec.NewInitial(AudioAdmissionDecisionDocument{
		Signal:   testutil.ArtifactID(t, artifact.KindEvidence, "missing-signal"),
		Policy:   testutil.ArtifactID(t, artifact.KindProfile, "missing-policy"),
		Decision: recipecontract.AudioAdmissionDecision{Outcome: recipecontract.AudioAdmissionAccepted},
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := audioAdmissionDecisionCodec.Content(decision)
	if err != nil {
		t.Fatal(err)
	}
	unbound, err := artifact.NewDocumentBatch("fixture/unbound-audio-admission", []artifact.Content{content}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
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
