package runrecord

import (
	"overgo/internal/artifact"
	"overgo/internal/testutil"
	"testing"
)

func TestAttemptDirectionNovelty(t *testing.T) {
	input := AttemptDirectionInput{Strategy: testutil.ArtifactID(t, artifact.KindProfile, "strategy"), BaseManifest: testutil.ArtifactID(t, artifact.KindProfile, "base"),
		CandidateManifest: testutil.ArtifactID(t, artifact.KindProfile, "candidate"), Symbols: []string{"pkg.B", "pkg.A"}, Checks: []string{"tests"}, Objective: testutil.ArtifactID(t, artifact.KindRecipe, "task")}
	firstAttempt := fixtureAttempt(t)
	firstAttempt, _ = NewAttemptRecord(firstAttempt)
	first, err := ObserveAttemptDirection(input, firstAttempt, nil)
	if err != nil {
		t.Fatal(err)
	}
	failed := fixtureAttempt(t)
	failed.Outcome, failed.Failure = OutcomeFailed, "tests"
	failed, _ = NewAttemptRecord(failed)
	repeated, err := ObserveAttemptDirection(input, failed, []AttemptDirection{first})
	if err != nil {
		t.Fatal(err)
	}
	if repeated.Fingerprint != first.Fingerprint || repeated.Count != 2 || !repeated.Pivot || repeated.First != firstAttempt.ID || repeated.Last != failed.ID {
		t.Fatalf("direction = %+v", repeated)
	}
	input.Checks = append(input.Checks, "device")
	novel, err := ObserveAttemptDirection(input, failed, []AttemptDirection{first})
	if err != nil || novel.Fingerprint == first.Fingerprint || novel.Pivot {
		t.Fatalf("novel direction = %+v %v", novel, err)
	}
}
