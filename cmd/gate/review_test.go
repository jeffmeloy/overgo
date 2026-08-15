package main

import (
	"context"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
)

func TestReviewAdmissionStore(t *testing.T) {
	const base = "0123456789abcdef0123456789abcdef01234567"
	const target = "89abcdef0123456789abcdef0123456789abcdef"
	developer := mustReview(runrecord.NewReviewActor("local:developer", runrecord.ReviewDeveloper))
	reviewer := mustReview(runrecord.NewReviewActor("local:sqa", runrecord.ReviewSQA))
	developerTree := mustReview(runrecord.NewReviewWorktree("C:/repo/dev", "codex/dev", target, true))
	reviewTree := mustReview(runrecord.NewReviewWorktree("C:/repo/review", "codex/review", target, true))
	evaluator := mustReview(runrecord.NewReviewEvaluator("gate", reviewArtifactID(t, artifact.KindRecipe, "definition"), base))
	candidate := mustReview(runrecord.NewReviewCandidate(runrecord.ReviewCandidate{
		BaseCommit: base, CodeCommit: target, Developer: developer.ID, Worktree: developerTree.ID, Evaluator: evaluator.ID,
		GateResult: reviewArtifactID(t, artifact.KindEvidence, "result"), GateRun: reviewArtifactID(t, artifact.KindEvidence, "run"),
	}))
	finding := mustReview(runrecord.NewReviewFinding(runrecord.ReviewFinding{
		Candidate: candidate.ID, Reviewer: reviewer.ID, Evaluator: evaluator.ID,
		Severity: runrecord.ReviewMajor, Status: runrecord.ReviewFindingResolved, Summary: "fixed", Check: "go test ./...",
	}))
	verdict := mustReview(runrecord.NewReviewVerdict(runrecord.ReviewVerdict{
		Candidate: candidate.ID, Reviewer: reviewer.ID, Worktree: reviewTree.ID, Evaluator: evaluator.ID,
		TargetHead: target, Findings: []artifact.ID{finding.ID}, Outcome: runrecord.ReviewApproved,
	}))
	contents := mustReview(runrecord.ReviewContents(developer, reviewer, developerTree, reviewTree, evaluator, candidate, finding, verdict))
	batch := mustReview(artifact.NewDocumentBatch("test/review-admission", contents, nil, nil))
	root := t.TempDir()
	store := mustReview(repodb.Open(filepath.Join(root, "store")))
	if _, err := store.Commit(context.Background(), batch); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if err := admitStoredReview(root, "store", verdict.ID.String(), target); err != nil {
		t.Fatal(err)
	}
	if err := admitStoredReview(root, "store", verdict.ID.String(), base); err == nil {
		t.Fatal("stored review admitted at the wrong target")
	}
}

func mustReview[T any](value T, err error) T {
	if err != nil {
		panic(err)
	}
	return value
}

func reviewArtifactID(t *testing.T, kind artifact.Kind, value string) artifact.ID {
	t.Helper()
	id, err := artifact.IdentifyBytes(kind, []byte(value))
	if err != nil {
		t.Fatal(err)
	}
	return id
}
