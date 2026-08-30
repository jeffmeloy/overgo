package main

import (
	"context"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestReviewAdmissionStore(t *testing.T) {
	const base = "0123456789abcdef0123456789abcdef01234567"
	const target = "89abcdef0123456789abcdef0123456789abcdef"
	developer := mustGateValue(runrecord.NewReviewActor("local:developer", runrecord.ReviewDeveloper))
	reviewer := mustGateValue(runrecord.NewReviewActor("local:sqa", runrecord.ReviewSQA))
	developerTree := mustGateValue(runrecord.NewReviewWorktree("C:/repo/dev", "codex/dev", target, true))
	reviewTree := mustGateValue(runrecord.NewReviewWorktree("C:/repo/review", "codex/review", target, true))
	evaluator := mustGateValue(runrecord.NewReviewEvaluator("gate", testutil.ArtifactID(t, artifact.KindRecipe, "definition"), base))
	candidate := mustGateValue(runrecord.NewReviewCandidate(runrecord.ReviewCandidate{
		BaseCommit: base, CodeCommit: target, Developer: developer.ID, Worktree: developerTree.ID, Evaluator: evaluator.ID,
		GateResult: testutil.ArtifactID(t, artifact.KindEvidence, "result"), GateRun: testutil.ArtifactID(t, artifact.KindEvidence, "run"),
	}))
	finding := mustGateValue(runrecord.NewReviewFinding(runrecord.ReviewFinding{
		Candidate: candidate.ID, Reviewer: reviewer.ID, Evaluator: evaluator.ID,
		Severity: runrecord.ReviewMajor, Status: runrecord.ReviewFindingResolved, Summary: "fixed", Check: "go test ./...",
	}))
	verdict := mustGateValue(runrecord.NewReviewVerdict(runrecord.ReviewVerdict{
		Candidate: candidate.ID, Reviewer: reviewer.ID, Worktree: reviewTree.ID, Evaluator: evaluator.ID,
		TargetHead: target, Findings: []artifact.ID{finding.ID}, Outcome: runrecord.ReviewApproved,
	}))
	contents := mustGateValue(runrecord.ReviewContents(developer, reviewer, developerTree, reviewTree, evaluator, candidate, finding, verdict))
	batch := mustGateValue(artifact.NewDocumentBatch("test/review-admission", contents, nil, nil))
	root := t.TempDir()
	store := mustGateValue(overgodb.Open(filepath.Join(root, "store")))
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

func mustGateValue[T any](value T, err error) T {
	if err != nil {
		panic(err)
	}
	return value
}
