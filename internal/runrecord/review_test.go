package runrecord

import (
	"testing"

	"overgo/internal/artifact"
)

func TestIndependentSQAIdentities(t *testing.T) {
	developer, err := NewReviewActor("local:developer-a", ReviewDeveloper)
	if err != nil {
		t.Fatal(err)
	}
	reviewer, err := NewReviewActor("local:sqa-b", ReviewSQA)
	if err != nil {
		t.Fatal(err)
	}
	devTree, err := NewReviewWorktree("C:/repo/dev", "codex/dev", reviewCommitA, true)
	if err != nil {
		t.Fatal(err)
	}
	sqaTree, err := NewReviewWorktree("C:/repo/sqa", "codex/sqa", reviewCommitB, true)
	if err != nil {
		t.Fatal(err)
	}
	definition := reviewID(t, artifact.KindRecipe, "frozen evaluator definition")
	evaluator, err := NewReviewEvaluator("automation-foundation", definition, reviewCommitA)
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := NewReviewCandidate(ReviewCandidate{
		BaseCommit: reviewCommitA, CodeCommit: reviewCommitB,
		Developer: developer.ID, Worktree: devTree.ID, Evaluator: evaluator.ID,
		GateResult: reviewID(t, artifact.KindEvidence, "gate-result"),
		GateRun:    reviewID(t, artifact.KindEvidence, "gate-run"),
	})
	if err != nil {
		t.Fatal(err)
	}
	finding, err := NewReviewFinding(ReviewFinding{
		Candidate: candidate.ID, Reviewer: reviewer.ID, Evaluator: evaluator.ID,
		Severity: ReviewMajor, Status: ReviewFindingResolved,
		Summary: "retry identity omitted GOFLAGS", Check: "go test ./cmd/gate -run TestEnvironmentBoundRetry -v",
	})
	if err != nil {
		t.Fatal(err)
	}
	verdict, err := NewReviewVerdict(ReviewVerdict{
		Candidate: candidate.ID, Reviewer: reviewer.ID, Worktree: sqaTree.ID,
		Evaluator: evaluator.ID, TargetHead: reviewCommitB,
		Findings: []artifact.ID{finding.ID}, Outcome: ReviewApproved,
	})
	if err != nil {
		t.Fatal(err)
	}
	contents, err := ReviewContents(developer, reviewer, devTree, sqaTree, evaluator, candidate, finding, verdict)
	if err != nil {
		t.Fatal(err)
	}
	if len(contents) != 8 || contents[5].Descriptor.MediaType != ReviewCandidateMediaType || contents[7].Descriptor.MediaType != ReviewVerdictMediaType {
		t.Fatalf("review contents do not preserve typed identities: %+v", contents)
	}
	if len(candidate.Lineage()) != 5 || len(finding.Lineage()) != 3 || len(verdict.Lineage()) != 5 {
		t.Fatal("review identity graph omitted a required dependency")
	}

	mutated := verdict
	mutated.TargetHead = reviewCommitA
	if err := mutated.ValidateIdentity(); err == nil {
		t.Fatal("mutated verdict retained its immutable identity")
	}
	if _, err := NewReviewWorktree(`C:\repo\sqa`, "codex/sqa", reviewCommitB, true); err == nil {
		t.Fatal("non-normalized worktree identity accepted")
	}
	if _, err := NewReviewVerdict(ReviewVerdict{
		Candidate: candidate.ID, Reviewer: reviewer.ID, Worktree: sqaTree.ID,
		Evaluator: evaluator.ID, TargetHead: reviewCommitB, Findings: nil, Outcome: ReviewApproved,
	}); err == nil {
		t.Fatal("verdict with an ambiguous nil findings set accepted")
	}
}

const (
	reviewCommitA = "0123456789abcdef0123456789abcdef01234567"
	reviewCommitB = "89abcdef0123456789abcdef0123456789abcdef"
)

func reviewID(t *testing.T, kind artifact.Kind, value string) artifact.ID {
	t.Helper()
	id, err := artifact.IdentifyBytes(kind, []byte(value))
	if err != nil {
		t.Fatal(err)
	}
	return id
}
