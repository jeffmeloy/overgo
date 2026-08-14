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

func TestReviewDocumentFamily(t *testing.T) {
	actor, err := NewReviewActor("local:sqa", ReviewSQA)
	if err != nil {
		t.Fatal(err)
	}
	content, err := actor.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseReviewActor(content.Data)
	if err != nil || parsed.ID != actor.ID {
		t.Fatalf("shared document codec round trip = (%+v, %v)", parsed, err)
	}
	if _, err := ParseReviewActor([]byte(`{"version":1,"principal":"local:sqa","role":"sqa","unknown":true}`)); err == nil {
		t.Fatal("shared document codec accepted an unknown field")
	}

	findings := []artifact.ID{reviewID(t, artifact.KindEvidence, "finding")}
	verdict, err := NewReviewVerdict(ReviewVerdict{
		Candidate:  reviewID(t, artifact.KindEvidence, "candidate"),
		Reviewer:   reviewID(t, artifact.KindEvidence, "reviewer"),
		Worktree:   reviewID(t, artifact.KindEvidence, "worktree"),
		Evaluator:  reviewID(t, artifact.KindEvidence, "evaluator"),
		TargetHead: reviewCommitA,
		Findings:   findings,
		Outcome:    ReviewApproved,
	})
	if err != nil {
		t.Fatal(err)
	}
	findings[0] = reviewID(t, artifact.KindEvidence, "mutated finding")
	if verdict.Findings[0] == findings[0] {
		t.Fatal("shared document codec retained mutable input")
	}
	mutated := actor
	mutated.Principal = "local:someone-else"
	if err := mutated.ValidateIdentity(); err == nil {
		t.Fatal("identity validation accepted mutated review content")
	}
	if got := dependencyLineage(actor.ID, reviewID(t, artifact.KindEvidence, "parent")); len(got) != 1 || got[0].Child != actor.ID {
		t.Fatalf("shared dependency lineage = %+v", got)
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
