package runrecord

import (
	"context"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
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
	devTree, err := NewReviewWorktree("C:/repo/dev", "codex/dev", reviewCommitB, true)
	if err != nil {
		t.Fatal(err)
	}
	sqaTree, err := NewReviewWorktree("C:/repo/sqa", "codex/sqa", reviewCommitB, true)
	if err != nil {
		t.Fatal(err)
	}
	definition := testutil.ArtifactID(t, artifact.KindRecipe, "frozen evaluator definition")
	evaluator, err := NewReviewEvaluator("automation-foundation", definition, reviewCommitA)
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := NewReviewCandidate(ReviewCandidate{
		BaseCommit: reviewCommitA, CodeCommit: reviewCommitB,
		Developer: developer.ID, Worktree: devTree.ID, Evaluator: evaluator.ID,
		GateResult: testutil.ArtifactID(t, artifact.KindEvidence, "gate-result"),
		GateRun:    testutil.ArtifactID(t, artifact.KindEvidence, "gate-run"),
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
	if err := AdmitReview(reviewCommitB, ReviewAdmission{
		Developer: developer, Reviewer: reviewer, DeveloperWorktree: devTree, ReviewWorktree: sqaTree,
		Evaluator: evaluator, Candidate: candidate, Findings: []ReviewFinding{finding}, Verdict: verdict,
	}); err != nil {
		t.Fatalf("valid review admission failed: %v", err)
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

func TestReviewAdmission(t *testing.T) {
	developer, _ := NewReviewActor("local:developer", ReviewDeveloper)
	reviewer, _ := NewReviewActor("local:sqa", ReviewSQA)
	devTree, _ := NewReviewWorktree("C:/repo/dev", "codex/dev", reviewCommitB, true)
	reviewTree, _ := NewReviewWorktree("C:/repo/review", "codex/review", reviewCommitB, true)
	evaluator, _ := NewReviewEvaluator("review", testutil.ArtifactID(t, artifact.KindRecipe, "definition"), reviewCommitA)
	candidate, _ := NewReviewCandidate(ReviewCandidate{
		BaseCommit: reviewCommitA, CodeCommit: reviewCommitB, Developer: developer.ID,
		Worktree: devTree.ID, Evaluator: evaluator.ID,
		GateResult: testutil.ArtifactID(t, artifact.KindEvidence, "result"), GateRun: testutil.ArtifactID(t, artifact.KindEvidence, "run"),
	})
	resolved, _ := NewReviewFinding(ReviewFinding{
		Candidate: candidate.ID, Reviewer: reviewer.ID, Evaluator: evaluator.ID,
		Severity: ReviewMajor, Status: ReviewFindingResolved, Summary: "fixed", Check: "go test ./...",
	})
	verdict, _ := NewReviewVerdict(ReviewVerdict{
		Candidate: candidate.ID, Reviewer: reviewer.ID, Worktree: reviewTree.ID, Evaluator: evaluator.ID,
		TargetHead: reviewCommitB, Findings: []artifact.ID{resolved.ID}, Outcome: ReviewApproved,
	})
	valid := ReviewAdmission{
		Developer: developer, Reviewer: reviewer, DeveloperWorktree: devTree, ReviewWorktree: reviewTree,
		Evaluator: evaluator, Candidate: candidate, Findings: []ReviewFinding{resolved}, Verdict: verdict,
	}
	if err := AdmitReview(reviewCommitB, valid); err != nil {
		t.Fatal(err)
	}
	wrongActor := valid
	wrongActor.Reviewer = developer
	if err := AdmitReview(reviewCommitB, wrongActor); err == nil {
		t.Fatal("developer admitted as own SQA reviewer")
	}
	dirtyTree, _ := NewReviewWorktree("C:/repo/review", "codex/review", reviewCommitB, false)
	dirty := valid
	dirty.ReviewWorktree = dirtyTree
	dirty.Verdict, _ = NewReviewVerdict(ReviewVerdict{
		Candidate: candidate.ID, Reviewer: reviewer.ID, Worktree: dirtyTree.ID, Evaluator: evaluator.ID,
		TargetHead: reviewCommitB, Findings: []artifact.ID{resolved.ID}, Outcome: ReviewApproved,
	})
	if err := AdmitReview(reviewCommitB, dirty); err == nil {
		t.Fatal("dirty review worktree admitted")
	}
	if err := AdmitReview(reviewCommitA, valid); err == nil {
		t.Fatal("verdict admitted at a different target head")
	}
	open, _ := NewReviewFinding(ReviewFinding{
		Candidate: candidate.ID, Reviewer: reviewer.ID, Evaluator: evaluator.ID,
		Severity: ReviewMajor, Status: ReviewFindingOpen, Summary: "open", Check: "go test ./...",
	})
	openPacket := valid
	openPacket.Findings = []ReviewFinding{open}
	openPacket.Verdict, _ = NewReviewVerdict(ReviewVerdict{
		Candidate: candidate.ID, Reviewer: reviewer.ID, Worktree: reviewTree.ID, Evaluator: evaluator.ID,
		TargetHead: reviewCommitB, Findings: []artifact.ID{open.ID}, Outcome: ReviewApproved,
	})
	if err := AdmitReview(reviewCommitB, openPacket); err == nil {
		t.Fatal("open finding admitted")
	}
	omitted := valid
	omitted.Findings = nil
	if err := AdmitReview(reviewCommitB, omitted); err == nil {
		t.Fatal("omitted finding admitted")
	}
	notApproved := valid
	notApproved.Verdict, _ = NewReviewVerdict(ReviewVerdict{
		Candidate: candidate.ID, Reviewer: reviewer.ID, Worktree: reviewTree.ID, Evaluator: evaluator.ID,
		TargetHead: reviewCommitB, Findings: []artifact.ID{resolved.ID}, Outcome: ReviewChangesRequired,
	})
	if err := AdmitReview(reviewCommitB, notApproved); err == nil {
		t.Fatal("changes-required verdict admitted")
	}
}

func TestReviewPriority(t *testing.T) {
	developer, _ := NewReviewActor("local:developer", ReviewDeveloper)
	reviewer, _ := NewReviewActor("local:sqa", ReviewSQA)
	devTree, _ := NewReviewWorktree("C:/repo/dev", "codex/dev", reviewCommitB, true)
	reviewTree, _ := NewReviewWorktree("C:/repo/review", "codex/review", reviewCommitB, true)
	evaluator, _ := NewReviewEvaluator("review", testutil.ArtifactID(t, artifact.KindRecipe, "priority definition"), reviewCommitA)
	candidate, _ := NewReviewCandidate(ReviewCandidate{
		BaseCommit: reviewCommitA, CodeCommit: reviewCommitB, Developer: developer.ID,
		Worktree: devTree.ID, Evaluator: evaluator.ID,
		GateResult: testutil.ArtifactID(t, artifact.KindEvidence, "priority result"), GateRun: testutil.ArtifactID(t, artifact.KindEvidence, "priority run"),
	})
	verdict, _ := NewReviewVerdict(ReviewVerdict{
		Candidate: candidate.ID, Reviewer: reviewer.ID, Worktree: reviewTree.ID, Evaluator: evaluator.ID,
		TargetHead: reviewCommitB, Findings: []artifact.ID{}, Outcome: ReviewApproved,
	})
	contents, err := ReviewContents(developer, reviewer, devTree, reviewTree, evaluator, candidate, verdict)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch("test/review-priority", contents, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(filepath.Join(t.TempDir(), "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.Commit(context.Background(), batch); err != nil {
		t.Fatal(err)
	}
	priority, err := DeriveReviewPriority(context.Background(), store, reviewCommitB, []ReviewCandidate{candidate}, nil)
	if err != nil || priority.Phase != ReviewPhaseSQA || priority.Candidate != candidate.ID {
		t.Fatalf("candidate phase = (%+v, %v)", priority, err)
	}
	priority, err = DeriveReviewPriority(context.Background(), store, reviewCommitB, []ReviewCandidate{candidate}, []ReviewVerdict{verdict})
	if err != nil || priority.Phase != ReviewPhasePriority || priority.Candidate != candidate.ID || priority.Verdict != verdict.ID {
		t.Fatalf("admitted phase = (%+v, %v)", priority, err)
	}
	priority, err = DeriveReviewPriority(context.Background(), store, reviewCommitA, []ReviewCandidate{candidate}, []ReviewVerdict{verdict})
	if err != nil || priority.Phase != ReviewPhaseImplementation {
		t.Fatalf("unreviewed head phase = (%+v, %v)", priority, err)
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
	parsed, err := reviewActorCodec.Parse(content.Data)
	if err != nil || parsed.ID != actor.ID {
		t.Fatalf("shared document codec round trip = (%+v, %v)", parsed, err)
	}
	if _, err := reviewActorCodec.Parse([]byte(`{"version":1,"principal":"local:sqa","role":"sqa","unknown":true}`)); err == nil {
		t.Fatal("shared document codec accepted an unknown field")
	}

	findings := []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "finding")}
	verdict, err := NewReviewVerdict(ReviewVerdict{
		Candidate:  testutil.ArtifactID(t, artifact.KindEvidence, "candidate"),
		Reviewer:   testutil.ArtifactID(t, artifact.KindEvidence, "reviewer"),
		Worktree:   testutil.ArtifactID(t, artifact.KindEvidence, "worktree"),
		Evaluator:  testutil.ArtifactID(t, artifact.KindEvidence, "evaluator"),
		TargetHead: reviewCommitA,
		Findings:   findings,
		Outcome:    ReviewApproved,
	})
	if err != nil {
		t.Fatal(err)
	}
	findings[0] = testutil.ArtifactID(t, artifact.KindEvidence, "mutated finding")
	if verdict.Findings[0] == findings[0] {
		t.Fatal("shared document codec retained mutable input")
	}
	mutated := actor
	mutated.Principal = "local:someone-else"
	if err := mutated.ValidateIdentity(); err == nil {
		t.Fatal("identity validation accepted mutated review content")
	}
	if got := artifact.DependencyLineage(actor.ID, testutil.ArtifactID(t, artifact.KindEvidence, "parent")); len(got) != 1 || got[0].Child != actor.ID {
		t.Fatalf("shared dependency lineage = %+v", got)
	}
}

const (
	reviewCommitA = "0123456789abcdef0123456789abcdef01234567"
	reviewCommitB = "89abcdef0123456789abcdef0123456789abcdef"
)
