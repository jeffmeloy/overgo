package gate

import (
	"os"
	"strings"
	"testing"
)

func TestPreparedGateReferenceRefusesDifferentBranchAtSameParent(t *testing.T) {
	repository, intent := newReferenceUpdateFixture(t)
	reference := intent.HeadReference
	runGitFixture(t, repository, "branch", "other", intent.Parent)
	runGitFixture(t, repository, "checkout", "-q", "other")

	err := installCompletionIndexAndAdvanceGateReference(repository, intent)
	if err == nil || !strings.Contains(err.Error(), "reference or parent moved") {
		t.Fatalf("different-branch update error = %v", err)
	}
	if got := recoveryGit(t, repository, "rev-parse", reference); got != intent.Parent {
		t.Fatalf("captured branch = %s, want unchanged parent %s", got, intent.Parent)
	}
	if got := recoveryGit(t, repository, "rev-parse", "HEAD"); got != intent.Parent {
		t.Fatalf("current branch = %s, want unchanged parent %s", got, intent.Parent)
	}
}

func TestPreparedGateReferenceRefusesDetachedHead(t *testing.T) {
	repository, intent := newReferenceUpdateFixture(t)
	runGitFixture(t, repository, "checkout", "-q", "--detach", intent.Parent)

	err := installCompletionIndexAndAdvanceGateReference(repository, intent)
	if err == nil || !strings.Contains(err.Error(), "reference or parent moved") {
		t.Fatalf("detached update error = %v", err)
	}
	if got := recoveryGit(t, repository, "rev-parse", "HEAD"); got != intent.Parent {
		t.Fatalf("detached HEAD = %s, want unchanged parent %s", got, intent.Parent)
	}
}

func TestExactGateHeadDetectsAdvanceAfterAcceptedCommit(t *testing.T) {
	repository, intent := newReferenceUpdateFixture(t)
	if err := installCompletionIndexAndAdvanceGateReference(repository, intent); err != nil {
		t.Fatal(err)
	}
	if !exactGateHead(repository, intent.HeadReference, intent.Commit) {
		t.Fatal("exact accepted commit was not current")
	}
	runGitFixture(t, repository, "commit", "--allow-empty", "-q", "-m", "concurrent advance")
	advanced := recoveryGit(t, repository, "rev-parse", "HEAD")
	if exactGateHead(repository, intent.HeadReference, intent.Commit) {
		t.Fatal("advanced branch still matched the accepted commit")
	}
	if got := recoveryGit(t, repository, "rev-parse", intent.HeadReference); got != advanced {
		t.Fatalf("final binding check rewound branch to %s, want %s", got, advanced)
	}
}

func newReferenceUpdateFixture(t *testing.T) (string, gateCommitIntent) {
	t.Helper()
	repository, before, after := newIndexCASFixture(t)
	if err := os.MkdirAll(repository+"/tmp", 0o755); err != nil {
		t.Fatal(err)
	}
	intent := newKeepaliveTransactionIntent(t, repository, before, after, nil)
	if err := writeGateCommitIntent(repository, intent); err != nil {
		t.Fatal(err)
	}
	if err := readJSON(repository, gateCommitIntentFile, &intent); err != nil {
		t.Fatal(err)
	}
	return repository, intent
}
