package gate

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/authoritylock"
	"overgo/internal/clioptions"
	"overgo/internal/processlock"
)

// TestToolStateLivesInTheRuntimeDirectory holds every piece of state the gate
// keeps between processes -- its locks, its crash locators, its retry cache,
// its plan transaction scratch -- to the checkout's runtime directory, never
// to tmp: tmp is scrap and may be emptied at any time, and a lock must not be
// beside what is. The repository here has no ignore file at all, as a checkout
// of an older commit would not name the directory: the writers still leave
// git nothing to report, and none of them creates tmp.
func TestToolStateLivesInTheRuntimeDirectory(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	runGitFixture(t, repo, "init", "--quiet")
	state := []string{
		gateDebtFile, gateCommitIntentFile, gateHeartbeatFile, gateRetryFile, gatePreparedSelectionFile,
		gateLanesLocatorFile, gateLanesLogFile, gateLanesLockFile, gatePlanScratchDir,
	}
	for _, name := range state {
		if !strings.HasPrefix(name, processlock.StateDirectory+"/") {
			t.Errorf("%s is kept outside %s", name, processlock.StateDirectory)
		}
	}
	for _, name := range []string{gateDebtFile, gateCommitIntentFile, gateHeartbeatFile, gateRetryFile, gatePreparedSelectionFile, gateLanesLocatorFile} {
		if err := writeJSON(repo, name, map[string]string{"writer": name}, clioptions.OutputFileMode); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}
	runner, err := acquireLaneRunner(repo)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	authority, err := authoritylock.Acquire(repo)
	if err != nil {
		t.Fatal(err)
	}
	defer authority.Close()
	preparation, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("preparation"))
	if err != nil {
		t.Fatal(err)
	}
	scratch, err := gatePlanScratchPath(repo, preparation)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(scratch, filepath.Join(repo, processlock.StateDirectory)+string(filepath.Separator)) {
		t.Fatalf("plan transaction scratch %s is outside the runtime directory", scratch)
	}
	if _, err := os.Lstat(filepath.Join(repo, "tmp")); !os.IsNotExist(err) {
		t.Fatalf("a state writer created tmp: %v", err)
	}
	status, err := exec.Command("git", "-C", repo, "status", "--porcelain", "--untracked-files=all").CombinedOutput()
	if err != nil {
		t.Fatalf("git status: %v: %s", err, status)
	}
	if strings.TrimSpace(string(status)) != "" {
		t.Fatalf("tool state is visible to git:\n%s", status)
	}
}
