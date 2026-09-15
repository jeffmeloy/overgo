package gate

import (
	"bytes"
	"errors"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/gitauthority"
	"overgo/internal/plan"
)

func TestCandidateDriftRefused(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	runGitFixture(t, repo, "init")
	runGitFixture(t, repo, "config", "user.email", "gate@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Gate Test")
	path := filepath.Join(repo, "candidate.go")
	if err := os.WriteFile(path, []byte("package candidate\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "candidate.go")
	runGitFixture(t, repo, "commit", "-m", "base")
	gate := gateContext{repo: repo, paths: []string{"candidate.go"}}
	planned, err := gate.treeStateKey()
	if err != nil {
		t.Fatal(err)
	}
	if err := gate.requireCandidateTree(planned); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("package candidate\n\nconst changed = true\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := gate.requireCandidateTree(planned); err == nil || !strings.Contains(err.Error(), "candidate drifted after manifest planning") {
		t.Fatalf("drift result = %v", err)
	}
}

func TestPreparedCandidateCannotMoveBeforePlanning(t *testing.T) {
	t.Parallel()
	gate := gateContext{}
	gate.preparation.TreeKey = strings.Repeat("a", 64)
	if err := gate.requirePreparedCandidate(strings.Repeat("a", 64)); err != nil {
		t.Fatal(err)
	}
	if err := gate.requirePreparedCandidate(strings.Repeat("b", 64)); err == nil ||
		!strings.Contains(err.Error(), "changed after durable preparation") {
		t.Fatalf("moved prepared candidate result = %v", err)
	}
}

func TestPlannedPathsRejectGitPathspecMagic(t *testing.T) {
	t.Parallel()
	for _, candidate := range []string{":(glob)**/*.md", ":!docs/rogue.md", "docs/*.md", "docs/[ab].md"} {
		if err := validatePlannedPaths([]string{candidate}); err == nil ||
			!strings.Contains(err.Error(), "canonical literal repository path") {
			t.Fatalf("magic path %q result = %v", candidate, err)
		}
	}
	if err := validatePlannedPaths([]string{"cmd/gate", "docs/plan.json"}); err != nil {
		t.Fatalf("literal planned paths refused: %v", err)
	}
}

func TestCandidateTreeKeyFramesUntrackedPathsAndContent(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "gate@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Gate Test")
	if err := os.WriteFile(filepath.Join(repo, "base"), []byte("base"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "--", "base")
	runGitFixture(t, repo, "commit", "-q", "-m", "base")
	gate := gateContext{repo: repo, paths: []string{"a", "bc"}}
	if err := os.WriteFile(filepath.Join(repo, "a"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "bc"), []byte("bcZ"), 0o644); err != nil {
		t.Fatal(err)
	}
	first, err := gate.treeStateKey()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "a"), []byte("bc"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "bc"), []byte("Z"), 0o644); err != nil {
		t.Fatal(err)
	}
	second, err := gate.treeStateKey()
	if err != nil {
		t.Fatal(err)
	}
	if first == second {
		t.Fatal("different path/content framing produced one candidate key")
	}
}

func TestCandidateVerifierExcludesAmbientWorktreeInputs(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "gate@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Gate Test")
	for name, content := range map[string]string{
		".gitignore":     "acceptance.flag\nignored.go\n",
		"planned.flag":   "base",
		"unplanned.flag": "base",
	} {
		if err := os.WriteFile(filepath.Join(repo, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	runGitFixture(t, repo, "add", "--", ".gitignore", "planned.flag", "unplanned.flag")
	runGitFixture(t, repo, "commit", "-q", "-m", "base")
	if err := os.WriteFile(filepath.Join(repo, "planned.flag"), []byte("planned"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "unplanned.flag"), []byte("ambient"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "acceptance.flag"), []byte("ambient"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "ignored.go"), []byte("package ignored\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	gate := gateContext{repo: repo, paths: []string{"planned.flag"}}
	tree, err := gate.plannedTree()
	if err != nil {
		t.Fatal(err)
	}
	verify := `test "$(cat planned.flag)" = planned && test "$(cat unplanned.flag)" = base && test ! -e acceptance.flag && test ! -e ignored.go`
	if _, err := gate.executeCandidateVerifier(tree, verify); err != nil {
		t.Fatalf("isolated candidate verifier: %v", err)
	}
}

func TestCandidateVerifierCannotMutateAcceptedTree(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "gate@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Gate Test")
	if err := os.WriteFile(filepath.Join(repo, "planned.flag"), []byte("accepted"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "--", "planned.flag")
	runGitFixture(t, repo, "commit", "-q", "-m", "base")
	gate := gateContext{repo: repo, paths: []string{"planned.flag"}}
	tree, err := gate.plannedTree()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := gate.executeCandidateVerifier(tree, `printf changed > planned.flag`); err == nil ||
		!strings.Contains(err.Error(), "mutated its immutable candidate worktree") {
		t.Fatalf("mutating verifier result = %v", err)
	}
}

func TestBuildAcceptedCompletionTreeIsolatesSharedIndexAndIgnoresLaterWorktreeEdit(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "gate@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Gate Test")
	if err := os.Mkdir(filepath.Join(repo, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	before := []byte("{\"campaign\":\"before\"}\n")
	if err := os.WriteFile(filepath.Join(repo, filepath.FromSlash(plan.Path)), before, 0o644); err != nil {
		t.Fatal(err)
	}
	wantCandidate := []byte("package candidate\n\nconst value = 1\n")
	if err := os.WriteFile(filepath.Join(repo, "candidate.go"), wantCandidate, 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "--", plan.Path, "candidate.go")
	runGitFixture(t, repo, "commit", "-q", "-m", "base")
	gate := gateContext{repo: repo, paths: []string{"candidate.go", plan.Path}}
	accepted, err := gate.plannedTree()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		filepath.Join(repo, "candidate.go"), []byte("package candidate\n\nconst value = 2\n"), 0o644,
	); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		filepath.Join(repo, filepath.FromSlash(plan.Path)), []byte("{\"campaign\":\"ambient\"}\n"), 0o644,
	); err != nil {
		t.Fatal(err)
	}
	acceptedPlan, err := acceptedPlanBytes(repo, accepted)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(acceptedPlan, before) {
		t.Fatalf("accepted plan = %q, want snapshotted bytes %q", acceptedPlan, before)
	}
	planAfter := []byte("{\"campaign\":\"after\"}\n")
	indexBefore, err := captureGateIndex(repo)
	if err != nil {
		t.Fatal(err)
	}
	completionIndex, err := buildAcceptedCompletionIndex(repo, accepted, planAfter)
	if err != nil {
		t.Fatal(err)
	}
	currentIndex, err := captureGateIndex(repo)
	if err != nil {
		t.Fatal(err)
	}
	if currentIndex.Tree != indexBefore.Tree || !bytes.Equal(currentIndex.Data, indexBefore.Data) {
		t.Fatalf("isolated completion construction changed shared index from %s to %s", indexBefore.Tree, currentIndex.Tree)
	}
	stagedCandidate, err := command(repo, "git", "show", completionIndex.Tree+":candidate.go")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal([]byte(stagedCandidate), wantCandidate) {
		t.Fatalf("staged candidate = %q, want accepted bytes %q", stagedCandidate, wantCandidate)
	}
	stagedPlan, err := command(repo, "git", "show", completionIndex.Tree+":"+plan.Path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal([]byte(stagedPlan), planAfter) {
		t.Fatalf("staged plan = %q, want exact transition %q", stagedPlan, planAfter)
	}
	restoreIndex, err := buildGateIndexForTree(repo, indexBefore.Tree)
	if err != nil {
		t.Fatal(err)
	}
	intent := bindGateIntentKeepaliveForTest(t, repo, gateCommitIntent{
		IndexTree: indexBefore.Tree, Tree: completionIndex.Tree,
		IndexBefore: indexBefore.Data, IndexAfter: completionIndex.Data, IndexRestore: restoreIndex.Data,
		IndexMode: uint32(indexBefore.Mode.Perm()),
	})
	if err := installCompletionIndexForTest(repo, intent); err != nil {
		t.Fatal(err)
	}
	installed, err := captureGateIndex(repo)
	if err != nil {
		t.Fatal(err)
	}
	if installed.Tree != completionIndex.Tree {
		t.Fatalf("installed completion index = %s, want %s", installed.Tree, completionIndex.Tree)
	}
}

func TestInstallCompletionIndexRefusesWriterBeforeLock(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	repo, before, after := newIndexCASFixture(t)
	wantConcurrent := []byte("concurrent install staging\n")
	if err := os.WriteFile(filepath.Join(repo, "candidate.txt"), wantConcurrent, 0o644); err != nil {
		t.Fatal(err)
	}
	gateIndexCASBeforeLockHook = func(repository string) {
		runGitFixture(t, repository, "add", "--", "candidate.txt")
	}
	t.Cleanup(func() { gateIndexCASBeforeLockHook = nil })

	err := installCompletionIndexForTest(repo, exactIndexIntent(t, repo, before, after))
	if err == nil || !strings.Contains(err.Error(), "index moved") {
		t.Fatalf("concurrent install result = %v", err)
	}
	staged, err := command(repo, "git", "show", ":candidate.txt")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal([]byte(staged), wantConcurrent) {
		t.Fatalf("refused install replaced concurrent staged bytes %q", staged)
	}
}

func TestGateStartIndexRefusesAndPreservesLaterUnplannedStaging(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	repo, before, _ := newIndexCASFixture(t)
	unplanned := []byte("staged after gate start\n")
	if err := os.WriteFile(filepath.Join(repo, "unplanned.txt"), unplanned, 0o644); err != nil {
		t.Fatal(err)
	}
	gateBeforeCommitStateHook = func(repository string) {
		runGitFixture(t, repository, "add", "--", "unplanned.txt")
	}
	t.Cleanup(func() { gateBeforeCommitStateHook = nil })
	gate := gateContext{repo: repo, indexBefore: before}
	if err := gate.requireGateStartState(); err == nil || !strings.Contains(err.Error(), "index moved after") {
		t.Fatalf("late unplanned staging result = %v", err)
	}
	staged, err := command(repo, "git", "show", ":unplanned.txt")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal([]byte(staged), unplanned) {
		t.Fatalf("refused gate changed late staged content %q", staged)
	}
}

func TestRestoreCapturedIndexRefusesWriterBeforeLock(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	repo, before, after := newIndexCASFixture(t)
	intent := exactIndexIntent(t, repo, before, after)
	if err := installCompletionIndexForTest(repo, intent); err != nil {
		t.Fatal(err)
	}
	wantConcurrent := []byte("concurrent recovery staging\n")
	if err := os.WriteFile(filepath.Join(repo, "candidate.txt"), wantConcurrent, 0o644); err != nil {
		t.Fatal(err)
	}
	gateIndexCASBeforeLockHook = func(repository string) {
		runGitFixture(t, repository, "add", "--", "candidate.txt")
	}
	t.Cleanup(func() { gateIndexCASBeforeLockHook = nil })

	err := restoreCapturedIndex(repo, intent)
	if err == nil || !strings.Contains(err.Error(), "index moved") {
		t.Fatalf("concurrent restore result = %v", err)
	}
	staged, err := command(repo, "git", "show", ":candidate.txt")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal([]byte(staged), wantConcurrent) {
		t.Fatalf("refused restore replaced concurrent staged bytes %q", staged)
	}
}

func TestExactIndexCASBlocksWriterWhileLocked(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	repo, before, after := newIndexCASFixture(t)
	if err := os.WriteFile(filepath.Join(repo, "candidate.txt"), []byte("locked writer\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var competingErr error
	var competingOutput []byte
	gateIndexCASLockedHook = func(repository string) {
		process := exec.Command("git", "add", "--", "candidate.txt")
		process.Dir = repository
		competingOutput, competingErr = process.CombinedOutput()
	}
	t.Cleanup(func() { gateIndexCASLockedHook = nil })

	if err := installCompletionIndexForTest(repo, exactIndexIntent(t, repo, before, after)); err != nil {
		t.Fatal(err)
	}
	if competingErr == nil || !strings.Contains(string(competingOutput), "index.lock") {
		t.Fatalf("writer under exact index lock = (%v, %q)", competingErr, competingOutput)
	}
	indexPath, err := gateIndexPath(repo)
	if err != nil {
		t.Fatal(err)
	}
	installed, err := os.ReadFile(indexPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(installed, after.Data) {
		t.Fatal("locked writer changed the exact installed index bytes")
	}
}

func TestCaptureGateIndexRejectsSemanticEntryFlags(t *testing.T) {
	t.Parallel()
	repo, _, _ := newIndexCASFixture(t)
	runGitFixture(t, repo, "update-index", "--assume-unchanged", "--", "candidate.txt")
	indexPath, err := gateIndexPath(repo)
	if err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(indexPath)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := captureGateIndex(repo); err == nil || !strings.Contains(err.Error(), "special Git index entry flags") {
		t.Fatalf("semantic index capture result = %v", err)
	}
	after, err := os.ReadFile(indexPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(after, before) {
		t.Fatal("refused semantic index capture changed the shared index bytes")
	}
}

func TestCaptureGateIndexRejectsIntentToAddMetadata(t *testing.T) {
	t.Parallel()
	repo, _, _ := newIndexCASFixture(t)
	intentPath := filepath.Join(repo, "intent.txt")
	if err := os.WriteFile(intentPath, []byte("intent\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "--intent-to-add", "--", "intent.txt")
	indexPath, err := gateIndexPath(repo)
	if err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(indexPath)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := captureGateIndex(repo); err == nil || !strings.Contains(err.Error(), "intent-to-add") {
		t.Fatalf("intent-to-add index capture result = %v", err)
	}
	after, err := os.ReadFile(indexPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(after, before) {
		t.Fatal("refused intent-to-add capture changed the shared index bytes")
	}
}

func TestCaptureGateIndexRejectsResolveUndoMetadata(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	runGitFixture(t, repo, "init", "-q", "-b", "main")
	runGitFixture(t, repo, "config", "user.email", "resolve-undo@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Resolve Undo Test")
	path := filepath.Join(repo, "candidate.txt")
	if err := os.WriteFile(path, []byte("base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "--", "candidate.txt")
	runGitFixture(t, repo, "commit", "-q", "-m", "base")
	runGitFixture(t, repo, "checkout", "-q", "-b", "side")
	if err := os.WriteFile(path, []byte("side\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "commit", "-q", "-am", "side")
	runGitFixture(t, repo, "checkout", "-q", "main")
	if err := os.WriteFile(path, []byte("main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "commit", "-q", "-am", "main")
	merge := exec.Command("git", "merge", "--no-commit", "side")
	merge.Dir = repo
	if output, err := merge.CombinedOutput(); err == nil {
		t.Fatalf("conflict fixture unexpectedly merged: %s", output)
	}
	if err := os.WriteFile(path, []byte("resolved\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "--", "candidate.txt")
	indexPath, err := gateIndexPath(repo)
	if err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(indexPath)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := captureGateIndex(repo); err == nil || !strings.Contains(err.Error(), "resolve-undo") {
		t.Fatalf("resolve-undo index capture result = %v", err)
	}
	after, err := os.ReadFile(indexPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(after, before) {
		t.Fatal("refused resolve-undo capture changed the shared index bytes")
	}
}

func TestGateGitAuthorityIgnoresAmbientRepositoryOverrides(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	requested := t.TempDir()
	foreign := t.TempDir()
	for repository, content := range map[string]string{requested: "requested\n", foreign: "foreign\n"} {
		runGitFixture(t, repository, "init", "-q")
		runGitFixture(t, repository, "config", "user.email", "authority@example.invalid")
		runGitFixture(t, repository, "config", "user.name", "Authority Test")
		if err := os.WriteFile(filepath.Join(repository, "marker.txt"), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		runGitFixture(t, repository, "add", "--", "marker.txt")
		runGitFixture(t, repository, "commit", "-q", "-m", "marker")
	}
	wantHead, err := command(requested, "git", "rev-parse", "HEAD")
	if err != nil {
		t.Fatal(err)
	}
	wantTree, err := command(requested, "git", "rev-parse", "HEAD^{tree}")
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("GIT_DIR", filepath.Join(foreign, ".git"))
	t.Setenv("GIT_WORK_TREE", foreign)
	t.Setenv("GIT_INDEX_FILE", filepath.Join(foreign, ".git", "index"))
	t.Setenv("GIT_CONFIG_COUNT", "1")
	t.Setenv("GIT_CONFIG_KEY_0", "core.bare")
	t.Setenv("GIT_CONFIG_VALUE_0", "true")
	if err := gitauthority.RequireRepositoryRoot(t.Context(), requested); err != nil {
		t.Fatalf("requested repository root was redirected: %v", err)
	}
	head, err := gitAuthorityOutput(requested, "rev-parse", "HEAD")
	if err != nil || strings.TrimSpace(string(head)) != strings.TrimSpace(wantHead) {
		t.Fatalf("ambient-overridden authority HEAD = (%q, %v), want %q", head, err, wantHead)
	}
	marker, err := command(requested, "git", "show", "HEAD:marker.txt")
	if err != nil || marker != "requested\n" {
		t.Fatalf("ambient-overridden marker = (%q, %v)", marker, err)
	}
	index, err := captureGateIndex(requested)
	if err != nil || index.Tree != strings.TrimSpace(wantTree) {
		t.Fatalf("ambient-overridden index = (%s, %v), want tree %s", index.Tree, err, wantTree)
	}
	gate := gateContext{repo: requested}
	if _, err := gate.executeCandidateVerifier(
		index.Tree,
		`test "$(git show HEAD:marker.txt)" = requested && printf 'verified requested repository\n'`,
	); err != nil {
		t.Fatalf("candidate verifier was redirected by ambient Git state: %v", err)
	}
	object, err := gitHashObject(requested, []byte("requested authority object\n"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := command(requested, "git", "cat-file", "-e", object); err != nil {
		t.Fatalf("authority object was not written to requested repository: %v", err)
	}
	if _, err := command(foreign, "git", "cat-file", "-e", object); err == nil {
		t.Fatal("authority object was redirected into ambient foreign repository")
	}
}

func newIndexCASFixture(t *testing.T) (string, gateIndexSnapshot, gateIndexSnapshot) {
	t.Helper()
	repo := t.TempDir()
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "index-cas@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Index CAS Test")
	runGitFixture(t, repo, "config", "core.autocrlf", "false")
	path := filepath.Join(repo, "candidate.txt")
	if err := os.WriteFile(path, []byte("baseline\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "--", "candidate.txt")
	runGitFixture(t, repo, "commit", "-q", "-m", "baseline")
	before, err := captureGateIndex(repo)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("accepted\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	tree, err := (&gateContext{repo: repo, paths: []string{"candidate.txt"}}).plannedTree()
	if err != nil {
		t.Fatal(err)
	}
	after, err := buildGateIndexForTree(repo, tree)
	if err != nil {
		t.Fatal(err)
	}
	return repo, before, after
}

func exactIndexIntent(t *testing.T, repo string, before, after gateIndexSnapshot) gateCommitIntent {
	t.Helper()
	restore, err := buildGateIndexForTree(repo, before.Tree)
	if err != nil {
		t.Fatal(err)
	}
	return bindGateIntentKeepaliveForTest(t, repo, gateCommitIntent{
		IndexTree: before.Tree, Tree: after.Tree,
		IndexBefore: before.Data, IndexAfter: after.Data, IndexRestore: restore.Data,
		IndexMode: uint32(before.Mode.Perm()),
	})
}

// installCompletionIndexForTest isolates the index half of the production
// prepared index-and-reference transaction so crash seams can be constructed
// without publishing the fixture branch.
func installCompletionIndexForTest(repo string, intent gateCommitIntent) error {
	err := withGateGitStateLock(
		repo, fs.FileMode(intent.IndexMode), intent,
		gateIndexCASBeforeLockHook, gateIndexCASLockedHook,
		func(indexPath string) error {
			return installCompletionIndexUnderLock(repo, indexPath, intent)
		},
	)
	if errors.Is(err, errGateIndexMoved) {
		return errors.New("commit admission: Git index moved before exact completion staging")
	}
	return err
}

func TestPlannedTreeHandlesRenameWithUnstagedEdit(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "gate@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Gate Test")
	oldPath, newPath := "old path.go", "new path.go"
	if err := os.WriteFile(filepath.Join(repo, oldPath), []byte("package candidate\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "--", oldPath)
	runGitFixture(t, repo, "commit", "-q", "-m", "base")
	if err := os.Rename(filepath.Join(repo, oldPath), filepath.Join(repo, newPath)); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "-A", "--", oldPath, newPath)
	want := []byte("package candidate\n\nconst changedAfterStaging = true\n")
	if err := os.WriteFile(filepath.Join(repo, newPath), want, 0o644); err != nil {
		t.Fatal(err)
	}

	tree, err := (&gateContext{repo: repo, paths: []string{oldPath, newPath}}).plannedTree()
	if err != nil {
		t.Fatal(err)
	}
	stagedText, err := command(repo, "git", "show", tree+":"+newPath)
	if err != nil {
		t.Fatal(err)
	}
	staged := []byte(stagedText)
	if !bytes.Equal(staged, want) {
		t.Fatalf("staged candidate = %q, want worktree bytes %q", staged, want)
	}
	if _, err := command(repo, "git", "cat-file", "-e", tree+":"+oldPath); err == nil {
		t.Fatal("planned tree retained the deleted rename source")
	}
}

func TestScopeRefusesUnplannedVerificationInputButAllowsResearchDocument(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "gate@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Gate Test")
	for name, content := range map[string]string{
		"candidate.go": "package candidate\n",
		"helper.go":    "package candidate\n\nconst helper = false\n",
	} {
		if err := os.WriteFile(filepath.Join(repo, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	runGitFixture(t, repo, "add", "--", "candidate.go", "helper.go")
	runGitFixture(t, repo, "commit", "-q", "-m", "base")
	if err := os.Mkdir(filepath.Join(repo, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "docs", "research.pdf"), []byte("untracked research"), 0o644); err != nil {
		t.Fatal(err)
	}
	gate := gateContext{repo: repo, paths: []string{"candidate.go"}}
	if _, err := gate.stepScope(); err != nil {
		t.Fatalf("untracked research document blocked an isolated code gate: %v", err)
	}
	if err := os.WriteFile(
		filepath.Join(repo, "helper.go"), []byte("package candidate\n\nconst helper = true\n"), 0o644,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := gate.stepScope(); err == nil || !strings.Contains(err.Error(), "unplanned verification input") {
		t.Fatalf("unplanned helper scope result = %v", err)
	}
}

func runGitFixture(t *testing.T, repo string, arguments ...string) {
	t.Helper()
	command := exec.Command("git", arguments...)
	command.Dir = repo
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("git %v: %v: %s", arguments, err, output)
	}
}

// TestGateStartIndexAdoptsStatRefreshUnderSameTree pins stat-only refresh recovery.
func TestGateStartIndexAdoptsStatRefreshUnderSameTree(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	repo, _, _ := newIndexCASFixture(t)
	steady := filepath.Join(repo, "steady.txt")
	if err := os.WriteFile(steady, []byte("steady\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "--", "steady.txt")
	runGitFixture(t, repo, "commit", "-q", "-m", "steady")
	before, err := captureGateIndex(repo)
	if err != nil {
		t.Fatal(err)
	}
	refreshed := time.Now().Add(24 * time.Hour).Truncate(time.Second)
	if err := os.Chtimes(steady, refreshed, refreshed); err != nil {
		t.Fatal(err)
	}
	gateBeforeCommitStateHook = func(repository string) {
		runGitFixture(t, repository, "status", "--porcelain=v1", "--untracked-files=no")
	}
	t.Cleanup(func() { gateBeforeCommitStateHook = nil })
	gate := gateContext{repo: repo, indexBefore: before}
	if err := gate.requireGateStartState(); err != nil {
		t.Fatalf("stat refresh under the same tree was refused: %v", err)
	}
	after, err := captureGateIndex(repo)
	if err != nil {
		t.Fatal(err)
	}
	if after.Tree != before.Tree || !bytes.Equal(gate.indexBefore.Data, after.Data) {
		t.Fatalf("adopted snapshot does not match the refreshed index: tree %s vs %s", gate.indexBefore.Tree, after.Tree)
	}
	if bytes.Equal(after.Data, before.Data) {
		t.Fatal("stat refresh did not change index bytes; adoption was not exercised")
	}
}
