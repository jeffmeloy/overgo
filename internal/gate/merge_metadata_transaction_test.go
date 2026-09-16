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
)

// restoreInterruptedMergeForTest isolates the merge-metadata half of the
// production interrupted-state transaction while retaining its real locks,
// keepalive authority, and exact-state classifier.
func restoreInterruptedMergeForTest(repo string, intent gateCommitIntent) error {
	if intent.Merge == nil {
		return nil
	}
	err := withGateMergeMetadataLock(repo, fs.FileMode(intent.IndexMode), intent, func(indexPath string) error {
		if err := requireGateIntentKeepalive(repo, intent); err != nil {
			return err
		}
		return restoreInterruptedMergeLocked(repo, intent, indexPath)
	})
	return err
}

func TestGitMetadataPathsMatchGitInLinkedWorktree(t *testing.T) {
	t.Parallel()
	repo, _, _ := newIndexCASFixture(t)
	worktree := filepath.Join(t.TempDir(), "linked worktree")
	runGitFixture(t, repo, "worktree", "add", "--detach", worktree, "HEAD")
	names := []string{"index", "MERGE_HEAD", "MERGE_MSG", "AUTO_MERGE", "refs/heads/metadata-test"}
	for _, directory := range []string{repo, worktree} {
		paths, err := gitMetadataPaths(directory, names)
		if err != nil {
			t.Fatal(err)
		}
		for _, name := range names {
			output, err := command(directory, "git", "rev-parse", "--path-format=absolute", "--git-path", name)
			if err != nil {
				t.Fatal(err)
			}
			if want := filepath.Clean(strings.TrimSpace(output)); paths[name] != want {
				t.Fatalf("%s/%s = %q, want %q", directory, name, paths[name], want)
			}
		}
	}
	for _, names := range [][]string{nil, {""}, {"MERGE_HEAD\nindex"}} {
		if _, err := gitMetadataPaths(repo, names); err == nil {
			t.Fatalf("accepted invalid names %q", names)
		}
	}
}

func TestClearCommittedMergeStateRefusesConcurrentMetadataBeforeLock(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	repo, intent := newMergeMetadataTransactionFixture(t)
	writeMergeMetadataFixture(t, repo, intent.Merge)
	concurrent := []byte("concurrent merge message\n")
	gateMergeCASBeforeLockHook = func(repository string) {
		if err := os.WriteFile(mustGateValue(gitMetadataPath(repository, "MERGE_MSG")), concurrent, 0o600); err != nil {
			t.Errorf("concurrent metadata writer: %v", err)
		}
	}
	t.Cleanup(func() { gateMergeCASBeforeLockHook = nil })

	err := clearCommittedMergeState(repo, intent)
	if !errors.Is(err, errGateMergeMetadataMoved) {
		t.Fatalf("cleanup race result = %v, want merge metadata moved", err)
	}
	assertMergeMetadataFixture(t, repo, "MERGE_MSG", concurrent, true)
	assertMergeMetadataFixture(t, repo, "MERGE_MODE", intent.Merge.Mode, true)
	assertMergeMetadataFixture(t, repo, "MERGE_HEAD", intent.Merge.Head, true)
}

func TestRestoreInterruptedMergeRefusesConcurrentMetadataBeforeLock(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	repo, intent := newMergeMetadataTransactionFixture(t)
	concurrent := []byte("concurrent recovery message\n")
	gateMergeCASBeforeLockHook = func(repository string) {
		if err := os.WriteFile(mustGateValue(gitMetadataPath(repository, "MERGE_MSG")), concurrent, 0o600); err != nil {
			t.Errorf("concurrent metadata writer: %v", err)
		}
	}
	t.Cleanup(func() { gateMergeCASBeforeLockHook = nil })

	err := restoreInterruptedMergeForTest(repo, intent)
	if !errors.Is(err, errGateMergeMetadataMoved) {
		t.Fatalf("restore race result = %v, want merge metadata moved", err)
	}
	assertMergeMetadataFixture(t, repo, "MERGE_MSG", concurrent, true)
	assertMergeMetadataFixture(t, repo, "MERGE_MODE", nil, false)
	assertMergeMetadataFixture(t, repo, "MERGE_HEAD", nil, false)
}

func TestMergeMetadataTransactionHoldsActualIndexLock(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	repo, intent := newMergeMetadataTransactionFixture(t)
	writeMergeMetadataFixture(t, repo, intent.Merge)
	if err := os.WriteFile(filepath.Join(repo, "candidate.txt"), []byte("competing index writer\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var competingErr error
	var competingOutput []byte
	gateMergeCASLockedHook = func(repository string) {
		process := exec.Command("git", "add", "--", "candidate.txt")
		process.Dir = repository
		competingOutput, competingErr = process.CombinedOutput()
	}
	t.Cleanup(func() { gateMergeCASLockedHook = nil })

	if err := clearCommittedMergeState(repo, intent); err != nil {
		t.Fatal(err)
	}
	if competingErr == nil || !strings.Contains(string(competingOutput), "index.lock") {
		t.Fatalf("writer under merge transaction lock = (%v, %q)", competingErr, competingOutput)
	}
	assertMergeMetadataFixture(t, repo, "MERGE_MODE", nil, false)
	assertMergeMetadataFixture(t, repo, "MERGE_MSG", nil, false)
	assertMergeMetadataFixture(t, repo, "MERGE_HEAD", nil, false)
}

func TestMergeMetadataTransactionHoldsAutoMergeRefLock(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	repo, intent := newRealConflictMergeTransactionFixture(t)
	var competingErr error
	var competingOutput []byte
	gateMergeCASLockedHook = func(repository string) {
		process := exec.Command("git", "update-ref", "AUTO_MERGE", intent.IndexTree)
		process.Dir = repository
		competingOutput, competingErr = process.CombinedOutput()
	}
	t.Cleanup(func() { gateMergeCASLockedHook = nil })

	if err := clearCommittedMergeState(repo, intent); err != nil {
		t.Fatal(err)
	}
	if competingErr == nil || !strings.Contains(string(competingOutput), "AUTO_MERGE.lock") {
		t.Fatalf("AUTO_MERGE writer under ref lock = (%v, %q)", competingErr, competingOutput)
	}
	assertMergeMetadataFixture(t, repo, "AUTO_MERGE", nil, false)
}

func TestMergeMetadataTransactionBlocksConcurrentMergeQuit(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	repo, intent := newMergeMetadataTransactionFixture(t)
	writeMergeMetadataFixture(t, repo, intent.Merge)
	var competingErr error
	var competingOutput []byte
	gateMergeCASLockedHook = func(repository string) {
		process := exec.Command("git", "merge", "--quit")
		process.Dir = repository
		competingOutput, competingErr = process.CombinedOutput()
	}
	t.Cleanup(func() { gateMergeCASLockedHook = nil })

	if err := clearCommittedMergeState(repo, intent); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(competingOutput), "AUTO_MERGE.lock") {
		t.Fatalf("merge --quit under exact Git-state locks = (%v, %q)", competingErr, competingOutput)
	}
	assertMergeMetadataFixture(t, repo, "MERGE_HEAD", nil, false)
}

func TestMergeMetadataPresenceMarkerIsLast(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	repo, intent := newMergeMetadataTransactionFixture(t)
	writeMergeMetadataFixture(t, repo, intent.Merge)
	var cleanup, restore []string
	gateMergeCASAfterStepHook = func(repository, operation, name string) {
		headPath := mustGateValue(gitMetadataPath(repository, "MERGE_HEAD"))
		_, err := os.Stat(headPath)
		headPresent := err == nil
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			t.Errorf("inspect MERGE_HEAD after %s/%s: %v", operation, name, err)
		}
		switch operation {
		case "cleanup":
			cleanup = append(cleanup, name)
			if name != "MERGE_HEAD" && !headPresent {
				t.Errorf("MERGE_HEAD disappeared before cleanup removed %s", name)
			}
			if name == "MERGE_HEAD" && headPresent {
				t.Error("MERGE_HEAD remains after its cleanup step")
			}
		case "restore":
			restore = append(restore, name)
			if name != "MERGE_HEAD" && headPresent {
				t.Errorf("MERGE_HEAD was published while restoring %s", name)
			}
			if name == "MERGE_HEAD" && !headPresent {
				t.Error("MERGE_HEAD is absent after its restore step")
			}
		}
	}
	t.Cleanup(func() { gateMergeCASAfterStepHook = nil })

	if err := clearCommittedMergeState(repo, intent); err != nil {
		t.Fatal(err)
	}
	if err := restoreInterruptedMergeForTest(repo, intent); err != nil {
		t.Fatal(err)
	}
	want := []string{"MERGE_MODE", "MERGE_MSG", "MERGE_HEAD"}
	if strings.Join(cleanup, ",") != strings.Join(want, ",") {
		t.Fatalf("cleanup order = %v, want %v", cleanup, want)
	}
	if strings.Join(restore, ",") != strings.Join(want, ",") {
		t.Fatalf("restore order = %v, want %v", restore, want)
	}
}

func TestRestoreInterruptedMergeResumesExactCleanupProgress(t *testing.T) {
	t.Parallel()
	repo, intent := newMergeMetadataTransactionFixture(t)
	writeMergeMetadataFixture(t, repo, intent.Merge)
	if err := os.Remove(mustGateValue(gitMetadataPath(repo, "MERGE_MODE"))); err != nil {
		t.Fatal(err)
	}
	if err := requireRecoverableGitState(repo, intent, intent.Commit); err != nil {
		t.Fatalf("exact cleanup progress was not recoverable: %v", err)
	}
	if err := restoreInterruptedMergeForTest(repo, intent); err != nil {
		t.Fatal(err)
	}
	assertMergeMetadataFixture(t, repo, "MERGE_MODE", intent.Merge.Mode, true)
	assertMergeMetadataFixture(t, repo, "MERGE_MSG", intent.Merge.Message, true)
	assertMergeMetadataFixture(t, repo, "MERGE_HEAD", intent.Merge.Head, true)
}

func TestMergeMetadataRestoresCapturedPermissionModes(t *testing.T) {
	t.Parallel()
	repo, intent := newMergeMetadataTransactionFixture(t)
	writeMergeMetadataFixture(t, repo, intent.Merge)
	if err := clearCommittedMergeState(repo, intent); err != nil {
		t.Fatal(err)
	}
	if err := restoreInterruptedMergeForTest(repo, intent); err != nil {
		t.Fatal(err)
	}
	for _, metadata := range []struct {
		name string
		mode uint32
	}{
		{name: "MERGE_MODE", mode: intent.Merge.ModeFileMode},
		{name: "MERGE_MSG", mode: intent.Merge.MessageFileMode},
		{name: "MERGE_HEAD", mode: intent.Merge.HeadFileMode},
	} {
		info, err := os.Stat(mustGateValue(gitMetadataPath(repo, metadata.name)))
		if err != nil {
			t.Fatal(err)
		}
		if got := uint32(info.Mode().Perm()); got != metadata.mode {
			t.Fatalf("restored %s mode = %#o, want %#o", metadata.name, got, metadata.mode)
		}
		if metadata.mode == 0o600 {
			t.Fatalf("%s fixture did not exercise a non-0600 mode", metadata.name)
		}
	}
}

func TestConflictAutoMergeIsCleanedWithMergeMetadata(t *testing.T) {
	t.Parallel()
	repo, intent := newRealConflictMergeTransactionFixture(t)
	wantAutoMerge := bytes.Clone(intent.Merge.AutoMerge)
	if len(wantAutoMerge) == 0 {
		t.Fatal("real conflict merge did not publish AUTO_MERGE")
	}

	if err := clearCommittedMergeState(repo, intent); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"MERGE_MODE", "MERGE_MSG", "AUTO_MERGE", "MERGE_HEAD"} {
		assertMergeMetadataFixture(t, repo, name, nil, false)
	}
}

func TestConflictAutoMergeIsRestoredBeforeMergeHead(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	repo, intent := newRealConflictMergeTransactionFixture(t)
	wantAutoMerge := bytes.Clone(intent.Merge.AutoMerge)
	if err := clearCommittedMergeState(repo, intent); err != nil {
		t.Fatal(err)
	}
	var restored []string
	gateMergeCASAfterStepHook = func(_ string, operation, name string) {
		if operation != "restore" {
			return
		}
		restored = append(restored, name)
		if name == "AUTO_MERGE" {
			assertMergeMetadataFixture(t, repo, "AUTO_MERGE", wantAutoMerge, true)
			assertMergeMetadataFixture(t, repo, "MERGE_HEAD", nil, false)
		}
	}
	t.Cleanup(func() { gateMergeCASAfterStepHook = nil })

	if err := restoreInterruptedMergeForTest(repo, intent); err != nil {
		t.Fatal(err)
	}
	wantOrder := []string{"MERGE_MODE", "MERGE_MSG", "AUTO_MERGE", "MERGE_HEAD"}
	if strings.Join(restored, ",") != strings.Join(wantOrder, ",") {
		t.Fatalf("real conflict restore order = %v, want %v", restored, wantOrder)
	}
	assertMergeMetadataFixture(t, repo, "AUTO_MERGE", wantAutoMerge, true)
	assertMergeMetadataFixture(t, repo, "MERGE_HEAD", intent.Merge.Head, true)
}

func TestCapturePendingMergeRejectsAndPreservesNonemptyMergeRR(t *testing.T) {
	t.Parallel()
	repo := newRealConflictMergeFixture(t, true)
	mergeRRPath := mustGateValue(gitMetadataPath(repo, "MERGE_RR"))
	want, err := os.ReadFile(mergeRRPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(want) == 0 {
		t.Fatal("rerere conflict did not publish nonempty MERGE_RR")
	}

	if _, _, err := captureGateStartState(repo); err == nil || !strings.Contains(err.Error(), "nonempty MERGE_RR") {
		t.Fatalf("nonempty MERGE_RR gate-start result = %v", err)
	}
	if _, err := capturePendingMerge(repo); err == nil || !strings.Contains(err.Error(), "nonempty MERGE_RR") {
		t.Fatalf("nonempty MERGE_RR capture result = %v", err)
	}
	if got, err := os.ReadFile(mergeRRPath); err != nil {
		t.Fatal(err)
	} else if !bytes.Equal(got, want) {
		t.Fatalf("MERGE_RR changed on refusal: got %x want %x", got, want)
	}
}

func TestEmptyMergeRRAcceptedAndPreservedAcrossAdmissionRecovery(t *testing.T) {
	t.Parallel()
	repo, index, _ := newIndexCASFixture(t)
	mergeRRPath := mustGateValue(gitMetadataPath(repo, "MERGE_RR"))
	if err := os.WriteFile(mergeRRPath, nil, 0o600); err != nil {
		t.Fatal(err)
	}

	merge, captured, err := captureGateStartState(repo)
	if err != nil {
		t.Fatalf("empty MERGE_RR gate-start admission: %v", err)
	}
	if merge != nil || !bytes.Equal(captured.Data, index.Data) {
		t.Fatalf("empty MERGE_RR capture = (%+v, %d bytes), want nonmerge exact index", merge, len(captured.Data))
	}
	intent := gateCommitIntent{
		IndexBefore: bytes.Clone(index.Data), IndexAfter: bytes.Clone(index.Data),
		IndexRestore: bytes.Clone(index.Data), IndexMode: uint32(index.Mode.Perm()),
	}
	if err := requireRecoverableGitState(repo, intent, ""); err != nil {
		t.Fatalf("empty MERGE_RR recovery admission: %v", err)
	}
	if info, err := os.Stat(mergeRRPath); err != nil {
		t.Fatal(err)
	} else if info.Size() != 0 {
		t.Fatalf("empty MERGE_RR was rewritten to %d bytes", info.Size())
	}
}

func TestCapturePendingMergeRejectsAndPreservesOrphanAutoMerge(t *testing.T) {
	t.Parallel()
	repo, index, _ := newIndexCASFixture(t)
	autoMergePath := mustGateValue(gitMetadataPath(repo, "AUTO_MERGE"))
	want := []byte(index.Tree + "\n")
	if err := os.WriteFile(autoMergePath, want, 0o600); err != nil {
		t.Fatal(err)
	}

	if _, err := capturePendingMerge(repo); err == nil || !strings.Contains(err.Error(), "orphan AUTO_MERGE") {
		t.Fatalf("orphan AUTO_MERGE capture result = %v", err)
	}
	if got, err := os.ReadFile(autoMergePath); err != nil {
		t.Fatal(err)
	} else if !bytes.Equal(got, want) {
		t.Fatalf("AUTO_MERGE changed on refusal: got %q want %q", got, want)
	}
}

func TestNonMergeFinalStateRefusesConcurrentAutoMerge(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	repo, index, _ := newIndexCASFixture(t)
	intent := bindGateIntentKeepaliveForTest(t, repo, gateCommitIntent{
		IndexTree: index.Tree, Tree: index.Tree,
		IndexBefore: bytes.Clone(index.Data), IndexAfter: bytes.Clone(index.Data),
		IndexRestore: bytes.Clone(index.Data), IndexMode: uint32(index.Mode.Perm()),
	})
	gateMergeCASBeforeLockHook = func(repository string) {
		runGitFixture(t, repository, "update-ref", "AUTO_MERGE", index.Tree)
	}
	t.Cleanup(func() { gateMergeCASBeforeLockHook = nil })

	err := clearCommittedMergeState(repo, intent)
	if !errors.Is(err, errGateMergeMetadataMoved) {
		t.Fatalf("nonmerge concurrent AUTO_MERGE result = %v, want metadata moved", err)
	}
	if got := recoveryGit(t, repo, "rev-parse", "--verify", "AUTO_MERGE"); got != index.Tree {
		t.Fatalf("refused final state changed AUTO_MERGE to %s, want %s", got, index.Tree)
	}
}

func TestGateStartRejectsReftableRepository(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	initCommand := exec.Command("git", "init", "-q", "--ref-format=reftable")
	initCommand.Dir = repo
	if output, err := initCommand.CombinedOutput(); err != nil {
		t.Skipf("installed Git does not support reftable fixtures: %v: %s", err, output)
	}

	if _, _, err := captureGateStartState(repo); err == nil || !strings.Contains(err.Error(), "reference format") {
		t.Fatalf("reftable gate-start result = %v", err)
	}
}

func TestGateStartAndRecoveryRejectAndPreserveSquashMessage(t *testing.T) {
	t.Parallel()
	repo := newSquashMergeFixture(t)
	squashPath := mustGateValue(gitMetadataPath(repo, "SQUASH_MSG"))
	want, err := os.ReadFile(squashPath)
	if err != nil {
		t.Fatal(err)
	}
	index, err := captureGateIndex(repo)
	if err != nil {
		t.Fatal(err)
	}
	intent := gateCommitIntent{
		IndexBefore: bytes.Clone(index.Data), IndexAfter: bytes.Clone(index.Data),
		IndexRestore: bytes.Clone(index.Data), IndexMode: uint32(index.Mode.Perm()),
	}

	if _, _, err := captureGateStartState(repo); err == nil || !strings.Contains(err.Error(), "SQUASH_MSG") {
		t.Fatalf("squash gate-start result = %v", err)
	}
	if err := requireRecoverableGitState(repo, intent, ""); err == nil || !strings.Contains(err.Error(), "SQUASH_MSG") {
		t.Fatalf("squash recovery result = %v", err)
	}
	if got, err := os.ReadFile(squashPath); err != nil {
		t.Fatal(err)
	} else if !bytes.Equal(got, want) {
		t.Fatalf("SQUASH_MSG changed on refusal: got %q want %q", got, want)
	}
}

func newRealConflictMergeTransactionFixture(t *testing.T) (string, gateCommitIntent) {
	t.Helper()
	repo := newRealConflictMergeFixture(t, false)
	if err := os.WriteFile(filepath.Join(repo, "conflict.txt"), []byte("resolved by gate\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "--", "conflict.txt")
	runGitFixture(t, repo, "update-index", "--clear-resolve-undo")
	merge, err := capturePendingMerge(repo)
	if err != nil {
		t.Fatal(err)
	}
	if merge == nil || len(merge.AutoMerge) == 0 {
		t.Fatal("resolved real conflict has no exact AUTO_MERGE authority")
	}
	index, err := captureGateIndex(repo)
	if err != nil {
		t.Fatal(err)
	}
	return repo, bindGateIntentKeepaliveForTest(t, repo, gateCommitIntent{
		Merge: merge, IndexTree: index.Tree, Tree: index.Tree,
		IndexBefore: bytes.Clone(index.Data), IndexAfter: bytes.Clone(index.Data),
		IndexRestore: bytes.Clone(index.Data), IndexMode: uint32(index.Mode.Perm()),
	})
}

func newRealConflictMergeFixture(t *testing.T, rerere bool) string {
	t.Helper()
	repo := t.TempDir()
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "merge-state@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Merge State Test")
	runGitFixture(t, repo, "config", "core.autocrlf", "false")
	if rerere {
		runGitFixture(t, repo, "config", "rerere.enabled", "true")
	}
	conflictPath := filepath.Join(repo, "conflict.txt")
	if err := os.WriteFile(conflictPath, []byte("base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "--", "conflict.txt")
	runGitFixture(t, repo, "commit", "-q", "-m", "base")
	baseBranch := recoveryGit(t, repo, "rev-parse", "--abbrev-ref", "HEAD")
	runGitFixture(t, repo, "checkout", "-q", "-b", "conflict-side")
	if err := os.WriteFile(conflictPath, []byte("side\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "commit", "-q", "-am", "side")
	runGitFixture(t, repo, "checkout", "-q", baseBranch)
	if err := os.WriteFile(conflictPath, []byte("main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "commit", "-q", "-am", "main")

	merge := exec.Command("git", "merge", "--no-ff", "--no-commit", "conflict-side")
	merge.Dir = repo
	output, err := merge.CombinedOutput()
	if err == nil || !strings.Contains(string(output), "CONFLICT") {
		t.Fatalf("real conflict merge = (%v, %q)", err, output)
	}
	return repo
}

func newSquashMergeFixture(t *testing.T) string {
	t.Helper()
	repo := t.TempDir()
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "squash-state@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Squash State Test")
	if err := os.WriteFile(filepath.Join(repo, "base.txt"), []byte("base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "--", "base.txt")
	runGitFixture(t, repo, "commit", "-q", "-m", "base")
	baseBranch := recoveryGit(t, repo, "rev-parse", "--abbrev-ref", "HEAD")
	runGitFixture(t, repo, "checkout", "-q", "-b", "squash-side")
	if err := os.WriteFile(filepath.Join(repo, "side.txt"), []byte("side\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "--", "side.txt")
	runGitFixture(t, repo, "commit", "-q", "-m", "side")
	runGitFixture(t, repo, "checkout", "-q", baseBranch)
	runGitFixture(t, repo, "merge", "--squash", "squash-side")
	return repo
}

func newMergeMetadataTransactionFixture(t *testing.T) (string, gateCommitIntent) {
	t.Helper()
	repo, index, _ := newIndexCASFixture(t)
	merge := &gateMergeIntent{
		IndexTree:       index.Tree,
		Head:            []byte(strings.Repeat("a", 40) + "\n"),
		HeadFileMode:    0o666,
		Mode:            []byte{},
		ModeFileMode:    0o666,
		Message:         []byte("exact interrupted merge message\n"),
		MessageFileMode: 0o666,
	}
	return repo, bindGateIntentKeepaliveForTest(t, repo, gateCommitIntent{
		Merge:        merge,
		IndexTree:    index.Tree,
		Tree:         index.Tree,
		IndexBefore:  bytes.Clone(index.Data),
		IndexAfter:   bytes.Clone(index.Data),
		IndexRestore: bytes.Clone(index.Data),
		IndexMode:    uint32(index.Mode.Perm()),
	})
}

func writeMergeMetadataFixture(t *testing.T, repo string, merge *gateMergeIntent) {
	t.Helper()
	for _, metadata := range []struct {
		name string
		data []byte
		mode uint32
	}{
		{name: "MERGE_MODE", data: merge.Mode, mode: merge.ModeFileMode},
		{name: "MERGE_MSG", data: merge.Message, mode: merge.MessageFileMode},
		{name: "MERGE_HEAD", data: merge.Head, mode: merge.HeadFileMode},
	} {
		path := mustGateValue(gitMetadataPath(repo, metadata.name))
		if err := os.WriteFile(path, metadata.data, os.FileMode(metadata.mode)); err != nil {
			t.Fatal(err)
		}
		if err := os.Chmod(path, os.FileMode(metadata.mode)); err != nil {
			t.Fatal(err)
		}
	}
}

func assertMergeMetadataFixture(t *testing.T, repo, name string, want []byte, present bool) {
	t.Helper()
	data, err := os.ReadFile(mustGateValue(gitMetadataPath(repo, name)))
	if !present && errors.Is(err, os.ErrNotExist) {
		return
	}
	if err != nil {
		t.Fatalf("read %s: %v", name, err)
	}
	if !present {
		t.Fatalf("%s unexpectedly exists with %q", name, data)
	}
	if !bytes.Equal(data, want) {
		t.Fatalf("%s = %q, want %q", name, data, want)
	}
}
