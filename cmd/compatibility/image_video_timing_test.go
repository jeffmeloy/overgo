package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

const mediaTimingSHA256 = "1a1ebaf1e98a84e3c4fa0da0ae39b7d00825643e6c8c2c326466bd95bf860a4b"

// Bind the entire numerical scope and the newly used measurement owner. Old
// resource measurements retain their original source; this proves output reuse.
func checkMediaTimingSource(root, revision string, paths []string) (string, error) {
	raw, err := os.ReadFile(filepath.Join(root, "docs/image_video_timing_reconciliation.json"))
	if err != nil {
		return "", err
	}
	if err := checkMediaProtocolIdentity(raw, mediaTimingSHA256); err != nil {
		return "", err
	}
	var proof struct {
		Before, After, Scope string
		Evidence             artifact.ID
		AssertionSources     map[string]string `json:"assertion_sources"`
	}
	if err := json.Unmarshal(raw, &proof); err != nil {
		return "", err
	}
	scope := append(slices.Clone(paths), "internal/processmeasure", "internal/workflowruntime")
	changed, err := mediaRuntimeChanges(root, proof.After, revision, scope)
	if err != nil {
		return "", err
	}
	if len(changed) != 0 {
		return "", fmt.Errorf("source exceeds retained timing reconciliation: %s", strings.Join(changed, ", "))
	}
	roots, err := dataroot.Resolve(root)
	if err != nil {
		return "", err
	}
	store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
	if err != nil {
		return "", err
	}
	defer store.Close()
	content, found, err := artifact.ReadContent(context.Background(), store, proof.Evidence)
	if err != nil {
		return "", err
	}
	var receipt struct{ Source, Command, TestSHA256, Output string }
	if !found || json.Unmarshal(content.Data, &receipt) != nil || receipt.Source != proof.After {
		return "", errors.New("missing source-bound timing reconciliation receipt")
	}
	if receipt.TestSHA256 != proof.AssertionSources["internal/hostmath/dispatch_test.go"] {
		return "", errors.New("reconciliation overlay differs from acquisition")
	}
	for path, expected := range proof.AssertionSources {
		source, err := os.ReadFile(filepath.Join(root, path))
		source = []byte(strings.ReplaceAll(string(source), "\r\n", "\n"))
		if err != nil || fmt.Sprintf("%x", sha256.Sum256(source)) != expected {
			return "", fmt.Errorf("reconciliation assertion source differs: %s", path)
		}
	}
	if err := checkMediaTestReceipt("timing reconciliation", []byte(receipt.Output), map[string][]string{
		"overgo/internal/hostmath":        {"TestDispatchSchedulingPreservesResults"},
		"overgo/internal/processmeasure":  {"TestStopwatchReadsCounter", "TestStopwatchRetainsFailure"},
		"overgo/internal/workflowruntime": {"TestRuntimePublishesCancelledRun", "TestWorkflowRestartSkipsCompletedStages", "TestCancellationCauseReachesTerminalReceipt", "TestStageProgressPrecedesSiblingCompletion", "TestRemoteStageAttemptLineage", "TestRemoteCancellation", "TestRemoteArtifactIdentity"},
	}); err != nil {
		return "", err
	}
	return proof.Before, nil
}

// Empty revision includes worktree and untracked inputs; test-only edits do not
// turn historical numerical outputs into fresh measurements.
func mediaRuntimeChanges(root, before, revision string, paths []string) ([]string, error) {
	git := func(args ...string) ([]byte, error) {
		c := exec.Command("git", args...)
		c.Dir = root
		return c.Output()
	}
	// Inspect both sides: a rename from source to a non-source extension is a
	// deletion from the execution scope, even when Git detects identical bytes.
	args := []string{"diff", "--no-renames", "--name-only", "-z", before}
	if revision != "" {
		args = append(args, revision)
	}
	changed, err := git(append(append(args, "--"), paths...)...)
	if err != nil {
		return nil, err
	}
	if revision == "" {
		untracked, err := git(append([]string{"ls-files", "--others", "--exclude-standard", "-z", "--"}, paths...)...)
		if err != nil {
			return nil, err
		}
		changed = append(changed, untracked...)
	}
	return slices.DeleteFunc(strings.Split(string(changed), "\x00"), func(path string) bool { return !mediaLifecycleProductionPath(path) }), nil
}

func TestMediaTimingSourceReconciliation(t *testing.T) {
	root := testutil.RepoRoot(t)
	if _, err := checkMediaTimingSource(root, "", []string{"internal/hostmath"}); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(root, "docs/image_video_timing_reconciliation.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"before", "after", "evidence", "scope", "assertion_sources"} {
		var altered map[string]any
		if err := json.Unmarshal(raw, &altered); err != nil {
			t.Fatal(err)
		}
		delete(altered, field)
		encoded, err := json.Marshal(altered)
		if err != nil {
			t.Fatal(err)
		}
		if checkMediaProtocolIdentity(encoded, mediaTimingSHA256) == nil {
			t.Fatalf("missing %s accepted", field)
		}
	}
	fixture := t.TempDir()
	git := func(args ...string) {
		t.Helper()
		c := exec.Command("git", args...)
		c.Dir = fixture
		if out, err := c.CombinedOutput(); err != nil {
			t.Fatalf("git: %v %s", err, out)
		}
	}
	write := func(name, text string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(fixture, name), []byte(text), 0600); err != nil {
			t.Fatal(err)
		}
	}
	git("init", "-q")
	git("config", "user.email", "fixture@example.invalid")
	git("config", "user.name", "Fixture")
	write("kernel.go", "package fixture\n")
	git("add", ".")
	git("commit", "-qm", "baseline")
	baseline, err := mediaRuntimeIdentity(fixture, "HEAD", []string{"."})
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"kernel_test.go", "kernel.go", "added.go", "added file.go", "café.go"} {
		write(name, "package fixture\n// changed\n")
		changed, err := mediaRuntimeChanges(fixture, "HEAD", "", []string{"."})
		if err != nil || slices.Contains(changed, name) == strings.HasSuffix(name, "_test.go") {
			t.Fatalf("source selection %s: %v %v", name, changed, err)
		}
	}
	if _, err := mediaRuntimeChanges(fixture, "missing-reference", "", []string{"."}); err == nil {
		t.Fatal("invalid reference accepted")
	}
	git("add", ".")
	git("commit", "-qm", "source additions")
	added, err := mediaRuntimeIdentity(fixture, "HEAD", []string{"."})
	if err != nil || added == baseline {
		t.Fatal("source additions did not change identity", err)
	}
	for _, name := range []string{"kernel.go", "added file.go", "café.go"} {
		git("mv", name, name+".txt")
	}
	if err := os.Remove(filepath.Join(fixture, "added.go")); err != nil {
		t.Fatal(err)
	}
	changed, err := mediaRuntimeChanges(fixture, "HEAD", "", []string{"."})
	slices.Sort(changed)
	if err != nil || !slices.Equal(changed, []string{"added file.go", "added.go", "café.go", "kernel.go"}) {
		t.Fatalf("source removals/renames lost: %v, %v", changed, err)
	}
	git("add", ".")
	git("commit", "-qm", "source removals")
	if _, err := mediaRuntimeIdentity(fixture, "HEAD", []string{"."}); err == nil {
		t.Fatal("test-only tree accepted as a production source identity")
	}
}
