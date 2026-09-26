package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/plan"
)

func TestSnapshotReusePreflight(t *testing.T) {
	t.Parallel()
	for _, conflict := range []bool{false, true} {
		t.Run(map[bool]string{false: "contained", true: "source_conflict"}[conflict], func(t *testing.T) {
			root, git := mergeTestRepository(t)
			if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
				t.Fatal(err)
			}
			document := mutationPlan(t, "snapshot preflight")
			if err := plan.Save(filepath.Join(root, filepath.FromSlash(plan.Path)), document); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(root, ".gitignore"), []byte("overgodb-store/\ntmp/\n"), 0o644); err != nil {
				t.Fatal(err)
			}
			git("add", ".")
			git("commit", "-m", "plan")
			source := filepath.Join(t.TempDir(), "source")
			git("worktree", "add", "-b", "topic", source)
			if conflict {
				if err := os.WriteFile(filepath.Join(root, "fixture"), []byte("target\n"), 0o644); err != nil {
					t.Fatal(err)
				}
				git("commit", "-am", "target")
				if err := os.WriteFile(filepath.Join(source, "fixture"), []byte("source\n"), 0o644); err != nil {
					t.Fatal(err)
				}
				if out, err := commandOutput(source, "git", "commit", "-am", "source"); err != nil {
					t.Fatalf("source commit: %s %v", out, err)
				}
			}
			// A capture would fail on this uninitialized store. Admission must
			// finish without opening it, let alone copying its contents.
			if err := os.Mkdir(filepath.Join(source, "overgodb-store"), 0o755); err != nil {
				t.Fatal(err)
			}
			head := git("rev-parse", "HEAD")
			var output bytes.Buffer
			err := prepareMerge(root, "topic", &output)
			if conflict {
				if err == nil || !strings.Contains(err.Error(), "source conflict") {
					t.Fatalf("expected source preflight refusal, got %v", err)
				}
			} else if err != nil || !strings.Contains(output.String(), "already contains") {
				t.Fatalf("contained source: %s %v", output.String(), err)
			}
			if git("rev-parse", "HEAD") != head || git("status", "--porcelain") != "" {
				t.Fatal("preflight mutated the target")
			}
			if _, err := gitOutput(root, "rev-parse", "--verify", "MERGE_HEAD"); err == nil {
				t.Fatal("preflight staged a merge")
			}
		})
	}
}

func TestMergePreflightOwnedDocuments(t *testing.T) {
	t.Parallel()
	for _, path := range []string{apiManifestJSONPath, harnessSurfaceBaselinePath, "source conflict.txt"} {
		t.Run(path, func(t *testing.T) {
			root, git := mergeTestRepository(t)
			branch := git("branch", "--show-current")
			file := filepath.Join(root, filepath.FromSlash(path))
			if err := os.MkdirAll(filepath.Dir(file), 0o755); err != nil {
				t.Fatal(err)
			}
			write := func(value string) {
				t.Helper()
				if err := os.WriteFile(file, []byte(value), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			write("base\n")
			git("add", ".")
			git("commit", "-m", "base document")
			git("checkout", "-b", "topic")
			write("source\n")
			git("commit", "-am", "source")
			git("checkout", branch)
			write("target\n")
			git("commit", "-am", "target")
			head := git("rev-parse", "HEAD")
			err := preflightMergeConflicts(root, head, "topic")
			if mergeOwnedDocument(path) {
				if err != nil {
					t.Fatalf("owned document must reach semantic regeneration: %v", err)
				}
			} else if err == nil || !strings.Contains(err.Error(), path) {
				t.Fatalf("source conflict path lost: %v", err)
			}
			if err := preflightMergeConflicts(root, head, "absent-source"); err == nil || !strings.Contains(err.Error(), "conflict preflight") {
				t.Fatalf("Git error was accepted as a merge: %v", err)
			}
			if git("rev-parse", "HEAD") != head || git("status", "--porcelain") != "" {
				t.Fatal("preflight changed target state")
			}
		})
	}
}

// TestMergeHarnessBaselineConflictResolution proves a generated harness
// ratchet conflict stays on the lane's reviewed baseline until gate repair.
func TestMergeHarnessBaselineConflictResolution(t *testing.T) {
	t.Parallel()
	root, git := mergeTestRepository(t)
	branch := git("branch", "--show-current")
	for _, path := range []string{harnessSurfaceBaselinePath, modernGoBaselinePath} {
		file := filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(file), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(file, []byte("base\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	git("add", ".")
	git("commit", "-m", "base baselines")
	git("checkout", "-b", "topic")
	harness := filepath.Join(root, filepath.FromSlash(harnessSurfaceBaselinePath))
	if err := os.WriteFile(harness, []byte("source\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("commit", "-am", "source baseline")
	git("checkout", branch)
	if err := os.WriteFile(harness, []byte("target\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("commit", "-am", "target baseline")
	head := git("rev-parse", "HEAD")
	if err := preflightMergeConflicts(root, head, "topic"); err != nil {
		t.Fatalf("generated conflict was rejected: %v", err)
	}
	if _, err := commandOutput(root, "git", "merge", "--no-commit", "topic"); err == nil {
		t.Fatal("fixture did not create the expected baseline conflict")
	}
	if err := restoreMergeTargetBaselines(root, head); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(harness)
	if err != nil || string(raw) != "target\n" {
		t.Fatalf("lane baseline = %q, %v", raw, err)
	}
	if _, err := commandOutput(root, "git", "add", "--", harnessSurfaceBaselinePath); err != nil {
		t.Fatal(err)
	}
	if unmerged := git("ls-files", "-u"); unmerged != "" {
		t.Fatalf("generated conflict remains: %s", unmerged)
	}
}
