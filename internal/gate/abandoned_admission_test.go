package gate

import (
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/authoritylock"
	"overgo/internal/overgodb"
	"overgo/internal/processlock"
	"overgo/internal/runrecord"
)

func TestAbandonedGateProcess(t *testing.T) {
	repo := os.Getenv("OVERGO_TEST_ABANDONED_GATE")
	if repo == "" {
		return
	}
	lock, err := authoritylock.Acquire(repo)
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Close()
	g := gateContext{repo: repo, storePath: StorePath, start: time.Now(), environment: lifecycleTestEnvironment(t)}
	if err := g.prepare(); err != nil {
		t.Fatal(err)
	}
	if err := json.NewEncoder(os.Stdout).Encode(g.preparation); err != nil {
		t.Fatal(err)
	}
	_, _ = io.Copy(io.Discard, os.Stdin)
}

func TestAbandonedGateAdmission(t *testing.T) {
	for _, obstacle := range []string{"missing locator", "corrupt locator", "record debt", "commit intent"} {
		t.Run(obstacle, func(t *testing.T) {
			repo := newAbandonedGateRepo(t)
			lock, err := authoritylock.Acquire(repo)
			if err != nil {
				t.Fatal(err)
			}
			defer lock.Close()
			g := gateContext{repo: repo, storePath: StorePath, start: time.Now(), environment: lifecycleTestEnvironment(t)}
			if err := g.prepare(); err != nil {
				t.Fatal(err)
			}
			store, err := overgodb.Open(filepath.Join(repo, StorePath))
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			filename := gateHeartbeatFile
			switch obstacle {
			case "missing locator":
				if err := os.Remove(filepath.Join(repo, filename)); err != nil {
					t.Fatal(err)
				}
			case "record debt":
				filename = gateDebtFile
			case "commit intent":
				filename = gateCommitIntentFile
			}
			if obstacle != "missing locator" {
				if err := os.WriteFile(filepath.Join(repo, filename), []byte("preserve unresolved authority"), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			ledger, inputs := terminalEvidenceFixture(t, repo)
			evidence, err := ledger.openPackageEvidence()
			if err != nil {
				t.Fatal(err)
			}
			defer evidence.store.Close()
			packages := []string{"fixture/good", "fixture/pending"}
			if err := evidence.prepare(t.Context(), packages, "complete", inputs, ledger.retryCache); err != nil {
				t.Fatal(err)
			}
			if err := evidence.record(t.Context(), "fixture/good", true); err != nil {
				t.Fatal(err)
			}
			if err := store.Refresh(t.Context()); err != nil {
				t.Fatal(err)
			}
			head, sequence := store.Head()
			_, err = admitPendingGateState(repo, StorePath, store)
			if obstacle == "missing locator" {
				if err != nil {
					t.Fatal(err)
				}
				finalized, found, err := runrecord.GateFinalizationForPreparation(t.Context(), store, g.preparation.ID)
				if err != nil || !found || finalized.Outcome != runrecord.OutcomeCancelled {
					t.Fatalf("recovery = %+v, %v, %v", finalized, found, err)
				}
			} else {
				if err == nil {
					t.Fatal("unresolved authority admitted")
				}
				if err := store.Refresh(t.Context()); err != nil {
					t.Fatal(err)
				}
				if after, number := store.Head(); after != head || number != sequence {
					t.Fatal("refusal mutated store authority")
				}
				if bytes, err := os.ReadFile(filepath.Join(repo, filename)); err != nil || string(bytes) != "preserve unresolved authority" {
					t.Fatalf("refusal changed recovery file: %s, %v", bytes, err)
				}
			}
			if err := evidence.prepare(t.Context(), packages, "complete", inputs, ledger.retryCache); err != nil {
				t.Fatal(err)
			}
			pending, reused, err := ledger.packageCachePartition(packages, "complete", inputs)
			if err != nil || reused != 1 || len(pending) != 1 || pending[0] != "fixture/pending" {
				t.Fatalf("recovery lost package evidence: pending=%v reused=%d err=%v", pending, reused, err)
			}
			requireStoredPackageObligation(t, filepath.Join(repo, StorePath), evidence.obligations["fixture/pending"].ID)
		})
	}
	t.Run("ambiguous preparations", func(t *testing.T) {
		fixture := newStaleLifecycleRecoveryFixture(t, 2, true)
		lock, err := authoritylock.Acquire(fixture.repo)
		if err != nil {
			t.Fatal(err)
		}
		defer lock.Close()
		store, err := overgodb.Open(filepath.Join(fixture.repo, fixture.storePath))
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		head, sequence := store.Head()
		if _, err := admitPendingGateState(fixture.repo, fixture.storePath, store); err == nil || !strings.Contains(err.Error(), "2 unresolved") {
			t.Fatalf("ambiguous recovery = %v", err)
		}
		if err := store.Refresh(t.Context()); err != nil {
			t.Fatal(err)
		}
		if after, number := store.Head(); after != head || number != sequence {
			t.Fatal("ambiguous recovery changed authority")
		}
	})
	t.Run("incomplete finalization authority", func(t *testing.T) {
		fixture := newStaleLifecycleRecoveryFixture(t, 1, false)
		lock, err := authoritylock.Acquire(fixture.repo)
		if err != nil {
			t.Fatal(err)
		}
		defer lock.Close()
		store, err := overgodb.Open(filepath.Join(fixture.repo, fixture.storePath))
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		head, sequence := store.Head()
		if _, err := admitPendingGateState(fixture.repo, fixture.storePath, store); err == nil {
			t.Fatal("incomplete finalization admitted")
		}
		if err := store.Refresh(t.Context()); err != nil {
			t.Fatal(err)
		}
		if after, number := store.Head(); after != head || number != sequence {
			t.Fatal("invalid finalization was rewritten")
		}
	})
	t.Run("hard exit with recent heartbeat", func(t *testing.T) {
		repo := newAbandonedGateRepo(t)
		child := exec.Command(os.Args[0], "-test.run=^TestAbandonedGateProcess$", "-test.timeout=1m")
		child.Env = append(os.Environ(), "OVERGO_TEST_ABANDONED_GATE="+repo)
		input, err := child.StdinPipe()
		if err != nil {
			t.Fatal(err)
		}
		output, err := child.StdoutPipe()
		if err != nil {
			t.Fatal(err)
		}
		child.Stderr = os.Stderr
		if err := child.Start(); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { input.Close(); child.Process.Kill(); child.Wait() })
		var encoded json.RawMessage
		if err := json.NewDecoder(output).Decode(&encoded); err != nil {
			t.Fatal(err)
		}
		preparation, err := runrecord.ParseGateLifecycle(encoded)
		if err != nil {
			t.Fatal(err)
		}
		t.Chdir(repo)
		options := Options{StorePath: StorePath, PlanRef: "fixture/do", PathsCSV: "candidate.go", MessageFile: "tmp/message.txt"}
		if err := Run(options); !errors.Is(err, processlock.ErrBusy) {
			t.Fatalf("live owner admission = %v; want OS contention", err)
		}
		if err := child.Process.Kill(); err != nil {
			t.Fatal(err)
		}
		_ = child.Wait()
		before, err := command(repo, "git", "status", "--porcelain")
		if err != nil {
			t.Fatal(err)
		}
		// No plan exists: recovery must complete before plan binding refuses.
		if err := Run(options); err == nil {
			t.Fatal("missing plan admitted")
		}
		store, err := overgodb.Open(filepath.Join(repo, StorePath))
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		finalization, found, err := runrecord.GateFinalizationForPreparation(t.Context(), store, preparation.ID)
		if err != nil || !found || finalization.Outcome != runrecord.OutcomeCancelled {
			t.Fatalf("dead owner was not cancelled: %+v, %v, %v", finalization, found, err)
		}
		head, sequence := store.Head()
		if err := Run(options); err == nil {
			t.Fatal("missing plan admitted on retry")
		}
		if err := store.Refresh(t.Context()); err != nil {
			t.Fatal(err)
		}
		if after, number := store.Head(); after != head || number != sequence {
			t.Fatal("repeated admission appended another recovery")
		}
		after, err := command(repo, "git", "status", "--porcelain")
		if err != nil || after != before {
			t.Fatalf("recovery changed candidate: %q -> %q, %v", before, after, err)
		}
	})
}

func newAbandonedGateRepo(t *testing.T) string {
	t.Helper()
	t.Setenv("OVERGO_STRATEGY_ID", "")
	repo := newLifecycleRepo(t)
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "admission@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Gate Admission Test")
	if err := os.WriteFile(filepath.Join(repo, ".gitignore"), []byte("tmp/\novergodb-store/\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "candidate.go"), []byte("package candidate\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", ".gitignore", "candidate.go")
	runGitFixture(t, repo, "commit", "-qm", "baseline")
	if err := os.WriteFile(filepath.Join(repo, "candidate.go"), []byte("package candidate\n// pending work\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return repo
}
