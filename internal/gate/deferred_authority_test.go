package gate

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/authoritylock"
	"overgo/internal/automationcheck"
	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
	"overgo/internal/processlock"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

const deferredAuthorityRepo = "OVERGO_TEST_DEFERRED_AUTHORITY_REPO"

func TestDeferredAuthorityLifetime(t *testing.T) {
	if repo := os.Getenv(deferredAuthorityRepo); repo != "" {
		err := runDeferredLanes(repo, "store", func(g *gateContext, _ runrecord.GateLaneObligation) (runrecord.Outcome, string, []runrecord.GateStep, error) {
			fmt.Println(g.sourceRoot())
			_, err := io.Copy(io.Discard, os.Stdin)
			return deferredAuthorityResult(g, err)
		})
		if err != nil {
			t.Fatal(err)
		}
		return
	}
	if isolatedProcess(t) {
		return
	}
	t.Setenv(dataroot.Env, "")
	t.Setenv(audioReferenceEnv, "")
	for _, scenario := range []string{"independent-write", "terminal-conflict", "cancelled", "process-death"} {
		t.Run(scenario, func(t *testing.T) {
			repo, store, pending := deferredAuthorityFixture(t)
			if scenario == "process-death" {
				child := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestDeferredAuthorityLifetime$", "-test.count=1")
				child.Env = append(os.Environ(), deferredAuthorityRepo+"="+repo)
				input, err := child.StdinPipe()
				if err != nil {
					t.Fatal(err)
				}
				defer input.Close()
				output, err := child.StdoutPipe()
				if err != nil {
					t.Fatal(err)
				}
				child.Stderr = os.Stderr
				if err := child.Start(); err != nil {
					t.Fatal(err)
				}
				defer func() { _ = child.Process.Kill(); _ = child.Wait() }()
				// The candidate path is a readiness receipt, not a timer.
				candidate, err := bufio.NewReader(output).ReadString('\n')
				if err != nil {
					t.Fatal(err)
				}
				candidate = strings.TrimSpace(candidate)
				if lock, err := acquireLaneRunner(repo); !errors.Is(err, processlock.ErrBusy) {
					_ = lock.Close()
					t.Fatalf("child did not own runner: %v", err)
				}
				pending = deferredAuthorityCurrent(t, store)
				if pending.State != runrecord.LaneObligationRunning {
					t.Fatal("child did not persist running state")
				}
				if err := child.Process.Kill(); err != nil {
					t.Fatal(err)
				}
				_ = child.Wait()
				// Remove only the exact fixture worktree returned by this child.
				if filepath.Base(candidate) != "candidate" || !strings.HasPrefix(filepath.Base(filepath.Dir(candidate)), "overgo-gate-acceptance-") {
					t.Fatalf("unexpected fixture worktree %q", candidate)
				}
				runGitFixture(t, repo, "worktree", "remove", "--force", candidate)
				if err := os.Remove(filepath.Dir(candidate)); err != nil {
					t.Fatal(err)
				}
			}
			var executed bool
			var competing runrecord.GateLaneObligation
			independent := testutil.ArtifactID(t, artifact.KindEvidence, "independent plan writer")
			err := runDeferredLanes(repo, "store", func(g *gateContext, running runrecord.GateLaneObligation) (runrecord.Outcome, string, []runrecord.GateStep, error) {
				executed = true
				if scenario == "process-death" && running.ID != pending.ID {
					t.Fatal("restart changed the running obligation")
				}
				lock, err := authoritylock.Acquire(repo)
				if err != nil {
					t.Fatalf("independent dispatch blocked during execution: %v", err)
				}
				defer lock.Close()
				duplicate := runDeferredLanes(repo, "store", func(*gateContext, runrecord.GateLaneObligation) (runrecord.Outcome, string, []runrecord.GateStep, error) {
					t.Fatal("duplicate runner executed")
					return "", "", nil, nil
				})
				if !errors.Is(duplicate, processlock.ErrBusy) {
					t.Fatalf("duplicate runner admission: %v", duplicate)
				}
				if err := store.Refresh(t.Context()); err != nil {
					t.Fatal(err)
				}
				if _, err := requireLaneObligationsResolved(repo, store); err == nil || !strings.Contains(err.Error(), "still running") {
					t.Fatalf("next gate failed to report outstanding work: %v", err)
				}
				before, err := g.sourceEnvironment()
				if err != nil {
					t.Fatal(err)
				}
				fingerprint, err := g.phaseInputFingerprint("build")
				if err != nil {
					t.Fatal(err)
				}
				inputs, err := g.manifestExternalInputs(true)
				if err != nil || len(inputs) != 1 {
					t.Fatalf("external input fixture: %v %v", inputs, err)
				}
				for name, body := range map[string]string{"fixture.go": "uncommitted later source", "input.json": "later input", dataroot.ConfigFile: `{"models":"later-models"}`} {
					if err := os.WriteFile(filepath.Join(repo, name), []byte(body), 0o600); err != nil {
						t.Fatal(err)
					}
				}
				changedFingerprint, err := g.phaseInputFingerprint("build")
				if err != nil || changedFingerprint != fingerprint {
					t.Fatalf("source fingerprint drifted: %v", err)
				}
				changedInputs, err := g.manifestExternalInputs(true)
				if err != nil || !slices.Equal(inputs, changedInputs) {
					t.Fatalf("manifest inputs drifted: %v", err)
				}
				after, err := g.sourceEnvironment()
				if err != nil || !slices.Equal(before, after) {
					t.Fatalf("environment drifted: %v", err)
				}
				data, err := os.ReadFile(filepath.Join(g.sourceRoot(), "fixture.go"))
				if err != nil || string(data) != "package fixture\n" {
					t.Fatalf("candidate source drifted: %s %v", data, err)
				}
				for _, binding := range after {
					name, value, _ := strings.Cut(binding, "=")
					if strings.HasPrefix(name, "OVERGO_") {
						t.Setenv(name, value)
					}
				}
				roots, err := dataroot.Resolve(g.sourceRoot())
				if err != nil || roots.Models != filepath.Join(repo, "models") {
					t.Fatalf("child roots drifted: %+v %v", roots, err)
				}
				if scenario == "terminal-conflict" {
					competing, err = running.Transition(runrecord.LaneObligationFailed, independent, time.Now())
					if err != nil {
						t.Fatal(err)
					}
					batch, err := competing.Batch(&running.ID)
					if err != nil {
						t.Fatal(err)
					}
					if _, err := store.Commit(t.Context(), batch); err != nil {
						t.Fatal(err)
					}
				} else {
					_, err := store.Commit(t.Context(), artifact.Batch{Key: "independent/write", Artifacts: []artifact.Descriptor{{ID: independent}}})
					if err != nil {
						t.Fatal(err)
					}
				}
				if scenario == "cancelled" {
					return deferredAuthorityResult(g, context.Canceled)
				}
				return deferredAuthorityResult(g, nil)
			})
			if !executed {
				t.Fatalf("runner did not execute: %v", err)
			}
			current := deferredAuthorityCurrent(t, store)
			if scenario != "terminal-conflict" {
				if _, found, err := store.Artifact(t.Context(), independent); err != nil || !found {
					t.Fatalf("concurrent store write lost: %v", err)
				}
			}
			switch scenario {
			case "terminal-conflict":
				if err == nil || !strings.Contains(err.Error(), "obligation changed") || current.ID != competing.ID {
					t.Fatalf("competing publication overwritten: %+v %v", current, err)
				}
			case "cancelled":
				if !errors.Is(err, context.Canceled) || current.State != runrecord.LaneObligationFailed {
					t.Fatalf("cancellation credited: %+v %v", current, err)
				}
			default:
				if err != nil || current.State != runrecord.LaneObligationPassed {
					t.Fatalf("completion: %+v %v", current, err)
				}
			}
			for _, acquire := range []func(string) (*processlock.Lock, error){authoritylock.Acquire, acquireLaneRunner} {
				lock, err := acquire(repo)
				if err != nil {
					t.Fatalf("terminal path retained ownership: %v", err)
				}
				if err := lock.Close(); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}

func deferredAuthorityResult(g *gateContext, err error) (runrecord.Outcome, string, []runrecord.GateStep, error) {
	wall, measurementErr := g.clock.Elapsed()
	err = errors.Join(err, measurementErr)
	outcome, failure := runrecord.OutcomeSucceeded, ""
	if err != nil {
		outcome, failure = runrecord.OutcomeFailed, testDeviceCheckName
	}
	return outcome, failure, []runrecord.GateStep{gateEvidenceRecord(testDeviceCheckName, runrecord.PhaseTest, automationcheck.Evidence{DurationNS: wall}, err, "")}, err
}

func deferredAuthorityCurrent(t *testing.T, store *overgodb.Store) runrecord.GateLaneObligation {
	t.Helper()
	if err := store.Refresh(t.Context()); err != nil {
		t.Fatal(err)
	}
	current, found, err := runrecord.CurrentGateLaneObligation(t.Context(), store)
	if err != nil || !found {
		t.Fatalf("current obligation: %+v %v", current, err)
	}
	return current
}

func deferredAuthorityFixture(t *testing.T) (string, *overgodb.Store, runrecord.GateLaneObligation) {
	t.Helper()
	repo := newLifecycleRepo(t)
	if err := os.WriteFile(filepath.Join(repo, "fixture.go"), []byte("package fixture\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "input.json"), []byte(`{"value":"original"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "core.autocrlf", "false")
	runGitFixture(t, repo, "config", "user.email", "deferred@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Deferred Test")
	runGitFixture(t, repo, "add", "fixture.go", "input.json")
	runGitFixture(t, repo, "commit", "-q", "-m", "fixture")
	commit := strings.TrimSpace(recoveryGit(t, repo, "rev-parse", "HEAD"))
	store, err := overgodb.Open(filepath.Join(repo, "store"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Error(err)
		}
	})
	pending, err := runrecord.NewGateLaneObligation(commit, testutil.ArtifactID(t, artifact.KindEvidence, "preparation"), testutil.ArtifactID(t, artifact.KindEvidence, "result"), []string{testDeviceCheckName}, []string{"fixture.go", "input.json"}, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	batch, err := pending.Batch(nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	return repo, store, pending
}
