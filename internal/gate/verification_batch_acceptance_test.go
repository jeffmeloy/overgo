package gate

import (
	"bytes"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestGovernedBatchDeclarationAcceptance(t *testing.T) {
	t.Parallel()
	for _, mode := range []string{"pass", "failure", "skip", "unavailable", "no-match", "mutation"} {
		t.Run(mode, func(t *testing.T) {
			t.Parallel()
			g, batch, tree := verificationBatchFixture(t, mode)
			before, err := os.ReadFile(filepath.Join(g.repo, plan.Path))
			if err != nil {
				t.Fatal(err)
			}
			// These are real acceptance executions, not mocked positive results.
			checks, err := g.batchAcceptanceChecks([]automationcheck.Check{
				gateCheck("acceptance", runrecord.PhaseTest, g.stepAcceptance),
			}, batch)
			if err != nil {
				t.Fatal(err)
			}
			invocations, err := automationcheck.Plan(checks, automationcheck.Impact{})
			if err != nil {
				t.Fatal(err)
			}
			invocations = append(invocations, automationcheck.Invocation{
				ID:    testutil.ArtifactID(t, artifact.KindRecipe, "batch commit"),
				Check: automationcheck.Descriptor{Name: "commit", Phase: runrecord.PhasePackage, Dependencies: []string{"acceptance"}},
			})
			manifest, err := automationcheck.BindManifestPlan(
				testutil.ArtifactID(t, artifact.KindProfile, "batch base"),
				testutil.ArtifactID(t, artifact.KindProfile, "batch candidate"),
				strings.Repeat("a", 64), candidateTreeKey(tree),
				automationcheck.Surface{Identity: "batch acceptance fixture"}, automationcheck.Impact{}, invocations,
			)
			if err != nil {
				t.Fatal(err)
			}
			g.manifestPlan = &manifest
			acceptances := invocations[:len(invocations)-1]
			for index, invocation := range acceptances {
				acceptances[index], err = g.bindCheckExecution(manifest, invocation, manifest.CandidateManifest)
				if err != nil {
					t.Fatal(err)
				}
			}
			results, err := automationcheck.ExecuteDAG(t.Context(), acceptances, nil, automationcheck.Run)
			if err != nil {
				t.Fatal(err)
			}
			ran := 0
			for _, result := range results {
				if result.Invocation.ID.Valid() {
					ran++
				}
				if result.Evidence.ID.Valid() {
					g.terminal[result.Invocation.Check.Name] = result.Evidence
				}
			}
			if results[0].Err != nil || !results[0].Evidence.ID.Valid() {
				t.Fatalf("first member did not execute: %+v", results[0])
			}
			if mode == "pass" {
				if ran != len(acceptances) || g.acceptedTree != tree {
					t.Fatalf("full batch acceptance: ran=%d/%d accepted=%s", ran, len(acceptances), g.acceptedTree)
				}
				if err := validateManifestCommitAdmission(manifest, g.terminal, nil); err != nil {
					t.Fatal(err)
				}
				delete(g.terminal, batch.Checkpoints[0].GateCheckName())
				if err := validateManifestCommitAdmission(manifest, g.terminal, nil); err == nil {
					t.Fatal("commit admitted without the first member's terminal evidence")
				}
			} else {
				if ran != len(batch.Checkpoints) || results[1].Err == nil || g.acceptedTree != "" || g.stepEvidence["acceptance"] != "" {
					t.Fatalf("rejected member escaped barrier: ran=%d, result=%+v accepted=%q", ran, results[1], g.acceptedTree)
				}
				if err := validateManifestCommitAdmission(manifest, g.terminal, nil); err == nil {
					t.Fatal("commit admitted despite a rejected subordinate verifier")
				}
			}
			after, err := os.ReadFile(filepath.Join(g.repo, plan.Path))
			if err != nil || !bytes.Equal(before, after) || recoveryGit(t, g.repo, "rev-parse", "HEAD") != g.planHead {
				t.Fatalf("acceptance changed parent plan or branch: %v", err)
			}
			if output := recoveryGit(t, g.repo, "status", "--porcelain"); output != "" {
				t.Fatalf("acceptance changed candidate worktree: %s", output)
			}
			t.Logf("executed=%d required=%d parent-completed=false reduced-gating=false", ran, len(acceptances))
		})
	}
	t.Run("full gate coverage preserved", func(t *testing.T) {
		g, batch, _ := verificationBatchFixture(t, "pass")
		original := g.pipelineChecks()
		checks, err := g.batchAcceptanceChecks(g.pipelineChecks(), batch)
		if err != nil {
			t.Fatal(err)
		}
		for _, required := range original {
			if !slices.ContainsFunc(checks, func(check automationcheck.Check) bool {
				return check.Descriptor.Name == required.Descriptor.Name &&
					check.Descriptor.Always == required.Descriptor.Always
			}) {
				t.Fatalf("batch removed full-gate requirement %s", required.Descriptor.Name)
			}
		}
		for _, checkpoint := range batch.Checkpoints {
			if phaseReusesEvidence(checkpoint.GateCheckName()) {
				t.Fatal("subordinate acceptance unexpectedly reused partial checkpoint evidence")
			}
		}
		if _, err := g.batchAcceptanceChecks(nil, batch); err == nil {
			t.Fatal("batch accepted without the parent integration check")
		}
		if _, err := automationcheck.Plan(checks, automationcheck.Impact{}); err != nil {
			t.Fatalf("full batch dependency graph: %v", err)
		}
	})
	t.Run("recovery uses complete batch acceptance", testBatchCompletionRecovery)
}

func testBatchCompletionRecovery(t *testing.T) {
	t.Parallel()
	for _, complete := range []bool{false, true} {
		batch := &plan.VerificationBatch{
			Scope: []string{"candidate.go"}, Rationale: "Prove exact recovery.", ReopenWhen: "The accepted tree changes.",
			Checkpoints: []plan.VerificationCheckpoint{{ID: "source", Title: "Source", Verify: "go test ./x -run '^TestSource$'"}},
		}
		fixture := newInterruptedCommitFixtureWithHook(t, nil, batch)
		var members []runrecord.GateStep
		if complete {
			checkpoint := batch.Checkpoints[0]
			evidence, err := runrecord.FormatCompletionAcceptanceEvidence(runrecord.CurrentVerifyPolicy, "ratchet/recover", checkpoint.Verify)
			if err != nil {
				t.Fatal(err)
			}
			members = append(members, runrecord.GateStep{
				Name: checkpoint.GateCheckName(), Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: 1, Evidence: evidence,
			})
		}
		_, publication := successfulAttemptBatchForReference(t, fixture, "ratchet", "recover", "go test ./cmd/gate", true, "batch recovery", members...)
		store := openRecoveryStore(t, fixture)
		_, err := store.Commit(t.Context(), publication)
		closeErr := store.Close()
		if err != nil || closeErr != nil {
			t.Fatalf("publish recovery fixture: %v, %v", err, closeErr)
		}
		if _, err := recoverInterruptedCommit(fixture.repo, fixture.storePath); err != nil {
			t.Fatalf("recover complete=%t: %v", complete, err)
		}
		wantHead, wantPlan := fixture.parent, fixture.planBefore
		if complete {
			wantHead, wantPlan = fixture.commit, fixture.planAfter
		}
		if got := recoveryGit(t, fixture.repo, "rev-parse", "HEAD"); got != wantHead {
			t.Fatalf("recover complete=%t HEAD=%s, want %s", complete, got, wantHead)
		}
		gotPlan, err := os.ReadFile(filepath.Join(fixture.repo, plan.Path))
		if err != nil || !bytes.Equal(gotPlan, wantPlan) {
			t.Fatalf("recover complete=%t changed exact parent contract: %v", complete, err)
		}
	}
}

func verificationBatchFixture(t *testing.T, mode string) (*gateContext, *plan.VerificationBatch, string) {
	t.Helper()
	repo := t.TempDir()
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "batch@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Batch Test")
	if err := os.Mkdir(filepath.Join(repo, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	batch := &plan.VerificationBatch{
		Scope: []string{"unit_test.go", "mode"}, Rationale: "Verify producer and consumer together.",
		ReopenWhen: "The source or acceptance changes.",
		Checkpoints: []plan.VerificationCheckpoint{
			{ID: "producer", Title: "Producer", Verify: "go test . -run '^TestProducer$' -count=1 -v"},
			{ID: "consumer", Title: "Consumer", Verify: "go test . -run '^TestConsumer$' -count=1 -v", DependsOn: []string{"producer"}},
		},
	}
	if mode == "no-match" {
		batch.Checkpoints[1].Verify = "go test . -run '^TestAbsent$' -count=1 -v"
	}
	document := plan.Plan{Items: []plan.Item{{ID: "audio", Status: plan.StatusOpen, Steps: []plan.Step{{
		ID: "dataset", Status: plan.StatusOpen, Verify: "go test . -run '^TestIntegration$' -count=1 -v", VerificationBatch: batch,
	}}}}}
	if err := plan.Save(filepath.Join(repo, plan.Path), document); err != nil {
		t.Fatal(err)
	}
	for path, content := range map[string]string{
		"go.mod": "module batchfixture\n\ngo 1.25\n", "mode": mode, ".gitignore": "tmp/\n",
		"unit_test.go": `package batchfixture
import ("os"; "testing")
func TestProducer(t *testing.T) { if _, err := os.ReadFile("mode"); err != nil { t.Fatal(err) } }
func TestConsumer(t *testing.T) {
    mode, err := os.ReadFile("mode"); if err != nil { t.Fatal(err) }
    switch string(mode) {
    case "failure": t.Fatal("consumer failed")
    case "skip": t.Skip("missing fixture")
    case "unavailable": t.Log("UNAVAILABLE")
    case "mutation": if err := os.WriteFile("mode", []byte("changed"), 0600); err != nil { t.Fatal(err) }
    }
}
func TestIntegration(t *testing.T) { if _, err := os.ReadFile("go.mod"); err != nil { t.Fatal(err) } }
`,
	} {
		if err := os.WriteFile(filepath.Join(repo, path), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	runGitFixture(t, repo, "add", "--", ".")
	runGitFixture(t, repo, "commit", "-q", "-m", "batch fixture")
	return verificationBatchContext(t, repo)
}

// verificationBatchContext: gate context over an existing fixture repo; the
// crash helper builds it in another process over the parent's repo.
func verificationBatchContext(t *testing.T, repo string) (*gateContext, *plan.VerificationBatch, string) {
	t.Helper()
	document, err := plan.Load(filepath.Join(repo, plan.Path))
	if err != nil {
		t.Fatal(err)
	}
	head := recoveryGit(t, repo, "rev-parse", "HEAD")
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	authority, err := plan.ResolveCompletionAuthority(t.Context(), repo, head, document, store)
	if err != nil {
		t.Fatal(err)
	}
	g := &gateContext{
		repo: repo, planRef: "audio/dataset", planHead: head, completionAuthority: authority,
		storePath: filepath.Join("tmp", "batch-store"), environment: lifecycleTestEnvironment(t),
		paths:        []string{plan.Path, "go.mod", "unit_test.go", "mode"},
		stepEvidence: map[string]string{}, terminal: map[string]automationcheck.Evidence{},
	}
	t.Cleanup(func() { _ = g.closeStore() })
	tree, err := g.plannedTree()
	if err != nil {
		t.Fatal(err)
	}
	declared, err := planVerificationBatch(repo, g.planRef)
	if err != nil || declared == nil {
		t.Fatalf("load declared batch: %v", err)
	}
	return g, declared, tree
}
