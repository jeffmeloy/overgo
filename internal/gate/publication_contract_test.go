package gate

import (
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/dispatchreadiness"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// TestCheapAdmissionReadiness drives the cheap admission projection the gate
// resolves before any repair, replay, or analysis: it reports branch and HEAD,
// classifies a dirty input outside the declared scope while leaving one inside
// it uncounted, opens no store, and marks an undeclared scope provisional rather
// than inventing an out-of-scope refusal.
func TestCheapAdmissionReadiness(t *testing.T) {
	repo := t.TempDir()
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "gate@test")
	runGitFixture(t, repo, "config", "user.name", "gate")
	if err := os.WriteFile(filepath.Join(repo, "seed.go"), []byte("package seed\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "-A")
	runGitFixture(t, repo, "commit", "-q", "-m", "baseline")
	for _, path := range []string{"in/changed.go", "out/stray.go"} {
		if err := os.MkdirAll(filepath.Join(repo, filepath.Dir(path)), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(repo, path), []byte("package p\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	readiness, err := dispatchreadiness.Resolve(repo, []string{"in"})
	if err != nil {
		t.Fatal(err)
	}
	if readiness.Branch == "" || readiness.Head == "" {
		t.Fatalf("cheap facts missing: %+v", readiness)
	}
	if readiness.Provisional {
		t.Fatal("a declared scope was reported provisional")
	}
	if !slices.Contains(readiness.OutOfScope, "out/stray.go") || slices.Contains(readiness.OutOfScope, "in/changed.go") {
		t.Fatalf("dirty scope classification wrong: %+v", readiness.OutOfScope)
	}

	provisional, err := dispatchreadiness.Resolve(repo, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !provisional.Provisional || len(provisional.OutOfScope) != 0 {
		t.Fatalf("undeclared scope must be provisional without an out-of-scope refusal: %+v", provisional)
	}
}

// Explicit campaign acceptance reads retained records; ordinary package tests
// exercise accounting without depending on an operator's evidence store.
var retainedPublicationCost = flag.Bool("retained-publication-cost", false, "verify frozen publication attempts in the canonical data-root store")

func TestPlanOnlyPublicationReusesUnchangedPhases(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	t.Run("restart and live admission", testModernCensusRestartAndLiveAdmission)
	t.Run("missing and unproven results", testModernCensusMissingAndUnprovenResults)
	t.Run("source and build invalidation", testModernCensusSourceAndBuildInvalidation)
}

func TestPlanPublicationCostBreakdown(t *testing.T) {
	t.Parallel()
	t.Run("overlapping work is not elapsed wall", func(t *testing.T) {
		steps := costSteps(runrecord.StepSucceeded, runrecord.StepSucceeded, 5*uint64(time.Second), 3*uint64(time.Second))
		g := &gateContext{planRef: "publication/cost"}
		store, err := overgodb.Open(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		g.batchCostAudit(t.Context(), store, steps, uint64(time.Second))
		cost := batchCostOf(steps)
		if cost.StepNS != 8*uint64(time.Second)+16 || cost.ExecutedNS != 8*uint64(time.Second)+7 {
			t.Fatalf("overlapping work lost: %+v", cost)
		}
		if len(auditLines(g.audit)) != 1 || !strings.Contains(auditLines(g.audit)[0], "total_wall=1s summed_step_time=8.000000016s") {
			t.Fatalf("audit = %v", auditLines(g.audit))
		}
		t.Log(auditLines(g.audit)[0])
	})
	t.Run("failed and reused work stay separate", func(t *testing.T) {
		cost := batchCostOf(costSteps(runrecord.StepFailed, runrecord.StepReused, 5*uint64(time.Second), 1))
		if cost.Failed != 1 || cost.FailedNS != 5*uint64(time.Second) || cost.Reused != 1 || cost.Accepted != 2 || cost.ExecutedNS != 7 {
			t.Fatalf("failure or reuse counted as fresh execution: %+v", cost)
		}
		if saved, unknown := reuseSavings(nil, cost); saved != 0 || len(unknown) != 1 || unknown[0] != "acceptance-consumer" {
			t.Fatalf("unmeasured reuse invented savings: %d %v", saved, unknown)
		}
	})
	if *retainedPublicationCost {
		t.Run("retained publication attempts", verifyPublicationAttempts)
	}
}

func verifyPublicationAttempts(t *testing.T) {
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	// Only identities and independent expected measurements are retained here;
	// source snapshots and complete results remain in their original store.
	for _, expected := range []struct {
		attempt, result, commit, planRef string
		wallNS, modernGoNS               uint64
	}{
		{"evidence:sha256:5d442c10f3c19a125e6afc67bc11d8393589d22721b70912b77b50078960a406", "evidence:sha256:b430b034e4c2369cda371324bcdb4d1cbda9126d2fd711033db574726a0d0bf9", "7367a3f86bfacad45349d97b3076b852d1d1cd54", "documentation-cost-publication/do", 86189940200, 17900956400},
		{"evidence:sha256:92916fd3376eac8b43ea38013dbf9b5441b3fbedf8eb4a08c273b35db410aa42", "evidence:sha256:7d2aad4caee87b8908c57c6f375f9abf780ce971cda8bd0ea8c1b150b63e1beb", "5564cad105beddb927947ab15cdcc174956fda90", "testing-reduction-replan/do", 107224226600, 24733207500},
		{"evidence:sha256:f64ad341bdb1521937002235ce63c393478a979ef933fc5e63fdc2884ccff0fe", "evidence:sha256:9bcdca177846ae78313dbcdacc050efd6630c81485fcfa2ea9c6f82a3003a142", "6ca71474f53f4861648d152aa42f0d7f7a9d5e00", "testing-reduction-split/do", 97506357900, 29200303100},
	} {
		t.Run(expected.planRef, func(t *testing.T) {
			id, err := artifact.ParseID(expected.attempt)
			if err != nil {
				t.Fatal(err)
			}
			attempt, err := runrecord.RequireAttemptRecord(t.Context(), store, id)
			if err != nil {
				t.Fatal(err)
			}
			if attempt.ID != id || attempt.Result.String() != expected.result || attempt.CodeCommit != expected.commit || attempt.PlanItem+"/"+attempt.PlanStep != expected.planRef || attempt.Outcome != runrecord.OutcomeSucceeded || attempt.WallNS != expected.wallNS {
				t.Fatalf("wrong publication attempt: %+v", attempt)
			}
			result, err := runrecord.RequireGateResult(t.Context(), store, attempt.Result)
			if err != nil {
				t.Fatal(err)
			}
			if result.Recipe != attempt.Recipe || result.CodeCommit != attempt.CodeCommit || result.Outcome != attempt.Outcome {
				t.Fatalf("attempt/result binding mismatch: %+v", result)
			}
			var modernGoNS uint64
			outcomes := map[runrecord.StepOutcome]int{}
			for _, phase := range result.Steps {
				outcomes[phase.Outcome]++
				if phase.Name == "modern-go" && phase.Outcome == runrecord.StepSucceeded {
					modernGoNS = phase.DurationNS
				}
			}
			if modernGoNS != expected.modernGoNS {
				t.Fatalf("modern-Go phase = %d, want retained %d", modernGoNS, expected.modernGoNS)
			}
			report := struct {
				Attempt    string                        `json:"attempt"`
				Result     string                        `json:"result"`
				WallNS     uint64                        `json:"attempt_wall_ns"`
				Selection  runrecord.AttemptSelection    `json:"selection"`
				Outcomes   map[runrecord.StepOutcome]int `json:"recorded_phase_outcomes"`
				Work       batchCost                     `json:"summed_phase_work"`
				ModernGoNS uint64                        `json:"modern_go_phase_ns"`
				Unknown    []string                      `json:"unknown"`
			}{expected.attempt, expected.result, attempt.WallNS, attempt.Selection, outcomes, batchCostOf(result.Steps), modernGoNS,
				[]string{"phase interval union", "process wall", "compilation cost", "unattributed residual", "census-only cost", "package execution denominators"}}
			encoded, err := json.Marshal(report)
			if err != nil {
				t.Fatal(err)
			}
			t.Log(string(encoded))
		})
	}
}
