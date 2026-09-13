package main

import (
	"bytes"
	"encoding/json"
	"os"
	"overgo/internal/plan"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestRetainedGatePhaseReport(t *testing.T) {
	t.Run("missing or contradictory result", testRetainedGateResultIntegrity)
	root := t.TempDir()
	t.Setenv(dataroot.Env, root)
	repository := filepath.Join(root, "overgodb-store")
	store, err := overgodb.Open(repository)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	commit := strings.Repeat("ab", 20)
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "report-environment")
	recipe := testutil.ArtifactID(t, artifact.KindRecipe, "report-recipe")
	put := func(content artifact.Content) {
		t.Helper()
		_, err := store.Commit(t.Context(), artifact.Batch{Key: "report/" + content.Descriptor.ID.String(), Artifacts: []artifact.Descriptor{content.Descriptor}, Contents: []artifact.Content{content}})
		if err != nil {
			t.Fatal(err)
		}
	}
	var results []artifact.ID
	for _, fixture := range []struct {
		outcome runrecord.Outcome
		step    runrecord.StepOutcome
		commit  string
	}{
		{runrecord.OutcomeFailed, runrecord.StepFailed, commit},
		{runrecord.OutcomeSucceeded, runrecord.StepSucceeded, commit},
		{runrecord.OutcomeCancelled, runrecord.StepCancelled, commit},
		{runrecord.OutcomeSucceeded, runrecord.StepSucceeded, strings.Repeat("cd", 20)},
	} {
		failure := ""
		if fixture.outcome == runrecord.OutcomeFailed {
			failure = "unit"
		}
		record, err := runrecord.NewGateRecord(recipe, environment, fixture.commit, fixture.outcome, failure, 10, []runrecord.GateStep{
			{Name: "unit", Phase: runrecord.PhaseTest, Outcome: fixture.step, DurationNS: 8},
			{Name: "owner", Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: 7},
			{Name: "retained", Phase: runrecord.PhaseTest, Outcome: runrecord.StepReused, DurationNS: 1},
			{Name: "unstarted", Phase: runrecord.PhaseTest, Outcome: runrecord.StepSkipped},
		})
		if err != nil {
			t.Fatal(err)
		}
		content, err := record.Result.Content()
		if err != nil {
			t.Fatal(err)
		}
		put(content)
		results = append(results, record.Result.ID)
		preparation, err := runrecord.NewGatePreparation(strings.TrimPrefix(record.Result.ID.String(), "evidence:sha256:"), environment, time.Unix(1, 0))
		if err != nil {
			t.Fatal(err)
		}
		content, err = preparation.Content()
		if err != nil {
			t.Fatal(err)
		}
		put(content)
		finalization, err := runrecord.NewGateFinalization(preparation, fixture.commit, record.Result.ID, fixture.outcome)
		if err != nil {
			t.Fatal(err)
		}
		content, err = finalization.Content()
		if err != nil {
			t.Fatal(err)
		}
		put(content)
		// Interrupted gates can lack an attempt: the attempt schema admits only
		// success or failure. The result must still remain visible.
		if fixture.outcome == runrecord.OutcomeCancelled {
			continue
		}
		attempt, err := runrecord.NewAttemptRecord(runrecord.AttemptRecord{
			PlanItem: "report", PlanStep: "do", Recipe: recipe, Result: record.Result.ID, CodeCommit: fixture.commit,
			Outcome: fixture.outcome, Failure: failure, WallNS: 10,
			Selection: runrecord.AttemptSelection{Defined: 5, Selected: 4, Excluded: 1, Uncertainty: 1, CacheEligible: 2, CacheHits: 1, PlanningNS: 3},
		})
		if err != nil {
			t.Fatal(err)
		}
		content, err = attempt.Content()
		if err != nil {
			t.Fatal(err)
		}
		put(content)
	}
	preparation, err := runrecord.NewGatePreparation(strings.Repeat("a", 64), environment, time.Unix(1, 0))
	if err != nil {
		t.Fatal(err)
	}
	content, err := preparation.Content()
	if err != nil {
		t.Fatal(err)
	}
	put(content)
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	reader, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	report, err := loadGatePhaseHistory(t.Context(), reader, cli{history: "all", historyCommit: commit})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Gates) != 3 || len(report.Unfinalized) != 1 {
		t.Fatalf("retries or unfinished preparation lost: %+v", report)
	}
	for _, gate := range report.Gates {
		if gate.Result.CodeCommit != commit || gate.SummedStepNS != 16 || gate.WaitingNS != nil || gate.UnattributedNS != nil || len(gate.Finalizations) != 1 {
			t.Fatalf("filter or overlap fabricated measurement: %+v", gate)
		}
		if gate.Outcomes[runrecord.StepReused] != 1 || gate.Outcomes[runrecord.StepSkipped] != 1 {
			t.Fatalf("check dispositions lost: %+v", gate.Outcomes)
		}
	}
	selected, err := loadGatePhaseHistory(t.Context(), reader, cli{history: "report", historyResult: results[1].String()})
	if err != nil || len(selected.Gates) != 1 || selected.Gates[0].Result.Outcome != runrecord.OutcomeSucceeded {
		t.Fatalf("exact retry = %+v, %v", selected, err)
	}
	missing, err := loadGatePhaseHistory(t.Context(), reader, cli{history: "all", historyCommit: commit[:8]})
	if err != nil || len(missing.Gates) != 0 {
		t.Fatalf("prefix matched exact commit: %+v, %v", missing, err)
	}
	var output bytes.Buffer
	if err := writeGatePhaseHistory(&output, report); err != nil {
		t.Fatal(err)
	}
	for _, text := range []string{"outer_wall=10ns", "summed_step_work=16ns", "outcome=cancelled", "outer wall unavailable", "unmeasured", "commit=unbound", "defined=5 selected=4 excluded=1 uncertain=1 cache_hits=1/2 planning=3ns"} {
		if !strings.Contains(output.String(), text) {
			t.Fatalf("missing %q in %s", text, output.String())
		}
	}
	output.Reset()
	if err := printGatePhaseHistory(cli{history: "all", historyCommit: commit, json: true}, &output); err != nil {
		t.Fatal(err)
	}
	encoded := output.Bytes()
	var decoded gatePhaseReport
	if err := json.Unmarshal(encoded, &decoded); err != nil || len(decoded.Gates) != len(report.Gates) {
		t.Fatalf("JSON command projection: %+v, %v", decoded, err)
	}
	if !bytes.Contains(encoded, []byte(`"waiting_ns":null`)) || !bytes.Contains(encoded, []byte(`"attempt_id"`)) {
		t.Fatalf("typed absence or identity lost: %s", encoded)
	}
	if _, err := loadGatePhaseHistory(t.Context(), reader, cli{history: "all", historyResult: "not-an-id"}); err == nil {
		t.Fatal("invalid result selector accepted")
	}
	// Historical diagnosis remains usable without a live plan document.
	t.Chdir(root)
	text := captureStdout(t, func() {
		if err := run(cli{history: "all", phases: true, historyResult: results[1].String(), retireLegacyLeases: noLegacyLeaseRetirement}, nil); err != nil {
			t.Fatal(err)
		}
	})
	if !strings.Contains(text, results[1].String()) {
		t.Fatal("command did not project the selected retained result")
	}
	for _, invalid := range []cli{
		{history: "all", phases: true, add: true, retireLegacyLeases: noLegacyLeaseRetirement},
		{history: "all", phases: true, prepareMerge: "master", retireLegacyLeases: noLegacyLeaseRetirement},
		{history: "all", phases: true, verify: true, retireLegacyLeases: noLegacyLeaseRetirement},
		{history: "all", phases: true, retireLegacyLeases: 0},
	} {
		if err := run(invalid, nil); err == nil || !strings.Contains(err.Error(), "incompatible") {
			t.Fatalf("history accepted a mixed operation: %v", err)
		}
	}
}

func testRetainedGateResultIntegrity(t *testing.T) {
	for _, name := range []string{"missing result", "contradictory result", "contradictory finalization"} {
		t.Run(name, func(t *testing.T) {
			store, err := overgodb.Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			commit := strings.Repeat("ab", 20)
			recipe := testutil.ArtifactID(t, artifact.KindRecipe, "integrity-recipe")
			environment := testutil.ArtifactID(t, artifact.KindEvidence, "integrity-environment")
			result, err := runrecord.NewGateRecord(recipe, environment, commit, runrecord.OutcomeSucceeded, "", 10,
				[]runrecord.GateStep{{Name: "unit", Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: 10}})
			if err != nil {
				t.Fatal(err)
			}
			outcome, failure := runrecord.OutcomeFailed, "unit"
			if name == "contradictory finalization" {
				outcome, failure = runrecord.OutcomeSucceeded, ""
			}
			attempt, err := runrecord.NewAttemptRecord(runrecord.AttemptRecord{
				PlanItem: "report", PlanStep: "do", Recipe: recipe, Result: result.Result.ID,
				CodeCommit: commit, Outcome: outcome, Failure: failure, WallNS: 10,
			})
			if err != nil {
				t.Fatal(err)
			}
			content, err := attempt.Content()
			if err != nil {
				t.Fatal(err)
			}
			batch := artifact.Batch{Key: "report/integrity", Artifacts: []artifact.Descriptor{content.Descriptor}, Contents: []artifact.Content{content}}
			if name != "missing result" {
				content, err = result.Result.Content()
				if err != nil {
					t.Fatal(err)
				}
				batch.Artifacts = append(batch.Artifacts, content.Descriptor)
				batch.Contents = append(batch.Contents, content)
			}
			if name == "contradictory finalization" {
				preparation, err := runrecord.NewGatePreparation(strings.Repeat("a", 64), environment, time.Unix(1, 0))
				if err != nil {
					t.Fatal(err)
				}
				finalization, err := runrecord.NewGateFinalization(preparation, strings.Repeat("cd", 20), result.Result.ID, outcome)
				if err != nil {
					t.Fatal(err)
				}
				for _, lifecycle := range []runrecord.GateLifecycle{preparation, finalization} {
					content, err := lifecycle.Content()
					if err != nil {
						t.Fatal(err)
					}
					batch.Artifacts = append(batch.Artifacts, content.Descriptor)
					batch.Contents = append(batch.Contents, content)
				}
			}
			if _, err := store.Commit(t.Context(), batch); err != nil {
				t.Fatal(err)
			}
			_, err = loadGatePhaseHistory(t.Context(), store, cli{history: "all"})
			if err == nil || !strings.Contains(err.Error(), result.Result.ID.String()) {
				t.Fatalf("unbound result accepted: %v", err)
			}
		})
	}
}

func TestPlanEditCommand(t *testing.T) {
	t.Setenv(plan.AutomationRoleEnvironment, plan.UnassignedRole)
	t.Setenv(plan.AutomationWorkerEnvironment, "")
	document := mutationPlan(t, "Existing work")
	root := initializePlanTestRepository(t, document)
	t.Chdir(root)
	editPath := filepath.Join(root, "edit.json")
	writeEdit := func(edit plan.StepEdit) {
		t.Helper()
		data, err := json.Marshal(edit)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(editPath, data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	readPlan := func() []byte {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(root, plan.Path))
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	edit := plan.StepEdit{ExpectedPlan: document.Digest(), Item: "row", Create: true, Step: plan.Step{ID: "second", Title: "Review second", Status: plan.StatusOpen, Verify: "go test ./internal/plan -run '^TestAtomicStepEdit$'", DependsOn: []string{"row/do"}, Rationale: "Keep an explicit prerequisite."}}
	writeEdit(edit)
	output := captureStdout(t, func() {
		if err := run(cli{edit: editPath, json: true, retireLegacyLeases: noLegacyLeaseRetirement}, nil); err != nil {
			t.Fatal(err)
		}
	})
	var result struct {
		Before, After       string
		Created             bool
		AcceptanceChanged   bool `json:"acceptance_changed"`
		PublicationRequired bool `json:"publication_required"`
	}
	if err := json.Unmarshal([]byte(output), &result); err != nil {
		t.Fatal(err)
	}
	updated, err := plan.Load(filepath.Join(root, plan.Path))
	if err != nil {
		t.Fatal(err)
	}
	if len(updated.Items[0].Steps) != 2 || result.Before != document.Digest() || result.After != updated.Digest() || !result.Created || !result.AcceptanceChanged || !result.PublicationRequired {
		t.Fatalf("edit result %s", output)
	}
	// The fixture has no Go module. Successful editing proves the verifier did not execute.
	before := readPlan()
	if err := editPlanStep(root, editPath, &bytes.Buffer{}); err == nil {
		t.Fatal("stale edit accepted")
	}
	if !bytes.Equal(before, readPlan()) {
		t.Fatal("stale edit changed original bytes")
	}
	contextOutput := captureStdout(t, func() {
		if err := run(cli{context: true, retireLegacyLeases: noLegacyLeaseRetirement}, nil); err != nil {
			t.Fatal(err)
		}
	})
	var projected plan.AutomationContext
	if err := json.Unmarshal([]byte(contextOutput), &projected); err != nil || projected.PlanDigest != updated.Digest() {
		t.Fatalf("context identity: %s %v", contextOutput, err)
	}
	edit = plan.StepEdit{ExpectedPlan: updated.Digest(), Item: "row", Step: updated.Items[0].Steps[1]}
	edit.Step.Title = "Revised contract"
	writeEdit(edit)
	if err := editPlanStep(root, editPath, &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	updated, err = plan.Load(filepath.Join(root, plan.Path))
	if err != nil {
		t.Fatal(err)
	}
	for name, change := range map[string]func(*plan.StepEdit){
		"unproven dependency": func(e *plan.StepEdit) { e.Step.DependsOn = []string{"missing/do"} },
		"completion":          func(e *plan.StepEdit) { e.Step.Status = plan.StatusDone },
		"batch":               func(e *plan.StepEdit) { e.Step.VerificationBatch = &plan.VerificationBatch{} },
	} {
		t.Run(name, func(t *testing.T) {
			bad := edit
			bad.ExpectedPlan = updated.Digest()
			change(&bad)
			writeEdit(bad)
			before := readPlan()
			if err := editPlanStep(root, editPath, &bytes.Buffer{}); err == nil {
				t.Fatal("invalid edit accepted")
			}
			if !bytes.Equal(before, readPlan()) {
				t.Fatal("refusal changed bytes")
			}
		})
	}
	if err := os.WriteFile(editPath, []byte(`{"unknown_field":true}`), 0600); err != nil {
		t.Fatal(err)
	}
	if err := editPlanStep(root, editPath, &bytes.Buffer{}); err == nil {
		t.Fatal("unknown schema accepted")
	}
	if err := run(cli{edit: editPath, next: true, retireLegacyLeases: noLegacyLeaseRetirement}, nil); err == nil {
		t.Fatal("mixed edit and dispatch accepted")
	}
	claimed, err := plan.ResolveDispatch(t.Context(), root, plan.DispatchRequest{Acquire: true, Worker: "other-worker", Reference: "row/do"})
	if err != nil || claimed.Claim == nil {
		t.Fatalf("claim: %+v %v", claimed, err)
	}
	edit = plan.StepEdit{ExpectedPlan: updated.Digest(), Item: "row", Step: updated.Items[0].Steps[0]}
	edit.Step.Title = "Cannot change active work"
	writeEdit(edit)
	before = readPlan()
	if err := editPlanStep(root, editPath, &bytes.Buffer{}); err == nil || !strings.Contains(err.Error(), "claimed by worker other-worker") {
		t.Fatalf("active claim not protected: %v", err)
	}
	if !bytes.Equal(before, readPlan()) {
		t.Fatal("claimed edit changed bytes")
	}
}
