package plan

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
	"overgo/internal/worklease"
)

func batchPlanFixture() Plan {
	return Plan{Items: []Item{{ID: "audio", Status: StatusOpen, Steps: []Step{{
		ID: "dataset", Title: "Materialize audio", Status: StatusOpen,
		Verify: "go test ./internal/dataset -run '^TestIntegration$' -count=1 -v",
		VerificationBatch: &VerificationBatch{
			Scope:      []string{"internal/dataset", "internal/evaluation"},
			Rationale:  "Share cumulative verification across source traversal and its consumer.",
			ReopenWhen: "Ownership or the declared acceptance changes.",
			Checkpoints: []VerificationCheckpoint{
				{ID: "traversal", Title: "Traverse source rows", Verify: "go test ./internal/dataset -run '^TestTraversal$' -count=1 -v"},
				{ID: "consumer", Title: "Migrate evaluation", Verify: "go test ./internal/evaluation -run '^TestConsumer$' -count=1 -v", DependsOn: []string{"traversal"}},
			},
		},
	}}}}}
}

func TestVerificationBatchDeclaration(t *testing.T) {
	document := batchPlanFixture()
	data, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := Parse(data)
	if err != nil || !reflect.DeepEqual(parsed, document) {
		t.Fatalf("declaration round trip = %+v, %v", parsed, err)
	}
	for _, test := range []struct {
		name string
		edit func(*VerificationBatch)
	}{
		{"empty scope", func(b *VerificationBatch) { b.Scope = nil }},
		{"empty members", func(b *VerificationBatch) { b.Checkpoints = nil }},
		{"missing rationale", func(b *VerificationBatch) { b.Rationale = "" }},
		{"missing reopen", func(b *VerificationBatch) { b.ReopenWhen = "" }},
		{"duplicate id", func(b *VerificationBatch) { b.Checkpoints[1].ID = b.Checkpoints[0].ID }},
		{"cross row dependency", func(b *VerificationBatch) { b.Checkpoints[1].DependsOn = []string{"audio/traversal"} }},
		{"unknown dependency", func(b *VerificationBatch) { b.Checkpoints[1].DependsOn = []string{"absent"} }},
		{"self dependency", func(b *VerificationBatch) { b.Checkpoints[0].DependsOn = []string{"traversal"} }},
		{"cycle", func(b *VerificationBatch) { b.Checkpoints[0].DependsOn = []string{"consumer"} }},
		{"duplicate dependency", func(b *VerificationBatch) { b.Checkpoints[1].DependsOn = []string{"traversal", "traversal"} }},
		{"missing target", func(b *VerificationBatch) { b.Checkpoints[0].Verify = "go test ./..." }},
		{"non test", func(b *VerificationBatch) { b.Checkpoints[0].Verify = "true" }},
		{"invalid pattern", func(b *VerificationBatch) { b.Checkpoints[0].Verify = "go test ./x -run '['" }},
		{"unsafe id", func(b *VerificationBatch) { b.Checkpoints[0].ID = "../acceptance" }},
	} {
		t.Run(test.name, func(t *testing.T) {
			candidate := batchPlanFixture()
			test.edit(candidate.Items[0].Steps[0].VerificationBatch)
			if err := Validate(candidate); err == nil {
				t.Fatal("invalid batch admitted")
			}
		})
	}
	for _, scope := range []string{".", "..", "../escape", "/absolute", "C:/absolute", `internal\dataset`, "internal//dataset", "internal/dataset/", ":(glob)**", "internal/*", " internal/dataset", "internal/dataset\n"} {
		t.Run("scope "+scope, func(t *testing.T) {
			candidate := batchPlanFixture()
			candidate.Items[0].Steps[0].VerificationBatch.Scope = []string{scope}
			if err := Validate(candidate); err == nil {
				t.Fatal("nonliteral or unbounded scope admitted")
			}
		})
	}
	for _, field := range []string{`"completed":true,`, `"policy":"skip-device",`, `"checkpoint_status":"done",`} {
		candidate := strings.Replace(string(data), `"scope":`, field+`"scope":`, 1)
		if _, err := Parse([]byte(candidate)); err == nil {
			t.Fatalf("declaration accepted authority field %s", field)
		}
	}
	// Declarations cannot complete their parent or its external consumers.
	document.Items[0].Steps = append(document.Items[0].Steps, Step{
		ID: "train", Status: StatusOpen, Verify: "go test ./x -run '^TestTrain$'", DependsOn: []string{"audio/dataset"},
	})
	authority := testCompletionAuthority(t, document)
	_, step, open := Current(document, worklease.UnassignedRole, authority)
	if !open || step.ID != "dataset" {
		t.Fatalf("batch declaration advanced parent: %+v, %t", step, open)
	}
	document.Items[0].Steps[0].VerificationBatch.Checkpoints[0].Verify += " -race"
	if _, _, open := Current(document, worklease.UnassignedRole, authority); open {
		t.Fatal("changed batch reused prior completion authority")
	}
}

func TestVerificationBatchMerge(t *testing.T) {
	base, local, upstream := batchPlanFixture(), batchPlanFixture(), batchPlanFixture()
	local.Items[0].Steps[0].VerificationBatch.Rationale = "Local measured batch cost."
	upstream.Items[0].Steps[0].Title = "Updated parent title"
	merged, err := MergeDocuments(base, local, upstream)
	if err != nil {
		t.Fatal(err)
	}
	if merged.Items[0].Steps[0].Title != upstream.Items[0].Steps[0].Title ||
		!reflect.DeepEqual(merged.Items[0].Steps[0].VerificationBatch, local.Items[0].Steps[0].VerificationBatch) {
		t.Fatal("merge discarded batch beside independent parent edits")
	}
	upstream.Items[0].Steps[0].VerificationBatch.Checkpoints[0].Verify += " -race"
	if _, err := MergeDocuments(base, local, upstream); err == nil {
		t.Fatal("concurrent batch authorities silently combined")
	}
	upstream.Items[0].Steps = nil
	if _, err := MergeDocuments(base, local, upstream); err == nil {
		t.Fatal("edited batch disappeared beside unverified deletion")
	}
}

func TestVerificationBatchRequiresExactMemberEvidence(t *testing.T) {
	checkpoint := batchPlanFixture().Items[0].Steps[0].VerificationBatch.Checkpoints[0]
	evidence, err := runrecord.FormatCompletionAcceptanceEvidence(testevidence.CurrentVerifyPolicy, "audio/dataset", checkpoint.Verify)
	if err != nil {
		t.Fatal(err)
	}
	member := runrecord.GateStep{Name: checkpoint.GateCheckName(), Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, Evidence: evidence}
	gate := runrecord.GateResult{Steps: []runrecord.GateStep{member}}
	if err := requireNamedCompletionAcceptance(gate, checkpoint.GateCheckName(), "audio/dataset", checkpoint.Verify); err != nil {
		t.Fatal(err)
	}
	if err := requireCompletionAcceptance(gate, "audio/dataset", checkpoint.Verify); err == nil {
		t.Fatal("member evidence counted as parent completion")
	}
	for _, outcome := range []runrecord.StepOutcome{runrecord.StepSkipped, runrecord.StepInapplicable, runrecord.StepFailed, runrecord.StepCancelled} {
		gate.Steps[0].Outcome = outcome
		if err := requireNamedCompletionAcceptance(gate, checkpoint.GateCheckName(), "audio/dataset", checkpoint.Verify); err == nil {
			t.Fatalf("member credited with %s", outcome)
		}
	}
	gate.Steps = []runrecord.GateStep{member, member}
	if err := requireNamedCompletionAcceptance(gate, checkpoint.GateCheckName(), "audio/dataset", checkpoint.Verify); err == nil {
		t.Fatal("duplicate member evidence admitted")
	}
	gate.Steps = []runrecord.GateStep{member}
	if err := requireNamedCompletionAcceptance(gate, checkpoint.GateCheckName(), "audio/dataset", checkpoint.Verify+" -race"); err == nil {
		t.Fatal("old verifier evidence admitted for changed contract")
	}
}

func TestVerificationBatchHistoryRefusesParentOnlyReceipt(t *testing.T) {
	document := batchPlanFixture()
	fixture := newCompletionFixture(t, document, "audio", "dataset")
	fixture.commit(fixture.canonicalMessage(), true)
	if _, err := ResolveCompletionAuthority(t.Context(), fixture.repository, fixture.completionHash, fixture.child, fixture.store); err == nil || !strings.Contains(err.Error(), "checkpoint traversal") {
		t.Fatalf("batch completed with only parent acceptance: %v", err)
	}
}
