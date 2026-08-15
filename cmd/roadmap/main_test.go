package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/plan"
)

func TestRoadmapDAGAndReadiness(t *testing.T) {
	path := filepath.Join("..", "..", "docs", "rsi_plan.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	evidence := plan.RoadmapEvidence{
		Live:     map[string]bool{"roadmap-validator": true},
		Landed:   map[string]bool{"go-hygiene-policy-baseline": true, "go-readability-gate": true},
		InFlight: map[string]bool{}, ProbeBound: map[string]string{
			"refusal-ledger": "go test ./internal/runrecord ./internal/recipe -run '^TestRefusalDecisionCarriesMeasuredReason$' -count=1 -v",
		},
	}
	report, err := plan.EvaluateRoadmap(data, evidence)
	if err != nil {
		t.Fatal(err)
	}
	if report.States["roadmap-validator"] != "open" || report.States["refusal-ledger"] != "ready" || len(report.Proposed) != 1 {
		t.Fatalf("derived roadmap state = %+v; proposals = %+v", report.States, report.Proposed)
	}
	encoded, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), `"disposition":"PROPOSED"`) || !strings.Contains(string(encoded), `"status":"open"`) {
		t.Fatalf("proposal is not an advisory live-plan insertion: %s", encoded)
	}
	staleProbe := evidence
	staleProbe.ProbeBound = map[string]string{"refusal-ledger": "go test ./internal/runrecord -run '^TestOther$' -count=1"}
	staleReport, err := plan.EvaluateRoadmap(data, staleProbe)
	if err != nil || staleReport.States["refusal-ledger"] != "new" {
		t.Fatalf("stale verifier probe changed readiness: state=%q err=%v", staleReport.States["refusal-ledger"], err)
	}

	tests := []struct {
		name, row string
		mutate    func(map[string]any)
		want      string
	}{
		{"cycle", "go-hygiene-policy-baseline", func(row map[string]any) { row["depends_on"] = []any{"go-readability-gate"} }, "cycle"},
		{"unknown dependency", "roadmap-validator", func(row map[string]any) { row["depends_on"] = []any{"absent"} }, "unknown"},
		{"missing safety reachability", "composition-viability", func(row map[string]any) {
			dependencies := row["depends_on"].([]any)
			row["depends_on"] = slices.DeleteFunc(dependencies, func(value any) bool { return value == "evaluation-isolation" })
		}, "mandatory dependency"},
		{"invalid verifier", "roadmap-validator", func(row map[string]any) { row["verify"] = "go test ./cmd/roadmap" }, "-run"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			changed := mutateRoadmap(t, data, test.row, test.mutate)
			if _, err := plan.EvaluateRoadmap(changed, evidence); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
		})
	}

	t.Run("stored stage mismatch", func(t *testing.T) {
		var document map[string]any
		if err := json.Unmarshal(data, &document); err != nil {
			t.Fatal(err)
		}
		execution := document["execution_order"].(map[string]any)
		execution["stage_1"] = execution["stage_1"].([]any)[1:]
		changed, _ := json.Marshal(document)
		if _, err := plan.EvaluateRoadmap(changed, evidence); err == nil || !strings.Contains(err.Error(), "omit") {
			t.Fatalf("stage mismatch error = %v", err)
		}
	})

	t.Run("live dependency must be landed", func(t *testing.T) {
		unsafe := evidence
		unsafe.Live = map[string]bool{"go-readability-gate": true}
		unsafe.Landed = map[string]bool{}
		if _, err := plan.EvaluateRoadmap(data, unsafe); err == nil || !strings.Contains(err.Error(), "unsatisfied dependencies") {
			t.Fatalf("live safety error = %v", err)
		}
	})
}

func mutateRoadmap(t *testing.T, data []byte, id string, mutate func(map[string]any)) []byte {
	t.Helper()
	var document map[string]any
	if err := json.Unmarshal(data, &document); err != nil {
		t.Fatal(err)
	}
	for _, item := range document["items"].([]any) {
		for _, step := range item.(map[string]any)["steps"].([]any) {
			row := step.(map[string]any)
			if row["id"] == id {
				mutate(row)
				changed, err := json.Marshal(document)
				if err != nil {
					t.Fatal(err)
				}
				return changed
			}
		}
	}
	t.Fatalf("row %s not found", id)
	return nil
}
