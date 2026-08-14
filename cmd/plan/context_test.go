package main

import (
	"bytes"
	"encoding/json"
	"testing"

	"overgo/internal/plan"
)

func TestAutomationContextCommandEncoding(t *testing.T) {
	document := plan.Plan{Items: []plan.Item{{
		ID: "automation", Title: "Automation", Status: "open",
		Steps: []plan.Step{{ID: "context", Title: "Context", Status: "open", Verify: "go test ./..."}},
	}}}
	facts := plan.ContextFacts{
		Head: "0123456789abcdef0123456789abcdef01234567", Branch: "codex/automation",
		Worktree: "C:/repo", Role: "sqa",
		EvidenceDebt: plan.EvidenceDebt{State: "possible", Source: "bin/gate_status.json", Reason: "fixture"},
	}
	context, err := plan.BuildAutomationContext(document, facts)
	if err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := json.NewEncoder(&output).Encode(context); err != nil {
		t.Fatal(err)
	}
	var decoded plan.AutomationContext
	if err := json.Unmarshal(output.Bytes(), &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Role != "sqa" || decoded.CurrentTask == nil || decoded.CurrentTask.ItemID != "automation" || decoded.CurrentTask.StepID != "context" {
		t.Fatalf("encoded context = %+v", decoded)
	}
}
