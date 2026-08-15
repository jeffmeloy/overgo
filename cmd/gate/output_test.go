package main

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"overgo/internal/runrecord"
)

func TestCompactAgentOutput(t *testing.T) {
	gate := gateContext{
		start: time.Now(),
		steps: []runrecord.GateStep{
			{Name: "build", Outcome: runrecord.StepSucceeded},
			{Name: "test", Outcome: runrecord.StepSucceeded},
			{Name: "claims", Outcome: runrecord.StepSkipped},
		},
		honesty: []string{
			"code profile: production=700 files/1000000 nodes and a large routine baseline",
			"code profile delta vs HEAD: production=+0 files/-20 nodes duplicate_excess=-12",
			"test scope: 2 direct + 1 dependent packages (derived from import graph)",
			"claims skipped: no changed path appears in compatibility.json",
			"magic backlog: 3 inherited uncatalogued constants",
		},
	}
	var output bytes.Buffer
	gate.printSummary(&output, runrecord.OutcomeSucceeded, "")
	text := output.String()
	if len(text) > 1000 || !strings.Contains(text, "GATE SUCCEEDED") || !strings.Contains(text, "delta:") || !strings.Contains(text, "warning:") {
		t.Fatalf("gate output is not compact and decision-complete (%d bytes):\n%s", len(text), text)
	}
	if strings.Contains(text, "routine baseline") || strings.Contains(text, "claims skipped:") {
		t.Fatalf("gate output repeated stored routine evidence:\n%s", text)
	}
}
