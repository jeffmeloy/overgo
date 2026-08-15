package main

import (
	"testing"

	"overgo/internal/runrecord"
)

func TestGateFailureCodeUsesTerminalStep(t *testing.T) {
	steps := []runrecord.GateStep{
		{Name: "scope", Outcome: runrecord.StepSucceeded},
		{Name: "claims", Outcome: runrecord.StepFailed},
	}
	if got := terminalFailureCode(steps); got != "claims" {
		t.Fatalf("failure code = %q", got)
	}
	if got := terminalFailureCode(nil); got != "gate" {
		t.Fatalf("empty failure code = %q", got)
	}
}
