package main

import (
	"strings"
	"testing"

	"overgo/internal/plan"
)

func TestRunVerifyRejectsSkippedGoTest(t *testing.T) {
	item := plan.Item{ID: "gate"}
	step := plan.Step{
		ID:     "skip",
		Verify: "go test ../../internal/testevidence/testdata/skipfixture -count=1 -v",
	}
	err := runVerify(item, step)
	if err == nil {
		t.Fatal("skipped go test passed plan verification")
	}
	if !strings.Contains(err.Error(), "VACUOUS") {
		t.Fatalf("runVerify error = %v, want vacuous evidence refusal", err)
	}
}
