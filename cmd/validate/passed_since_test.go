package main

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"overgo/internal/landing"
	"overgo/internal/plan"
)

// TestOneWholeSlowRunIsReadFromTheLanesRecords holds -lanes-passed-at to the results the lanes keep (slow-whole,
// lanes-record): it answers that every lane run whole has a pass current at the commit named, and every kept test of a
// lane run test by test is one; a test that is not, and a lane run whole that is not, are named, and a lane with no
// result kept is no pass.
func TestOneWholeSlowRunIsReadFromTheLanesRecords(t *testing.T) {
	t.Parallel()
	lanes := []landing.CycleLane{
		{Name: "audio"},
		{Name: "models", Each: [][]string{{"test", "./internal/inference"}, {"test", "./cmd/evaluate"}}},
	}
	inference, evaluate := testKey("./internal/inference", "TestA"), testKey("./cmd/evaluate", "TestB")
	results := map[string]plan.TestResult{testKey("", "audio"): {Commit: "c1"}, inference: {Commit: "c1"}, evaluate: {Commit: "c0"}}
	held := map[string]bool{testKey("", "audio"): true, inference: true, evaluate: true}
	current := func(_, _, key string) bool { return held[key] }
	var output bytes.Buffer
	if err := lanesPassedAt(&output, results, lanes, "c1", current); err != nil || !strings.Contains(output.String(), "models: 2 tests kept passed and current at c1, 0 not") {
		t.Fatalf("every result current at c1: %v\n%s", err, output.String())
	}
	held[testKey("", "audio")], held[evaluate] = false, false
	output.Reset()
	if err := lanesPassedAt(&output, results, lanes, "c1", current); err == nil || !strings.HasSuffix(err.Error(), ": audio, models") || !strings.Contains(output.String(), "1 not") {
		t.Fatalf("a lane run whole and a test that are not current are named: %v\n%s", err, output.String())
	}
	output.Reset()
	if err := lanesPassedAt(&output, nil, lanes, "c1", func(string, string, string) bool { return false }); err == nil || !strings.Contains(output.String(), "audio: no pass current") {
		t.Fatalf("lanes with no result kept: %v\n%s", err, output.String())
	}
}

// TestATestsPassSinceACommitIsReadFromTheKeptPasses holds -passed-since to the results kept per test
// (plan.RecordTestResults): a test whose last pass is at the commit named or after it answers, one whose last pass is
// before it, whose last result failed, or that has none kept is named in the error, and a name that is not
// <package>.<Test> is refused.
func TestATestsPassSinceACommitIsReadFromTheKeptPasses(t *testing.T) {
	t.Parallel()
	ended := time.Date(2026, 11, 21, 9, 0, 0, 0, time.UTC)
	results := map[string]plan.TestResult{
		plan.PassedTest("overgo/internal/inference", "TestAfter"):  {Commit: "c2", Ended: ended},
		plan.PassedTest("overgo/internal/inference", "TestAt"):     {Commit: "c1", Ended: ended},
		plan.PassedTest("overgo/internal/inference", "TestBefore"): {Commit: "c0", Ended: ended},
		plan.PassedTest("overgo/internal/inference", "TestBroke"):  {Commit: "c2", Failed: true, Ended: ended},
	}
	// The history is c0 → c1 → c2.
	since := func(earlier, later string) bool { return earlier <= later }
	var output bytes.Buffer
	if err := testsPassedSince(&output, results, "c1", []string{"overgo/internal/inference.TestAfter", "overgo/internal/inference.TestAt"}, since); err != nil {
		t.Fatalf("tests that passed at and after the commit: %v\n%s", err, output.String())
	}
	output.Reset()
	err := testsPassedSince(&output, results, "c1",
		[]string{"overgo/internal/inference.TestBefore", "overgo/internal/inference.TestMissing", "overgo/internal/inference.TestBroke"}, since)
	if err == nil || !strings.HasSuffix(err.Error(), "TestBefore, overgo/internal/inference.TestMissing, overgo/internal/inference.TestBroke") {
		t.Fatalf("a pass before the commit, a test with none kept and a failure are named: %v\n%s", err, output.String())
	}
	if err := testsPassedSince(&output, results, "c1", []string{"TestAt"}, since); err == nil {
		t.Fatal("a name that is not <package>.<Test> was admitted")
	}
}
