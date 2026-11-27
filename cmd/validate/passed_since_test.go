package main

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"overgo/internal/landing"
	"overgo/internal/plan"
)

// TestASlowRunMissingARequiredTestDidNotPass holds -lanes-passed-at to every test the lanes require (slow-whole,
// SQA-517): a lane run test by test passes where each test its packages list has a pass current at the commit named, a
// listed test with no result is no pass beside the others' passes, and a result of a test no package lists any more is
// not read; a lane run whole is judged by its own result. Counted from the record's keys, one current pass had passed a
// lane missing a required result, and a deleted test's failure had kept one red.
func TestASlowRunMissingARequiredTestDidNotPass(t *testing.T) {
	t.Parallel()
	lanes := []landing.CycleLane{
		{Name: "audio"},
		{Name: "models", Each: [][]string{{"test", "./internal/inference"}, {"test", "./cmd/evaluate"}}},
	}
	listed := func(arguments, _ []string) ([]string, error) {
		return map[string][]string{"./internal/inference": {"TestA"}, "./cmd/evaluate": {"TestB", "TestC"}}[arguments[len(arguments)-1]], nil
	}
	held := map[string]bool{testKey("", "audio"): true, testKey("./internal/inference", "TestA"): true, testKey("./cmd/evaluate", "TestB"): true}
	current := func(_, _, key string) bool { return held[key] }
	var output bytes.Buffer
	if err := lanesPassedAt(&output, lanes, "c1", current, listed); err == nil || !strings.HasSuffix(err.Error(), ": models") || !strings.Contains(output.String(), "models: 2 listed tests passed and current at c1, 1 not") {
		t.Fatalf("a listed test with no result beside current passes: %v\n%s", err, output.String())
	}
	held[testKey("./cmd/evaluate", "TestC")], held[testKey("./cmd/evaluate", "TestGone")] = true, false
	output.Reset()
	if err := lanesPassedAt(&output, lanes, "c1", current, listed); err != nil {
		t.Fatalf("every listed test current, a test no package lists failed: %v\n%s", err, output.String())
	}
	held[testKey("", "audio")] = false
	output.Reset()
	if err := lanesPassedAt(&output, lanes, "c1", current, listed); err == nil || !strings.Contains(output.String(), "audio: no pass current") {
		t.Fatalf("a lane run whole with no pass current: %v\n%s", err, output.String())
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
