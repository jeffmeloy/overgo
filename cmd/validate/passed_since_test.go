package main

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"overgo/internal/plan"
)

// TestATestsPassSinceACommitIsReadFromTheKeptPasses holds -passed-since to the passes kept per test
// (plan.RecordTestPasses): a test whose last pass is at the commit named or after it answers, one whose last pass is
// before it or that has none kept is named in the error, and a name that is not <package>.<Test> is refused.
func TestATestsPassSinceACommitIsReadFromTheKeptPasses(t *testing.T) {
	t.Parallel()
	ended := time.Date(2026, 11, 21, 9, 0, 0, 0, time.UTC)
	passes := map[string]plan.TestPass{
		plan.PassedTest("overgo/internal/inference", "TestAfter"):  {Commit: "c2", Ended: ended},
		plan.PassedTest("overgo/internal/inference", "TestAt"):     {Commit: "c1", Ended: ended},
		plan.PassedTest("overgo/internal/inference", "TestBefore"): {Commit: "c0", Ended: ended},
	}
	// The history is c0 → c1 → c2.
	order := map[string]int{"c0": 0, "c1": 1, "c2": 2}
	since := func(earlier, later string) bool { return order[earlier] <= order[later] }

	var output bytes.Buffer
	if err := testsPassedSince(&output, passes, "c1", []string{"overgo/internal/inference.TestAt", "overgo/internal/inference.TestAfter"}, since); err != nil {
		t.Fatalf("tests passed at c1 and after it: %v\n%s", err, output.String())
	}
	if !strings.Contains(output.String(), "overgo/internal/inference.TestAfter: passed at c2") {
		t.Fatalf("each test's last pass is said:\n%s", output.String())
	}

	output.Reset()
	err := testsPassedSince(&output, passes, "c1", []string{"overgo/internal/inference.TestAt", "overgo/internal/inference.TestBefore", "overgo/internal/inference.TestNever"}, since)
	if err == nil || !strings.Contains(err.Error(), "overgo/internal/inference.TestBefore, overgo/internal/inference.TestNever") || strings.Contains(err.Error(), "TestAt,") {
		t.Fatalf("a test last passed before c1 and one with no pass kept are named, and only they: %v", err)
	}
	if !strings.Contains(output.String(), "TestBefore: last passed at c0") || !strings.Contains(output.String(), "TestNever: no pass kept") {
		t.Fatalf("each test's answer is said:\n%s", output.String())
	}

	for _, named := range []string{"TestAt", "overgo/internal/inference.helper"} {
		if err := testsPassedSince(&output, passes, "c1", []string{named}, since); err == nil || !strings.Contains(err.Error(), "names no test") {
			t.Fatalf("%q is no <package>.<Test>: %v", named, err)
		}
	}
	if err := testsPassedSince(&output, passes, "c1", nil, since); err == nil || !strings.Contains(err.Error(), "usage") {
		t.Fatalf("no test named: %v", err)
	}
}
