package testevidence

import (
	"strings"
	"testing"

	"overgo/internal/testskip"
)

// TestCompleteRunRecordsDeclaredExclusions holds a complete run to marking a
// package passed when its only skip states why it cannot apply here, with
// that test in the verdicts as a skip, and to leaving the package unpassed
// when a skip states no such reason.
func TestCompleteRunRecordsDeclaredExclusions(t *testing.T) {
	t.Parallel()
	stream := func(reason string) string {
		return packageEvent("start", "example", "", "") +
			packageEvent("run", "example", "TestApplies", "") + packageEvent("pass", "example", "TestApplies", "") +
			packageEvent("run", "example", "TestElsewhere", "") + packageEvent("output", "example", "TestElsewhere", reason) +
			packageEvent("skip", "example", "TestElsewhere", "") + packageEvent("pass", "example", "", "")
	}
	var observed []string
	observe := func(pkg string, passed bool, tests map[string]string) error {
		observed = append(observed, pkg+" passed="+map[bool]string{true: "true", false: "false"}[passed]+" elsewhere="+tests["TestElsewhere"]+" applies="+tests["TestApplies"])
		return nil
	}
	report, err := readGoTestJSON(strings.NewReader(stream(testskip.Inapplicable+": applies to another campaign lane")), false, false, 0, observe, nil)
	if err != nil {
		t.Fatal(err)
	}
	if report.PassedPackages != 1 || report.PassedTests != 1 || len(report.ClassifiedSkipped) != 1 || len(report.Skipped) != 0 {
		t.Fatalf("declared exclusion report = %+v", report)
	}
	if len(observed) != 1 || observed[0] != "example passed=true elsewhere=skip applies=pass" {
		t.Fatalf("observed = %v", observed)
	}
	if err := RequireComplete(report); err != nil {
		t.Fatalf("a declared exclusion is not complete: %v", err)
	}
	observed = nil
	report, err = readGoTestJSON(strings.NewReader(stream("integration: needs a device")), false, false, 0, observe, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Skipped) != 1 || len(report.ClassifiedSkipped) != 0 || len(observed) != 0 {
		t.Fatalf("unowned skip report = %+v observed = %v", report, observed)
	}
	if err := RequireComplete(report); err == nil {
		t.Fatal("an unowned skip was complete")
	}
}
