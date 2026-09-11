package testevidence

import (
	"fmt"
	"slices"
	"testing"

	"overgo/internal/processcontrol"
)

// TestGoTestJSONReportNamesRefusedResourceClaims pins the typed contention:
// a test whose output carries the refused physical resource claim names its
// package once in Contended, a run failing only that way is contention
// only, and a run with any other failure or an unfinished test is not.
func TestGoTestJSONReportNamesRefusedResourceClaims(t *testing.T) {
	refused := fmt.Sprintf("{\"Action\":\"run\",\"Package\":\"x\",\"Test\":\"TestA\"}\n"+
		"{\"Action\":\"output\",\"Package\":\"x\",\"Test\":\"TestA\",\"Output\":%q}\n"+
		"{\"Action\":\"fail\",\"Package\":\"x\",\"Test\":\"TestA\"}\n"+
		"{\"Action\":\"run\",\"Package\":\"x\",\"Test\":\"TestB\"}\n"+
		"{\"Action\":\"output\",\"Package\":\"x\",\"Test\":\"TestB\",\"Output\":%q}\n"+
		"{\"Action\":\"fail\",\"Package\":\"x\",\"Test\":\"TestB\"}\n"+
		"{\"Action\":\"fail\",\"Package\":\"x\"}\n",
		"    trainer_test.go:10: processcontrol: "+processcontrol.ErrResourceBusy.Error()+": \"GPU-1\"\n",
		"    trainer_test.go:20: processcontrol: "+processcontrol.ErrResourceBusy.Error()+": \"GPU-1\"\n",
	)
	report, err := GoTestJSONReport(refused)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(report.Contended, []string{"x"}) || !report.ContentionOnly() {
		t.Fatalf("refused claims = contended %v, contention only %t; report %+v", report.Contended, report.ContentionOnly(), report)
	}
	mixed := refused + "{\"Action\":\"run\",\"Package\":\"y\",\"Test\":\"TestC\"}\n" +
		"{\"Action\":\"output\",\"Package\":\"y\",\"Test\":\"TestC\",\"Output\":\"    other_test.go:5: wrong value\\n\"}\n" +
		"{\"Action\":\"fail\",\"Package\":\"y\",\"Test\":\"TestC\"}\n"
	report, err = GoTestJSONReport(mixed)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(report.Contended, []string{"x"}) || report.ContentionOnly() {
		t.Fatalf("a real failure beside the refusals counted as contention only: %+v", report)
	}
	unfinished := refused + "{\"Action\":\"run\",\"Package\":\"z\",\"Test\":\"TestD\"}\n"
	if report, err := GoTestJSONReport(unfinished); err != nil || report.ContentionOnly() {
		t.Fatalf("an unfinished test beside the refusals counted as contention only: %+v, %v", report, err)
	}
	if report, err := GoTestJSONReport("{\"Action\":\"run\",\"Package\":\"x\",\"Test\":\"TestA\"}\n{\"Action\":\"pass\",\"Package\":\"x\",\"Test\":\"TestA\"}\n"); err != nil || report.ContentionOnly() || len(report.Contended) != 0 {
		t.Fatalf("a passing run reported contention: %+v, %v", report, err)
	}
}
