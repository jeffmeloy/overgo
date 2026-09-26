package testevidence

import (
	"strings"
	"testing"
)

// TestVerifyReadsTypedUnavailability holds unavailability to go test's typed
// actions: a test whose prerequisite is absent skips or fails, and that action
// alone withholds credit. No output text decides -- a passing test is credited
// whatever it prints, and a skipped one is not, whatever it prints.
func TestVerifyReadsTypedUnavailability(t *testing.T) {
	t.Parallel()
	stream := func(action, output string) string {
		return packageEvent("start", "example", "", "") + packageEvent("run", "example", "TestFixture", "") +
			packageEvent("output", "example", "TestFixture", output) +
			packageEvent(action, "example", "TestFixture", "") + packageEvent(action, "example", "", "")
	}
	for _, tc := range []struct {
		action, output string
		credited       bool
	}{
		{"pass", "UNAVAILABLE: parity NOT verified\n", true},
		{"skip", "fixture present\n", false},
		{"fail", "fixture present\n", false},
	} {
		report, err := readGoTestJSON(strings.NewReader(stream(tc.action, tc.output)), false, false, 0, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		if credited := RequireComplete(report) == nil && report.PackagePassed("example"); credited != tc.credited {
			t.Fatalf("%s printing %q credited=%t, want %t: %+v", tc.action, tc.output, credited, tc.credited, report)
		}
	}
}
