package main

import (
	"strings"
	"testing"

	"overgo/internal/runrecord"
)

// TestHistoryReportsShadowIsolation holds the phase history to reporting the
// isolation measure a gate recorded: how many of the packages it ran the
// rule under trial would have left out, and what those cost, summed from the
// packages the selection record marks and from nothing else.
func TestHistoryReportsShadowIsolation(t *testing.T) {
	t.Parallel()
	histogram := runrecord.SelectionHistogram{Packages: []runrecord.SelectionPackageCauses{
		{Package: "overgo/internal/confined", Step: "test", ShadowIsolated: true, ElapsedSeconds: new(1.5)},
		{Package: "overgo/internal/other", Step: "test", ShadowIsolated: true, ElapsedSeconds: new(2.5)},
		{Package: "overgo/internal/reaching", Step: "test", ElapsedSeconds: new(9.0)},
	}}
	var output strings.Builder
	writeSelectionCauses(&output, &histogram)
	if want := "would leave out packages=2 of 3 elapsed=4.0s"; !strings.Contains(output.String(), want) {
		t.Fatalf("history = %q, want it to report %q", output.String(), want)
	}
}
