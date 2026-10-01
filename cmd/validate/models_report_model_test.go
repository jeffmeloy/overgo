//go:build modeltest

package main

import (
	"testing"
)

// TestModelsReportRendersFromItsInputs holds the model report to rendering on
// demand from its inputs -- the typed verification specifications joined
// against the store -- into the typed report a reader decodes.
func TestModelsReportRendersFromItsInputs(t *testing.T) {
	t.Parallel()
	if report := renderedModelsReport(t); len(report.Prototypes) == 0 {
		t.Fatal("the rendered model report is empty")
	}
}
