package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// The README's speed section runs from its served-models heading to the next section's.
const (
	servedModelsHeading = "### Served models"
	sampleOutputHeading = "### Sample outputs"
)

// TestTheReadmeReportsNoColdSpeed holds the README's served-models section to the most recent warm speed alone: no
// table column reports a first output or throughput, which were the first request after a fresh load, no line calls
// a figure cold or a fresh load's, and no line dates a figure, since only the most recent results are reported.
func TestTheReadmeReportsNoColdSpeed(t *testing.T) {
	t.Parallel()
	readme := testcheck.Must(os.ReadFile(filepath.Join("..", "..", "README.md")))(t)
	_, section, found := strings.Cut(string(readme), servedModelsHeading)
	section, _, ended := strings.Cut(section, sampleOutputHeading)
	if !found || !ended {
		t.Fatalf("the README holds no %q section ending at %q", servedModelsHeading, sampleOutputHeading)
	}
	for line := range strings.SplitSeq(section, "\n") {
		lower := strings.ToLower(line)
		header := strings.HasPrefix(line, "| Model |")
		if header && (strings.Contains(line, "First output") || strings.Contains(line, "Throughput")) ||
			strings.Contains(lower, "cold") || strings.Contains(lower, "fresh load") || datesAFigure(line) {
			t.Errorf("the served-models section reports a cold, first-request or dated figure: %q", line)
		}
	}
}

// datesAFigure says a line holds a date written year-month-day, four digits, a dash, two digits, a dash, two digits.
func datesAFigure(line string) bool {
	const width = len("2026-11-07")
	for start := 0; start+width <= len(line); start++ {
		candidate := line[start : start+width]
		dated := true
		for index := range width {
			dash := index == len("2026") || index == len("2026-11")
			if dash != (candidate[index] == '-') || !dash && (candidate[index] < '0' || candidate[index] > '9') {
				dated = false
				break
			}
		}
		if dated {
			return true
		}
	}
	return false
}
