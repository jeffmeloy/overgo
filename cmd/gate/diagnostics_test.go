package main

import (
	"strings"
	"testing"

	"overgo/internal/closurescan"
)

func TestDiagnosticsPrioritizeChangedDebt(t *testing.T) {
	baseline := []closurescan.Candidate{
		{Name: "inheritedLimit", File: "internal/x/x.go", Value: "8"},
		{Name: "changedLimit", File: "internal/x/x.go", Value: "16"},
	}
	current := []closurescan.Candidate{
		{Name: "newLimit", File: "internal/x/x.go", Value: "32"},
		{Name: "inheritedLimit", File: "internal/x/x.go", Value: "8"},
		{Name: "changedLimit", File: "internal/x/x.go", Value: "24"},
		{Name: "cataloguedLimit", File: "internal/x/x.go", Value: "64"},
	}
	lines := magicDiagnostics(current, baseline, map[string]bool{"cataloguedLimit": true})
	if len(lines) != 3 || !strings.Contains(lines[0], "newLimit") || !strings.Contains(lines[1], "changedLimit") {
		t.Fatalf("changed debt was not first: %v", lines)
	}
	if !strings.Contains(lines[2], "1 inherited") || strings.Contains(strings.Join(lines, "\n"), "inheritedLimit=") {
		t.Fatalf("inherited debt was not summarized: %v", lines)
	}
}
