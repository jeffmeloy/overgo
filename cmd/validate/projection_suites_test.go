package main

import (
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/jsonfile"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestEveryCommittedProjectionSuiteIsNamed holds ocr-projection: every native projection suite committed under
// docs/verification/fixtures is one projectionSuites names, so a projection cell whose recipe's exact verification ran
// it is grounded and the full run verifies it again. Unlimited-OCR's suite, committed with its reader
// (drawn-text-reading), and Gemma 4 12B's were named nowhere, and the full measurement's preflight refused the OCR
// cell (2026-11-23).
func TestEveryCommittedProjectionSuiteIsNamed(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	paths := testcheck.Must(filepath.Glob(filepath.Join(root, "docs", "verification", "fixtures", "*", "suite.json")))(t)
	committed := 0
	for _, path := range paths {
		var suite struct {
			Schema string `json:"schema"`
		}
		testcheck.NoError(t, jsonfile.Decode(path, &suite))
		if suite.Schema != "overgo/native-projection/v1" {
			continue
		}
		committed++
		if relative := filepath.ToSlash(testcheck.Must(filepath.Rel(root, path))(t)); !slices.Contains(projectionSuites, relative) {
			t.Errorf("%s is a committed projection suite projectionSuites does not name, so no cell it grounds is verified again", relative)
		}
	}
	if committed == 0 {
		t.Fatal("no committed projection suite was found under docs/verification/fixtures")
	}
}
