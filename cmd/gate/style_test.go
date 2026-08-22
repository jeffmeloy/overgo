package main

import (
	"strings"
	"testing"

	"overgo/internal/repoanalysis"
)

func TestGoStyleChangedFileGate(t *testing.T) {
	baseline := styleSnapshot(t, "// Package sample owns a fixture.\npackage sample\n\nfunc local() error { return nil }\n")
	legacy := styleSnapshot(t, "package sample\n\nfunc local() error { return nil }\n")
	if err := repoanalysis.ValidateGoStyleDelta(legacy, baseline); err == nil {
		t.Fatal("new package documentation finding was admitted")
	}
	shifted := styleSnapshot(t, "// Package sample owns a fixture.\npackage sample\n\n// Fixture boundary.\nfunc local() error { return nil }\n")
	if err := repoanalysis.ValidateGoStyleDelta(shifted, baseline); err != nil {
		t.Fatalf("line-only change rejected: %v", err)
	}
	capitalized := styleSnapshot(t, "// Package sample owns a fixture.\npackage sample\n\nimport \"errors\"\n\nfunc local() error { return errors.New(\"Bad value\") }\n")
	if err := repoanalysis.ValidateGoStyleDelta(capitalized, baseline); err == nil || !strings.Contains(err.Error(), "error-text") {
		t.Fatalf("capitalized error finding = %v", err)
	}
}

func styleSnapshot(t *testing.T, source string) repoanalysis.SourceSnapshot {
	t.Helper()
	snapshot, err := (repoanalysis.SourceSnapshot{}).Overlay(map[string][]byte{"sample/sample.go": []byte(source)})
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}
