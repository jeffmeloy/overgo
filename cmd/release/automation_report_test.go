package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAutomationReportNamesLiveOwners(t *testing.T) {
	path := filepath.Join("..", "..", "automation_report.md")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	text := string(raw)
	for _, owner := range []string{
		"`cmd/finding`", "`cmd/plan -sync-master`", "`cmd/closure-scan -raw`",
		"per-phase input fingerprints", "symbol-scoped", "`docs/.bounded_request`",
	} {
		if !strings.Contains(text, owner) {
			t.Errorf("automation report omits live owner/capability %s", owner)
		}
	}
	for _, stale := range []string{"cached against a hash of HEAD", "docs/findings.json"} {
		if strings.Contains(text, stale) {
			t.Errorf("automation report retains superseded claim %q", stale)
		}
	}
}
