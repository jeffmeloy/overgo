package repoanalysis

import (
	"path/filepath"
	"testing"
)

func TestAgentHarnessConsolidation(t *testing.T) {
	root := filepath.Join("..", "..")
	snapshot, err := DiscoverGo(root, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	report, err := AuditAgentHarnessConsolidation(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if err := report.Error(); err != nil {
		t.Fatal(err)
	}
	if report.Declarations != report.Rules {
		t.Fatalf("declarations=%d rules=%d", report.Declarations, report.Rules)
	}

	mutated, err := snapshot.Overlay(map[string][]byte{
		"internal/agentloop/duplicate_effect.go": []byte("package agentloop\ntype InvocationEffect struct{}\n"),
		"internal/automationpolicy/parallel.go":  []byte("package automationpolicy\ntype Lifecycle struct{}\n"),
	})
	if err != nil {
		t.Fatal(err)
	}
	broken, err := AuditAgentHarnessConsolidation(mutated)
	if err != nil {
		t.Fatal(err)
	}
	if len(broken.Findings) < 2 || broken.Error() == nil {
		t.Fatalf("mutation escaped audit: %+v", broken)
	}
}
