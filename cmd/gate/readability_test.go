package main

import (
	"encoding/json"
	"strings"
	"testing"

	"overgo/internal/testutil"
)

func TestGoReadabilityCandidateDelta(t *testing.T) {
	root := t.TempDir()
	testutil.WriteTextFile(t, root, "go.mod", "module fixture\n\ngo 1.26\n")
	testutil.WriteTextFile(t, root, "internal/p/a.go", "package p\nfunc before() {}\n")
	testutil.WriteTextFile(t, root, "internal/p/stable.go", "package p\nfunc stable() {}\n")
	for _, args := range [][]string{
		{"init"}, {"add", "."}, {"-c", "user.name=fixture", "-c", "user.email=fixture@example.com", "commit", "-m", "base"},
	} {
		if _, err := command(root, "git", args...); err != nil {
			t.Fatal(err)
		}
	}
	testutil.WriteTextFile(t, root, "internal/p/a.go", "package p\nfunc after() { panic(\"observe only\") }\n")
	gate := gateContext{
		repo: root, paths: []string{"internal/p/a.go"}, stepEvidence: map[string]string{},
	}
	skipped, err := gate.stepReadability()
	if err != nil || skipped {
		t.Fatalf("readability step = skipped %t, err %v", skipped, err)
	}
	var evidence readabilityEvidence
	rawEvidence := []byte(gate.stepEvidence["readability"])
	if err := json.Unmarshal(rawEvidence, &evidence); err != nil {
		t.Fatal(err)
	}
	if len(rawEvidence) > 2048 || evidence.BaseIdentity == evidence.CandidateIdentity || evidence.Analysis.FilesReused != 1 || len(evidence.Rules) == 0 || len(evidence.Blocking) != 0 {
		t.Fatalf("readability evidence = %+v", evidence)
	}
	if !strings.Contains(gate.honesty[0], "observe-only") || !strings.Contains(gate.honesty[0], "reused=1") {
		t.Fatalf("readability honesty = %v", gate.honesty)
	}
	for _, rule := range evidence.Rules {
		if rule.Maturity == "enforceable" {
			t.Fatalf("first gate revision promoted %s", rule.ID)
		}
		if !rule.Selected && rule.SkipReason == "" {
			t.Fatalf("rule %s was silently skipped", rule.ID)
		}
	}
}
