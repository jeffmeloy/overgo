package main

import (
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/runrecord"
)

// TestArchitectureRatchetAlwaysRequired pins the permanent entry-authority
// ratchet: it is a manifest-owned, always-required member of every gate
// pipeline, ordered inside the cheap validation prefix, and it executes for
// documentation-only commits that skip the go-source magic scan — measured
// in-process against the gate's own cached source inventory, with no second
// verification process.
func TestArchitectureRatchetAlwaysRequired(t *testing.T) {
	checks := (&gateContext{}).pipelineChecks()
	positions := make(map[string]int, len(checks))
	var architecture, found = checks[0], false
	for index, check := range checks {
		positions[check.Descriptor.Name] = index
		if check.Descriptor.Name == "architecture" {
			architecture, found = check, true
		}
	}
	if !found {
		t.Fatal("gate pipeline lacks the architecture entry-authority ratchet")
	}
	if !architecture.Descriptor.Always {
		t.Fatal("architecture ratchet is not always required")
	}
	if architecture.Descriptor.Phase != runrecord.PhaseValidate {
		t.Fatalf("architecture ratchet phase = %s", architecture.Descriptor.Phase)
	}
	if positions["architecture"] <= positions["magics"] {
		t.Fatal("architecture ratchet does not follow the magic authority scan")
	}
	for _, expensive := range []string{"acceptance", "vet", "build", "test", "device"} {
		if positions["architecture"] >= positions[expensive] {
			t.Fatalf("architecture ratchet at %d follows %s at %d", positions["architecture"], expensive, positions[expensive])
		}
	}

	documentationOnly := &gateContext{repo: filepath.Join("..", ".."), paths: []string{"README.md"}}
	if skipped, err := documentationOnly.stepMagics(); err != nil || !skipped {
		t.Fatalf("magic scan on documentation-only paths = (skipped=%t, %v)", skipped, err)
	}
	skipped, err := documentationOnly.stepArchitecture()
	if err != nil {
		t.Fatalf("architecture ratchet refused the current tree: %v", err)
	}
	if skipped {
		t.Fatal("architecture ratchet skipped a documentation-only commit")
	}
	measured := false
	for _, line := range documentationOnly.honesty {
		if strings.HasPrefix(line, "entry authority ratchet:") && strings.Contains(line, "wall=") {
			measured = true
		}
	}
	if !measured {
		t.Fatalf("architecture ratchet did not record its measured wall cost: %q", documentationOnly.honesty)
	}
}
