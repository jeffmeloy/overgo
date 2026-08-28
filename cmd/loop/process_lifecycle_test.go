package main

import (
	"strings"
	"testing"
)

// TestExternalProcessLifecycle proves loop tool invocations ride the
// shared process supervisor end to end.
func TestExternalProcessLifecycle(t *testing.T) {
	out, err := runTool("go", "version")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "go version") {
		t.Fatalf("tool output = %q", out)
	}
}
