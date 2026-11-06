package main

import (
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestExternalProcessLifecycle proves loop tool invocations ride the
// shared process supervisor end to end.
func TestExternalProcessLifecycle(t *testing.T) {
	out := testcheck.Must(runTool("go", "version"))(t)
	if !strings.Contains(out, "go version") {
		t.Fatalf("tool output = %q", out)
	}
}
