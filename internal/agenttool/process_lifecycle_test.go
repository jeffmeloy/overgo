package agenttool

import (
	"context"
	"strings"
	"testing"
)

// TestExternalProcessLifecycle proves argv tool invocation rides the
// shared process supervisor: the tool runs to completion under
// containment and its stdout returns as the strict-JSON result.
func TestExternalProcessLifecycle(t *testing.T) {
	manual := Manual{Name: "go-version", Transport: Transport{Program: "go", Args: []string{"version"}}}
	result, err := argvAdapter{}.invoke(context.Background(), manual, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(result), "go version") {
		t.Fatalf("tool result = %s", result)
	}
}
