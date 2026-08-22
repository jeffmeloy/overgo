package main

import (
	"bytes"
	"strings"
	"testing"

	"overgo/internal/plan"
)

func TestCompactAgentOutput(t *testing.T) {
	document := plan.Plan{Campaign: "campaign", Doctrine: strings.Repeat("long doctrine ", 200), Items: []plan.Item{{
		ID: "item", Title: "item title", Status: "open", Steps: []plan.Step{{
			ID: "do", Title: "make the bounded change", Status: "open", Verify: "go test ./x -run '^TestX$'",
		}},
	}}}
	var output bytes.Buffer
	printPrompt(document, plan.UnassignedRole, &output)
	text := output.String()
	if len(text) > 700 || !strings.Contains(text, "TASK item/do") || !strings.Contains(text, "VERIFY go test") || !strings.Contains(text, "COMMIT go run ./cmd/gate") {
		t.Fatalf("prompt is not compact and decision-complete (%d bytes):\n%s", len(text), text)
	}
	if strings.Contains(text, "PROTOCOL") || strings.Contains(text, "long doctrine") {
		t.Fatalf("prompt repeated full doctrine:\n%s", text)
	}
}
