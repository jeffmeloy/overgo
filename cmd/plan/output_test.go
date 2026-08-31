package main

import (
	"bytes"
	"io"
	"os"
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
	authority := mustTestCompletionAuthority(t, document)
	var output bytes.Buffer
	printPrompt(document, plan.UnassignedRole, &output, authority)
	text := output.String()
	if len(text) > 700 || !strings.Contains(text, "TASK item/do") || !strings.Contains(text, "VERIFY go test") || !strings.Contains(text, "COMMIT go run ./cmd/gate") {
		t.Fatalf("prompt is not compact and decision-complete (%d bytes):\n%s", len(text), text)
	}
	if strings.Contains(text, "PROTOCOL") || strings.Contains(text, "long doctrine") {
		t.Fatalf("prompt repeated full doctrine:\n%s", text)
	}
}

func TestStatusCountsRetainedOpenRows(t *testing.T) {
	document := plan.Plan{Items: []plan.Item{
		{ID: "complete", Title: "complete", Status: plan.StatusDone, Steps: []plan.Step{{ID: "done", Status: plan.StatusDone}}},
		{ID: "active", Title: "active", Status: plan.StatusOpen, Steps: []plan.Step{
			{ID: "done", Status: plan.StatusDone},
			{ID: "open", Status: plan.StatusOpen},
		}},
	}}
	output := captureStdout(t, func() { printStatus(document) })
	if !strings.Contains(output, "complete") || !strings.Contains(output, "0 open rows") ||
		!strings.Contains(output, "active") || !strings.Contains(output, "1 open rows") {
		t.Fatalf("status did not distinguish retained and open rows:\n%s", output)
	}
}

func captureStdout(t *testing.T, run func()) string {
	t.Helper()
	read, write, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	original := os.Stdout
	os.Stdout = write
	defer func() { os.Stdout = original }()
	run()
	if err := write.Close(); err != nil {
		t.Fatal(err)
	}
	data, err := io.ReadAll(read)
	if err != nil {
		t.Fatal(err)
	}
	if err := read.Close(); err != nil {
		t.Fatal(err)
	}
	return string(data)
}
