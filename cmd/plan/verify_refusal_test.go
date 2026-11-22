package main

import (
	"strings"
	"testing"

	"overgo/internal/plan"
)

// TestACheckReadingARuntimeFileIsRefused holds the checks a step may take (verifyRefusal): one reading a file under
// the runtime state directory by hand is refused, naming validate -passed-since, which answers from the passes kept;
// the browser rule still holds; and rows filed whole (-add -from) are refused by the first step whose check is.
func TestACheckReadingARuntimeFileIsRefused(t *testing.T) {
	t.Parallel()
	refused := `awk '/TestFlashNextHomesEveryExpert/' "$(git rev-parse --show-toplevel)/.overgo-runtime/lanes.log" | grep -q '^ok'`
	err := verifyRefusal(refused)
	if err == nil || !strings.Contains(err.Error(), "validate -passed-since") {
		t.Fatalf("a check reading the lanes' log is refused, naming the query: %v", err)
	}
	for _, command := range []string{
		"go run ./cmd/validate -passed-since 32cf1a5d0 overgo/internal/inference.TestFlashNextHomesEveryExpert",
		"go test ./cmd/plan -count=1 -run '^TestACheckReadingARuntimeFileIsRefused$'",
	} {
		if err := verifyRefusal(command); err != nil {
			t.Fatalf("%q may be a check: %v", command, err)
		}
	}
	if verifyRefusal("go test ./internal/server -run '^TestWebUIBrowserFirstRun$' -count=1") == nil {
		t.Fatal("a browser test named outside the lane is refused")
	}

	items := []plan.Item{
		{ID: "kept", Steps: []plan.Step{{ID: "do", Verify: "go test ./cmd/plan -count=1"}}},
		{ID: "read", Steps: []plan.Step{{ID: "do", Verify: "go test ./cmd/plan -count=1"}, {ID: "read", Verify: refused}}},
	}
	if err := checksRefused(items[:1]); err != nil {
		t.Fatalf("rows whose checks may be filed: %v", err)
	}
	if err := checksRefused(items); err == nil || !strings.HasPrefix(err.Error(), "read/read: ") {
		t.Fatalf("rows filed whole are refused by the step whose check reads a runtime file: %v", err)
	}
}
