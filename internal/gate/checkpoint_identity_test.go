package gate

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/plan"
)

// TestCheckpointMemoCoversEveryCommand pins: every go test segment's packages
// enter the memo input, so a later package's change invalidates it; a covered
// file guard is accepted; an unsupported segment, an uncovered guard, or an
// absent package refuses the memo with an audited reason.
func TestCheckpointMemoCoversEveryCommand(t *testing.T) {
	g, batch, _ := verificationBatchFixture(t, "pass")
	subDir := filepath.Join(g.repo, "sub")
	if err := os.MkdirAll(subDir, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, content := range map[string]string{
		"sub.go":      "package sub\n\n// Value: fixture constant.\nfunc Value() int { return 1 }\n",
		"sub_test.go": "package sub\nimport \"testing\"\nfunc TestSub(t *testing.T) { if Value() != 1 { t.Fatal(Value()) } }\n",
	} {
		if err := os.WriteFile(filepath.Join(subDir, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	batch.Flush = &plan.BatchFlush{Key: "identity", MaxSize: 2, MaxInterval: "1m", MaxBytes: 1 << 20}
	compound := "go test . -run '^TestProducer$' -count=1 -v && go test ./sub -run '^TestSub$' -count=1"
	batch.Checkpoints = []plan.VerificationCheckpoint{
		{ID: "compound", Title: "Both packages", Verify: compound},
		{ID: "guarded", Title: "Guarded", Verify: "test -f unit_test.go && go test . -run '^TestProducer$' -count=1 -v"},
		{ID: "mixed", Title: "Mixed", Verify: "echo prepared && go test . -run '^TestProducer$' -count=1 -v"},
		{ID: "stray-guard", Title: "Stray guard", Verify: "test -f sub/sub.go && go test . -run '^TestProducer$' -count=1 -v"},
		{ID: "absent", Title: "Absent package", Verify: "go test ./missing -run '^TestNone$' -count=1 -v"},
	}
	g.packageGraph = nil
	first, err := g.checkpointMemoInputs(batch)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"acceptance-compound", "acceptance-guarded"} {
		if _, found := first[want]; !found {
			t.Fatalf("%s was not memoised; audit=%v", want, g.audit)
		}
	}
	for _, refused := range []string{"acceptance-mixed", "acceptance-stray-guard", "acceptance-absent"} {
		if _, found := first[refused]; found {
			t.Fatalf("%s was memoised despite an uncoverable verifier", refused)
		}
	}
	audit := strings.Join(g.audit, "\n")
	for _, reason := range []string{"mixed: segment", "stray-guard: guard", "absent: package"} {
		if !strings.Contains(audit, reason) {
			t.Fatalf("audit lacks refusal %q:\n%s", reason, audit)
		}
	}

	if err := os.WriteFile(filepath.Join(subDir, "sub.go"), []byte("package sub\n\n// Value: fixture constant.\nfunc Value() int { return 2 }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	g.packageGraph, g.audit = nil, nil
	second, err := g.checkpointMemoInputs(batch)
	if err != nil {
		t.Fatal(err)
	}
	if second["acceptance-compound"].slot != first["acceptance-compound"].slot || second["acceptance-compound"].input == first["acceptance-compound"].input {
		t.Fatal("a change in the second package did not invalidate the compound memo input")
	}
	if second["acceptance-guarded"].input != first["acceptance-guarded"].input {
		t.Fatal("a change outside the guarded package altered its memo input")
	}
}
