package gate

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestGateRefusalsCarryDeterministicRemediation pins the throughput contract:
// a refusal whose fix is deterministic either applies that fix inside the run
// or names the one exact command. Stale closure authority triggers the safe
// same-store rebind and revalidation with a recorded remediation receipt;
// ambiguous lifecycle debt enumerates one argument-bound recovery command per
// stale preparation; an unformatted file names its exact gofmt invocation.
func TestGateRefusalsCarryDeterministicRemediation(t *testing.T) {
	for message, remediable := range map[string]bool{
		"permanent authority: 3 stale active binding(s), first=orphan/literal.1":         true,
		"permanent authority: uncatalogued production policy literal.5 at cmd/x/y.go:12": true,
		"magic scan: active document identity mismatch":                                  false,
	} {
		if got := staleClosureAuthorityFailure(errors.New(message)); got != remediable {
			t.Fatalf("remediation classifier(%q) = %t", message, got)
		}
	}

	recorded := [][]string{}
	gate := &gateContext{repo: t.TempDir(), storePath: gateStorePath, runCommand: func(repo, name string, args ...string) (string, error) {
		recorded = append(recorded, append([]string{name}, args...))
		return "imported 3 closure document(s), unmatched=0", nil
	}}
	if err := gate.remediateStaleClosureBindings(); err != nil {
		t.Fatal(err)
	}
	if len(recorded) != 1 || strings.Join(recorded[0], " ") != "go run ./cmd/closure-scan -import-store "+filepath.Join(gate.repo, gate.storePath) {
		t.Fatalf("rebind remediation command = %v", recorded)
	}
	receipt := false
	for _, line := range gate.audit {
		if strings.HasPrefix(line, "remediation: closure rebind applied") && strings.Contains(line, "wall=") {
			receipt = true
		}
	}
	if !receipt {
		t.Fatalf("rebind remediation left no receipt: %q", gate.audit)
	}

	fixture := newStaleLifecycleRecoveryFixture(t, 2, true)
	err := requireNoPendingGateState(fixture.repo, fixture.storePath)
	if err == nil {
		t.Fatal("ambiguous lifecycle debt admitted a new gate")
	}
	for _, stale := range fixture.stale {
		exact := "go run ./cmd/gate -record-failure -preparation " + stale.ID.String()
		if !strings.Contains(err.Error(), exact) {
			t.Fatalf("ambiguous debt refusal omits %q: %v", exact, err)
		}
	}

	repo := t.TempDir()
	unformatted := filepath.Join(repo, "rogue.go")
	if err := os.WriteFile(unformatted, []byte("package rogue\nvar  x  =  1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	formatting := &gateContext{repo: repo, paths: []string{"rogue.go"}}
	_, fmtErr := formatting.stepFmt()
	if fmtErr == nil || !strings.Contains(fmtErr.Error(), "remediate with `gofmt -w rogue.go`") {
		t.Fatalf("format refusal omits its exact remediation: %v", fmtErr)
	}
}
