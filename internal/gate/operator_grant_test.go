package gate

import (
	"strings"
	"testing"

	"overgo/internal/plan"
)

// TestGateTestsStartWithoutOperatorGrant holds the environment of every
// process the gate runs to the code under test: an operator's maintenance
// grant authorizes the gate's admission, and a test that inherited it
// refused its own plan binding (the audio merge gate, 2026-09-25).
func TestGateTestsStartWithoutOperatorGrant(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	t.Setenv(plan.AutomationMaintenanceEnvironment, "evidence:sha256:"+strings.Repeat("0", 64))
	g := &gateContext{repo: t.TempDir()}
	environment, err := g.sourceEnvironment()
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range environment {
		if strings.HasPrefix(entry, plan.AutomationMaintenanceEnvironment+"=") {
			t.Fatalf("a gate-run process inherits the operator's grant: %q", entry)
		}
	}
}
