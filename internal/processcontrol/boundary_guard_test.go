package processcontrol

import (
	"path/filepath"
	"testing"

	"overgo/internal/repoanalysis"
)

// TestProductionCommandsUseSupervisor is the process boundary: outside
// this package, production code neither spawns processes through
// os/exec nor kills them directly. The production architecture analyzer owns
// the exact call-site allowances so the gate and this historical package pin
// cannot drift into two different policies.
func TestProductionCommandsUseSupervisor(t *testing.T) {
	root := filepath.Join("..", "..")
	snapshot, err := repoanalysis.DiscoverGo(root, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	report, err := repoanalysis.AuditProductionAuthorityBoundaries(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	for _, finding := range report.Findings {
		if finding.Family == "process" {
			t.Errorf("process authority finding: %+v", finding)
		}
	}
}
