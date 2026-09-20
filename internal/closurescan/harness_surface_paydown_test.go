package closurescan

import (
	"path/filepath"
	"testing"

	"overgo/internal/jsonfile"
	"overgo/internal/testutil"
)

// harnessSurfacePaidDown is the production-node count the dead-code paydown
// left in the nine harness packages. The committed baseline may only fall
// below it: the gate refuses a raise without a recorded reason, and this
// ceiling refuses a reason that undoes the paydown.
const harnessSurfacePaidDown = 182159

// TestHarnessSurfacePaydown holds the committed agent harness baseline at or
// below the paid-down surface.
func TestHarnessSurfacePaydown(t *testing.T) {
	var baseline HarnessSurface
	path := filepath.Join(testutil.RepoRoot(t), "docs", "harness_surface_baseline.json")
	if err := jsonfile.DecodeStrict(path, &baseline); err != nil {
		t.Fatal(err)
	}
	if baseline.ProductionNodes <= 0 || baseline.ProductionNodes > harnessSurfacePaidDown {
		t.Fatalf("harness production nodes = %d, want at most the paid-down %d", baseline.ProductionNodes, harnessSurfacePaidDown)
	}
}
