package gate

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// liveTreeScanOwners are the analysis owners whose tests may scan the live
// repository tree. Any other package doing so names the internal and cmd
// trees as its runtime input and runs its whole suite on every Go change.
var liveTreeScanOwners = []string{
	"cmd/closure-scan", "internal/codemanifest", "internal/codeprofile", "internal/gate", "internal/repoanalysis",
	// The projection authority ratchet still lives beside its subject; it
	// moves to an analysis owner with the next selection row.
	"internal/runrecord",
}

// TestLiveTreeScanRatchet pins two facts: only the analysis owners scan the
// live tree from a test, and a change to one gate file selects no direct
// owner beyond the gate and the remaining ratchet host.
func TestLiveTreeScanRatchet(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	var scanners []string
	for _, tree := range []string{"internal", "cmd"} {
		err := filepath.WalkDir(filepath.Join(root, tree), func(path string, entry os.DirEntry, err error) error {
			if err != nil || entry.IsDir() || !strings.HasSuffix(entry.Name(), "_test.go") {
				return err
			}
			data, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			source := string(data)
			if !strings.Contains(source, "repoanalysis.DiscoverGo(") || !strings.Contains(source, `"..", ".."`) {
				return nil
			}
			owner, err := filepath.Rel(root, filepath.Dir(path))
			if err != nil {
				return err
			}
			scanners = append(scanners, filepath.ToSlash(owner))
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	slices.Sort(scanners)
	scanners = slices.Compact(scanners)
	for _, owner := range scanners {
		if !slices.Contains(liveTreeScanOwners, owner) {
			t.Errorf("%s scans the live tree from a test; the gate architecture phase already validates every entry authority on each commit", owner)
		}
	}
	g := &gateContext{repo: root, paths: []string{"internal/gate/preflight.go"}}
	scope, err := g.deriveTestScope()
	if err != nil {
		t.Fatal(err)
	}
	for _, direct := range scope.direct {
		if !slices.Contains(liveTreeScanOwners, strings.TrimPrefix(direct, "overgo/")) {
			t.Errorf("gate-only change selected %s as a direct owner", direct)
		}
	}
	if !slices.Contains(scope.direct, "overgo/internal/gate") {
		t.Fatalf("gate-only change lost its own owner: %v", scope.direct)
	}
	t.Logf("live-tree scanners=%v; gate-only change: direct=%d dependent=%d excluded=%d", scanners, len(scope.direct), len(scope.dependent), scope.excluded)
}
