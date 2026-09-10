package gate

import (
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestDevicePackagesRunFirstUnderTheLease pins the one device-need fact and
// the batch order it drives: a package that is or transitively depends on
// internal/cuda needs the device; a test group splits into its device
// packages first, run as their own batch under the shared lease, and the
// host-only packages after with no lease; a group without device packages
// stays one batch and an empty group has none. The batch admission reads
// the same fact.
func TestDevicePackagesRunFirstUnderTheLease(t *testing.T) {
	root := filepath.Join(t.TempDir(), "repo")
	node := func(importPath, dir string, imports ...string) goPackageInput {
		return goPackageInput{ImportPath: importPath, Dir: filepath.Join(root, filepath.FromSlash(dir)), Match: []string{importPath}, Imports: imports}
	}
	graph := packageInputGraph{root: root, nodes: []goPackageInput{
		node("overgo/internal/cuda/driver", "internal/cuda/driver"),
		node("overgo/internal/devicemath", "internal/devicemath", "overgo/internal/cuda/driver"),
		node("overgo/internal/latentimage", "internal/latentimage", "overgo/internal/devicemath"),
		node("overgo/internal/plan", "internal/plan"),
		node("overgo/cmd/plan", "cmd/plan", "overgo/internal/plan"),
	}}
	g := &gateContext{repo: root, packageGraph: &graph}
	group := []string{"overgo/cmd/plan", "overgo/internal/latentimage", "overgo/internal/plan", "overgo/internal/devicemath"}
	devices, err := graph.devicePackages(group)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(devices, []string{"overgo/internal/latentimage", "overgo/internal/devicemath"}) {
		t.Fatalf("device packages = %v", devices)
	}
	batches, err := g.deviceFirst(group)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]string{{"overgo/internal/latentimage", "overgo/internal/devicemath"}, {"overgo/cmd/plan", "overgo/internal/plan"}}
	if len(batches) != 2 || !slices.Equal(batches[0], want[0]) || !slices.Equal(batches[1], want[1]) {
		t.Fatalf("batches = %v, want %v", batches, want)
	}
	if !slices.ContainsFunc(g.audit, func(line string) bool {
		return strings.HasPrefix(line, "test order: 2 device packages first under the shared lease")
	}) {
		t.Fatalf("audit = %v, want the device-first order", g.audit)
	}
	host := []string{"overgo/internal/plan", "overgo/cmd/plan"}
	if batches, err := g.deviceFirst(host); err != nil || len(batches) != 1 || !slices.Equal(batches[0], host) {
		t.Fatalf("host-only group = %v, %v; want one batch", batches, err)
	}
	if batches, err := g.deviceFirst(nil); err != nil || batches != nil {
		t.Fatalf("empty group = %v, %v; want no batch", batches, err)
	}
	only := []string{"overgo/internal/cuda/driver"}
	if batches, err := g.deviceFirst(only); err != nil || len(batches) != 1 || !slices.Equal(batches[0], only) {
		t.Fatalf("device-only group = %v, %v; want one batch under the lease", batches, err)
	}
	// The batch admission reads the same fact: a host-only batch needs no
	// lease and returns at once on any platform.
	release, err := g.admitTestResources(t.Context(), host, nil)
	if err != nil || release == nil {
		t.Fatalf("host-only admission = %v, released %t", err, release != nil)
	}
	if err := release(); err != nil {
		t.Fatal(err)
	}
}
