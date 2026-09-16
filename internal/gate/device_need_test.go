package gate

import (
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestDevicePackagesRunFirstUnderTheLease pins the one device-need fact and
// the batch order it drives: a package needs the device when it is part of
// the runtime, when its tests import the driver, the worker or the kernel
// test gate, or when its tests run a command that creates device contexts;
// a package that only links the runtime does not. A test group splits into
// its device packages first, run as their own batch under the shared lease,
// and the host-only packages after with no lease; a group without device
// packages stays one batch and an empty group has none. The batch admission
// reads the same fact.
func TestDevicePackagesRunFirstUnderTheLease(t *testing.T) {
	t.Parallel()
	root := filepath.Join(t.TempDir(), "repo")
	node := func(importPath, dir string, imports ...string) goPackageInput {
		return goPackageInput{ImportPath: importPath, Dir: filepath.Join(root, filepath.FromSlash(dir)), Match: []string{importPath}, Imports: imports}
	}
	devicemath := node("overgo/internal/devicemath", "internal/devicemath", "overgo/internal/cuda/driver")
	devicemath.TestImports = []string{"overgo/internal/cuda/testutil"}
	smoke := node("overgo/internal/smoketest", "internal/smoketest")
	smoke.executionDependencies = []string{"overgo/cmd/cuda-smoke"}
	graph := packageInputGraph{root: root, nodes: []goPackageInput{
		node("overgo/internal/cuda/driver", "internal/cuda/driver"),
		node("overgo/internal/cuda/device", "internal/cuda/device", "overgo/internal/cuda/driver"),
		node("overgo/cmd/cuda-smoke", "cmd/cuda-smoke", "overgo/internal/cuda/device"),
		devicemath,
		smoke,
		// latentimage links the runtime through devicemath; its tests declare no device use.
		node("overgo/internal/latentimage", "internal/latentimage", "overgo/internal/devicemath"),
		node("overgo/internal/plan", "internal/plan"),
		node("overgo/cmd/plan", "cmd/plan", "overgo/internal/plan"),
	}}
	g := &gateContext{repo: root, packageGraph: &graph}
	group := []string{"overgo/cmd/plan", "overgo/internal/latentimage", "overgo/internal/smoketest", "overgo/internal/plan", "overgo/internal/devicemath"}
	devices, err := graph.devicePackages(group)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(devices, []string{"overgo/internal/smoketest", "overgo/internal/devicemath"}) {
		t.Fatalf("device packages = %v", devices)
	}
	want := [][]string{{"overgo/internal/smoketest", "overgo/internal/devicemath"}, {"overgo/cmd/plan", "overgo/internal/latentimage", "overgo/internal/plan"}}
	if devices, hostPart, err := g.splitDevice(group); err != nil || !slices.Equal(devices, want[0]) || !slices.Equal(hostPart, want[1]) {
		t.Fatalf("split = %v / %v, %v; want %v", devices, hostPart, err, want)
	}
	if !slices.ContainsFunc(g.audit, func(line string) bool {
		return strings.HasPrefix(line, "test order: 2 device packages first under the shared lease")
	}) {
		t.Fatalf("audit = %v, want the device-first order", g.audit)
	}
	host := []string{"overgo/internal/plan", "overgo/cmd/plan", "overgo/internal/latentimage"}
	if devices, hostPart, err := g.splitDevice(host); err != nil || len(devices) != 0 || !slices.Equal(hostPart, host) {
		t.Fatalf("host-only split = %v / %v, %v", devices, hostPart, err)
	}
	if devices, hostPart, err := g.splitDevice(nil); err != nil || devices != nil || hostPart != nil {
		t.Fatalf("empty split = %v / %v, %v", devices, hostPart, err)
	}
	only := []string{"overgo/internal/cuda/driver", "overgo/cmd/cuda-smoke"}
	if devices, hostPart, err := g.splitDevice(only); err != nil || !slices.Equal(devices, only[:1]) || !slices.Equal(hostPart, only[1:]) {
		t.Fatalf("runtime split = %v / %v, %v; want the runtime package alone under the lease", devices, hostPart, err)
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
