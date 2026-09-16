package gate

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestDeviceGroupLinksOnlyKernelTests holds the device group to the tests
// that execute kernels, on the live graph: every package the group holds
// declares its device use, every package that only links the runtime runs
// as a host package, and no test outside the runtime opens the device or
// creates a context without the gate that skips it outside the device lane.
func TestDeviceGroupLinksOnlyKernelTests(t *testing.T) {
	t.Parallel()
	live := liveRepositoryFixture(t)
	graph, err := live.context().inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	var roots []string
	byPath := map[string]goPackageInput{}
	for _, node := range graph.nodes {
		if node.ForTest != "" || len(node.Match) == 0 || strings.HasSuffix(node.ImportPath, ".test") {
			continue
		}
		roots = append(roots, node.ImportPath)
		byPath[node.ImportPath] = node
	}
	slices.Sort(roots)
	linkerDirectories, err := graph.dependentDirectories("internal/cuda")
	if err != nil {
		t.Fatal(err)
	}
	linkers := map[string]bool{}
	for _, directory := range linkerDirectories {
		linkers["overgo/"+directory] = true
	}
	devices, err := graph.devicePackages(roots)
	if err != nil {
		t.Fatal(err)
	}
	workers := graph.productionDependents(deviceWorkerPackage)
	runtime := func(pkg string) bool {
		return pkg == deviceRuntimeRoot || strings.HasPrefix(pkg, deviceRuntimeRoot+"/")
	}
	declares := func(node goPackageInput) string {
		switch {
		case runtime(node.ImportPath):
			return "runtime package"
		case slices.ContainsFunc(slices.Concat(node.TestImports, node.XTestImports), func(imported string) bool {
			return slices.Contains(deviceTestPackages, imported)
		}):
			return "test import"
		case slices.ContainsFunc(node.executionDependencies, func(command string) bool { return workers[command] }):
			return "runs a context-creating command"
		}
		return ""
	}
	for _, pkg := range devices {
		if !linkers[pkg] {
			t.Errorf("%s is in the device group yet links no runtime", pkg)
		}
		if declares(byPath[pkg]) == "" {
			t.Errorf("%s is in the device group without declaring device use", pkg)
		}
	}
	var linking, host []string
	for _, pkg := range roots {
		if !linkers[pkg] {
			continue
		}
		linking = append(linking, pkg)
		if !slices.Contains(devices, pkg) {
			host = append(host, pkg)
		}
	}
	// A test that opens the device or creates a context outside the runtime
	// names the gate that skips it without the device lane's environment,
	// and its package is in the group; a host run would otherwise claim the
	// device beside the lane.
	entries := []string{"driver.Open(", "device.New(", ".ReserveDevice(", ".ContextCreate("}
	gates := []string{`"overgo/internal/cuda/testutil"`, "OVERGO_CUDA_TEST"}
	for _, pkg := range roots {
		node := byPath[pkg]
		if runtime(pkg) {
			continue
		}
		for _, file := range slices.Concat(node.TestGoFiles, node.XTestGoFiles) {
			text, err := os.ReadFile(filepath.Join(node.Dir, file))
			if err != nil {
				t.Fatal(err)
			}
			source := string(text)
			if !slices.ContainsFunc(entries, func(entry string) bool { return strings.Contains(source, entry) }) {
				continue
			}
			if !slices.ContainsFunc(gates, func(gate string) bool { return strings.Contains(source, gate) }) {
				t.Errorf("%s/%s opens the device without the kernel test gate", pkg, file)
			}
			if !slices.Contains(devices, pkg) {
				t.Errorf("%s opens the device in %s yet runs as a host package", pkg, file)
			}
		}
	}
	for _, required := range []string{"overgo/internal/devicemath", "overgo/internal/cuda/executor", "overgo/internal/inference", "overgo/internal/optimizer"} {
		if !slices.Contains(devices, required) {
			t.Errorf("%s runs kernels yet left the device group", required)
		}
	}
	for _, hostOnly := range []string{"overgo/cmd/api-manifest", "overgo/cmd/overgodb-import", "overgo/internal/overgodbimport", "overgo/cmd/store-check"} {
		if slices.Contains(devices, hostOnly) {
			t.Errorf("%s links the runtime without a kernel test yet stays in the device group (%s)", hostOnly, declares(byPath[hostOnly]))
		}
	}
	if len(host) == 0 {
		t.Fatal("no package links the runtime without a kernel test; the group is the link closure again")
	}
	t.Logf("device group: %d of the %d packages linking the runtime [%s]; %d link it and run as host packages: %s", len(devices), len(linking), strings.Join(devices, ","), len(host), strings.Join(host, ","))
}
