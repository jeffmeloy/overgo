package gate

import (
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// devicePackages lists the root packages whose production import closure
// reaches the device runtime under internal/cuda, the packages whose test
// binaries link it and run under the device lease.
func devicePackages(graph packageInputGraph) []string {
	imports := map[string][]string{}
	var roots []string
	for _, node := range graph.nodes {
		if node.ForTest != "" || strings.HasSuffix(node.ImportPath, ".test") {
			continue
		}
		imports[node.ImportPath] = node.Imports
		if len(node.Match) != 0 {
			roots = append(roots, node.ImportPath)
		}
	}
	memo := map[string]bool{}
	var reach func(string, map[string]bool) bool
	reach = func(pkg string, visiting map[string]bool) bool {
		if strings.HasPrefix(pkg, "overgo/internal/cuda") {
			return true
		}
		if value, ok := memo[pkg]; ok {
			return value
		}
		if visiting[pkg] {
			return false
		}
		visiting[pkg] = true
		result := slices.ContainsFunc(imports[pkg], func(edge string) bool { return reach(edge, visiting) })
		memo[pkg] = result
		return result
	}
	var device []string
	for _, pkg := range roots {
		if reach(pkg, map[string]bool{}) {
			device = append(device, strings.TrimPrefix(pkg, "overgo/"))
		}
	}
	slices.Sort(device)
	return device
}

// TestDeviceGroupRatchet pins the device group on the live graph: the host
// optimizer core and the model inventory link no device runtime, so the
// packages that reach the device only through them leave the group, and the
// group cannot regrow past the measured ceiling.
func TestDeviceGroupRatchet(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	g := &gateContext{repo: root}
	graph, err := g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	device := devicePackages(graph)
	total := 0
	for _, node := range graph.nodes {
		if len(node.Match) != 0 && node.ForTest == "" {
			total++
		}
	}
	// Measured 2026-09-14: 151 of 257 packages before the two cuts, 106 of
	// 259 after (hostoptimizer and modeldevice added). A package joins the
	// group only by importing the device runtime on purpose.
	const ceiling = 106
	if len(device) > ceiling {
		t.Fatalf("device group regrew to %d of %d packages, ceiling %d", len(device), total, ceiling)
	}
	for _, host := range []string{"internal/hostoptimizer", "internal/model", "internal/trainingprogram", "internal/runrecord", "internal/plan", "internal/testevidence"} {
		if slices.Contains(device, host) {
			t.Errorf("%s links the device runtime", host)
		}
	}
	for _, required := range []string{"internal/optimizer", "internal/modeldevice", "internal/inference", "internal/cuda/executor"} {
		if !slices.Contains(device, required) {
			t.Errorf("%s no longer links the device runtime", required)
		}
	}
	t.Logf("device group: %d of %d packages", len(device), total)
}
