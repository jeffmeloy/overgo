package gate

import (
	"maps"
	"path/filepath"
	"slices"
	"sync"
	"testing"
)

// One graph per test process. Source is fixed for a verification run; no
// graph or file-content cache survives into another run or candidate.
var livePackageGraph = sync.OnceValues(func() (packageInputGraph, error) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		return packageInputGraph{}, err
	}
	return loadPackageInputGraph(root)
})

// liveGateContext shares discovery, not mutable scope or evidence state.
func liveGateContext(t testing.TB, paths ...string) *gateContext {
	t.Helper()
	graph, err := livePackageGraph()
	if err != nil {
		t.Fatal(err)
	}
	graph = graph.clone()
	return &gateContext{repo: graph.root, paths: paths, packageGraph: &graph}
}

// clone isolates the fields bindResourceFiles mutates for each path set.
func (graph packageInputGraph) clone() packageInputGraph {
	graph.nodes = slices.Clone(graph.nodes)
	for i := range graph.nodes {
		node := &graph.nodes[i]
		node.inputDependencies = slices.Clone(node.inputDependencies)
		node.testInputDependencies = slices.Clone(node.testInputDependencies)
		node.executionDependencies = slices.Clone(node.executionDependencies)
	}
	graph.resourceFiles = cloneSliceMap(graph.resourceFiles)
	graph.testResourceFiles = cloneSliceMap(graph.testResourceFiles)
	graph.runtimeResourceFiles = cloneSliceMap(graph.runtimeResourceFiles)
	graph.sourceDirectories = maps.Clone(graph.sourceDirectories)
	// Identity batches own their file bytes; reuse discovery only.
	graph.fileInputs = nil
	return graph
}

func cloneSliceMap[V any](values map[string][]V) map[string][]V {
	cloned := maps.Clone(values)
	for key, value := range cloned {
		cloned[key] = slices.Clone(value)
	}
	return cloned
}
