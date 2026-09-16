package gate

import (
	"maps"
	"slices"
)

// clone gives one live context its own graph: deriveTestScope rebinds
// resource files into the graph's maps and node slices, and the parallel
// acceptances that share the checkout must not write the same backing
// arrays.
func (graph packageInputGraph) clone() packageInputGraph {
	graph.nodes = slices.Clone(graph.nodes)
	for index := range graph.nodes {
		node := &graph.nodes[index]
		node.inputDependencies = slices.Clone(node.inputDependencies)
		node.testInputDependencies = slices.Clone(node.testInputDependencies)
		node.executionDependencies = slices.Clone(node.executionDependencies)
	}
	graph.byID = cloneSliceMap(graph.byID)
	graph.resourceFiles = cloneSliceMap(graph.resourceFiles)
	graph.testResourceFiles = cloneSliceMap(graph.testResourceFiles)
	graph.runtimeResourceFiles = cloneSliceMap(graph.runtimeResourceFiles)
	graph.sourceDirectories = maps.Clone(graph.sourceDirectories)
	// Share discovery; each candidate reads its own current file bytes.
	graph.fileInputs = nil
	return graph
}

func cloneSliceMap[V any](values map[string][]V) map[string][]V {
	if values == nil {
		return nil
	}
	cloned := make(map[string][]V, len(values))
	for key, value := range values {
		cloned[key] = slices.Clone(value)
	}
	return cloned
}
