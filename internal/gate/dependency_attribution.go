package gate

import (
	"path/filepath"
	"slices"

	"overgo/internal/runrecord"
)

// attributeInputs fills the shared selection record's input fields.
// Execution and receipt facts are bound separately; this grants no exclusion.
func (graph packageInputGraph) attributeInputs(target string, changed []string) (runrecord.SelectionPackage, error) {
	attribution := runrecord.SelectionPackage{Package: target, RuntimeReaders: map[string]string{}}
	compiled, err := graph.inputNodes(target, false)
	if err != nil {
		return attribution, err
	}
	inputs, err := graph.inputFiles(target)
	if err != nil {
		return attribution, err
	}
	compilerFiles := map[string]bool{}
	for index, tests := range compiled {
		node := graph.nodes[index]
		names := node.productionFiles()
		if tests {
			names = node.files()
		}
		for _, name := range names {
			compilerFiles[filepath.Join(node.Dir, filepath.FromSlash(name))] = true
		}
		if node.Module != nil && node.Module.GoMod != "" {
			compilerFiles[node.Module.GoMod] = true
		}
		if node.opaqueReader || tests && node.testOpaque {
			attribution.RuntimeReaders[node.ImportPath] = node.runtimeReason
		}
	}
	for _, name := range []string{"go.mod", "go.sum"} {
		compilerFiles[filepath.Join(graph.root, name)] = true
	}
	for _, name := range changed {
		path := filepath.Clean(filepath.Join(graph.root, filepath.FromSlash(name)))
		switch {
		case !inputs[path]:
			attribution.UnboundInputs = append(attribution.UnboundInputs, name)
		case compilerFiles[path]:
			attribution.CompilerInputs = append(attribution.CompilerInputs, name)
		default:
			attribution.RuntimeInputs = append(attribution.RuntimeInputs, name)
		}
	}
	for _, names := range []*[]string{&attribution.CompilerInputs, &attribution.RuntimeInputs, &attribution.UnboundInputs} {
		slices.Sort(*names)
		*names = slices.Compact(*names)
	}
	return attribution, nil
}
