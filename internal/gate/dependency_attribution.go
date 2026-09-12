package gate

import (
	"path/filepath"
	"slices"

	"overgo/internal/artifact"
)

// packageInputAttribution explains package-level binding, not function reach.
// A runtime-only input is a candidate for investigation, not an exclusion.
type packageInputAttribution struct {
	Package        string            `json:"package"`
	Input          artifact.ID       `json:"input,omitzero"`
	CompilerInputs []string          `json:"compiler_inputs"`
	RuntimeInputs  []string          `json:"runtime_inputs"`
	UnboundInputs  []string          `json:"unbound_inputs"`
	RuntimeReaders map[string]string `json:"runtime_readers"`
}

func (graph packageInputGraph) attributeInputs(target string, changed []string) (packageInputAttribution, error) {
	attribution := packageInputAttribution{Package: target, RuntimeReaders: map[string]string{}}
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
