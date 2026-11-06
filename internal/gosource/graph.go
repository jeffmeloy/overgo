package gosource

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// Package is one package of a package graph as go list reports it: the fields the code identities, the cycle's
// trigger and the lanes' reach read of it.
type Package struct {
	ImportPath, Dir                                              string
	Imports, Deps                                                []string
	GoFiles, CgoFiles, TestGoFiles, XTestGoFiles, IgnoredGoFiles []string
	EmbedPatterns, EmbedFiles                                    []string
	Module                                                       *struct{ Path, Version string }
	Error                                                        *struct{ Err string }
	DepsErrors                                                   []struct{ Err string }
}

// ListGraph is the package graph one go list of the checkout at root gives, in its order: the packages the arguments
// name and, with deps, every package they import, where a package or dependency that cannot load refuses the graph.
// Every code identity is a query over one: the inference surface's digest and its execution identities, the cycle's
// trigger and the lanes' reach (HostBuildClosure).
func ListGraph(ctx context.Context, root string, deps bool, arguments ...string) ([]Package, error) {
	flags := []string{"list", "-e", "-json"}
	if deps {
		flags = append(flags, "-deps")
	}
	output, err := goOutput(ctx, root, append(flags, arguments...)...)
	if err != nil {
		return nil, err
	}
	var graph []Package
	decoder := json.NewDecoder(bytes.NewReader(output))
	for {
		var pkg Package
		if err := decoder.Decode(&pkg); errors.Is(err, io.EOF) {
			return graph, nil
		} else if err != nil {
			return nil, err
		}
		if deps && pkg.Error != nil {
			return nil, fmt.Errorf("dependency source selection: %s: %s", pkg.ImportPath, pkg.Error.Err)
		}
		if deps && len(pkg.DepsErrors) != 0 {
			return nil, fmt.Errorf("dependency source selection: %s: %s", pkg.ImportPath, pkg.DepsErrors[0].Err)
		}
		graph = append(graph, pkg)
	}
}
