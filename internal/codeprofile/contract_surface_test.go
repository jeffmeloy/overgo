package codeprofile

import (
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/gosource"
	"overgo/internal/repoanalysis"
)

func TestOwnedPackagesHaveNoUnusedExports(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := repoanalysis.DiscoverGo(root, "cmd", "internal")
	if err != nil {
		t.Fatal(err)
	}
	selection, err := gosource.HostBuildSelection(root, "./...")
	if err != nil {
		t.Fatal(err)
	}
	declarations, _, err := ProductionConsumerCensus(snapshot, selection, nil)
	if err != nil {
		t.Fatal(err)
	}
	prefixes := []string{
		"overgo/internal/artifact", "overgo/internal/dataset", "overgo/internal/recipe",
		"overgo/internal/modelrecipe", "overgo/internal/runrecord", "overgo/internal/evaluation",
		"overgo/internal/tensor", "overgo/internal/cuda", "overgo/internal/graphruntime",
		"overgo/internal/inference", "overgo/internal/projector", "overgo/internal/server",
	}
	for _, declaration := range declarations {
		if !declaration.Exported || declaration.Boundary != "" ||
			declaration.ProductionReferences != 0 || declaration.TestReferences != 0 ||
			declaration.ExternalReferences != 0 {
			continue
		}
		for _, prefix := range prefixes {
			if declaration.Package == prefix || strings.HasPrefix(declaration.Package, prefix+"/") {
				t.Errorf("package-local export %s.%s in %s", declaration.Package, declaration.Name, declaration.File)
				break
			}
		}
	}
}
