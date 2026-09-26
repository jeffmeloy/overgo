// code-manifest emits the canonical structural authority for a source tree.
package main

import (
	"bytes"
	"cmp"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/automationcheck"
	"overgo/internal/clioptions"
	"overgo/internal/codemanifest"
	"overgo/internal/codeprofile"
	"overgo/internal/gosource"
	"overgo/internal/repoanalysis"
)

func main() {
	clioptions.MainNamed("code-manifest", run)
}

func run() error {
	root := flag.String("root", ".", "repository root")
	basePath := flag.String("base", "", "canonical base manifest to compare")
	closure := flag.Bool("closure", false, "with -base: emit reverse-reachable impact instead of the raw delta")
	ownershipSurface := flag.Bool("ownership-surface", false, "with -base and -closure: emit the automation ownership surface")
	unconsumed := flag.String("unconsumed", "", "instead of the manifest, list the production declarations under these comma-separated package prefixes that no production code references and no dispatch boundary explains, with the private islands only such code reaches, costliest first")
	flag.Parse()
	resolved, err := filepath.Abs(*root)
	if err != nil {
		return err
	}
	snapshot, err := repoanalysis.DiscoverGo(resolved, "internal", "cmd")
	if err != nil {
		return err
	}
	selection, err := gosource.HostBuildSelection(resolved, "./internal/...", "./cmd/...")
	if err != nil {
		return err
	}
	if *unconsumed != "" {
		return listUnconsumed(snapshot, selection, strings.Split(*unconsumed, ","))
	}
	manifest, err := codemanifest.Generate(snapshot, []gosource.BuildSelection{selection}, nil)
	if err != nil {
		return err
	}
	if *basePath != "" {
		data, err := readManifestFile(*basePath)
		if err != nil {
			return err
		}
		base, err := codemanifest.Parse(bytes.TrimSuffix(data, []byte{'\n'}))
		if err != nil {
			return err
		}
		delta, err := codemanifest.Diff(base, manifest)
		if err != nil {
			return err
		}
		if !*closure {
			return json.NewEncoder(os.Stdout).Encode(delta)
		}
		impact, err := codemanifest.Close(base, manifest, delta)
		if err != nil {
			return err
		}
		if *ownershipSurface {
			return json.NewEncoder(os.Stdout).Encode(automationcheck.ManifestSurface(impact))
		}
		return json.NewEncoder(os.Stdout).Encode(impact)
	}
	return json.NewEncoder(os.Stdout).Encode(manifest)
}

// listUnconsumed prints the declarations under the prefixes that only tests
// or nothing reference, with their node counts, costliest first: the ranked
// paydown list for a surface that must be reduced or held.
func listUnconsumed(snapshot repoanalysis.SourceSnapshot, selection gosource.BuildSelection, prefixes []string) error {
	declarations, references, _, err := codeprofile.ProductionConsumerGraph(snapshot, selection)
	if err != nil {
		return err
	}
	// An island is private code whose only references come from code no live
	// root reaches: unconsumed through its callers, not by itself.
	islands := map[string]bool{}
	for _, island := range codeprofile.PrivateIslands(declarations, references) {
		islands[island.File+":"+island.Receiver+"."+island.Name] = true
	}
	profile, err := codeprofile.Build(snapshot)
	if err != nil {
		return err
	}
	nodes := map[string]int{}
	for _, function := range profile.Functions {
		nodes[function.File+":"+function.Name] = function.Nodes
	}
	type row struct {
		name, class string
		nodes       int
	}
	var rows []row
	total := 0
	for _, declaration := range declarations {
		island := islands[declaration.File+":"+declaration.Receiver+"."+declaration.Name]
		if (declaration.ProductionReferences > 0 || declaration.Boundary != "") && !island ||
			!slices.ContainsFunc(prefixes, func(prefix string) bool { return strings.HasPrefix(declaration.File, strings.TrimSpace(prefix)+"/") }) {
			continue
		}
		name := declaration.Name
		if declaration.Receiver != "" {
			name = declaration.Receiver + "." + name
		}
		class := "unreferenced"
		switch {
		case island && declaration.ProductionReferences > 0:
			class = "island"
		case declaration.TestReferences > 0:
			class = "test-only"
		}
		entry := row{name: declaration.File + ":" + name, class: class, nodes: nodes[declaration.File+":"+name]}
		rows = append(rows, entry)
		total += entry.nodes
	}
	slices.SortFunc(rows, func(left, right row) int {
		return cmp.Or(cmp.Compare(right.nodes, left.nodes), strings.Compare(left.name, right.name))
	})
	for _, entry := range rows {
		fmt.Printf("%6d %-12s %s\n", entry.nodes, entry.class, entry.name)
	}
	fmt.Printf("unconsumed=%d nodes=%d\n", len(rows), total)
	return nil
}

func readManifestFile(path string) ([]byte, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Size() <= 0 {
		return nil, errors.New("code-manifest: base must be a nonempty regular analysis file")
	}
	return os.ReadFile(path)
}
