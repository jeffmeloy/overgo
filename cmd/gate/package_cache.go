package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"

	"overgo/internal/artifact"
)

type goModuleInput struct {
	GoMod string
}

type goPackageInput struct {
	ImportPath      string
	ForTest         string
	Dir             string
	Standard        bool
	Module          *goModuleInput
	Imports         []string
	TestImports     []string
	XTestImports    []string
	GoFiles         []string
	CgoFiles        []string
	CFiles          []string
	CXXFiles        []string
	MFiles          []string
	HFiles          []string
	FFiles          []string
	SFiles          []string
	SwigFiles       []string
	SwigCXXFiles    []string
	SysoFiles       []string
	EmbedFiles      []string
	TestGoFiles     []string
	XTestGoFiles    []string
	TestEmbedFiles  []string
	XTestEmbedFiles []string
}

type packageInputGraph struct {
	root  string
	nodes []goPackageInput
	byID  map[string][]int
}

func loadPackageInputGraph(root string) (packageInputGraph, error) {
	out, err := command(root, "go", "list", "-deps", "-test", "-json", "./...")
	if err != nil {
		return packageInputGraph{}, err
	}
	decoder := json.NewDecoder(bytes.NewBufferString(out))
	graph := packageInputGraph{root: root, byID: map[string][]int{}}
	for {
		var node goPackageInput
		if err := decoder.Decode(&node); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return packageInputGraph{}, fmt.Errorf("decode package inputs: %w", err)
		}
		index := len(graph.nodes)
		graph.nodes = append(graph.nodes, node)
		graph.byID[node.ImportPath] = append(graph.byID[node.ImportPath], index)
	}
	return graph, nil
}

func (graph packageInputGraph) identity(target string) (artifact.ID, error) {
	queue := append([]int(nil), graph.byID[target]...)
	for index, node := range graph.nodes {
		if node.ForTest == target {
			queue = append(queue, index)
		}
	}
	if len(queue) == 0 {
		return artifact.ID{}, fmt.Errorf("package input identity: package %q is absent", target)
	}
	rootNodes := make(map[int]bool, len(queue))
	for _, index := range queue {
		rootNodes[index] = true
	}
	seen := map[int]bool{}
	var visit func(int)
	visit = func(index int) {
		if seen[index] {
			return
		}
		seen[index] = true
		imports := append([]string(nil), graph.nodes[index].Imports...)
		if rootNodes[index] {
			imports = append(imports, graph.nodes[index].TestImports...)
			imports = append(imports, graph.nodes[index].XTestImports...)
		}
		for _, imported := range imports {
			for _, dependency := range graph.byID[imported] {
				visit(dependency)
			}
		}
	}
	for _, index := range queue {
		visit(index)
	}
	files := map[string]bool{}
	for index := range seen {
		node := graph.nodes[index]
		for _, name := range node.files() {
			files[filepath.Join(node.Dir, filepath.FromSlash(name))] = true
		}
		if node.Module != nil && node.Module.GoMod != "" {
			files[node.Module.GoMod] = true
		}
	}
	for _, name := range []string{"go.mod", "go.sum"} {
		path := filepath.Join(graph.root, name)
		if _, err := os.Stat(path); err == nil {
			files[path] = true
		} else if !errors.Is(err, os.ErrNotExist) {
			return artifact.ID{}, err
		}
	}
	var paths []string
	for path := range files {
		paths = append(paths, filepath.Clean(path))
	}
	sort.Strings(paths)
	hasher := sha256.New()
	hasher.Write([]byte(target))
	hasher.Write([]byte("\x00"))
	for _, path := range paths {
		content, err := os.ReadFile(path)
		if err != nil {
			return artifact.ID{}, fmt.Errorf("package input identity %s: %w", target, err)
		}
		hasher.Write([]byte(filepath.ToSlash(path)))
		hasher.Write([]byte("\x00"))
		hasher.Write(content)
		hasher.Write([]byte("\x00"))
	}
	return artifact.IdentifyBytes(artifact.KindEvidence, hasher.Sum(nil))
}

func (node goPackageInput) files() []string {
	var files []string
	for _, group := range [][]string{
		node.GoFiles, node.CgoFiles, node.CFiles, node.CXXFiles, node.MFiles,
		node.HFiles, node.FFiles, node.SFiles, node.SwigFiles, node.SwigCXXFiles,
		node.SysoFiles, node.EmbedFiles, node.TestGoFiles, node.XTestGoFiles,
		node.TestEmbedFiles, node.XTestEmbedFiles,
	} {
		files = append(files, group...)
	}
	return files
}

func packageInputIdentities(graph packageInputGraph, packages []string) (map[string]artifact.ID, error) {
	identities := make(map[string]artifact.ID, len(packages))
	for _, packagePath := range packages {
		identity, err := graph.identity(packagePath)
		if err != nil {
			return nil, err
		}
		identities[packagePath] = identity
	}
	return identities, nil
}
