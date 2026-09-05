package gate

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

type goModuleInput struct {
	GoMod string
}

type goPackageInput struct {
	ImportPath      string
	ForTest         string
	Match           []string
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
	root          string
	nodes         []goPackageInput
	byID          map[string][]int
	resourceFiles map[string][]string
}

// Presence tags frame the v2 cache input grammar independently of file bytes.
const (
	packageInputAbsent byte = iota
	packageInputPresent
)

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
	paths, err := gitLines(root, "ls-files", "-co", "--exclude-standard")
	if err != nil {
		return packageInputGraph{}, fmt.Errorf("derive repository inputs: %w", err)
	}
	graph.bindResourceFiles(paths)
	return graph, nil
}

// bindResourceFiles attaches undeclared repository assets to their nearest
// package. Compiler-declared inputs retain their production/test classification.
// This same ownership feeds selection and cache identity; ignored external data
// is outside the immutable source candidate and is not treated as source evidence.
func (graph *packageInputGraph) bindResourceFiles(paths []string) {
	if graph.resourceFiles == nil {
		graph.resourceFiles = map[string][]string{}
	}
	compiled := map[string]bool{}
	var directories []string
	for _, node := range graph.nodes {
		if len(node.Match) == 0 || node.ForTest != "" {
			continue
		}
		directories = append(directories, node.Dir)
		for _, name := range node.files() {
			compiled[filepath.Join(node.Dir, filepath.FromSlash(name))] = true
		}
	}
	for _, name := range paths {
		absolute := filepath.Join(graph.root, filepath.FromSlash(name))
		if strings.EqualFold(filepath.Ext(name), ".go") || compiled[absolute] {
			continue
		}
		owner := ""
		for _, directory := range directories {
			relative, err := filepath.Rel(directory, absolute)
			if err == nil && relative != ".." && !filepath.IsAbs(relative) && !strings.HasPrefix(relative, ".."+string(filepath.Separator)) && len(directory) > len(owner) {
				owner = directory
			}
		}
		if owner != "" && !slices.Contains(graph.resourceFiles[owner], absolute) {
			graph.resourceFiles[owner] = append(graph.resourceFiles[owner], absolute)
		}
	}
}

// dependentDirectories returns every repository package that is or transitively
// imports a package rooted in one of the declared repository directories.
func (graph packageInputGraph) dependentDirectories(roots ...string) ([]string, error) {
	owned := map[string]bool{}
	for _, node := range graph.nodes {
		relative, err := filepath.Rel(graph.root, node.Dir)
		if err != nil {
			return nil, err
		}
		relative = filepath.ToSlash(relative)
		for _, root := range roots {
			if relative == root || strings.HasPrefix(relative, root+"/") {
				owned[node.ImportPath] = true
				break
			}
		}
	}
	for changed := true; changed; {
		changed = false
		for _, node := range graph.nodes {
			if owned[node.ImportPath] {
				continue
			}
			for _, imported := range append(append(slices.Clone(node.Imports), node.TestImports...), node.XTestImports...) {
				if owned[imported] {
					owned[node.ImportPath] = true
					changed = true
					break
				}
			}
		}
	}
	directories := map[string]bool{}
	for _, node := range graph.nodes {
		if !owned[node.ImportPath] {
			continue
		}
		relative, err := filepath.Rel(graph.root, node.Dir)
		if err != nil {
			return nil, err
		}
		if relative == ".." || filepath.IsAbs(relative) || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
			continue
		}
		directories[filepath.ToSlash(relative)] = true
	}
	result := make([]string, 0, len(directories))
	for directory := range directories {
		result = append(result, directory)
	}
	slices.Sort(result)
	return result, nil
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
		for _, path := range graph.resourceFiles[node.Dir] {
			files[path] = true
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
	slices.Sort(paths)
	hasher := sha256.New()
	hasher.Write([]byte("go-test-inputs/v2\x00"))
	hasher.Write([]byte(target))
	hasher.Write([]byte("\x00"))
	for _, path := range paths {
		content, err := os.ReadFile(path)
		hasher.Write([]byte(filepath.ToSlash(path)))
		hasher.Write([]byte("\x00"))
		if errors.Is(err, os.ErrNotExist) {
			hasher.Write([]byte{packageInputAbsent})
			continue
		}
		if err != nil {
			return artifact.ID{}, fmt.Errorf("package input identity %s: %w", target, err)
		}
		hasher.Write([]byte{packageInputPresent})
		digest := sha256.Sum256(content)
		hasher.Write(digest[:]) // Fixed-width framing prevents cross-file ambiguity.
	}
	return artifact.IdentifyBytes(artifact.KindEvidence, hasher.Sum(nil))
}

func (node goPackageInput) productionFiles() []string {
	var files []string
	for _, group := range [][]string{
		node.GoFiles, node.CgoFiles, node.CFiles, node.CXXFiles, node.MFiles,
		node.HFiles, node.FFiles, node.SFiles, node.SwigFiles, node.SwigCXXFiles,
		node.SysoFiles, node.EmbedFiles,
	} {
		files = append(files, group...)
	}
	return files
}

func (node goPackageInput) files() []string {
	files := node.productionFiles()
	for _, group := range [][]string{node.TestGoFiles, node.XTestGoFiles, node.TestEmbedFiles, node.XTestEmbedFiles} {
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
