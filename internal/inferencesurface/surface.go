// Package inferencesurface owns what the inference code surface is: the
// sources whose change can move a rate or a long-context read, and so expire
// validation evidence keyed to them. It imports no part of that surface, so a
// tool that only needs to know whether a change moves it -- the landing driver,
// the validation planner -- does not link the device stack to find out.
package inferencesurface

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	"overgo/internal/processcontrol"
)

// surfaceRoots are the packages whose dependency closure is the
// inference code surface: the runner, the CUDA executor, and the recipe
// resolution that binds a model to them. Every Go file the closure
// compiles is part of the surface.
var surfaceRoots = []string{"./internal/inference", "./internal/cuda/executor", "./internal/modelrecipe"}

// surfaceKernels are the kernel sources the closure does not import
// but the executor runs: the manifest that names them and the CUDA
// sources they are built from.
var surfaceKernels = []string{"kernels/manifest.json", "kernels/cuda"}

// identityAddressed are the packages the runner reaches only to resolve and
// record artifacts by content identity: the store and the run records. A
// change to them cannot alter the bytes a decode computes over, which the
// record pins by identity, so the surface does not descend into them. They
// were half of what expired text evidence: of 67 surface moves in 400
// commits, 21 touched nothing else.
var identityAddressed = []string{"internal/overgodb", "internal/runrecord"}

// Digest digests the inference code surface of the tree at root: the
// non-test Go sources and embedded files of every module-local package in
// the closure of surfaceRoots, the kernel sources, and the version of every
// other module the closure compiles. A long-form record keyed to
// the surface survives commits that cannot move a rate or a long-context
// read, and expires on any that could; HEAD alone expired every record
// on the other lane's documentation commits.
func Digest(ctx context.Context, root string) (string, error) {
	// The Go tool reports absolute directories; the root is made
	// absolute so every file resolves relative to it.
	root, err := filepath.Abs(root)
	if err != nil {
		return "", err
	}
	surfaceMu.Lock()
	defer surfaceMu.Unlock()
	if entry, memoized := surfaceMemo[root]; memoized {
		return entry.digest, entry.err
	}
	digest, err := computeSurface(ctx, root)
	surfaceMemo[root] = surfaceEntry{digest: digest, err: err}
	return digest, err
}

// surfaceEntry memoizes one root's digest for the process: the sources
// do not change under a running pass, and the digest is read once per
// model.
type surfaceEntry struct {
	digest string
	err    error
}

var (
	surfaceMu   sync.Mutex
	surfaceMemo = map[string]surfaceEntry{}
)

// computeSurface digests the closure's sources, the files its packages
// embed -- the architecture catalog, the runtime policies, the compiled
// kernels, read at run time though they are not Go -- and the version of
// every other module the closure compiles, such as the tokenizer's regexp
// engine: a change to any of them can move a rate or a read.
func computeSurface(ctx context.Context, root string) (string, error) {
	packages, modules, err := closure(ctx, root)
	if err != nil {
		return "", err
	}
	var files []string
	for _, pkg := range packages {
		sources, err := packageFiles(pkg)
		if err != nil {
			return "", err
		}
		files = append(files, sources...)
	}
	kernels, err := kernelFiles(root)
	if err != nil {
		return "", err
	}
	return digestFiles(root, append(files, kernels...), modules...)
}

// packageFiles are the files of one package a build of it reads: its non-test
// Go sources, generated ones among them, and the files it embeds.
func packageFiles(pkg Package) ([]string, error) {
	entries, err := os.ReadDir(pkg.Dir)
	if err != nil {
		return nil, err
	}
	var files []string
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		files = append(files, filepath.Join(pkg.Dir, name))
	}
	for _, embedded := range pkg.EmbedFiles {
		files = append(files, filepath.Join(pkg.Dir, embedded))
	}
	return files, nil
}

// kernelFiles are the kernel sources of the tree at root: the manifest and
// the CUDA sources beside it.
func kernelFiles(root string) ([]string, error) {
	var files []string
	for _, kernel := range surfaceKernels {
		path := filepath.Join(root, kernel)
		info, err := os.Stat(path)
		if err != nil {
			return nil, err
		}
		if !info.IsDir() {
			files = append(files, path)
			continue
		}
		entries, err := os.ReadDir(path)
		if err != nil {
			return nil, err
		}
		for _, entry := range entries {
			if !entry.IsDir() {
				files = append(files, filepath.Join(path, entry.Name()))
			}
		}
	}
	return files, nil
}

// Package is one module-local package of the inference closure: its directory
// and the files it embeds, which are runtime inputs though they are not source,
// with the patterns that name them.
type Package struct {
	Dir           string
	ImportPath    string
	Imports       []string
	EmbedPatterns []string
	EmbedFiles    []string
	Module        *struct{ Path, Version string }
}

// Packages lists the module-local packages in the closure of the surface roots,
// through the Go tool's own dependency resolution, without descending into
// the identity-addressed packages.
func Packages(ctx context.Context, root string) ([]Package, error) {
	packages, _, err := closure(ctx, root)
	return packages, err
}

// closure walks the dependency closure of the surface roots, without
// descending into the identity-addressed packages: it returns the
// module-local packages and, as module@version, every other module the walk
// reaches. The standard library belongs to the toolchain and is not listed.
func closure(ctx context.Context, root string) ([]Package, []string, error) {
	listed, module, err := list(ctx, root, surfaceRoots...)
	if err != nil {
		return nil, nil, err
	}
	var roots []string
	for _, root := range surfaceRoots {
		roots = append(roots, module+strings.TrimPrefix(root, "."))
	}
	packages, modules := reach(listed, module, roots...)
	if len(packages) == 0 {
		return nil, nil, errors.New("longform: the inference surface lists no module-local package")
	}
	return packages, modules, nil
}

// list asks the Go tool for every package the roots depend on, themselves
// among them, by import path, with the path of the module at root. The
// standard library belongs to the toolchain and is not listed.
func list(ctx context.Context, root string, roots ...string) (map[string]Package, string, error) {
	arguments := append([]string{"list", "-deps", "-json"}, roots...)
	var stdout, stderr bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: "go", Args: arguments, Dir: root, Env: os.Environ(), Stdout: &stdout, Stderr: &stderr,
	})
	if err != nil {
		return nil, "", fmt.Errorf("inference surface: go list: %w: %s", err, strings.TrimSpace(stderr.String()))
	}
	if receipt.ExitCode != 0 {
		return nil, "", fmt.Errorf("inference surface: go list: exit=%d: %s", receipt.ExitCode, strings.TrimSpace(stderr.String()))
	}
	module, err := modulePath(root)
	if err != nil {
		return nil, "", err
	}
	listed := map[string]Package{}
	decoder := json.NewDecoder(&stdout)
	for {
		var pkg Package
		if err := decoder.Decode(&pkg); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return nil, "", err
		}
		if pkg.Module != nil {
			listed[pkg.ImportPath] = pkg
		}
	}
	return listed, module, nil
}

// reach walks the listed packages from the roots, by import path, without
// descending into the identity-addressed packages: it returns the packages of
// module and, as module@version, every other module the walk reaches.
func reach(listed map[string]Package, module string, roots ...string) ([]Package, []string) {
	pruned := map[string]bool{}
	for _, name := range identityAddressed {
		pruned[module+"/"+name] = true
	}
	var packages []Package
	var modules []string
	reached := map[string]bool{}
	queue := slices.Clone(roots)
	for len(queue) > 0 {
		path := queue[0]
		queue = queue[1:]
		pkg, found := listed[path]
		if !found || reached[path] || pruned[path] {
			continue
		}
		reached[path] = true
		if pkg.Module.Path == module {
			packages = append(packages, pkg)
		} else {
			modules = append(modules, pkg.Module.Path+"@"+pkg.Module.Version)
		}
		queue = append(queue, pkg.Imports...)
	}
	slices.Sort(modules)
	return packages, slices.Compact(modules)
}

// modulePath reads the module path from the root's go.mod.
func modulePath(root string) (string, error) {
	data, err := os.ReadFile(filepath.Join(root, "go.mod"))
	if err != nil {
		return "", err
	}
	for line := range strings.SplitSeq(string(data), "\n") {
		if path, found := strings.CutPrefix(strings.TrimSpace(line), "module "); found {
			return strings.TrimSpace(path), nil
		}
	}
	return "", errors.New("longform: go.mod names no module")
}

// digestFiles hashes the files in path order, each as its root-relative
// slash path and its content digest, then the module@version of each other
// module, so the digest is the same on every host that holds the same
// sources and resolves the same modules.
func digestFiles(root string, files []string, modules ...string) (string, error) {
	digest, _, err := sources{}.digest(root, files, modules, "")
	return digest, err
}
