// Package inferencesurface owns what the inference code surface is: the
// sources whose change can move a rate or a long-context read, and so expire
// validation evidence keyed to them. It imports no part of that surface, so a
// tool that only needs to know whether a change moves it -- the landing driver,
// the validation planner -- does not link the device stack to find out.
package inferencesurface

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
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
		entries, err := os.ReadDir(pkg.Dir)
		if err != nil {
			return "", err
		}
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
	}
	for _, kernel := range surfaceKernels {
		path := filepath.Join(root, kernel)
		info, err := os.Stat(path)
		if err != nil {
			return "", err
		}
		if !info.IsDir() {
			files = append(files, path)
			continue
		}
		entries, err := os.ReadDir(path)
		if err != nil {
			return "", err
		}
		for _, entry := range entries {
			if !entry.IsDir() {
				files = append(files, filepath.Join(path, entry.Name()))
			}
		}
	}
	return digestFiles(root, files, modules...)
}

// Moves returns, of the repository-relative paths a change touches, the
// ones that move the inference surface: a non-test Go source in a package of
// the closure, a file one of them embeds, a kernel source, or go.mod, which
// pins the versions of the other modules the closure compiles. A record keyed
// to the surface expires when one of them changes, so a landing can name what
// it is about to expire before it commits rather than learn it from a review
// candidate afterwards. A path in the closure counts whether the change adds,
// edits or removes it.
func Moves(ctx context.Context, root string, paths []string) ([]string, error) {
	root, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	packages, _, err := closure(ctx, root)
	if err != nil {
		return nil, err
	}
	directories := map[string]bool{}
	embedded := map[string]bool{}
	for _, pkg := range packages {
		directory, err := filepath.Rel(root, pkg.Dir)
		if err != nil {
			return nil, err
		}
		directories[surfaceKey(directory)] = true
		for _, name := range pkg.EmbedFiles {
			embedded[surfaceKey(filepath.Join(directory, name))] = true
		}
	}
	var moves []string
	for _, changed := range paths {
		key := surfaceKey(changed)
		if key == "." || filepath.IsAbs(changed) || key == ".." || strings.HasPrefix(key, "../") || strings.Contains(key, ":") {
			return nil, fmt.Errorf("inference surface: changed path %q is not repository-relative", changed)
		}
		kernel := slices.ContainsFunc(surfaceKernels, func(source string) bool {
			return key == surfaceKey(source) || strings.HasPrefix(key, surfaceKey(source)+"/")
		})
		source := strings.HasSuffix(key, ".go") && !strings.HasSuffix(key, "_test.go") && directories[surfaceKey(filepath.Dir(key))]
		if kernel || source || embedded[key] || key == surfaceKey("go.mod") {
			moves = append(moves, filepath.ToSlash(changed))
		}
	}
	slices.Sort(moves)
	return slices.Compact(moves), nil
}

// surfaceKey is the one form a path is compared in: slash-separated and
// clean, either separator accepted, and case-folded where the filesystem
// folds case, so a spelling cannot evade the surface.
func surfaceKey(path string) string {
	key := filepath.ToSlash(filepath.Clean(strings.ReplaceAll(path, `\`, "/")))
	if runtime.GOOS == "windows" {
		key = strings.ToLower(key)
	}
	return key
}

// Package is one module-local package of the inference closure: its directory
// and the files it embeds, which are runtime inputs though they are not source.
type Package struct {
	Dir        string
	ImportPath string
	Imports    []string
	EmbedFiles []string
	Module     *struct{ Path, Version string }
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
	arguments := append([]string{"list", "-deps", "-json"}, surfaceRoots...)
	var stdout, stderr bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: "go", Args: arguments, Dir: root, Env: os.Environ(), Stdout: &stdout, Stderr: &stderr,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("inference surface: go list: %w: %s", err, strings.TrimSpace(stderr.String()))
	}
	if receipt.ExitCode != 0 {
		return nil, nil, fmt.Errorf("inference surface: go list: exit=%d: %s", receipt.ExitCode, strings.TrimSpace(stderr.String()))
	}
	modulePath, err := modulePath(root)
	if err != nil {
		return nil, nil, err
	}
	listed := map[string]Package{}
	decoder := json.NewDecoder(&stdout)
	for {
		var pkg Package
		if err := decoder.Decode(&pkg); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return nil, nil, err
		}
		if pkg.Module != nil {
			listed[pkg.ImportPath] = pkg
		}
	}
	pruned := map[string]bool{}
	for _, name := range identityAddressed {
		pruned[modulePath+"/"+name] = true
	}
	var packages []Package
	var modules []string
	reached := map[string]bool{}
	var queue []string
	for _, root := range surfaceRoots {
		queue = append(queue, modulePath+strings.TrimPrefix(root, "."))
	}
	for len(queue) > 0 {
		path := queue[0]
		queue = queue[1:]
		pkg, found := listed[path]
		if !found || reached[path] || pruned[path] {
			continue
		}
		reached[path] = true
		if pkg.Module.Path == modulePath {
			packages = append(packages, pkg)
		} else {
			modules = append(modules, pkg.Module.Path+"@"+pkg.Module.Version)
		}
		queue = append(queue, pkg.Imports...)
	}
	if len(packages) == 0 {
		return nil, nil, errors.New("longform: the inference surface lists no module-local package")
	}
	slices.Sort(modules)
	return packages, slices.Compact(modules), nil
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
	slices.Sort(files)
	surface := sha256.New()
	for _, file := range files {
		relative, err := filepath.Rel(root, file)
		if err != nil {
			return "", err
		}
		content := sha256.New()
		handle, err := os.Open(file)
		if err != nil {
			return "", err
		}
		_, copyErr := io.Copy(content, handle)
		closeErr := handle.Close()
		if copyErr != nil || closeErr != nil {
			return "", errors.Join(copyErr, closeErr)
		}
		fmt.Fprintf(surface, "%s\n%x\n", filepath.ToSlash(relative), content.Sum(nil))
	}
	for _, module := range modules {
		fmt.Fprintf(surface, "module %s\n", module)
	}
	return hex.EncodeToString(surface.Sum(nil)), nil
}
