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

// Digest digests the inference code surface of the tree at root: the
// non-test Go sources of every module-local package in the closure of
// surfaceRoots, plus the kernel sources. A long-form record keyed to
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

func computeSurface(ctx context.Context, root string) (string, error) {
	directories, err := surfaceDirectories(ctx, root)
	if err != nil {
		return "", err
	}
	var files []string
	for _, directory := range directories {
		entries, err := os.ReadDir(directory)
		if err != nil {
			return "", err
		}
		for _, entry := range entries {
			name := entry.Name()
			if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
				continue
			}
			files = append(files, filepath.Join(directory, name))
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
	return digestFiles(root, files)
}

// surfaceDirectories lists the module-local package directories in the
// closure of the surface roots, through the Go tool's own dependency
// resolution so the surface follows the imports rather than a list
// that would go stale.
func surfaceDirectories(ctx context.Context, root string) ([]string, error) {
	packages, err := Packages(ctx, root)
	if err != nil {
		return nil, err
	}
	var directories []string
	for _, pkg := range packages {
		directories = append(directories, pkg.Dir)
	}
	slices.Sort(directories)
	return slices.Compact(directories), nil
}

// Moves returns, of the repository-relative paths a change touches, the
// ones that move the inference surface: a non-test Go source in a package of
// the closure, or a kernel source. A record keyed to the surface expires when
// one of them changes, so a landing can name what it is about to expire before
// it commits rather than learn it from a review candidate afterwards. A path in
// the closure counts whether the change adds, edits or removes it.
func Moves(ctx context.Context, root string, paths []string) ([]string, error) {
	root, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	directories, err := surfaceDirectories(ctx, root)
	if err != nil {
		return nil, err
	}
	var moves []string
	for _, changed := range paths {
		changed = filepath.ToSlash(changed)
		kernel := slices.ContainsFunc(surfaceKernels, func(source string) bool {
			return changed == source || strings.HasPrefix(changed, source+"/")
		})
		source := strings.HasSuffix(changed, ".go") && !strings.HasSuffix(changed, "_test.go") &&
			slices.Contains(directories, filepath.Join(root, filepath.FromSlash(filepath.ToSlash(filepath.Dir(changed)))))
		if kernel || source {
			moves = append(moves, changed)
		}
	}
	slices.Sort(moves)
	return slices.Compact(moves), nil
}

// Package is one module-local package of the inference closure: its directory
// and the files it embeds, which are runtime inputs though they are not source.
type Package struct {
	Dir        string
	EmbedFiles []string
	Module     *struct{ Path string }
}

// Packages lists the module-local packages in the closure of the surface roots,
// through the Go tool's own dependency resolution.
func Packages(ctx context.Context, root string) ([]Package, error) {
	arguments := append([]string{"list", "-deps", "-json"}, surfaceRoots...)
	var stdout, stderr bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: "go", Args: arguments, Dir: root, Env: os.Environ(), Stdout: &stdout, Stderr: &stderr,
	})
	if err != nil {
		return nil, fmt.Errorf("inference surface: go list: %w: %s", err, strings.TrimSpace(stderr.String()))
	}
	if receipt.ExitCode != 0 {
		return nil, fmt.Errorf("inference surface: go list: exit=%d: %s", receipt.ExitCode, strings.TrimSpace(stderr.String()))
	}
	modulePath, err := modulePath(root)
	if err != nil {
		return nil, err
	}
	var packages []Package
	decoder := json.NewDecoder(&stdout)
	for {
		var pkg Package
		if err := decoder.Decode(&pkg); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return nil, err
		}
		if pkg.Module == nil || pkg.Module.Path != modulePath {
			continue
		}
		packages = append(packages, pkg)
	}
	if len(packages) == 0 {
		return nil, errors.New("longform: the inference surface lists no module-local package")
	}
	return packages, nil
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
// slash path and its content digest, so the digest is the same on every
// host that holds the same sources.
func digestFiles(root string, files []string) (string, error) {
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
	return hex.EncodeToString(surface.Sum(nil)), nil
}
