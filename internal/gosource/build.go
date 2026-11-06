// Package gosource selects and fingerprints producer inputs without AST policy.
package gosource

import (
	"bytes"
	"context"
	"fmt"
	"path/filepath"
	"strings"

	"overgo/internal/processcontrol"
)

// BuildSelection is the go command's host-context decision for repository Go
// files. Presence distinguishes a selected file from an unrelated file that
// go list did not classify.
type BuildSelection struct {
	Context string
	Root    string
	Files   map[string]bool
	// Packages maps repository files to the import path selected by go list.
	// It lets AST consumers resolve import aliases without rediscovering module
	// layout or maintaining package-path tables.
	Packages map[string]string
	// Dependencies is the toolchain's transitive import closure per package.
	Dependencies map[string][]string
	// BuildDependencies is each package's own import closure, as its binary builds it, without what a test
	// variant of it imports for its tests.
	BuildDependencies map[string][]string
}

// HostBuildSelection derives file membership through the Go toolchain so build
// tags, legacy tags, filename constraints, cgo, tests, and external tests share
// one repository owner: the files of the packages patterns name, their tests and
// the files their constraints leave out.
func HostBuildSelection(root string, patterns ...string) (BuildSelection, error) {
	return hostBuild(root, false, patterns)
}

// HostBuildClosure is HostBuildSelection over the closure of the packages patterns
// name: the compiled repository Go sources of each and of everything it imports;
// -test adds the requested test variants. Module, native and runtime inputs are
// separate.
func HostBuildClosure(root string, patterns ...string) (BuildSelection, error) {
	return hostBuild(root, true, patterns)
}

// hostBuild selects the files of the package graph ListGraph gives for patterns.
func hostBuild(root string, dependencies bool, patterns []string) (BuildSelection, error) {
	root, err := filepath.Abs(root)
	if err != nil {
		return BuildSelection{}, err
	}
	graph, err := ListGraph(context.Background(), root, dependencies, patterns...)
	if err != nil {
		return BuildSelection{}, err
	}
	selection := BuildSelection{
		Root: root, Files: map[string]bool{}, Packages: map[string]string{}, Dependencies: map[string][]string{}, BuildDependencies: map[string][]string{},
	}
	if target, err := goOutput(context.Background(), root, "env", "GOOS", "GOARCH"); err == nil {
		selection.Context = strings.Join(strings.Fields(string(target)), "/")
	}
	for _, pkg := range graph {
		if dependencies {
			var variant bool
			pkg.ImportPath, _, variant = strings.Cut(pkg.ImportPath, " [")
			if !variant {
				if relative, err := filepath.Rel(root, pkg.Dir); err == nil && filepath.IsLocal(relative) {
					selection.BuildDependencies[pkg.ImportPath] = pkg.Deps
				}
			}
			relative, err := filepath.Rel(root, pkg.Dir)
			if err != nil || !filepath.IsLocal(relative) {
				continue
			}
		}
		groups := [][]string{pkg.GoFiles, pkg.CgoFiles}
		if relative, err := filepath.Rel(root, pkg.Dir); err == nil && filepath.IsLocal(relative) {
			selection.Dependencies[pkg.ImportPath] = pkg.Deps
		}
		if !dependencies {
			groups = append(groups, pkg.TestGoFiles, pkg.XTestGoFiles)
		}
		for _, names := range groups {
			for _, name := range names {
				// With -test, Go emits a generated test main outside the repository.
				// Its repository inputs arrive through the selected test variant.
				if dependencies && filepath.IsAbs(name) {
					relative, err := filepath.Rel(root, name)
					if err != nil || !filepath.IsLocal(relative) {
						continue
					}
					name, err = filepath.Rel(pkg.Dir, name)
					if err != nil {
						return BuildSelection{}, err
					}
				}
				if err := recordBuildFile(root, pkg.Dir, pkg.ImportPath, name, true, selection); err != nil {
					return BuildSelection{}, err
				}
			}
		}
		if dependencies {
			continue
		}
		for _, name := range pkg.IgnoredGoFiles {
			if err := recordBuildFile(root, pkg.Dir, pkg.ImportPath, name, false, selection); err != nil {
				return BuildSelection{}, err
			}
		}
	}
	return selection, nil
}

func recordBuildFile(root, directory, packagePath, name string, selected bool, selection BuildSelection) error {
	relative, err := filepath.Rel(root, filepath.Join(directory, name))
	if err != nil {
		return err
	}
	if relative == ".." || filepath.IsAbs(relative) || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
		return fmt.Errorf("build file %s escapes repository", name)
	}
	path := filepath.ToSlash(relative)
	selection.Files[path] = selected
	selection.Packages[path] = packagePath
	return nil
}

func goOutput(ctx context.Context, root string, args ...string) ([]byte, error) {
	var output, diagnostic bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: "go", Args: args, Dir: root, Stdout: &output, Stderr: &diagnostic,
	})
	if err == nil && receipt.ExitCode != 0 {
		err = fmt.Errorf("go %s: exit %d: %s", strings.Join(args, " "), receipt.ExitCode, strings.TrimSpace(diagnostic.String()))
	}
	return output.Bytes(), err
}
