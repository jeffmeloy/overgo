// Package gosource selects and fingerprints producer inputs without AST policy.
package gosource

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"slices"
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
}

// HostBuildSelection derives file membership through the Go toolchain so build
// tags, legacy tags, filename constraints, cgo, tests, and external tests share
// one repository owner.
// With -deps, retain compiled repository Go sources only; -test adds the
// requested test variants. Module, native and runtime inputs are separate.
func HostBuildSelection(root string, patterns ...string) (BuildSelection, error) {
	root, err := filepath.Abs(root)
	if err != nil {
		return BuildSelection{}, err
	}
	output, err := goOutput(root, append([]string{"list", "-e", "-json"}, patterns...)...)
	if err != nil {
		return BuildSelection{}, err
	}
	selection := BuildSelection{Root: root, Files: map[string]bool{}, Packages: map[string]string{}, Dependencies: map[string][]string{}}
	if context, err := goOutput(root, "env", "GOOS", "GOARCH"); err == nil {
		selection.Context = strings.Join(strings.Fields(string(context)), "/")
	}
	decoder := json.NewDecoder(bytes.NewReader(output))
	dependencies := slices.Contains(patterns, "-deps")
	for {
		var pkg struct {
			Deps                                                         []string
			ImportPath                                                   string
			Dir                                                          string
			GoFiles, CgoFiles, TestGoFiles, XTestGoFiles, IgnoredGoFiles []string
			Error                                                        *struct{ Err string }
			DepsErrors                                                   []struct{ Err string }
		}
		if err := decoder.Decode(&pkg); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return BuildSelection{}, err
		}
		if dependencies {
			if pkg.Error != nil {
				return BuildSelection{}, fmt.Errorf("dependency source selection: %s: %s", pkg.ImportPath, pkg.Error.Err)
			}
			if len(pkg.DepsErrors) != 0 {
				return BuildSelection{}, fmt.Errorf("dependency source selection: %s: %s", pkg.ImportPath, pkg.DepsErrors[0].Err)
			}
			pkg.ImportPath, _, _ = strings.Cut(pkg.ImportPath, " [")
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

func goOutput(root string, args ...string) ([]byte, error) {
	var output, diagnostic bytes.Buffer
	receipt, err := processcontrol.Run(context.Background(), processcontrol.Command{
		Path: "go", Args: args, Dir: root, Stdout: &output, Stderr: &diagnostic,
	})
	if err == nil && receipt.ExitCode != 0 {
		err = fmt.Errorf("go %s: exit %d: %s", strings.Join(args, " "), receipt.ExitCode, strings.TrimSpace(diagnostic.String()))
	}
	return output.Bytes(), err
}
