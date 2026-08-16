package repoanalysis

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"path/filepath"
	"strings"
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
}

// HostBuildSelection derives file membership through the Go toolchain so build
// tags, legacy tags, filename constraints, cgo, tests, and external tests share
// one repository owner.
func HostBuildSelection(root string, patterns ...string) (BuildSelection, error) {
	root, err := filepath.Abs(root)
	if err != nil {
		return BuildSelection{}, err
	}
	args := append([]string{"list", "-e", "-json"}, patterns...)
	command := exec.Command("go", args...)
	command.Dir = root
	output, err := command.Output()
	if err != nil {
		return BuildSelection{}, err
	}
	selection := BuildSelection{Root: root, Files: map[string]bool{}, Packages: map[string]string{}}
	if context, err := goOutput(root, "env", "GOOS", "GOARCH"); err == nil {
		selection.Context = strings.Join(strings.Fields(context), "/")
	}
	decoder := json.NewDecoder(bytes.NewReader(output))
	for {
		var pkg struct {
			ImportPath                                                   string
			Dir                                                          string
			GoFiles, CgoFiles, TestGoFiles, XTestGoFiles, IgnoredGoFiles []string
		}
		if err := decoder.Decode(&pkg); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return BuildSelection{}, err
		}
		for _, names := range [][]string{pkg.GoFiles, pkg.CgoFiles, pkg.TestGoFiles, pkg.XTestGoFiles} {
			for _, name := range names {
				if err := recordBuildFile(root, pkg.Dir, pkg.ImportPath, name, true, selection); err != nil {
					return BuildSelection{}, err
				}
			}
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

func goOutput(root string, args ...string) (string, error) {
	command := exec.Command("go", args...)
	command.Dir = root
	output, err := command.Output()
	return string(bytes.TrimSpace(output)), err
}
