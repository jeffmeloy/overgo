package devicescope

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
)

// Launch-name ownership identifies producers. Follow imports from declared
// lane consumers so a kernel or Go change also exercises composed contracts.
// A lane package's test binary compiles all its test files, so their imports
// count for it once the package holds a test the lane runs; each package it
// imports contributes its production imports only. A recursive scope
// (`internal/cuda/...`) counts every package under it. Read all platform
// sources conservatively; host build tags cannot exclude a declared Windows
// device consumer from a plan prepared on another platform.
func addDeviceConsumers(root string, paths []string, packages map[string]bool) error {
	affected := make(map[string]bool, len(packages))
	for packagePath := range packages {
		affected[strings.TrimPrefix(packagePath, "./")] = true
	}
	for _, path := range paths {
		path = filepath.ToSlash(path)
		if strings.HasPrefix(path, "internal/") {
			affected[filepath.ToSlash(filepath.Dir(path))] = true
		}
	}
	imports := map[string][]string{}
	var depends func(string, map[string]bool) (bool, error)
	depends = func(packagePath string, seen map[string]bool) (bool, error) {
		if affected[packagePath] {
			return true, nil
		}
		if seen[packagePath] {
			return false, nil
		}
		seen[packagePath] = true
		dependencies, found := imports[packagePath]
		if !found {
			var err error
			dependencies, err = deviceImports(root, packagePath, nil)
			if err != nil {
				return false, err
			}
			imports[packagePath] = dependencies
		}
		for _, dependency := range dependencies {
			if affected, err := depends(dependency, seen); affected || err != nil {
				return affected, err
			}
		}
		return false, nil
	}
	scopes, err := scopePackages(root)
	if err != nil {
		return err
	}
	for _, scope := range scopes {
		if affected[scope.Package] {
			packages["./"+scope.Package] = true
			continue
		}
		tested, err := deviceImports(root, scope.Package, &scope)
		if err != nil {
			return err
		}
		seen := map[string]bool{scope.Package: true}
		for _, dependency := range tested {
			if reached, err := depends(dependency, seen); err != nil {
				return err
			} else if reached {
				packages["./"+scope.Package] = true
				break
			}
		}
	}
	return nil
}

// scopePackages lists the lane's scopes by package directory, a recursive
// scope as every directory under it that holds Go files, skipping those the
// go tool skips.
func scopePackages(root string) ([]TestScope, error) {
	var scopes []TestScope
	for _, declared := range deviceLaneScopes {
		base, recursive := strings.CutSuffix(strings.TrimPrefix(declared.Package, "./"), "/...")
		if !recursive {
			declared.Package = base
			scopes = append(scopes, declared)
			continue
		}
		listed := map[string]bool{}
		err := filepath.WalkDir(filepath.Join(root, filepath.FromSlash(base)), func(path string, entry fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if entry.IsDir() {
				if name := entry.Name(); name == "testdata" || strings.HasPrefix(name, ".") || strings.HasPrefix(name, "_") {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(entry.Name(), ".go") {
				return nil
			}
			relative, err := filepath.Rel(root, filepath.Dir(path))
			if err != nil {
				return err
			}
			if relative = filepath.ToSlash(relative); !listed[relative] {
				listed[relative] = true
				scopes = append(scopes, TestScope{Package: relative, Run: declared.Run, Tests: declared.Tests})
			}
			return nil
		})
		if err != nil && !os.IsNotExist(err) {
			return nil, err
		}
	}
	return scopes, nil
}

// deviceImports lists the repository packages a package's production files
// import, and, for a lane scope, its test files' imports too: once one of them
// declares a test the scope names, or always for a scope that names none (its
// -run pattern or the whole package runs).
func deviceImports(root, packagePath string, scope *TestScope) ([]string, error) {
	directory := filepath.Join(root, filepath.FromSlash(packagePath))
	entries, err := os.ReadDir(directory)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	selective := scope != nil && len(scope.Tests) > 0
	var imports, tested []string
	selected := !selective
	for _, entry := range entries {
		name := entry.Name()
		test := strings.HasSuffix(name, "_test.go")
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || test && scope == nil {
			continue
		}
		mode := parser.ImportsOnly
		if test && selective && !selected {
			mode = parser.SkipObjectResolution
		}
		file, err := parser.ParseFile(token.NewFileSet(), filepath.Join(directory, name), nil, mode)
		if err != nil {
			return nil, err
		}
		if mode != parser.ImportsOnly {
			selected = slices.ContainsFunc(file.Decls, func(declaration ast.Decl) bool {
				function, ok := declaration.(*ast.FuncDecl)
				return ok && function.Recv == nil && strings.HasPrefix(function.Name.Name, "Test") && slices.Contains(scope.Tests, function.Name.Name)
			})
		}
		for _, spec := range file.Imports {
			path, err := strconv.Unquote(spec.Path.Value)
			if err != nil {
				return nil, err
			}
			if relative, found := strings.CutPrefix(path, "overgo/internal/"); found {
				if test {
					tested = append(tested, "internal/"+relative)
				} else {
					imports = append(imports, "internal/"+relative)
				}
			}
		}
	}
	if selected {
		imports = append(imports, tested...)
	}
	return imports, nil
}
