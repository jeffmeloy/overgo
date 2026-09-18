package automationcheck

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Launch-name ownership identifies producers. Follow production imports from
// declared lane consumers so a kernel change also exercises composed contracts.
// Read all platform sources conservatively; host build tags cannot exclude a
// declared Windows device consumer from a plan prepared on another platform.
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
			dependencies, err = deviceProductionImports(root, packagePath)
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
	for _, packagePath := range DeviceLanePackages() {
		if affected, err := depends(packagePath, map[string]bool{}); err != nil {
			return err
		} else if affected {
			packages["./"+packagePath] = true
		}
	}
	return nil
}

func deviceProductionImports(root, packagePath string) ([]string, error) {
	directory := filepath.Join(root, filepath.FromSlash(packagePath))
	entries, err := os.ReadDir(directory)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var imports []string
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".go") || strings.HasSuffix(entry.Name(), "_test.go") {
			continue
		}
		file, err := parser.ParseFile(token.NewFileSet(), filepath.Join(directory, entry.Name()), nil, parser.ImportsOnly)
		if err != nil {
			return nil, err
		}
		for _, spec := range file.Imports {
			path, err := strconv.Unquote(spec.Path.Value)
			if err != nil {
				return nil, err
			}
			if relative, found := strings.CutPrefix(path, "overgo/internal/"); found {
				imports = append(imports, "internal/"+relative)
			}
		}
	}
	return imports, nil
}
