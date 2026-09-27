package gosyntax

import (
	"cmp"
	"path"
	"strconv"

	"overgo/internal/gosource"
)

// PackageNames maps import paths to their declared package names: the
// snapshot's own packages from their parsed declarations, every other import
// through gosource's go-list facts.
func PackageNames(snapshot Snapshot, selection gosource.BuildSelection) (map[string]string, error) {
	names := map[string]string{}
	imports := map[string]bool{}
	for _, source := range snapshot.Files {
		file, err := source.Syntax()
		if err != nil {
			return nil, err
		}
		packagePath := cmp.Or(selection.Packages[source.Path], path.Dir(source.Path)+"#"+file.Name.Name)
		// XTestGoFiles are owned by the production import path in the build
		// selection even though they declare package <name>_test. Preserve the
		// production declaration as the import qualifier regardless of source
		// ordering; otherwise an external test can make real consumers appear
		// unresolved and turn the consumer census into a false dead-export report.
		if names[packagePath] == "" || !source.Test {
			names[packagePath] = file.Name.Name
		}
		for _, spec := range file.Imports {
			importPath, err := strconv.Unquote(spec.Path.Value)
			if err == nil {
				imports[importPath] = true
			}
		}
	}
	var unresolved []string
	for importPath := range imports {
		if names[importPath] == "" {
			unresolved = append(unresolved, importPath)
		}
	}
	resolved, err := gosource.ImportNames(selection.Root, unresolved)
	if err != nil {
		return nil, err
	}
	for importPath, name := range resolved {
		names[importPath] = name
	}
	return names, nil
}
