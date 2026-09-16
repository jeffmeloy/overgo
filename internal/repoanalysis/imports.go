package repoanalysis

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"os/exec"
	"path"
	"slices"
	"strconv"

	"overgo/internal/gosource"
)

const importNameBatch = 32

// PackageNames maps import paths to their declared package names through the
// shared parsed snapshot and go-list ownership facts.
func PackageNames(snapshot SourceSnapshot, selection gosource.BuildSelection) (map[string]string, error) {
	names := map[string]string{}
	imports := map[string]bool{}
	for _, source := range snapshot.Files {
		file, err := source.Syntax()
		if err != nil {
			return nil, err
		}
		packagePath := selection.Packages[source.Path]
		if packagePath == "" {
			packagePath = path.Dir(source.Path) + "#" + file.Name.Name
		}
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
	paths := make([]string, 0, len(imports))
	for importPath := range imports {
		if names[importPath] == "" {
			paths = append(paths, importPath)
		}
	}
	slices.Sort(paths)
	for start := 0; start < len(paths); start += importNameBatch {
		end := min(start+importNameBatch, len(paths))
		command := exec.Command("go", append([]string{"list", "-e", "-json"}, paths[start:end]...)...)
		command.Dir = selection.Root
		output, err := command.Output()
		if err != nil {
			return nil, err
		}
		decoder := json.NewDecoder(bytes.NewReader(output))
		for {
			var pkg struct{ ImportPath, Name string }
			if err := decoder.Decode(&pkg); errors.Is(err, io.EOF) {
				break
			} else if err != nil {
				return nil, err
			}
			if pkg.ImportPath != "" && pkg.Name != "" {
				names[pkg.ImportPath] = pkg.Name
			}
		}
	}
	return names, nil
}
