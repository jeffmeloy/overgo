package gosource

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"slices"
	"strings"
	"sync"
)

const importNameBatch = 32

// ImportNames maps import paths to their declared package names through go
// list, run from root; a standard package's name comes from the process
// cache once any call has resolved it.
func ImportNames(root string, importPaths []string) (map[string]string, error) {
	names := map[string]string{}
	var paths []string
	standardMutex.Lock()
	for _, importPath := range importPaths {
		if name, known := standardNames[importPath]; known {
			names[importPath] = name
			continue
		}
		paths = append(paths, importPath)
	}
	standardMutex.Unlock()
	slices.Sort(paths)
	paths = slices.Compact(paths)
	for start := 0; start < len(paths); start += importNameBatch {
		end := min(start+importNameBatch, len(paths))
		output, err := goOutput(root, append([]string{"list", "-e", "-json"}, paths[start:end]...)...)
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
				rememberStandardName(pkg.ImportPath, pkg.Name)
			}
		}
	}
	return names, nil
}

var (
	standardMutex sync.Mutex
	standardNames = map[string]string{}
)

// rememberStandardName keeps a standard-library package's name for the rest
// of the process: the toolchain fixes it, while a module's package can change
// between the trees one process reads. Every manifest generation otherwise
// asked go list again for the same few hundred standard packages.
func rememberStandardName(importPath, name string) {
	first, _, _ := strings.Cut(importPath, "/")
	if strings.Contains(first, ".") {
		return
	}
	standardMutex.Lock()
	standardNames[importPath] = name
	standardMutex.Unlock()
}
