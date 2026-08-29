// buildscrub enforces the build/ boundary: build/RETAINED.json declares
// the artifacts tests and tools depend on, and everything else under
// build/ is development scrap this tool reports and, with -apply,
// removes. Durable knowledge lives in overgodb; per-session scratch
// lives in tmp/.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
)

type retainedDeclaration struct {
	Version  uint16 `json:"version"`
	Doc      string `json:"doc"`
	Retained []struct {
		Prefix       string   `json:"prefix"`
		Reason       string   `json:"reason"`
		ReferencedBy []string `json:"referenced_by"`
	} `json:"retained"`
}

func main() {
	clioptions.MainNamed("buildscrub", run)
}

func run() error {
	root := flag.String("root", "build", "build directory governed by RETAINED.json")
	apply := flag.Bool("apply", false, "remove the scrap; default reports only")
	flag.Parse()
	var declaration retainedDeclaration
	if err := jsonfile.Decode(filepath.Join(*root, "RETAINED.json"), &declaration); err != nil {
		return fmt.Errorf("buildscrub: %w", err)
	}
	if declaration.Version != 1 || len(declaration.Retained) == 0 {
		return fmt.Errorf("buildscrub: invalid retained declaration")
	}
	retained := func(name string) bool {
		if name == "RETAINED.json" {
			return true
		}
		for _, entry := range declaration.Retained {
			prefix := filepath.ToSlash(entry.Prefix)
			if name == prefix || strings.HasPrefix(name, prefix+"/") || strings.HasPrefix(prefix, name+"/") {
				return true
			}
		}
		return false
	}
	var scrap []string
	var scrapBytes int64
	kept := 0
	// Walk the boundary recursively: a directory on a retained path's
	// chain opens for inspection, a directory or file under a retained
	// prefix stays, and everything else is scrap at its highest
	// non-retained ancestor.
	var walk func(relative string) error
	walk = func(relative string) error {
		entries, err := os.ReadDir(filepath.Join(*root, filepath.FromSlash(relative)))
		if err != nil {
			return err
		}
		for _, entry := range entries {
			name := relative + entry.Name()
			if !retained(name) {
				scrap = append(scrap, filepath.FromSlash(name))
				scrapBytes += treeBytes(filepath.Join(*root, filepath.FromSlash(name)))
				continue
			}
			fullyRetained := false
			for _, declared := range declaration.Retained {
				prefix := filepath.ToSlash(declared.Prefix)
				if name == prefix || strings.HasPrefix(name, prefix+"/") {
					fullyRetained = true
				}
			}
			if name == "RETAINED.json" {
				fullyRetained = true
			}
			if fullyRetained || !entry.IsDir() {
				kept++
				continue
			}
			if err := walk(name + "/"); err != nil {
				return err
			}
		}
		return nil
	}
	if err := walk(""); err != nil {
		return err
	}
	slices.Sort(scrap)
	fmt.Printf("retained=%d scrap_entries=%d scrap_bytes=%.2f GB\n", kept, len(scrap), float64(scrapBytes)/1e9)
	for _, name := range scrap {
		fmt.Println("scrap:", name)
	}
	if !*apply {
		if len(scrap) > 0 {
			fmt.Println("re-run with -apply to remove the scrap")
		}
		return nil
	}
	for _, name := range scrap {
		if err := os.RemoveAll(filepath.Join(*root, name)); err != nil {
			return err
		}
	}
	fmt.Printf("removed %d scrap entries\n", len(scrap))
	return nil
}

func treeBytes(path string) int64 {
	var total int64
	_ = filepath.WalkDir(path, func(_ string, entry os.DirEntry, err error) error {
		if err != nil || entry.IsDir() {
			return nil
		}
		if info, err := entry.Info(); err == nil {
			total += info.Size()
		}
		return nil
	})
	return total
}
