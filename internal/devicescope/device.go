// Package devicescope is what the device lane verifies: the packages and
// tests it runs on the GPU, and which of them a change to kernels or device
// code reaches. The device lane runs them; the small gate reads the same
// scopes to decide when the lane must run.
package devicescope

import (
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

// VerificationPlan is the exact package/function scope for one device
// lane. Full is the fail-closed result when ownership cannot be proven.
type VerificationPlan struct {
	Packages  []string
	Functions []string
	Full      bool
	Reason    string
}

type deviceManifest struct {
	Modules []deviceModule `json:"modules"`
}

type deviceModule struct {
	Source             string `json:"source"`
	Asset              string `json:"asset"`
	Functions          []string
	SourceDependencies []deviceDependency `json:"sourceDependencies"`
}

type deviceDependency struct {
	Path string `json:"path"`
}

// PlanFor derives kernel-function ownership from the generated manifest and
// exact Go launch-name references. Unknown device surfaces fall back to full.
func PlanFor(root string, paths []string) (VerificationPlan, error) {
	if len(paths) == 0 {
		return VerificationPlan{Full: true, Reason: "no changed-path scope"}, nil
	}
	raw, err := os.ReadFile(filepath.Join(root, "kernels", "manifest.json"))
	if err != nil {
		return VerificationPlan{}, err
	}
	var manifest deviceManifest
	if err := json.Unmarshal(raw, &manifest); err != nil {
		return VerificationPlan{}, fmt.Errorf("device plan manifest: %w", err)
	}
	functions := map[string]bool{}
	packages := map[string]bool{}
	for _, changed := range paths {
		changed = filepath.ToSlash(changed)
		if direct := directDevicePackage(changed); direct != "" {
			packages[direct] = true
			continue
		}
		if changed == "kernels/manifest.json" || strings.HasPrefix(changed, "cmd/kernel-") || strings.HasPrefix(changed, "cmd/build-kernels/") {
			return VerificationPlan{Full: true, Reason: "kernel catalog authority changed"}, nil
		}
		matched := false
		for _, module := range manifest.Modules {
			if moduleOwnsPath(module, changed) {
				matched = true
				for _, function := range module.Functions {
					functions[function] = true
				}
			}
		}
		if !matched && (strings.HasPrefix(changed, "kernels/") || strings.HasPrefix(changed, "internal/cuda/")) {
			return VerificationPlan{Full: true, Reason: "device path has no manifest ownership"}, nil
		}
	}
	functionNames := slices.Sorted(maps.Keys(functions))
	if len(functionNames) > 0 {
		owned, complete, err := deviceFunctionPackages(root, functionNames)
		if err != nil {
			return VerificationPlan{}, err
		}
		if !complete {
			return VerificationPlan{Full: true, Reason: "kernel function ownership is incomplete"}, nil
		}
		for _, packagePath := range owned {
			packages[packagePath] = true
		}
	}
	if err := addDeviceConsumers(root, paths, packages); err != nil {
		return VerificationPlan{}, err
	}
	return VerificationPlan{Packages: slices.Sorted(maps.Keys(packages)), Functions: functionNames}, nil
}

func directDevicePackage(path string) string {
	if !strings.Contains(path, "_cuda_windows") || !strings.HasPrefix(path, "internal/") {
		return ""
	}
	parts := strings.Split(path, "/")
	if len(parts) < 3 {
		return ""
	}
	return "./" + strings.Join(parts[:len(parts)-1], "/")
}

func moduleOwnsPath(module deviceModule, path string) bool {
	if filepath.ToSlash(module.Source) == path || filepath.ToSlash(module.Asset) == path {
		return true
	}
	return slices.ContainsFunc(module.SourceDependencies, func(dependency deviceDependency) bool {
		return filepath.ToSlash(dependency.Path) == path
	})
}

func deviceFunctionPackages(root string, functions []string) ([]string, bool, error) {
	packages := map[string]bool{}
	ownedFunctions := map[string]bool{}
	err := filepath.WalkDir(filepath.Join(root, "internal"), func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".go") {
			return nil
		}
		content, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		for _, function := range functions {
			if !strings.Contains(string(content), `"`+function+`"`) {
				continue
			}
			ownedFunctions[function] = true
			relative, err := filepath.Rel(root, filepath.Dir(path))
			if err != nil {
				return err
			}
			packages["./"+filepath.ToSlash(relative)] = true
		}
		return nil
	})
	if errors.Is(err, os.ErrNotExist) {
		return nil, false, nil
	}
	return slices.Sorted(maps.Keys(packages)), len(ownedFunctions) == len(functions), err
}
