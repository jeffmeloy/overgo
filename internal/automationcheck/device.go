package automationcheck

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/runrecord"
)

const (
	deviceCheckName      = "device"
	deviceImpact    Fact = "capability:device"
)

// DeviceVerificationPlan is the exact package/function scope for one device
// lane. Full is the fail-closed result when ownership cannot be proven.
type DeviceVerificationPlan struct {
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

// DeviceCheck returns the shared-device verification adapter.
func DeviceCheck(root string, paths, packages []string, command LaneCommand) Check {
	return Check{
		Descriptor: Descriptor{
			Name: deviceCheckName, Phase: runrecord.PhaseTest, Triggers: []Fact{deviceImpact},
			Inapplicable: "no kernel or device implementation changed",
			Resources:    []Resource{{Name: "device"}},
			Requirements: Requirements{Process: ProcessDevice},
			Ownership: Ownership{
				Fact: deviceImpact, Packages: slices.Clone(packages),
				PackagePrefixes: []string{"internal/cuda"},
			},
		},
		Run: func(ctx context.Context, _ Invocation) (bool, string, error) {
			// The changed-path list rides in a file: a repo-wide commit can
			// carry more paths than the Windows command line admits.
			list, err := os.CreateTemp("", "device-paths-*.txt")
			if err != nil {
				return false, "", err
			}
			defer os.Remove(list.Name())
			if _, err := list.WriteString(strings.Join(paths, "\n")); err != nil {
				list.Close()
				return false, "", err
			}
			if err := list.Close(); err != nil {
				return false, "", err
			}
			receipt, err := command(ctx, root, "go", "run", "./cmd/device-lane", "-paths-file", list.Name())
			return false, receipt, err
		},
	}
}

// DevicePlan derives kernel-function ownership from the generated manifest and
// exact Go launch-name references. Unknown device surfaces fall back to full.
func DevicePlan(root string, paths []string) (DeviceVerificationPlan, error) {
	if len(paths) == 0 {
		return DeviceVerificationPlan{Full: true, Reason: "no changed-path scope"}, nil
	}
	raw, err := os.ReadFile(filepath.Join(root, "kernels", "manifest.json"))
	if err != nil {
		return DeviceVerificationPlan{}, err
	}
	var manifest deviceManifest
	if err := json.Unmarshal(raw, &manifest); err != nil {
		return DeviceVerificationPlan{}, fmt.Errorf("device plan manifest: %w", err)
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
			return DeviceVerificationPlan{Full: true, Reason: "kernel catalog authority changed"}, nil
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
			return DeviceVerificationPlan{Full: true, Reason: "device path has no manifest ownership"}, nil
		}
	}
	functionNames := slices.Sorted(maps.Keys(functions))
	if len(functionNames) > 0 {
		owned, complete, err := deviceFunctionPackages(root, functionNames)
		if err != nil {
			return DeviceVerificationPlan{}, err
		}
		if !complete {
			return DeviceVerificationPlan{Full: true, Reason: "kernel function ownership is incomplete"}, nil
		}
		for _, packagePath := range owned {
			packages[packagePath] = true
		}
	}
	return DeviceVerificationPlan{Packages: slices.Sorted(maps.Keys(packages)), Functions: functionNames}, nil
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
