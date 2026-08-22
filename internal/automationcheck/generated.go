package automationcheck

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/runrecord"
)

const (
	manifestCheckName                = "manifest"
	sbomCheckName                    = "sbom"
	compatibilityCheckName           = "claims"
	manifestExcludedReason           = "changed paths are independent of kernel authority"
	sbomExcludedReason               = "changed paths are independent of dependency authority"
	compatibilityExcludedReason      = "changed paths are independent of compatibility authority"
	manifestImpact              Fact = "authority:kernel-manifest"
	sbomImpact                  Fact = "authority:release-integrity"
	compatibilityImpact         Fact = "authority:compatibility"
)

// Command runs one check command from an explicit repository root.
type Command func(root, name string, arguments ...string) (string, error)

// GeneratedChecks returns freshness checks for generated repository authorities.
func GeneratedChecks(root string, command Command) []Check {
	return []Check{
		{
			Descriptor: Descriptor{Name: manifestCheckName, Phase: runrecord.PhaseValidate, Triggers: []Fact{manifestImpact}, Inapplicable: "no kernel authority changed"},
			Run: func(_ context.Context, _ Invocation) (bool, string, error) {
				if _, err := command(root, "go", "run", "./cmd/kernel-manifest"); err != nil {
					return false, "", err
				}
				_, err := command(root, "go", "test", "-run", "TestGeneratedBindingsMatchManifest", "-count=1", "./cmd/kernel-bindings")
				return false, "", err
			},
		},
		{
			Descriptor: Descriptor{Name: sbomCheckName, Phase: runrecord.PhaseValidate, Triggers: []Fact{sbomImpact}, Inapplicable: "no dependency authority changed"},
			Run:        commandRunner(root, command, "go", "run", "./cmd/sbom", "-check"),
		},
		{
			Descriptor: Descriptor{Name: compatibilityCheckName, Phase: runrecord.PhaseValidate, Triggers: []Fact{compatibilityImpact}, Inapplicable: "no compatibility evidence changed"},
			Run:        commandRunner(root, command, "go", "run", "./cmd/compatibility", "-check"),
		},
	}
}

// GeneratedImpact derives authority facts from shipped paths and manifests.
func GeneratedImpact(root string, paths []string) (Impact, error) {
	facts := map[Fact]bool{}
	searchCompatibility := false
	for _, path := range paths {
		path = filepath.ToSlash(path)
		if ownsManifest(path) {
			if _, err := os.Stat(filepath.Join(root, "kernels", "manifest.json")); err == nil {
				facts[manifestImpact] = true
			} else if errors.Is(err, os.ErrNotExist) {
				return Impact{}, errors.New("automation check: kernel authority changed but kernels/manifest.json is missing")
			} else {
				return Impact{}, err
			}
		}
		if path == "go.mod" || path == "go.sum" || path == "SBOM.cdx.json" || strings.HasPrefix(path, "cmd/sbom/") {
			facts[sbomImpact] = true
		}
		if path == "compatibility.json" || strings.HasPrefix(path, "cmd/compatibility/") || strings.HasPrefix(path, "internal/model/") {
			facts[compatibilityImpact] = true
		} else {
			searchCompatibility = true
		}
	}
	if !facts[compatibilityImpact] && searchCompatibility {
		compatibility, err := os.ReadFile(filepath.Join(root, "compatibility.json"))
		if err != nil {
			return Impact{}, err
		}
		manifestText := string(compatibility)
		for _, path := range paths {
			if strings.Contains(manifestText, filepath.ToSlash(path)) {
				facts[compatibilityImpact] = true
				break
			}
		}
	}
	result := Impact{Facts: make([]Fact, 0, len(facts))}
	for fact := range facts {
		result.Facts = append(result.Facts, fact)
	}
	slices.Sort(result.Facts)
	if len(paths) != 0 {
		if !facts[manifestImpact] {
			result.Exclusions = append(result.Exclusions, Exclusion{Check: manifestCheckName, Reason: manifestExcludedReason})
		}
		if !facts[sbomImpact] {
			result.Exclusions = append(result.Exclusions, Exclusion{Check: sbomCheckName, Reason: sbomExcludedReason})
		}
		if !facts[compatibilityImpact] {
			result.Exclusions = append(result.Exclusions, Exclusion{Check: compatibilityCheckName, Reason: compatibilityExcludedReason})
		}
	}
	return result, nil
}

func ownsManifest(path string) bool {
	return strings.HasPrefix(path, "kernels/") || strings.HasPrefix(path, "internal/cuda/kernel/") ||
		strings.HasPrefix(path, "cmd/kernel-manifest/") || strings.HasPrefix(path, "cmd/build-kernels/") ||
		strings.HasPrefix(path, "cmd/kernel-bindings/")
}

func commandRunner(root string, command Command, name string, arguments ...string) Runner {
	return func(context.Context, Invocation) (bool, string, error) {
		_, err := command(root, name, arguments...)
		return false, "", err
	}
}
