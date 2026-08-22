package automationcheck

import (
	"context"

	"overgo/internal/runrecord"
)

const (
	manifestCheckName           = "manifest"
	sbomCheckName               = "sbom"
	compatibilityCheckName      = "claims"
	manifestImpact         Fact = "authority:kernel-manifest"
	sbomImpact             Fact = "authority:release-integrity"
	compatibilityImpact    Fact = "authority:compatibility"
)

// Command runs one check command from an explicit repository root.
type Command func(root, name string, arguments ...string) (string, error)

// GeneratedChecks returns freshness checks for generated repository authorities.
func GeneratedChecks(root string, command Command) []Check {
	return []Check{
		{
			Descriptor: Descriptor{
				Name: manifestCheckName, Phase: runrecord.PhaseValidate,
				Triggers: []Fact{manifestImpact}, Inapplicable: "no kernel authority changed",
				Ownership: Ownership{Fact: manifestImpact, Packages: []string{
					"cmd/kernel-manifest", "cmd/build-kernels", "cmd/kernel-bindings", "internal/cuda/kernel",
				}},
			},
			Run: func(_ context.Context, _ Invocation) (bool, string, error) {
				if _, err := command(root, "go", "run", "./cmd/kernel-manifest"); err != nil {
					return false, "", err
				}
				_, err := command(root, "go", "test", "-run", "TestGeneratedBindingsMatchManifest", "-count=1", "./cmd/kernel-bindings")
				return false, "", err
			},
		},
		{
			Descriptor: Descriptor{
				Name: sbomCheckName, Phase: runrecord.PhaseValidate,
				Triggers: []Fact{sbomImpact}, Inapplicable: "no dependency authority changed",
				Ownership: Ownership{Fact: sbomImpact, Packages: []string{"cmd/sbom"}},
			},
			Run: commandRunner(root, command, "go", "run", "./cmd/sbom", "-check"),
		},
		{
			Descriptor: Descriptor{
				Name: compatibilityCheckName, Phase: runrecord.PhaseValidate,
				Triggers: []Fact{compatibilityImpact}, Inapplicable: "no compatibility evidence changed",
				Ownership: Ownership{Fact: compatibilityImpact, Packages: []string{"cmd/compatibility", "internal/model"}},
			},
			Run: commandRunner(root, command, "go", "run", "./cmd/compatibility", "-check"),
		},
	}
}

func commandRunner(root string, command Command, name string, arguments ...string) Runner {
	return func(context.Context, Invocation) (bool, string, error) {
		_, err := command(root, name, arguments...)
		return false, "", err
	}
}
