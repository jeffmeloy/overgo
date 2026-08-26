package main

import (
	"testing"

	"overgo/internal/automationcheck"
	"overgo/internal/codemanifest"
)

func TestFalseNegativeCorpus(t *testing.T) {
	cases := []struct {
		name string
		kind codemanifest.UncertaintyKind
		path string
	}{
		{"interface dispatch", codemanifest.UncertaintyInterface, "internal/model/interface.go"},
		{"reflection", codemanifest.UncertaintyReflection, "internal/recipe/reflect.go"},
		{"cgo export", codemanifest.UncertaintyCgo, "internal/cuda/export.go"},
		{"generated source", codemanifest.UncertaintyGenerated, "internal/generated/table.go"},
		{"build tag variant", codemanifest.UncertaintyBuildSelection, "internal/cuda/kernel_cuda_windows.go"},
		{"device input", codemanifest.UncertaintyNonGo, "kernels/kernel.cu"},
		{"schema input", codemanifest.UncertaintyNonGo, "architecture_profiles.json"},
		{"fixture input", codemanifest.UncertaintyNonGo, "internal/model/testdata/model.gguf"},
		{"external authority", codemanifest.UncertaintyOutsideSnapshot, "external/resource"},
	}
	checks := []automationcheck.Check{
		bootstrapCheck("model", "owner:model"), bootstrapCheck("device", "owner:device"), bootstrapCheck("schema", "owner:schema"),
	}
	for _, fixture := range cases {
		t.Run(fixture.name, func(t *testing.T) {
			surface := automationcheck.ManifestSurface(codemanifest.Impact{
				Base: "profile:base", Candidate: "profile:candidate", Packages: []string{"internal/model"},
				Uncertainty: []codemanifest.Uncertainty{{Kind: fixture.kind, Path: fixture.path, Reason: fixture.name}},
			})
			impact := automationcheck.OwnershipImpact(checks, surface)
			planned, err := automationcheck.Plan(checks, impact)
			if err != nil || len(planned) != len(checks) || len(impact.Exclusions) != 0 {
				t.Fatalf("boundary under-selected checks: planned=%d/%d impact=%+v err=%v", len(planned), len(checks), impact, err)
			}
		})
	}
}
