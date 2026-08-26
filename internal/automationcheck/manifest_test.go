package automationcheck

import (
	"testing"

	"overgo/internal/codemanifest"
)

func TestOwnershipJoinUsesManifestPackagesAndSymbols(t *testing.T) {
	checks := []Check{
		ownershipFixtureCheck("model", "owner:model", "internal/model"),
		ownershipFixtureCheck("cuda", "owner:cuda", "internal/cuda"),
	}
	impact := codemanifest.Impact{
		Base: "profile:sha256:base", Candidate: "profile:sha256:candidate",
		Packages: []string{"internal/model"},
		Reachable: []codemanifest.SymbolID{{
			Package: "internal/model", Context: "windows/amd64", Receiver: "Runtime", Name: "Run", Kind: codemanifest.SymbolMethod,
		}},
	}
	verdict := OwnershipImpact(checks, ManifestSurface(impact))
	planned, err := Plan(checks, verdict)
	if err != nil {
		t.Fatal(err)
	}
	if len(planned) != 1 || planned[0].Check.Name != "model" {
		t.Fatalf("planned=%+v verdict=%+v", planned, verdict)
	}
}

func TestOwnershipJoinUnknownCannotExclude(t *testing.T) {
	checks := []Check{ownershipFixtureCheck("model", "owner:model", "internal/model")}
	impact := codemanifest.Impact{
		Base: "profile:sha256:base", Candidate: "profile:sha256:candidate",
		Uncertainty: []codemanifest.Uncertainty{{Kind: codemanifest.UncertaintyReflection, Path: "internal/other/run.go", Reason: "fixture"}},
	}
	verdict := OwnershipImpact(checks, ManifestSurface(impact))
	if len(verdict.Exclusions) != 0 {
		t.Fatalf("unknown impact excluded checks: %+v", verdict)
	}
}
