package main

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/automationcheck"
	"overgo/internal/codemanifest"
	"overgo/internal/repoanalysis"
	"overgo/internal/runrecord"
)

// Mutation coverage of the automation itself: each case injects one
// classic semantic defect into the alpha fixture package and demands
// the manifest selector keep alpha's owning check selected -- the
// check that would catch the defect -- while still proving its
// precision by excluding the untouched beta package's check. A
// selector that dodges either direction is broken: missing the
// selection is a false negative, missing the exclusion is a selector
// that never earns its skips.

const mutationAlphaBase = `package alpha

func Clamp(value, limit int) int {
	if value > limit {
		return limit
	}
	return value
}

func Total(values []int, limit int) int {
	total := 0
	for _, value := range values {
		total += Clamp(value, limit)
	}
	return total
}
`

const mutationBetaBase = `package beta

func Identity(value int) int { return value }
`

func ownedCheck(name string, fact automationcheck.Fact, packagePath string) automationcheck.Check {
	return automationcheck.Check{Descriptor: automationcheck.Descriptor{
		Name: name, Phase: runrecord.PhaseTest,
		Triggers:     []automationcheck.Fact{fact},
		Ownership:    automationcheck.Ownership{Fact: fact, Packages: []string{packagePath}},
		Inapplicable: "proven independent",
	}, Run: func(context.Context, automationcheck.Invocation) (bool, string, error) { return false, "", nil }}
}

// TestMutationCorpusSelectsCatchingChecks pins the mutation contract
// per injected defect: the owning check stays selected and the
// untouched package's check is provably excluded.
func TestMutationCorpusSelectsCatchingChecks(t *testing.T) {
	mutations := []struct{ name, mutated string }{
		{"flipped comparison", mutationMutate(t, "value > limit", "value >= limit")},
		{"changed boundary return", mutationMutate(t, "return limit", "return limit - 1")},
		{"dropped clamp call", mutationMutate(t, "total += Clamp(value, limit)", "total += value")},
		{"zero seed defect", mutationMutate(t, "total := 0", "total := 1")},
	}
	checks := []automationcheck.Check{
		ownedCheck("alpha-check", "owner:alpha", "internal/alpha"),
		ownedCheck("beta-check", "owner:beta", "internal/beta"),
	}
	for _, mutation := range mutations {
		t.Run(mutation.name, func(t *testing.T) {
			impact := mutationImpact(t, map[string]string{
				"internal/alpha/alpha.go": mutation.mutated,
				"internal/beta/beta.go":   mutationBetaBase,
			})
			surface := automationcheck.ManifestSurface(impact)
			verdict := automationcheck.OwnershipImpact(checks, surface)
			if !mutationHasFact(verdict.Facts, "owner:alpha") {
				t.Fatalf("mutation escaped selection: %+v", verdict)
			}
			if !mutationHasExclusion(verdict.Exclusions, "beta-check") {
				t.Fatalf("selector proved no precision: %+v", verdict)
			}
			if mutationHasExclusion(verdict.Exclusions, "alpha-check") {
				t.Fatalf("the catching check was excluded: %+v", verdict)
			}
		})
	}
}

// TestMutationCorpusInverseDirection pins symmetry: a defect injected
// into beta selects beta's check and excludes alpha's, so the corpus
// exercises both directions of ownership rather than one lucky order.
func TestMutationCorpusInverseDirection(t *testing.T) {
	impact := mutationImpact(t, map[string]string{
		"internal/alpha/alpha.go": mutationAlphaBase,
		"internal/beta/beta.go":   mutationMutate(t, "return value", "return value + 1"),
	})
	checks := []automationcheck.Check{
		ownedCheck("alpha-check", "owner:alpha", "internal/alpha"),
		ownedCheck("beta-check", "owner:beta", "internal/beta"),
	}
	verdict := automationcheck.OwnershipImpact(checks, automationcheck.ManifestSurface(impact))
	if !mutationHasFact(verdict.Facts, "owner:beta") || !mutationHasExclusion(verdict.Exclusions, "alpha-check") {
		t.Fatalf("inverse direction verdict = %+v", verdict)
	}
}

func mutationMutate(t *testing.T, from, to string) string {
	t.Helper()
	base := mutationAlphaBase
	if from == "return value" {
		base = mutationBetaBase
	}
	mutated := ""
	if index := indexOf(base, from); index < 0 {
		t.Fatalf("mutation site %q is absent from the fixture", from)
	} else {
		mutated = base[:index] + to + base[index+len(from):]
	}
	return mutated
}

func indexOf(text, needle string) int {
	for i := 0; i+len(needle) <= len(text); i++ {
		if text[i:i+len(needle)] == needle {
			return i
		}
	}
	return -1
}

func mutationImpact(t *testing.T, candidate map[string]string) codemanifest.Impact {
	t.Helper()
	root := t.TempDir()
	baseSources := map[string]string{
		"internal/alpha/alpha.go": mutationAlphaBase,
		"internal/beta/beta.go":   mutationBetaBase,
	}
	for path, source := range baseSources {
		abs := filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(abs, []byte(source), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	baseSnapshot, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	for path, source := range candidate {
		if err := os.WriteFile(filepath.Join(root, filepath.FromSlash(path)), []byte(source), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	candidateSnapshot, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	selection := repoanalysis.BuildSelection{
		Context: "linux/amd64", Root: root,
		Files: map[string]bool{"internal/alpha/alpha.go": true, "internal/beta/beta.go": true},
		Packages: map[string]string{
			"internal/alpha/alpha.go": "internal/alpha",
			"internal/beta/beta.go":   "internal/beta",
		},
	}
	baseManifest, err := codemanifest.Generate(baseSnapshot, []repoanalysis.BuildSelection{selection}, nil)
	if err != nil {
		t.Fatal(err)
	}
	candidateManifest, err := codemanifest.Generate(candidateSnapshot, []repoanalysis.BuildSelection{selection}, nil)
	if err != nil {
		t.Fatal(err)
	}
	delta, err := codemanifest.Diff(baseManifest, candidateManifest)
	if err != nil {
		t.Fatal(err)
	}
	impact, err := codemanifest.Close(baseManifest, candidateManifest, delta)
	if err != nil {
		t.Fatal(err)
	}
	return impact
}

func mutationHasFact(facts []automationcheck.Fact, wanted automationcheck.Fact) bool {
	return slices.Contains(facts, wanted)
}

func mutationHasExclusion(exclusions []automationcheck.Exclusion, check string) bool {
	for _, exclusion := range exclusions {
		if exclusion.Check == check {
			return true
		}
	}
	return false
}
