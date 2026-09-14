package gate

import (
	"context"
	"testing"

	"overgo/internal/automationcheck"
	"overgo/internal/runrecord"
)

func TestGeneratorBootstrap(t *testing.T) {
	t.Parallel()
	for _, name := range []string{
		"internal/codemanifest/generate.go", "internal/codeprofile/profile.go",
		"internal/repoanalysis/source.go", "internal/automationcheck/manifest.go", "cmd/code-manifest/main.go",
	} {
		if !requiresManifestBootstrap([]string{name}) {
			t.Errorf("analyzer-owned path %s did not force bootstrap", name)
		}
	}
	// The gate is a planner consumer, not an analyzer: its change keeps the
	// ownership closure, and its own selected tests verify it.
	for _, name := range []string{"internal/model/model.go", "internal/gate/documentation_scope.go", "cmd/gate/main.go"} {
		if requiresManifestBootstrap([]string{name}) {
			t.Fatalf("%s forced analyzer bootstrap", name)
		}
	}
}

func TestAnalyzerFailureRunsFullPlan(t *testing.T) {
	t.Parallel()
	checks := []automationcheck.Check{
		bootstrapCheck("first", "owner:first"), bootstrapCheck("second", "owner:second"),
	}
	surface := automationcheck.Surface{Identity: "failed-analysis", Unknown: []string{"analyzer failed"}}
	impact := automationcheck.OwnershipImpact(checks, surface)
	planned, err := automationcheck.Plan(checks, impact)
	if err != nil || len(planned) != len(checks) || len(impact.Exclusions) != 0 {
		t.Fatalf("fallback plan = %+v, impact=%+v, err=%v", planned, impact, err)
	}
}

func bootstrapCheck(name string, fact automationcheck.Fact) automationcheck.Check {
	return automationcheck.Check{Descriptor: automationcheck.Descriptor{
		Name: name, Phase: runrecord.PhaseTest, Triggers: []automationcheck.Fact{fact}, Inapplicable: "proven independent",
	}, Run: func(context.Context, automationcheck.Invocation) (bool, string, error) { return false, "", nil }}
}
