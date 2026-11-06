package evaluation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
)

func TestMATHPinnedOracleParity(t *testing.T) {
	t.Parallel()
	const raw = "Work gives $\\frac12$."
	compiled := testcheck.Must(compileStructuredGenerated(structuredGeneratedSuite{
		Kind: structuredGeneratedKind, Schema: "lm-eval/hendrycks-math/v1.0", Source: "fixture",
		Extractor: extractorDollarSpan, Equivalence: equivalenceLatexSurface,
		Cases: []structuredGeneratedCase{{
			Name: "fraction", Group: "algebra", Prompt: "Problem: Divide one by two.\nAnswer:",
			MaxTokens: 8, Answers: []string{`\frac{1}{2}`},
		}},
	}))(t)
	plan := testcheck.Must(bindStructuredGenerated(compiled, ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"),
		RuntimeRecipe:   planID(t, artifact.KindRecipe, "recipe"), CodeCommit: planTestCommit,
		Environment: planID(t, artifact.KindEvidence, "environment"), Execution: ExecutionPolicy{Lifecycle: LifecycleResident},
	}))(t)
	store := overgodbtest.Open(t, t.TempDir())
	publishPlanFixtureAuthorities(t, store, plan)
	report := testcheck.Must(evaluateStructuredGenerated(
		t.Context(), store, exactGenerator{pieces: []string{raw}}, compiled, plan,
	))(t)
	observation := report.Observations[0]
	if report.Accuracy != 1 || len(report.Groups) != 1 || report.Groups[0].Name != "algebra" ||
		report.Groups[0].Accuracy != 1 || observation.Raw != raw || observation.Extracted != `\frac12` ||
		observation.Normalized != `\frac{1}{2}` || !observation.Accepted {
		t.Fatalf("MATH report = %+v", report)
	}
}
