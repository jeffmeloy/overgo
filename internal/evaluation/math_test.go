package evaluation

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
)

func TestMATHPinnedOracleParity(t *testing.T) {
	const raw = "Work gives $\\frac12$."
	compiled, err := CompileStructuredGenerated(StructuredGeneratedSuite{
		Kind: StructuredGeneratedKind, Schema: "lm-eval/hendrycks-math/v1.0", Source: "fixture",
		Extractor: ExtractorDollarSpan, Equivalence: EquivalenceLatexSurface,
		Cases: []StructuredGeneratedCase{{
			Name: "fraction", Group: "algebra", Prompt: "Problem: Divide one by two.\nAnswer:",
			MaxTokens: 8, Answers: []string{`\frac{1}{2}`},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	plan, err := BindStructuredGenerated(compiled, ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"),
		RuntimeRecipe:   planID(t, artifact.KindRecipe, "recipe"), CodeCommit: planTestCommit,
		Environment: planID(t, artifact.KindEvidence, "environment"), Execution: ExecutionPolicy{Lifecycle: LifecycleResident},
	})
	if err != nil {
		t.Fatal(err)
	}
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	report, err := EvaluateStructuredGenerated(
		context.Background(), store, exactGenerator{pieces: []string{raw}}, compiled, plan,
	)
	if err != nil {
		t.Fatal(err)
	}
	observation := report.Observations[0]
	if report.Accuracy != 1 || len(report.Groups) != 1 || report.Groups[0].Name != "algebra" ||
		report.Groups[0].Accuracy != 1 || observation.Raw != raw || observation.Extracted != `\frac12` ||
		observation.Normalized != `\frac{1}{2}` || !observation.Accepted {
		t.Fatalf("MATH report = %+v", report)
	}
}
