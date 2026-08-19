package evaluation

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
	"overgo/internal/sequencescore"
)

type fixedContinuationScorer struct {
	scores []sequencescore.Score
}

func (s fixedContinuationScorer) ScoreContinuations(
	context.Context,
	string,
	[]string,
) ([]sequencescore.Score, error) {
	return s.scores, nil
}

func TestMultipleChoiceScoringMatchesPinnedOracle(t *testing.T) {
	compiled, err := CompileMultipleChoice(MultipleChoiceSuite{
		Kind: MultipleChoiceKind, Schema: "fixture/v1", Source: "fixture",
		Normalization: sequencescore.NormalizationMean, Aggregation: AggregationAccuracy,
		Cases: []MultipleChoiceCase{{
			Name: "choice", Prompt: "prompt", Candidates: []string{" first", " second"}, Answer: 0,
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	plan, err := BindMultipleChoice(compiled, ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"),
		RuntimeRecipe:   planID(t, artifact.KindRecipe, "recipe"),
		CodeCommit:      planTestCommit,
		Environment:     planID(t, artifact.KindEvidence, "environment"),
		Execution:       ExecutionPolicy{Lifecycle: LifecycleResident},
	})
	if err != nil {
		t.Fatal(err)
	}
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	report, err := EvaluateMultipleChoice(
		context.Background(), store,
		fixedContinuationScorer{scores: []sequencescore.Score{
			{LogProbability: -4, Tokens: 2}, {LogProbability: -3, Tokens: 1},
		}},
		compiled, plan,
	)
	if err != nil {
		t.Fatal(err)
	}
	observation := report.Observations[0]
	if report.Accuracy != 1 || observation.Selected != 0 || observation.Tied ||
		observation.Values[0] != -2 || observation.Values[1] != -3 {
		t.Fatalf("multiple-choice report = %+v", report)
	}
}
