package modelrecipe

import (
	"errors"

	"overgo/internal/artifact"
)

// RecipeEvaluationSide is one measured evaluation of a recipe over an exact
// model and split: the evaluation evidence, the capability it measured, its
// quality score, its resource cost, and its case coverage.
type RecipeEvaluationSide struct {
	Recipe        artifact.ID `json:"recipe"`
	Model         artifact.ID `json:"model"`
	Split         artifact.ID `json:"split"`
	Evaluation    artifact.ID `json:"evaluation"`
	Capability    string      `json:"capability"`
	Quality       float64     `json:"quality"`
	ResourceBytes uint64      `json:"resource_bytes"`
	WallNS        uint64      `json:"wall_ns"`
	CoveredCases  uint64      `json:"covered_cases"`
	TotalCases    uint64      `json:"total_cases"`
}

// RecipeEvaluationAttribution is the typed verdict of a controlled recipe
// comparison: because both sides bind the same model and the same split,
// every reported delta attributes to recipe behavior — not weights,
// architecture, derivation policy, or another system change.
type RecipeEvaluationAttribution struct {
	Candidate     artifact.ID `json:"candidate"`
	Baseline      artifact.ID `json:"baseline"`
	Model         artifact.ID `json:"model"`
	Split         artifact.ID `json:"split"`
	Capability    string      `json:"capability"`
	QualityDelta  float64     `json:"quality_delta"`
	ResourceDelta int64       `json:"resource_delta"`
	WallDelta     int64       `json:"wall_delta"`
	Coverage      uint64      `json:"coverage"`
}

func validRecipeEvaluationSide(side RecipeEvaluationSide) error {
	if side.Recipe.Kind() != artifact.KindRecipe || side.Model.Kind() != artifact.KindModel ||
		!side.Split.Valid() || side.Evaluation.Kind() != artifact.KindEvaluation {
		return errors.New("model recipe: evaluation side requires exact recipe, model, split, and evaluation identities")
	}
	if side.Capability == "" || side.TotalCases == 0 {
		return errors.New("model recipe: evaluation side requires a capability and a case denominator")
	}
	return nil
}

// AttributeRecipeEvaluation judges one candidate recipe against a baseline.
// Both sides must bind the same model, the same split, and the same measured
// capability with complete case coverage — recipe behavior is the only
// candidate variable. A comparison that varies the model, the split, or the
// capability, or whose coverage is incomplete, refuses: it cannot attribute
// the delta to the recipe, and an unattributable gain is not evidence.
func AttributeRecipeEvaluation(
	candidate, baseline RecipeEvaluationSide,
) (RecipeEvaluationAttribution, error) {
	if err := errors.Join(validRecipeEvaluationSide(candidate), validRecipeEvaluationSide(baseline)); err != nil {
		return RecipeEvaluationAttribution{}, err
	}
	if candidate.Model != baseline.Model {
		return RecipeEvaluationAttribution{}, errors.New(
			"model recipe: sides bind different models; the delta would confound weights with recipe behavior",
		)
	}
	if candidate.Split != baseline.Split {
		return RecipeEvaluationAttribution{}, errors.New(
			"model recipe: sides bind different splits; the delta would confound data with recipe behavior",
		)
	}
	if candidate.Capability != baseline.Capability {
		return RecipeEvaluationAttribution{}, errors.New(
			"model recipe: sides measure different capabilities; the delta is not comparable",
		)
	}
	if candidate.Recipe == baseline.Recipe {
		return RecipeEvaluationAttribution{}, errors.New(
			"model recipe: sides bind one recipe; nothing varies to attribute",
		)
	}
	if candidate.CoveredCases != candidate.TotalCases || baseline.CoveredCases != baseline.TotalCases {
		return RecipeEvaluationAttribution{}, errors.New(
			"model recipe: incomplete case coverage blocks attribution",
		)
	}
	return RecipeEvaluationAttribution{
		Candidate: candidate.Recipe, Baseline: baseline.Recipe,
		Model: candidate.Model, Split: candidate.Split, Capability: candidate.Capability,
		QualityDelta:  candidate.Quality - baseline.Quality,
		ResourceDelta: int64(candidate.ResourceBytes) - int64(baseline.ResourceBytes),
		WallDelta:     int64(candidate.WallNS) - int64(baseline.WallNS),
		Coverage:      candidate.TotalCases,
	}, nil
}
