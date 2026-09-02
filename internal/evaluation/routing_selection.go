package evaluation

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// SelectServingRecipe derives which recipe serves a task from published
// evaluation and resource evidence and records the decision through the
// common publication path. Every cited evidence document admits through
// RequireEvaluationEvidence — the one ordered gate owner — and must share
// the baseline's split and measure the named metric in the baseline's
// direction, so the comparison never confounds data or objectives. Quality
// is the admitted measured value oriented in its improvement direction, the
// threshold is the baseline's own measured value, and resource is the
// candidate model's published artifact size; the registered derivation rule
// then selects and the decision commits before it is returned, so no live
// selection goes unrecorded.
func SelectServingRecipe(
	ctx context.Context,
	store artifact.Repository,
	task recipe.Task,
	metric string,
	baseline artifact.ID,
	candidates []artifact.ID,
) (modelrecipe.RecipeRoutingDecision, error) {
	if metric == "" || len(candidates) == 0 {
		return modelrecipe.RecipeRoutingDecision{}, errors.New("evaluation: selection requires a metric and candidate evidence")
	}
	base, err := RequireEvaluationEvidence(ctx, store, baseline)
	if err != nil {
		return modelrecipe.RecipeRoutingDecision{}, err
	}
	threshold, direction, measured := admittedMetric(base.Metrics, metric)
	if !measured {
		return modelrecipe.RecipeRoutingDecision{}, fmt.Errorf("evaluation: baseline evidence does not measure %q", metric)
	}
	routingCandidates := make([]modelrecipe.RoutingCandidate, 0, len(candidates))
	for _, id := range candidates {
		evidence, err := RequireEvaluationEvidence(ctx, store, id)
		if err != nil {
			return modelrecipe.RecipeRoutingDecision{}, err
		}
		if evidence.Split != base.Split {
			return modelrecipe.RecipeRoutingDecision{}, fmt.Errorf(
				"evaluation: candidate evidence %s was measured on another split; the comparison would confound data", id,
			)
		}
		value, candidateDirection, judged := admittedMetric(evidence.Metrics, metric)
		if !judged || candidateDirection != direction {
			return modelrecipe.RecipeRoutingDecision{}, fmt.Errorf(
				"evaluation: candidate evidence %s does not measure %q in the baseline direction", id, metric,
			)
		}
		definition, err := recipe.RequireDefinition(ctx, store, evidence.Recipe)
		if err != nil {
			return modelrecipe.RecipeRoutingDecision{}, err
		}
		resource, err := routingResourceBytes(ctx, store, definition.Model)
		if err != nil {
			return modelrecipe.RecipeRoutingDecision{}, err
		}
		routingCandidates = append(routingCandidates, modelrecipe.RoutingCandidate{
			Recipe: evidence.Recipe, Model: definition.Model,
			Evidence:      []artifact.ID{evidence.ID},
			Quality:       signedQuality(value, direction),
			ResourceBytes: resource,
		})
	}
	signal := modelrecipe.RoutingSignal{
		Task: task, Capability: metric,
		QualityThreshold: signedQuality(threshold, direction), ThresholdEvidence: baseline,
	}
	decision, err := modelrecipe.DeriveRecipeRoutingDecision(signal, routingCandidates)
	if err != nil {
		return modelrecipe.RecipeRoutingDecision{}, err
	}
	batch, err := decision.Batch("routing/decision/" + decision.ID.String())
	if err != nil {
		return modelrecipe.RecipeRoutingDecision{}, err
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		return modelrecipe.RecipeRoutingDecision{}, err
	}
	return decision, nil
}

// routingResourceBytes is a candidate model's resource cost: the recorded
// bytes of its weights. A model catalogued by component manifest has a
// 172-byte manifest artifact, so the descriptor size is not the model's
// cost; the sum of its components' recorded sizes is -- the same sizes
// the servable listing verifies against the files on disk. A model whose
// artifact is the raw weights costs its own recorded size; a manifest
// with no components, or an artifact with no recorded bytes, refuses.
func routingResourceBytes(ctx context.Context, store artifact.Repository, model artifact.ID) (uint64, error) {
	if manifest, found, err := store.Manifest(ctx, model); err != nil {
		return 0, err
	} else if found {
		var total uint64
		for _, component := range manifest.Components {
			descriptor, ok, err := store.Artifact(ctx, component.Artifact)
			if err != nil {
				return 0, err
			}
			if !ok || descriptor.Size == 0 {
				return 0, fmt.Errorf("evaluation: model %s component %s has no recorded bytes", model, component.Artifact)
			}
			total += descriptor.Size
		}
		if total == 0 {
			return 0, fmt.Errorf("evaluation: model %s has no published resource evidence", model)
		}
		return total, nil
	}
	descriptor, found, err := store.Artifact(ctx, model)
	if err != nil {
		return 0, err
	}
	if !found || descriptor.Size == 0 {
		return 0, fmt.Errorf("evaluation: model %s has no published resource evidence", model)
	}
	return descriptor.Size, nil
}

// signedQuality orients a measured value in its improvement direction so
// meeting the threshold is one comparison for both metric directions.
func signedQuality(value float64, direction runrecord.Direction) float64 {
	if direction == runrecord.DirectionMinimize {
		return -value
	}
	return value
}

func admittedMetric(metrics []runrecord.Metric, name string) (float64, runrecord.Direction, bool) {
	for _, metric := range metrics {
		if metric.Name == name {
			return metric.Value, metric.Direction, true
		}
	}
	return 0, "", false
}
