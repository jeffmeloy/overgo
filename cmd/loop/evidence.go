package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/jsonfile"
	"overgo/internal/loop"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// strategySpec names the authorities one published strategy binds: the
// worker agent definition, the tool catalog, and the loop budgets.
type strategySpec struct {
	Worker  artifact.ID `json:"worker"`
	Catalog artifact.ID `json:"catalog"`
	Loop    loop.Config `json:"loop"`
}

// publishStrategy is the one production door that mints a strategy: it
// resolves the exact worker definition, derives the content-addressed
// strategy, and commits it with its lineage. The loop then runs it by
// identity through the existing strategy_id configuration.
func publishStrategy(root, specPath string, output io.Writer) error {
	var spec strategySpec
	if err := jsonfile.DecodeStrict(specPath, &spec); err != nil {
		return err
	}
	store, err := overgodb.Open(root)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	worker, err := recipe.RequireAgentDefinition(ctx, store, spec.Worker)
	if err != nil {
		return err
	}
	strategy, err := loop.NewStrategy(worker, spec.Catalog, spec.Loop)
	if err != nil {
		return err
	}
	content, err := strategy.Content()
	if err != nil {
		return err
	}
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{
		Key:       "loop/strategy/" + strategy.ID.String(),
		Artifacts: []artifact.Descriptor{content.Descriptor},
		Contents:  []artifact.Content{content},
		Lineage:   strategy.Lineage(),
	}); err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "published strategy %s\n", strategy.ID)
	return err
}

// experimentSpec binds one strategy comparison: the task contract, the
// baseline commit, the competing candidates, and the fitness evidence.
type experimentSpec struct {
	Task       artifact.ID                        `json:"task"`
	Baseline   string                             `json:"baseline"`
	Candidates []loop.StrategyExperimentCandidate `json:"candidates"`
	Fitness    []artifact.ID                      `json:"fitness"`
}

// compareStrategies replays one strategy experiment from immutable store
// state and prints the typed comparison; it decides nothing and widens no
// budget on its own.
func compareStrategies(root, specPath string, output io.Writer) error {
	var spec experimentSpec
	if err := jsonfile.DecodeStrict(specPath, &spec); err != nil {
		return err
	}
	store, err := overgodb.OpenReadOnly(root)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	task, err := recipe.RequireAgentTaskContract(ctx, store, spec.Task)
	if err != nil {
		return err
	}
	experiment, comparison, err := loop.CompareStrategyExperiment(
		ctx, store, task, spec.Baseline, spec.Candidates, spec.Fitness,
	)
	if err != nil {
		return err
	}
	encoded, err := json.MarshalIndent(struct {
		Experiment loop.StrategyExperiment `json:"experiment"`
		Comparison loop.StrategyComparison `json:"comparison"`
	}{experiment, comparison}, "", " ")
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "%s\n", encoded)
	return err
}

// publishFitness commits one pairwise improvement-fitness proof against the
// exact journal head its coverage projection observed.
func publishFitness(root, specPath string, output io.Writer) error {
	var request evaluation.ImprovementFitnessRequest
	if err := jsonfile.DecodeStrict(specPath, &request); err != nil {
		return err
	}
	store, err := overgodb.Open(root)
	if err != nil {
		return err
	}
	defer store.Close()
	fitness, err := evaluation.PublishImprovementFitness(context.Background(), store, request)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "published improvement fitness %s\n", fitness.ID)
	return err
}

// resourceLanesSpec declares the compared workload lanes.
type resourceLanesSpec struct {
	Lanes []runrecord.ResourceFitnessLane `json:"lanes"`
}

// compareResourceLanes replays a resource no-regression proof over every
// declared lane and prints the typed comparison.
func compareResourceLanes(root, specPath string, output io.Writer) error {
	var spec resourceLanesSpec
	if err := jsonfile.DecodeStrict(specPath, &spec); err != nil {
		return err
	}
	if len(spec.Lanes) == 0 {
		return errors.New("loop: resource comparison requires at least one lane")
	}
	store, err := overgodb.OpenReadOnly(root)
	if err != nil {
		return err
	}
	defer store.Close()
	comparison, err := runrecord.CompareResourceFitness(context.Background(), store, spec.Lanes)
	if err != nil {
		return err
	}
	encoded, err := json.MarshalIndent(comparison, "", " ")
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "%s\n", encoded)
	return err
}
