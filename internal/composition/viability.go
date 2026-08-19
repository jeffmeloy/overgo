// Package composition evaluates catalog-proposed component bridges.
package composition

import (
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/densecausal"
	"overgo/internal/optimizer"
	"overgo/internal/organ"
	"overgo/internal/recipe"
	"overgo/internal/trainingprogram"
	"overgo/internal/workflowrecipe"
)

type Config struct {
	TargetDir   string
	DonorDir    string
	GraftLayer  int // Negative: target midpoint.
	DonorLayer  int // Negative: donor midpoint.
	DonorTensor string
	Seeds       []int64
	Steps       int
	BaseLR      float64 // Nonpositive: derive from live parameters.
	// LRScale scales the derived rate.
	LRScale  float64
	Momentum float64
	Train    [][]int
	HeldOut  []int
}

type SeedOutcome struct {
	Seed        int64
	TrainLosses []float64
	HeldOut     float64
}

type Result struct {
	Target        artifact.ID
	Donor         artifact.ID
	Component     artifact.ID
	Recipe        artifact.ID
	Dataset       artifact.ID
	Split         artifact.ID
	DonorContract organ.Contract
	DonorTensor   string
	GraftLayer    int
	Baseline      float64
	Outcomes      []SeedOutcome
	Ship          bool
	Reason        string
	definition    recipe.Definition
}

// RunViability: compile, train, evaluate, decide.
func RunViability(config Config) (Result, error) {
	if len(config.Seeds) == 0 || config.Steps <= 0 || len(config.Train) == 0 || len(config.HeldOut) < 2 {
		return Result{}, fmt.Errorf("composition: seeds, steps, train batches and a held-out batch are required")
	}
	target, err := densecausal.Load(config.TargetDir)
	if err != nil {
		return Result{}, fmt.Errorf("composition: load target: %w", err)
	}
	donor, err := densecausal.Load(config.DonorDir)
	if err != nil {
		return Result{}, fmt.Errorf("composition: load donor: %w", err)
	}
	targetID, err := identifyDenseModel(config.TargetDir)
	if err != nil {
		return Result{}, fmt.Errorf("composition: identify target: %w", err)
	}
	donorID, err := identifyDenseModel(config.DonorDir)
	if err != nil {
		return Result{}, fmt.Errorf("composition: identify donor: %w", err)
	}
	dataset, err := artifact.JSONID(artifact.KindDataset, config.Train)
	if err != nil {
		return Result{}, err
	}
	split, err := artifact.JSONID(artifact.KindDatasetShard, config.HeldOut)
	if err != nil {
		return Result{}, err
	}
	definition, err := bridgeRecipe(targetID, donorID, dataset)
	if err != nil {
		return Result{}, err
	}
	if _, err := recipe.CompileProgram(definition, workflowrecipe.Catalog()); err != nil {
		return Result{}, fmt.Errorf("composition: compile bridge recipe: %w", err)
	}
	graftLayer := config.GraftLayer
	if graftLayer < 0 {
		graftLayer = target.Dims.Layers / 2
	}
	names := make([]string, 0, len(donor.Weights))
	for name := range donor.Weights {
		names = append(names, name)
	}
	catalog, err := organ.CompileCatalog(names)
	if err != nil {
		return Result{}, err
	}
	donorLayer := config.DonorLayer
	if config.DonorTensor != "" {
		donorLayer = -1
		for index := range catalog.MLPCount() {
			candidate, _ := catalog.MLP(index)
			if candidate.Gate == config.DonorTensor || candidate.Up == config.DonorTensor || candidate.Down == config.DonorTensor {
				donorLayer = index
				break
			}
		}
	}
	if donorLayer < 0 && config.DonorTensor == "" {
		donorLayer = catalog.MLPCount() / 2
	}
	component, ok := catalog.MLP(donorLayer)
	if !ok {
		return Result{}, fmt.Errorf("composition: donor MLP index %d outside %d cataloged components", donorLayer, catalog.MLPCount())
	}
	gateName, upName, downName, contract := component.Gate, component.Up, component.Down, component.Contract
	componentID, err := artifact.JSONID(artifact.KindTensorSet, struct {
		Model artifact.ID `json:"model"`
		Gate  string      `json:"gate"`
		Up    string      `json:"up"`
		Down  string      `json:"down"`
	}{donorID, gateName, upName, downName})
	if err != nil {
		return Result{}, err
	}
	if config.BaseLR <= 0 && config.LRScale > 0 {
		bridgeWeights := 2 * target.Dims.Hidden * donor.Dims.Hidden
		config.BaseLR = config.LRScale * optimizer.DeriveBaseLR(bridgeWeights)
	}
	baseline, _, err := target.Loss(config.HeldOut)
	if err != nil {
		return Result{}, fmt.Errorf("composition: baseline held-out loss: %w", err)
	}
	result := Result{
		Target: targetID, Donor: donorID, Component: componentID, Recipe: definition.ID,
		Dataset: dataset, Split: split, definition: definition,
		DonorContract: contract, DonorTensor: gateName, GraftLayer: graftLayer, Baseline: baseline,
	}
	worst := math.Inf(-1)
	for _, seed := range config.Seeds {
		graft, err := densecausal.NewGraft(
			target, graftLayer,
			donor.Weights[gateName], donor.Weights[upName], donor.Weights[downName],
			donor.Dims.Hidden, donor.Dims.Intermediate, seed,
		)
		if err != nil {
			return Result{}, err
		}
		losses, err := target.TrainBridge(graft, config.Train, config.BaseLR, config.Momentum, config.Steps)
		if err != nil {
			return Result{}, fmt.Errorf("composition: seed %d: %w", seed, err)
		}
		heldOut, err := target.GraftLoss(graft, config.HeldOut)
		if err != nil {
			return Result{}, fmt.Errorf("composition: seed %d held-out: %w", seed, err)
		}
		if math.IsNaN(heldOut) || math.IsInf(heldOut, 0) {
			return Result{}, fmt.Errorf("composition: seed %d held-out loss is non-finite", seed)
		}
		result.Outcomes = append(result.Outcomes, SeedOutcome{Seed: seed, TrainLosses: losses, HeldOut: heldOut})
		worst = math.Max(worst, heldOut)
	}
	if worst < baseline {
		result.Ship = true
		result.Reason = fmt.Sprintf(
			"envelope separation: worst seed held-out %.6f < baseline %.6f across %d seeds",
			worst, baseline, len(config.Seeds))
	} else {
		result.Reason = fmt.Sprintf(
			"refusal: worst seed held-out %.6f does not beat baseline %.6f; failure mode: bridge training did not produce held-out gain within %d steps",
			worst, baseline, config.Steps)
	}
	return result, nil
}

func identifyDenseModel(directory string) (artifact.ID, error) {
	file, err := os.Open(filepath.Join(directory, trainingprogram.CheckpointWeights))
	if err != nil {
		return artifact.ID{}, err
	}
	id, _, identifyErr := artifact.Identify(artifact.KindModel, file)
	return id, errors.Join(identifyErr, file.Close())
}

func bridgeRecipe(target, donor, dataset artifact.ID) (recipe.Definition, error) {
	nodes := []recipe.Node{
		{ID: "select", Module: workflowrecipe.ModuleSelectComponent, Placement: recipe.PlacementHost, ModelSlot: 1},
		{ID: "train", Module: workflowrecipe.ModuleTrainBridge, Placement: recipe.PlacementHost},
		{ID: "evaluate", Module: workflowrecipe.ModuleEvaluateBridge, Placement: recipe.PlacementHost},
	}
	edge := func(from recipe.NodeID, fromPort recipe.PortName, to recipe.NodeID, toPort recipe.PortName) recipe.Edge {
		return recipe.Edge{From: recipe.Endpoint{Node: from, Port: fromPort}, To: recipe.Endpoint{Node: to, Port: toPort}}
	}
	return recipe.NewDefinitionWithDependencies(recipe.TaskTraining, []recipe.Dependency{
		{Role: recipe.DependencyModel, Artifact: target},
		{Role: recipe.DependencyModel, Slot: 1, Artifact: donor},
		{Role: recipe.DependencyDataset, Artifact: dataset},
	}, nodes, []recipe.Edge{
		edge("select", "component", "train", "component"),
		edge("train", "bridge", "evaluate", "bridge"),
	}, nil, []recipe.Output{{Name: "metrics", Data: recipe.DataMetrics,
		Source: recipe.Endpoint{Node: "evaluate", Port: "metrics"}}})
}
