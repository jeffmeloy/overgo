// Package composition evaluates catalog-proposed component bridges.
package composition

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/densecausal"
	"overgo/internal/modelrecipe"
	"overgo/internal/organ"
	"overgo/internal/recipe"
	"overgo/internal/safetensors"
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
	SessionPlan   modelrecipe.ComponentSessionPlan
	Dataset       artifact.ID
	Split         artifact.ID
	DonorContract organ.Contract
	DonorTensor   string
	GraftLayer    int
	Baseline      float64
	WorstHeldOut  float64
	Outcomes      []SeedOutcome
	Ship          bool
	Reason        string
	definition    recipe.Definition
}

// RunViability: compile, train, evaluate, decide.
func RunViability(config Config) (Result, error) {
	target, err := identifyDenseModel(config.TargetDir)
	if err != nil {
		return Result{}, fmt.Errorf("composition: identify target: %w", err)
	}
	donor, err := identifyDenseModel(config.DonorDir)
	if err != nil {
		return Result{}, fmt.Errorf("composition: identify donor: %w", err)
	}
	return runViability(config, target, donor)
}

func runViability(config Config, targetID, donorID artifact.ID) (result Result, err error) {
	if len(config.Seeds) == 0 || config.Steps <= 0 || len(config.Train) == 0 || len(config.HeldOut) < 2 {
		return Result{}, fmt.Errorf("composition: seeds, steps, train batches and a held-out batch are required")
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
	program, err := recipe.CompileProgram(definition, workflowrecipe.Catalog())
	if err != nil {
		return Result{}, fmt.Errorf("composition: compile bridge recipe: %w", err)
	}
	targetBytes, err := modelBytes(config.TargetDir)
	if err != nil {
		return Result{}, err
	}
	donorBytes, err := modelBytes(config.DonorDir)
	if err != nil {
		return Result{}, err
	}
	resources, err := modelrecipe.CompileComponentSessionPlanWithExtents(
		context.Background(), program, func(_ context.Context, id artifact.ID) (uint64, error) {
			switch id {
			case targetID:
				return targetBytes, nil
			case donorID:
				return donorBytes, nil
			default:
				return 0, errors.New("composition: recipe component model is unbound")
			}
		},
	)
	if err != nil {
		return Result{}, err
	}
	sessions, err := capabilityruntime.NewComponentSessionDirector[viabilityResource](
		"bridge-synthesis", string(recipe.PlacementHost), len(resources.Components),
	)
	if err != nil {
		return Result{}, err
	}
	leases := make([]*capabilityruntime.SessionLease[viabilityResource], 0, len(resources.Components))
	defer func() {
		for index := len(leases) - 1; index >= 0; index-- {
			err = errors.Join(err, leases[index].Release())
		}
		err = errors.Join(err, sessions.Close(context.Background()))
	}()
	var target *densecausal.Model
	var donor *donorComponent
	for _, component := range resources.Components {
		lease, leaseErr := sessions.LeaseComponent(context.Background(), component, func(context.Context) (viabilityResource, error) {
			switch component.Model {
			case targetID:
				model, loadErr := densecausal.Load(config.TargetDir)
				return viabilityResource{target: model}, loadErr
			case donorID:
				selected, loadErr := loadDonorComponent(config.DonorDir, config.DonorLayer, config.DonorTensor)
				return viabilityResource{donor: &selected}, loadErr
			default:
				return viabilityResource{}, errors.New("composition: recipe component model is unbound")
			}
		})
		if leaseErr != nil {
			return Result{}, leaseErr
		}
		leases = append(leases, lease)
		loaded := lease.Model()
		if loaded.target != nil {
			target = loaded.target
		}
		if loaded.donor != nil {
			donor = loaded.donor
		}
	}
	if target == nil || donor == nil {
		return Result{}, errors.New("composition: component session plan omitted a required model")
	}
	graftLayer := config.GraftLayer
	if graftLayer < 0 {
		graftLayer = target.Dims.Layers / 2
	}
	componentID, err := artifact.JSONID(artifact.KindTensorSet, struct {
		Model artifact.ID `json:"model"`
		Gate  string      `json:"gate"`
		Up    string      `json:"up"`
		Down  string      `json:"down"`
	}{donorID, donor.gateName, donor.upName, donor.downName})
	if err != nil {
		return Result{}, err
	}
	if config.BaseLR <= 0 && config.LRScale > 0 {
		bridgeWeights := 2 * target.Dims.Hidden * donor.hidden
		config.BaseLR = config.LRScale * trainingprogram.BuiltinOptimizerPolicy().BaseLearningRate(bridgeWeights)
	}
	baseline, _, err := target.Loss(config.HeldOut)
	if err != nil {
		return Result{}, fmt.Errorf("composition: baseline held-out loss: %w", err)
	}
	result = Result{
		Target: targetID, Donor: donorID, Component: componentID, Recipe: definition.ID,
		SessionPlan: resources,
		Dataset:     dataset, Split: split, definition: definition,
		DonorContract: donor.contract, DonorTensor: donor.gateName, GraftLayer: graftLayer, Baseline: baseline,
	}
	worst := math.Inf(-1)
	for _, seed := range config.Seeds {
		graft, err := densecausal.NewGraft(
			target, graftLayer,
			donor.gate, donor.up, donor.down, donor.hidden, donor.intermediate, seed,
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
		worst = max(worst, heldOut)
	}
	result.WorstHeldOut = worst
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

type viabilityResource struct {
	target *densecausal.Model
	donor  *donorComponent
}

func modelBytes(directory string) (uint64, error) {
	info, err := os.Stat(filepath.Join(directory, trainingprogram.CheckpointWeights))
	if err != nil || info.IsDir() || info.Size() <= 0 {
		return 0, errors.Join(errors.New("composition: model extent is unavailable"), err)
	}
	return uint64(info.Size()), nil
}

type donorComponent struct {
	gateName, upName, downName string
	gate, up, down             []float32
	hidden, intermediate       int
	contract                   organ.Contract
}

func loadDonorComponent(directory string, layer int, tensorName string) (_ donorComponent, err error) {
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		return donorComponent{}, err
	}
	defer func() { err = errors.Join(err, source.Close()) }()
	catalog, err := organ.CompileCatalog(source.Names())
	if err != nil {
		return donorComponent{}, err
	}
	if tensorName != "" {
		layer = -1
		for index := range catalog.MLPCount() {
			candidate, _ := catalog.MLP(index)
			if candidate.Gate == tensorName || candidate.Up == tensorName || candidate.Down == tensorName {
				layer = index
				break
			}
		}
	} else if layer < 0 {
		layer = catalog.MLPCount() / 2
	}
	component, ok := catalog.MLP(layer)
	if !ok {
		return donorComponent{}, fmt.Errorf("donor MLP index %d outside %d cataloged components", layer, catalog.MLPCount())
	}
	gate, gateRows, gateCols, err := readDonorMatrix(source, component.Gate)
	if err != nil {
		return donorComponent{}, err
	}
	up, upRows, upCols, err := readDonorMatrix(source, component.Up)
	if err != nil {
		return donorComponent{}, err
	}
	down, downRows, downCols, err := readDonorMatrix(source, component.Down)
	if err != nil {
		return donorComponent{}, err
	}
	if upRows != gateRows || upCols != gateCols || downRows != gateCols || downCols != gateRows {
		return donorComponent{}, fmt.Errorf("donor MLP tensor geometry differs")
	}
	return donorComponent{
		gateName: component.Gate, upName: component.Up, downName: component.Down,
		gate: gate, up: up, down: down, hidden: gateCols, intermediate: gateRows,
		contract: component.Contract,
	}, nil
}

func readDonorMatrix(source *safetensors.Source, name string) ([]float32, int, int, error) {
	tensor, ok := source.Tensors[name]
	if !ok || len(tensor.Shape) != 2 {
		return nil, 0, 0, fmt.Errorf("donor tensor %q is not a matrix", name)
	}
	rows, cols := int(tensor.Shape[0]), int(tensor.Shape[1])
	if rows <= 0 || cols <= 0 || uint64(rows) != tensor.Shape[0] || uint64(cols) != tensor.Shape[1] {
		return nil, 0, 0, fmt.Errorf("donor tensor %q geometry exceeds host range", name)
	}
	values, err := safetensors.ReadF32(tensor)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("donor tensor %q: %w", name, err)
	}
	return values, rows, cols, nil
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
		{ID: "select", Module: workflowrecipe.ModuleSelectComponent, Placement: recipe.PlacementHost, ModelSlot: 1,
			Session: recipe.SessionRequest, Residency: recipe.ResidencyStream},
		{ID: "train", Module: workflowrecipe.ModuleTrainBridge, Placement: recipe.PlacementHost,
			Session: recipe.SessionCapacity, Residency: recipe.ResidencyHostCache},
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
