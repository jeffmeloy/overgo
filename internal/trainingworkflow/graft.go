package trainingworkflow

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/composition"
	"overgo/internal/densecausal"
	"overgo/internal/modelrecipe"
	"overgo/internal/organ"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/safetensors"
	"overgo/internal/trainingprogram"
	"overgo/internal/workflowrecipe"
)

// Graft viability's geometry: a negative layer selects the middle one, the
// count halved; the bridge is two matrices, into and out of the donor's
// width; a donor tensor is a matrix; and a held-out loss predicts each token
// after the first, so it needs two.
const (
	graftMiddleDivisor  = 2
	graftBridgeMatrices = 2
	donorMatrixRank     = 2
	graftHeldOutMinimum = 2
)

// GraftConfig is one graft viability run: the models, the donor component,
// the seeds and budget each bridge trains with, and the token windows.
type GraftConfig struct {
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

// GraftSeedOutcome is one seed's bridge training and held-out loss.
type GraftSeedOutcome struct {
	Seed        int64
	TrainLosses []float64
	HeldOut     float64
	// Bridge is the seed's trained bridge, whose weights and optimizer
	// state a checkpoint publishes.
	Bridge *densecausal.Graft
}

// GraftResult is a viability run's verdict and the identities it binds.
type GraftResult struct {
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
	Outcomes      []GraftSeedOutcome
	Ship          bool
	Reason        string
	definition    recipe.Definition
}

// RunGraftViability compiles, trains, evaluates and decides.
func RunGraftViability(config GraftConfig) (GraftResult, error) {
	target, err := identifyModel(config.TargetDir)
	if err != nil {
		return GraftResult{}, fmt.Errorf("graft: identify target: %w", err)
	}
	donor, err := identifyModel(config.DonorDir)
	if err != nil {
		return GraftResult{}, fmt.Errorf("graft: identify donor: %w", err)
	}
	return runGraftViability(config, target, donor)
}

func runGraftViability(config GraftConfig, targetID, donorID artifact.ID) (result GraftResult, err error) {
	if len(config.Seeds) == 0 || config.Steps <= 0 || len(config.Train) == 0 || len(config.HeldOut) < graftHeldOutMinimum {
		return GraftResult{}, fmt.Errorf("graft: seeds, steps, train batches and a held-out batch are required")
	}
	dataset, err := artifact.JSONID(artifact.KindDataset, config.Train)
	if err != nil {
		return GraftResult{}, err
	}
	split, err := artifact.JSONID(artifact.KindDatasetShard, config.HeldOut)
	if err != nil {
		return GraftResult{}, err
	}
	definition, err := bridgeRecipe(targetID, donorID, dataset)
	if err != nil {
		return GraftResult{}, err
	}
	program, err := recipe.CompileProgram(definition, workflowrecipe.Catalog())
	if err != nil {
		return GraftResult{}, fmt.Errorf("graft: compile bridge recipe: %w", err)
	}
	targetBytes, err := modelBytes(config.TargetDir)
	if err != nil {
		return GraftResult{}, err
	}
	donorBytes, err := modelBytes(config.DonorDir)
	if err != nil {
		return GraftResult{}, err
	}
	resources, err := modelrecipe.CompileComponentSessionPlanWithExtents(
		context.Background(), program, func(_ context.Context, id artifact.ID) (uint64, error) {
			switch id {
			case targetID:
				return targetBytes, nil
			case donorID:
				return donorBytes, nil
			default:
				return 0, errors.New("graft: recipe component model is unbound")
			}
		},
	)
	if err != nil {
		return GraftResult{}, err
	}
	sessions, err := capabilityruntime.NewComponentSessionDirector[viabilityResource](
		"bridge-synthesis", string(recipe.PlacementHost), len(resources.Components),
	)
	if err != nil {
		return GraftResult{}, err
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
				return viabilityResource{}, errors.New("graft: recipe component model is unbound")
			}
		})
		if leaseErr != nil {
			return GraftResult{}, leaseErr
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
		return GraftResult{}, errors.New("graft: component session plan omitted a required model")
	}
	graftLayer := config.GraftLayer
	if graftLayer < 0 {
		graftLayer = target.Dims.Layers / graftMiddleDivisor
	}
	componentID, err := artifact.JSONID(artifact.KindTensorSet, struct {
		Model artifact.ID `json:"model"`
		Gate  string      `json:"gate"`
		Up    string      `json:"up"`
		Down  string      `json:"down"`
	}{donorID, donor.gateName, donor.upName, donor.downName})
	if err != nil {
		return GraftResult{}, err
	}
	if config.BaseLR <= 0 && config.LRScale > 0 {
		bridgeWeights := graftBridgeMatrices * target.Dims.Hidden * donor.hidden
		config.BaseLR = config.LRScale * trainingprogram.BuiltinOptimizerPolicy().InitializationLearningRate(bridgeWeights)
	}
	baseline, _, err := target.Loss(config.HeldOut)
	if err != nil {
		return GraftResult{}, fmt.Errorf("graft: baseline held-out loss: %w", err)
	}
	result = GraftResult{
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
			return GraftResult{}, err
		}
		losses, err := target.TrainBridge(graft, config.Train, config.BaseLR, config.Momentum, config.Steps)
		if err != nil {
			return GraftResult{}, fmt.Errorf("graft: seed %d: %w", seed, err)
		}
		heldOut, err := target.GraftLoss(graft, config.HeldOut)
		if err != nil {
			return GraftResult{}, fmt.Errorf("graft: seed %d held-out: %w", seed, err)
		}
		if math.IsNaN(heldOut) || math.IsInf(heldOut, 0) {
			return GraftResult{}, fmt.Errorf("graft: seed %d held-out loss is non-finite", seed)
		}
		result.Outcomes = append(result.Outcomes, GraftSeedOutcome{Seed: seed, TrainLosses: losses, HeldOut: heldOut, Bridge: graft})
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
		return 0, errors.Join(errors.New("graft: model extent is unavailable"), err)
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
		layer = catalog.MLPCount() / graftMiddleDivisor
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
	if !ok || len(tensor.Shape) != donorMatrixRank {
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

// SynthesisOutcome is a bridge synthesis's terminal decision and evidence.
type SynthesisOutcome struct {
	Decision recipe.Decision
	Record   runrecord.GenerationRecord
	Result   GraftResult
}

// SynthesizeBridge takes a blocked proposal to a measured terminal decision.
func SynthesizeBridge(
	store *overgodb.Store,
	proposal composition.BridgeProposal,
	config GraftConfig,
	decider recipe.Decider,
) (SynthesisOutcome, error) {
	if store == nil {
		return SynthesisOutcome{}, errors.New("graft: synthesis requires a store")
	}
	if !proposal.ID.Valid() {
		return SynthesisOutcome{}, errors.New("graft: synthesis candidate lacks identity")
	}
	if proposal.State != composition.ProposalPromotionBlocked {
		return SynthesisOutcome{}, errors.New("graft: synthesis input must be a blocked proposal")
	}
	target, err := identifyModel(config.TargetDir)
	if err != nil {
		return SynthesisOutcome{}, err
	}
	donor, err := identifyModel(config.DonorDir)
	if err != nil {
		return SynthesisOutcome{}, err
	}
	if target != proposal.Target {
		return SynthesisOutcome{}, errors.New("graft: proposal target differs from synthesis target")
	}
	selected := composition.BridgeCandidate{}
	for _, candidate := range proposal.Candidates {
		if candidate.Donor == donor {
			selected = candidate
			break
		}
	}
	if !selected.Donor.Valid() {
		return SynthesisOutcome{}, errors.New("graft: proposal does not admit the synthesis donor")
	}
	config.DonorTensor = selected.Component
	result, err := runGraftViability(config, target, donor)
	if err != nil {
		return SynthesisOutcome{}, err
	}
	record, err := RecordGraftViability(store, config, result)
	if err != nil {
		return SynthesisOutcome{}, err
	}
	outcome := recipe.DecisionRefused
	tier := recipe.EvidenceVerified
	reason := result.Reason
	if result.Ship {
		outcome = recipe.DecisionAccepted
		tier = recipe.EvidenceParity
		reason = "bridge synthesis produced held-out gain: " + result.Reason
	}
	decision, err := recipe.NewDecision(
		proposal.ID, outcome, tier, reason, decider,
		[]artifact.ID{record.ID, record.Run, record.Decision},
	)
	if err != nil {
		return SynthesisOutcome{}, err
	}
	batch, err := decision.Batch("bridge-synthesis/decision/" + decision.ID.String())
	if err != nil {
		return SynthesisOutcome{}, err
	}
	if _, err := artifact.CommitBatch(context.Background(), store, batch); err != nil {
		return SynthesisOutcome{}, err
	}
	return SynthesisOutcome{Decision: decision, Record: record, Result: result}, nil
}

var (
	viabilityEvaluatorContract = artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: "application/vnd.overgo.viability-evaluator+json", Schema: "overgo/viability-evaluator/v1",
	}
	viabilityVerdictContract = artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: "application/vnd.overgo.viability-verdict+json", Schema: "overgo/viability-verdict/v1",
	}
)

const (
	baselineHeldOutMetric = "baseline-heldout-loss"
	worstHeldOutMetric    = "worst-seed-heldout-loss"
)

type viabilityEvaluator struct {
	Recipe  artifact.ID `json:"recipe"`
	Dataset artifact.ID `json:"dataset"`
	Split   artifact.ID `json:"split"`
	Metrics []string    `json:"metrics"`
}

type viabilityVerdict struct {
	Evaluation artifact.ID `json:"evaluation"`
	Ship       bool        `json:"ship"`
	Reason     string      `json:"reason"`
}

// RecordGraftViability commits atomically a run's recipe, run, evaluation,
// verdict and lineage.
func RecordGraftViability(store *overgodb.Store, config GraftConfig, result GraftResult) (runrecord.GenerationRecord, error) {
	if result.Target.Kind() != artifact.KindModel || result.Donor.Kind() != artifact.KindModel ||
		result.Component.Kind() != artifact.KindTensorSet || result.Recipe.Kind() != artifact.KindRecipe ||
		result.SessionPlan.Identity.Kind() != artifact.KindProfile ||
		result.Dataset.Kind() != artifact.KindDataset || result.Split.Kind() != artifact.KindDatasetShard ||
		len(result.Outcomes) == 0 || result.definition.ID != result.Recipe ||
		result.WorstHeldOut < 0 || math.IsNaN(result.WorstHeldOut) || math.IsInf(result.WorstHeldOut, 0) {
		return runrecord.GenerationRecord{}, fmt.Errorf("graft: viability authority is incomplete")
	}
	protocol := fmt.Sprintf("target=%s;donor=%s;tensor=%s;layer=%d;steps=%d;seeds=%d;lrscale=%g",
		result.Target, result.Donor, result.DonorTensor, result.GraftLayer, config.Steps, len(config.Seeds), config.LRScale)
	seeds := make([]uint64, len(config.Seeds))
	for index, seed := range config.Seeds {
		seeds[index] = uint64(seed)
	}
	outcome := runrecord.OutcomeFailed
	failure := "held-out-gain-absent"
	if result.Ship {
		outcome = runrecord.OutcomeSucceeded
		failure = ""
	}
	bridge, err := artifact.JSONID(artifact.KindAdapter, struct {
		Recipe    artifact.ID `json:"recipe"`
		Component artifact.ID `json:"component"`
		Protocol  string      `json:"protocol"`
	}{result.Recipe, result.Component, protocol})
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	child, err := artifact.JSONID(artifact.KindModel, struct {
		Target artifact.ID `json:"target"`
		Donor  artifact.ID `json:"donor"`
		Bridge artifact.ID `json:"bridge"`
	}{result.Target, result.Donor, bridge})
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	run, err := runrecord.NewRun(result.Recipe, outcome,
		[]artifact.ID{result.Target, result.Donor, result.Component, result.SessionPlan.Identity, result.Dataset, result.Split},
		[]artifact.ID{child, bridge}, failure)
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	evaluation, err := runrecord.NewEvaluation(result.Recipe, run.ID, result.Dataset, []runrecord.Metric{
		{Name: baselineHeldOutMetric, Value: result.Baseline, Direction: runrecord.DirectionMinimize},
		{Name: worstHeldOutMetric, Value: result.WorstHeldOut, Direction: runrecord.DirectionMinimize},
	})
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	evaluator, err := artifact.JSONContent(viabilityEvaluatorContract, viabilityEvaluator{
		Recipe: result.Recipe, Dataset: result.Dataset, Split: result.Split,
		Metrics: []string{baselineHeldOutMetric, worstHeldOutMetric},
	})
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	verdict, err := artifact.JSONContent(viabilityVerdictContract, viabilityVerdict{
		Evaluation: evaluation.ID, Ship: result.Ship, Reason: result.Reason,
	})
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	identifyEvidence := func(payload string) (artifact.ID, error) {
		return artifact.IdentifyBytes(artifact.KindEvidence, []byte("graft-probe/v2:"+payload))
	}
	code, err := identifyEvidence("code:" + protocol)
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	environment, err := identifyEvidence("environment:host-reference")
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	budget, err := identifyEvidence(fmt.Sprintf("budget:seeds=%d;steps=%d", len(seeds), config.Steps))
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	record, err := runrecord.NewGenerationRecord(runrecord.GenerationRecord{
		Parents:      []artifact.ID{result.Target, result.Donor},
		Child:        child,
		Components:   []artifact.ID{result.Component},
		Bridge:       bridge,
		TrainingPlan: result.Recipe,
		Dataset:      result.Dataset,
		Split:        result.Split,
		Evaluator:    evaluator.Descriptor.ID,
		Code:         code,
		Environment:  environment,
		Seeds:        seeds,
		Budget:       budget,
		Run:          run.ID,
		Outcome:      outcome,
		Decision:     verdict.Descriptor.ID,
	})
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	static := append([]artifact.ID{record.Child, record.Bridge, record.Dataset,
		record.Split, record.Code, record.Environment, record.Budget}, record.Parents...)
	static = append(static, record.Components...)
	descriptors := make([]artifact.Descriptor, len(static))
	for index, id := range static {
		descriptors[index] = artifact.Descriptor{ID: id}
	}
	recipeContent, err := result.definition.ArtifactContent()
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	sessionContent, err := result.SessionPlan.Content()
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	runContent, err := run.Content()
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	evaluationContent, err := evaluation.Content()
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	recordContent, err := record.Content()
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	lineage := append(run.Lineage(), evaluation.Lineage()...)
	lineage = append(lineage, artifact.DependencyLineage(
		evaluator.Descriptor.ID, result.Recipe, result.Dataset, result.Split,
	)...)
	lineage = append(lineage, artifact.DependencyLineage(verdict.Descriptor.ID, evaluation.ID)...)
	lineage = append(lineage, record.Lineage()...)
	ctx := context.Background()
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "graft-probe/generation/" + record.ID.String(), Artifacts: descriptors,
		Contents: []artifact.Content{recipeContent, sessionContent, runContent, evaluationContent, evaluator, verdict, recordContent},
		Lineage:  lineage,
	}); err != nil {
		return runrecord.GenerationRecord{}, err
	}
	return record, nil
}
