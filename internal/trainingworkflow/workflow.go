// Package trainingworkflow executes compiled native training workflows.
package trainingworkflow

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/densecausal"
	artifactexport "overgo/internal/export"
	"overgo/internal/hfbpe"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/safetensors"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
	"overgo/internal/workflowrecipe"
)

type Request struct {
	Repository         artifact.Reader
	Recipe             artifact.ID
	ModelDirectory     string
	DatasetPath        string
	OutputDirectory    string
	ResumeDirectory    string
	ReferenceDirectory string
	Steps              int
	MaximumSequence    int
	LearningRate       float64
	Momentum           float64
	ObjectiveScale     float64
	Host               bool
	FreezeLexical      bool
	ObserveDPO         func(trainingprogram.DPOObservation)
	ObserveGRPO        func(trainingprogram.GRPOObservation)
	Progress           io.Writer
	MaxProjectedWall   time.Duration
}

func stepGuard(request Request, backend string) densecausal.TrainObserver {
	return func(step int, loss float64, stepWall time.Duration) error {
		if request.Progress != nil {
			fmt.Fprintf(request.Progress, "step %d/%d loss %.6f wall %s backend %s\n",
				step+1, request.Steps, loss, stepWall.Round(time.Millisecond), backend)
		}
		if request.MaxProjectedWall > 0 {
			projected := stepWall * time.Duration(request.Steps)
			if projected > request.MaxProjectedWall {
				return fmt.Errorf(
					"training workflow: step %d took %s on backend %s; %d steps project to %s, over the %s bound -- rerun with a longer -max-wall if intended",
					step+1, stepWall.Round(time.Second), backend, request.Steps,
					projected.Round(time.Minute), request.MaxProjectedWall)
			}
		}
		return nil
	}
}

type Result struct {
	Backend        string
	Objective      trainingprogram.ObjectiveKind
	Losses         []float64
	DPO            []trainingprogram.DPOObservation
	GRPO           []trainingprogram.GRPOObservation
	StreamPosition uint64
	Checkpoint     trainingprogram.Checkpoint
}

func Execute(ctx context.Context, request Request) (Result, error) {
	if ctx == nil || request.Repository == nil || request.Recipe.Kind() != artifact.KindRecipe ||
		request.ModelDirectory == "" && request.ResumeDirectory == "" ||
		request.DatasetPath == "" || request.OutputDirectory == "" || request.Steps <= 0 {
		return Result{}, errors.New("training workflow: repository, recipe, model or resume, dataset, output, and positive steps required")
	}
	if request.Host && request.FreezeLexical {
		return Result{}, errors.New("training workflow: host and frozen lexical execution are incompatible")
	}
	inputDirectory := request.ModelDirectory
	var resumed trainingprogram.Checkpoint
	var resumeStream *trainingdata.StreamState
	if request.ResumeDirectory != "" {
		var err error
		resumed, err = trainingprogram.LoadCheckpoint(request.ResumeDirectory)
		if err != nil {
			return Result{}, fmt.Errorf("training workflow: load checkpoint: %w", err)
		}
		inputDirectory = request.ResumeDirectory
		resumeStream = &trainingdata.StreamState{Identity: resumed.Stream.Identity, Position: resumed.Stream.Position}
	}
	modelID := resumed.Model
	if !modelID.Valid() {
		var err error
		modelID, err = identifyModel(inputDirectory)
		if err != nil {
			return Result{}, fmt.Errorf("training workflow: identify model: %w", err)
		}
	}
	_, runtime, err := modelrecipe.ResolveActiveCapability(ctx, request.Repository, modelID, recipe.TaskTraining)
	if err != nil {
		return Result{}, fmt.Errorf("training workflow: resolve active recipe: %w", err)
	}
	if runtime.Definition().ID != request.Recipe {
		return Result{}, errors.New("training workflow: active recipe differs")
	}
	objective, err := ProgramObjective(runtime)
	if err != nil {
		return Result{}, err
	}
	switch objective {
	case trainingprogram.ObjectiveDPO:
		if request.ReferenceDirectory == "" || request.ObjectiveScale <= 0 || request.FreezeLexical {
			return Result{}, errors.New("training workflow: DPO inputs differ from recipe")
		}
	case trainingprogram.ObjectiveGRPO:
		if request.ReferenceDirectory != "" || request.ObjectiveScale <= 0 || request.FreezeLexical {
			return Result{}, errors.New("training workflow: GRPO inputs differ from recipe")
		}
	default:
		if request.ReferenceDirectory != "" || request.ObjectiveScale != 0 {
			return Result{}, errors.New("training workflow: token recipe rejects RL inputs")
		}
	}
	model, err := densecausal.Load(inputDirectory)
	if err != nil {
		return Result{}, fmt.Errorf("training workflow: load model: %w", err)
	}
	tokenizer, err := hfbpe.Load(inputDirectory)
	if err != nil {
		return Result{}, fmt.Errorf("training workflow: load tokenizer: %w", err)
	}
	raw, err := os.ReadFile(request.DatasetPath)
	if err != nil {
		return Result{}, fmt.Errorf("training workflow: read dataset: %w", err)
	}
	return denseSession{
		ctx: ctx, request: request, runtime: runtime, objective: objective,
		inputDirectory: inputDirectory, model: model, encode: tokenizer.Encode, raw: raw,
		resumed: resumed, resumeStream: resumeStream,
	}.run()
}

type denseSession struct {
	ctx            context.Context
	request        Request
	runtime        recipe.Program
	objective      trainingprogram.ObjectiveKind
	inputDirectory string
	model          *densecausal.Model
	encode         func(string) ([]int, error)
	raw            []byte
	resumed        trainingprogram.Checkpoint
	resumeStream   *trainingdata.StreamState
}

type densePrepared struct {
	data       batchAuthority
	stream     trainingdata.StreamState
	preference *trainingprogram.PreferencePolicy
	reference  *densecausal.Model
	tokens     [][]int
	pairs      []trainingdata.PreferenceBatch
	groups     []trainingdata.RolloutGroup
	evaluators []artifact.ID
}

func (session denseSession) run() (Result, error) {
	prepared, err := session.prepare()
	if err != nil {
		return Result{}, err
	}
	authority, err := compileAuthority(
		session.ctx, session.request.Repository, session.runtime, session.model, session.inputDirectory,
		prepared.data, prepared.stream, session.request.LearningRate, session.resumed,
		session.objective, prepared.preference, prepared.evaluators,
	)
	if err != nil {
		return Result{}, fmt.Errorf("training workflow: compile authority: %w", err)
	}
	result, state, err := session.execute(prepared)
	if err != nil {
		return Result{}, err
	}
	result.Checkpoint, err = publishCheckpoint(
		session.inputDirectory, session.request.OutputDirectory, session.model, authority, state,
	)
	return result, err
}

func (session denseSession) prepare() (densePrepared, error) {
	if session.objective == trainingprogram.ObjectiveTokenPrediction {
		batches, stream, data, err := tokenBatchesResume(
			session.ctx, session.raw, session.request.Steps, session.request.MaximumSequence,
			session.encode, session.resumeStream,
		)
		return densePrepared{data: data, stream: stream, tokens: batches}, err
	}
	if session.objective == trainingprogram.ObjectiveGRPO {
		groups, stream, data, evaluators, err := groupedRolloutBatchesResume(
			session.ctx, session.raw, session.request.Steps, session.encode, session.resumeStream,
		)
		return densePrepared{data: data, stream: stream, groups: groups, evaluators: evaluators}, err
	}
	reference, err := densecausal.Load(session.request.ReferenceDirectory)
	if err != nil {
		return densePrepared{}, fmt.Errorf("training workflow: load reference: %w", err)
	}
	batches, stream, data, err := preferenceBatchesResume(
		session.ctx, session.raw, session.request.Steps, session.encode, session.resumeStream,
	)
	if err != nil {
		return densePrepared{}, err
	}
	referenceID, err := identifyModel(session.request.ReferenceDirectory)
	if err != nil {
		return densePrepared{}, fmt.Errorf("training workflow: identify reference: %w", err)
	}
	return densePrepared{
		data: data, stream: stream, reference: reference, pairs: batches,
		preference: &trainingprogram.PreferencePolicy{Reference: referenceID, Scale: session.request.ObjectiveScale},
	}, nil
}

func (session denseSession) execute(prepared densePrepared) (Result, densecausal.TrainState, error) {
	if session.objective == trainingprogram.ObjectiveGRPO {
		var resume *densecausal.RLState
		if session.resumed.ID().Valid() {
			resume = &densecausal.RLState{Optimizer: session.resumed.Optimizer, Stream: *session.resumeStream}
		}
		observations, state, err := session.model.TrainGRPOGroupsResume(
			prepared.groups, session.request.LearningRate, session.request.Momentum,
			session.request.ObjectiveScale, resume, session.request.ObserveGRPO,
		)
		return Result{Backend: "host", Objective: session.objective, GRPO: observations,
			StreamPosition: state.Stream.Position}, state.Optimizer, err
	}
	if session.objective == trainingprogram.ObjectiveDPO {
		var resume *densecausal.RLState
		if session.resumed.ID().Valid() {
			resume = &densecausal.RLState{Optimizer: session.resumed.Optimizer, Stream: *session.resumeStream}
		}
		observations, state, err := session.model.TrainDPOBatchesResume(
			prepared.reference, prepared.pairs, session.request.LearningRate, session.request.Momentum,
			session.request.ObjectiveScale, resume, session.request.ObserveDPO,
		)
		return Result{
			Backend: "host", Objective: session.objective, DPO: observations,
			StreamPosition: state.Stream.Position,
		}, state.Optimizer, err
	}
	var resume *densecausal.TrainState
	if session.resumed.ID().Valid() {
		state := densecausal.TrainState(session.resumed.Optimizer)
		resume = &state
	}
	lane := "cuda-resident"
	if session.request.Host {
		lane = "host-reference"
	}
	if session.request.Progress != nil {
		fmt.Fprintf(session.request.Progress, "lane %s: %d steps, %d batches, frozen_lexical=%t\n",
			lane, session.request.Steps, len(prepared.tokens), session.request.FreezeLexical)
	}
	losses, backend, state, err := runTrainingState(
		session.model, prepared.tokens, session.request.LearningRate, session.request.Momentum,
		!session.request.Host, session.request.FreezeLexical, resume, stepGuard(session.request, lane),
	)
	return Result{
		Backend: backend, Objective: session.objective, Losses: losses,
		StreamPosition: prepared.stream.Position,
	}, state, err
}

func publishCheckpoint(source, target string, model *densecausal.Model, authority compiledAuthority, state densecausal.TrainState) (trainingprogram.Checkpoint, error) {
	spec, err := authority.checkpointSpec(state)
	if err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	checkpoint, err := trainingprogram.PublishCheckpoint(target, spec, func(stage string) error {
		metadata := map[string]string{"format": "pt", "trainer": "overgo"}
		if err := safetensors.Save(filepath.Join(stage, trainingprogram.CheckpointWeights), model.Weights, model.Shapes, metadata); err != nil {
			return err
		}
		return artifactexport.CopyFiles(source, stage, []string{"config.json", "tokenizer.json"})
	})
	if err != nil {
		return trainingprogram.Checkpoint{}, fmt.Errorf("training workflow: publish checkpoint: %w", err)
	}
	return checkpoint, nil
}

func identifyModel(directory string) (artifact.ID, error) {
	path := filepath.Join(directory, trainingprogram.CheckpointWeights)
	if _, err := os.Stat(path); err != nil {
		// Single shard: direct identity. Multi-shard: manifest required.
		shards, globErr := filepath.Glob(filepath.Join(directory, "model-*-of-*.safetensors"))
		if globErr != nil || len(shards) == 0 {
			return artifact.ID{}, err
		}
		if len(shards) > 1 {
			return artifact.ID{}, fmt.Errorf(
				"training workflow: %d weight shards in %s; multi-shard identity requires a manifest", len(shards), directory)
		}
		path = shards[0]
	}
	file, err := os.Open(path)
	if err != nil {
		return artifact.ID{}, err
	}
	id, _, identifyErr := artifact.Identify(artifact.KindModel, file)
	return id, errors.Join(identifyErr, file.Close())
}

type compiledAuthority struct {
	model, program, runPlan artifact.ID
	optimizerPlan           string
	data                    batchAuthority
	stream                  trainingdata.StreamState
	rng                     []trainingprogram.RNGState
	lineage                 []trainingprogram.LineageParent
	projectors, codecs      []artifact.ID
}

func compileAuthority(
	ctx context.Context,
	repository artifact.Reader,
	runtime recipe.Program,
	model *densecausal.Model,
	modelDirectory string,
	data batchAuthority,
	stream trainingdata.StreamState,
	learningRate float64,
	resumed trainingprogram.Checkpoint,
	objective trainingprogram.ObjectiveKind,
	preference *trainingprogram.PreferencePolicy,
	evaluators []artifact.ID,
) (compiledAuthority, error) {
	modelID := resumed.Model
	var err error
	if !modelID.Valid() {
		modelID, err = identifyModel(modelDirectory)
		if err != nil {
			return compiledAuthority{}, err
		}
	}
	muonPlan, _, err := model.TrainingPlan(learningRate)
	if err != nil {
		return compiledAuthority{}, err
	}
	parameters := make([]trainingprogram.ParameterSpec, muonPlan.GroupCount())
	for index := range parameters {
		group, _ := muonPlan.Group(index)
		parameters[index] = trainingprogram.ParameterSpec{Name: group.Name, Rows: group.Rows, Cols: group.Cols, Trainable: !group.Frozen}
	}
	operators := tokenOperators
	if objective == trainingprogram.ObjectiveDPO {
		operators = dpoOperators
	} else if objective == trainingprogram.ObjectiveGRPO {
		operators = grpoOperators
	}
	program, err := trainingprogram.CompileTrainingProgram(trainingprogram.ProgramSpec{
		Objective: objective, Operators: operators, Parameters: parameters, Optimizer: muonPlan, Preference: preference,
	})
	if err != nil {
		return compiledAuthority{}, err
	}
	definition := runtime.Definition()
	if definition.Task != recipe.TaskTraining || definition.Model != modelID {
		return compiledAuthority{}, errors.New("training workflow: recipe model or task differs")
	}
	if preference != nil {
		reference, ok := definition.Dependency(recipe.DependencyModel, 1)
		if !ok || reference != preference.Reference {
			return compiledAuthority{}, errors.New("training workflow: recipe reference differs")
		}
	}
	if err := validateEvaluators(definition, evaluators); err != nil {
		return compiledAuthority{}, err
	}
	policies, err := trainingprogram.PoliciesFromRecipe(definition)
	if err != nil {
		return compiledAuthority{}, err
	}
	initial := trainingprogram.InitialStateSpec{Model: modelID}
	if resumed.ID().Valid() {
		initial = trainingprogram.InitialStateSpec{Checkpoint: resumed.ID()}
	}
	runPlan, err := trainingprogram.CompileTrainingRunPlanFromRepository(ctx, repository, trainingprogram.RunSpec{
		Recipe: definition.ID, Initial: initial, Dataset: data.Dataset, Split: data.Split,
		Signature:  recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{recipecontract.ModalityText}, Outputs: []recipecontract.Modality{recipecontract.ModalityText}},
		Processors: []artifact.ID{data.Processor},
		Policies:   policies,
		Program:    program,
	})
	if err != nil {
		return compiledAuthority{}, err
	}
	if resumed.ID().Valid() {
		if err := trainingprogram.ValidateResume(runPlan, resumed, stream.Identity); err != nil {
			return compiledAuthority{}, err
		}
	}
	rngAlgorithm, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("overgo/counter-rng/v1"))
	rng := append([]trainingprogram.RNGState(nil), resumed.RNG...)
	if len(rng) == 0 {
		rng = []trainingprogram.RNGState{{Name: "augmentation", Algorithm: rngAlgorithm}, {Name: "data", Algorithm: rngAlgorithm}}
	}
	lineage := []trainingprogram.LineageParent{
		{Artifact: modelID, Relation: artifact.RelationTrainedFrom},
		{Artifact: data.Dataset, Relation: artifact.RelationDependsOn},
		{Artifact: data.Split, Relation: artifact.RelationDependsOn},
		{Artifact: data.Processor, Relation: artifact.RelationTokenizedBy},
		{Artifact: program.ID(), Relation: artifact.RelationProducedBy},
		{Artifact: runPlan.ID(), Relation: artifact.RelationDependsOn},
	}
	if preference != nil {
		lineage = append(lineage, trainingprogram.LineageParent{Artifact: preference.Reference, Relation: artifact.RelationDependsOn})
	}
	for _, evaluator := range evaluators {
		lineage = append(lineage, trainingprogram.LineageParent{Artifact: evaluator, Relation: artifact.RelationDependsOn})
	}
	if resumed.ID().Valid() {
		lineage = append(lineage, trainingprogram.LineageParent{Artifact: resumed.ID(), Relation: artifact.RelationDerivedFrom})
	}
	return compiledAuthority{
		model: modelID, program: program.ID(), runPlan: runPlan.ID(), optimizerPlan: muonPlan.Identity(),
		data: data, stream: stream, rng: rng, lineage: lineage, projectors: resumed.Projectors, codecs: resumed.Codecs,
	}, nil
}

func validateEvaluators(definition recipe.Definition, evaluators []artifact.ID) error {
	for index, evaluator := range evaluators {
		declared, ok := definition.Dependency(recipe.DependencyEvaluator, uint32(index))
		if !ok || declared != evaluator {
			return errors.New("training workflow: rollout evaluator differs from recipe")
		}
	}
	if _, extra := definition.Dependency(recipe.DependencyEvaluator, uint32(len(evaluators))); extra {
		return errors.New("training workflow: recipe evaluator is unused")
	}
	return nil
}

func ProgramObjective(program recipe.Program) (trainingprogram.ObjectiveKind, error) {
	stages := program.Stages()
	modules := make([]recipe.ModuleID, len(stages))
	for index, stage := range stages {
		modules[index] = stage.Module.ID
	}
	if slices.Equal(modules, tokenModules) {
		return trainingprogram.ObjectiveTokenPrediction, nil
	}
	if slices.Equal(modules, dpoModules) {
		return trainingprogram.ObjectiveDPO, nil
	}
	if slices.Equal(modules, grpoModules) {
		return trainingprogram.ObjectiveGRPO, nil
	}
	return "", errors.New("training workflow: active recipe has no supported dense objective")
}

func (authority compiledAuthority) checkpointSpec(state densecausal.TrainState) (trainingprogram.CheckpointSpec, error) {
	if state.PlanIdentity != authority.optimizerPlan {
		return trainingprogram.CheckpointSpec{}, errors.New("training workflow: checkpoint optimizer plan differs")
	}
	rng := append([]trainingprogram.RNGState(nil), authority.rng...)
	for index := range rng {
		switch rng[index].Name {
		case "augmentation":
			rng[index].Counter = uint64(state.Step)
		case "data":
			rng[index].Counter = authority.stream.Position
		}
	}
	return trainingprogram.CheckpointSpec{
		RunPlan: authority.runPlan, Program: authority.program, Model: authority.model,
		Dataset: authority.data.Dataset, Split: authority.data.Split,
		Stream:    trainingprogram.DatasetState{Identity: authority.stream.Identity, Position: authority.stream.Position},
		Optimizer: state, ParameterCount: len(state.Momentum), RNG: rng,
		Processors: []artifact.ID{authority.data.Processor}, Projectors: authority.projectors, Codecs: authority.codecs,
		Lineage: authority.lineage,
	}, nil
}

var tokenOperators = []trainingprogram.OperatorSpec{
	{ID: "dense-forward", Phase: trainingprogram.PhaseForward},
	{ID: "dense-backward", Phase: trainingprogram.PhaseBackward},
	{ID: "muon", Phase: trainingprogram.PhaseOptimize},
}

var dpoOperators = []trainingprogram.OperatorSpec{
	{ID: "policy-score", Phase: trainingprogram.PhaseForward},
	{ID: "reference-score", Phase: trainingprogram.PhaseForward},
	{ID: "dpo", Phase: trainingprogram.PhaseLoss},
	{ID: "dense-backward", Phase: trainingprogram.PhaseBackward},
	{ID: "muon", Phase: trainingprogram.PhaseOptimize},
}

var grpoOperators = []trainingprogram.OperatorSpec{
	{ID: "policy-score", Phase: trainingprogram.PhaseForward},
	{ID: "grpo", Phase: trainingprogram.PhaseLoss},
	{ID: "dense-backward", Phase: trainingprogram.PhaseBackward},
	{ID: "muon", Phase: trainingprogram.PhaseOptimize},
}

var tokenModules = []recipe.ModuleID{
	workflowrecipe.ModuleBatchDataset, workflowrecipe.ModuleTrainingForward,
	workflowrecipe.ModuleBackward, workflowrecipe.ModuleOptimize,
}

var dpoModules = []recipe.ModuleID{
	workflowrecipe.ModuleBatchPreference, workflowrecipe.ModuleScorePolicy,
	workflowrecipe.ModuleScoreReference, workflowrecipe.ModuleDPOObjective,
	workflowrecipe.ModuleBackward, workflowrecipe.ModuleOptimize,
}

var grpoModules = []recipe.ModuleID{
	workflowrecipe.ModuleBatchRollout, workflowrecipe.ModuleScorePolicy,
	workflowrecipe.ModuleGRPOObjective, workflowrecipe.ModuleBackward, workflowrecipe.ModuleOptimize,
}
