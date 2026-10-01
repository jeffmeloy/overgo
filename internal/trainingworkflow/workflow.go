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
	"overgo/internal/dataset"
	"overgo/internal/densecausal"
	artifactexport "overgo/internal/export"
	"overgo/internal/hostoptimizer"
	"overgo/internal/modelintake"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/safetensors"
	"overgo/internal/tensorcatalog"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
	"overgo/internal/workflowrecipe"
)

type Request struct {
	// Audio selects recipe-bound CPU CTC adaptation; dense-only inputs refuse.
	Audio              *AudioTrainingSpec
	Repository         artifact.Reader
	Recipe             artifact.ID
	ModelDirectory     string
	DatasetPath        string
	OutputDirectory    string
	ResumeDirectory    string
	ReferenceDirectory string
	Steps              int
	MaximumSequence    int
	ObjectiveScale     float64
	Host               bool
	// Device trains the dense-causal model when Host is false.
	Device        DeviceTrainer
	FreezeLexical bool
	// PreferenceMode and PreferenceMargin select the DPO objective's loss
	// mode and SimPO's target reward margin; other objectives refuse them.
	// KTO's mode reads the dataset as unpaired completions.
	PreferenceMode   trainingprogram.PreferenceMode
	PreferenceMargin float64
	// Observations: optional typed run evidence.
	Observations artifact.Repository

	ObserveDPO       func(trainingprogram.DPOObservation)
	ObserveGRPO      func(trainingprogram.GRPOObservation)
	Progress         io.Writer
	MaxProjectedWall time.Duration
}

type Result struct {
	// Candidate is a published, inactive serving recipe for the checkpoint.
	Candidate      artifact.ID
	Backend        string
	Objective      trainingprogram.ObjectiveKind
	Optimizer      hostoptimizer.Config
	Plan           trainingprogram.TrainingSessionPlan
	Losses         []float64
	DPO            []trainingprogram.DPOObservation
	KTO            []trainingprogram.KTOObservation
	GRPO           []trainingprogram.GRPOObservation
	StreamPosition uint64
	Checkpoint     trainingprogram.Checkpoint
	// Observation: committed session evidence.
	Observation               artifact.ID
	RouterObservations        []artifact.ID
	RouterObservationCoverage artifact.ID
	authority                 compiledAuthority
	router                    []routerObservationRecord
	routerExpectation         routerObservationExpectation
}

func Execute(ctx context.Context, request Request) (Result, error) {
	if request.Audio != nil {
		return executeAudioTraining(ctx, request)
	}
	if ctx == nil || request.Repository == nil || request.Recipe.Kind() != artifact.KindRecipe ||
		request.ModelDirectory == "" && request.ResumeDirectory == "" ||
		request.DatasetPath == "" || request.OutputDirectory == "" || request.Steps < 0 {
		return Result{}, errors.New("training workflow: repository, recipe, model or resume, dataset, output, and nonnegative requested steps required")
	}
	if request.Host && request.FreezeLexical {
		return Result{}, errors.New("training workflow: host and frozen lexical execution are incompatible")
	}
	var observer *Observer
	if request.Observations != nil {
		var err error
		observer, err = NewObserver(request.Observations, request.Host)
		if err != nil {
			return Result{}, err
		}
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
		// A run over a GGUF resumes from the trained model it published.
		inputDirectory = trainedInput(request.ResumeDirectory)
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
	if err := observer.Admit(ctx, modelID, request.Recipe); err != nil {
		return Result{}, err
	}
	// Every exit after admission, a setup refusal included, finishes the
	// session it admitted.
	result, runErr := admittedRun(ctx, request, observer, inputDirectory, resumed, resumeStream, modelID)
	if observer != nil {
		// A cancelled run's outcome records too.
		observationID, routerIDs, routerCoverage, observeErr := observer.finishTraining(
			context.WithoutCancel(ctx), modelID, request.Recipe, runErr, result,
		)
		if observeErr != nil && runErr == nil {
			return Result{}, observeErr
		}
		result.Observation, result.RouterObservations = observationID, routerIDs
		result.RouterObservationCoverage = routerCoverage
		if runErr == nil {
			runErr = observer.Verify(ctx, request.Recipe, result.Observation, result.Checkpoint.ID())
		}
	}
	return result, runErr
}

// admittedRun resolves the admitted session's authority, loads its model,
// tokenizer and dataset, and trains them.
func admittedRun(ctx context.Context, request Request, observer *Observer, inputDirectory string, resumed trainingprogram.Checkpoint, resumeStream *trainingdata.StreamState, modelID artifact.ID) (Result, error) {
	authority, err := ResolveSessionAuthority(ctx, request.Repository, modelID, request.Recipe)
	if err != nil {
		return Result{}, err
	}
	runtime, objective, optimizerPolicy := authority.Program, authority.Objective, authority.Optimizer
	if objective != trainingprogram.ObjectiveDPO && (request.PreferenceMode != "" || request.PreferenceMargin != 0) {
		return Result{}, errors.New("training workflow: a preference mode or margin needs the DPO objective")
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
	loadStarted := time.Now()
	model, err := loadTrainableModel(inputDirectory)
	if err != nil {
		return Result{}, fmt.Errorf("training workflow: load model: %w", err)
	}
	if request.MaximumSequence <= 0 {
		request.MaximumSequence = model.Dims.ContextLength
	}
	muonPlan, err := model.TrainingPlan()
	if err != nil {
		return Result{}, fmt.Errorf("training workflow: compile optimizer plan: %w", err)
	}
	optimizerConfig, err := optimizerPolicy.Config(muonPlan.ParameterCount())
	if err != nil {
		return Result{}, fmt.Errorf("training workflow: compile optimizer config: %w", err)
	}
	encode, err := loadTrainableEncoder(inputDirectory)
	if err != nil {
		return Result{}, fmt.Errorf("training workflow: load tokenizer: %w", err)
	}
	raw, err := os.ReadFile(request.DatasetPath)
	if err != nil {
		return Result{}, fmt.Errorf("training workflow: read dataset: %w", err)
	}
	observer.Phase(runrecord.PhaseLoad, time.Since(loadStarted))
	trainStarted := time.Now()
	result, runErr := denseSession{
		ctx: ctx, request: request, runtime: runtime, objective: objective,
		optimizerPlan: muonPlan, optimizerConfig: optimizerConfig,
		inputDirectory: inputDirectory, model: model, encode: encode, raw: raw,
		resumed: resumed, resumeStream: resumeStream, stepSample: observer.SampleStep,
	}.run()
	observer.Phase(runrecord.PhaseForwardBackward, time.Since(trainStarted))
	return result, runErr
}

type denseSession struct {
	ctx             context.Context
	request         Request
	runtime         recipe.Program
	objective       trainingprogram.ObjectiveKind
	optimizerPlan   hostoptimizer.Plan
	optimizerConfig hostoptimizer.Config
	inputDirectory  string
	model           *densecausal.Model
	encode          func(string) ([]int, error)
	raw             []byte
	resumed         trainingprogram.Checkpoint
	resumeStream    *trainingdata.StreamState
	stepSample      func()
}

type densePrepared struct {
	units      int
	data       batchAuthority
	stream     trainingdata.StreamState
	preference *trainingprogram.PreferencePolicy
	reference  *densecausal.Model
	tokens     [][]int
	pairs      []trainingdata.PreferenceBatch
	unpaired   []trainingdata.UnpairedBatch
	groups     []trainingdata.RolloutGroup
	evaluators []artifact.ID
}

func (session denseSession) run() (Result, error) {
	prepared, err := session.prepare()
	if err != nil {
		return Result{}, err
	}
	plan, err := trainingprogram.CompileTrainingSessionPlan(trainingprogram.SessionSpec{
		Objective: session.objective, DatasetUnits: prepared.units, RequestedUpdates: session.request.Steps,
		MaximumSequence: session.request.MaximumSequence, ObjectiveScale: session.request.ObjectiveScale,
		MaxProjectedWall: session.request.MaxProjectedWall, Optimizer: session.optimizerConfig,
	})
	if err != nil {
		return Result{}, err
	}
	session.request.Steps = plan.Updates()
	session.optimizerConfig = plan.Optimizer()
	authority, err := compileAuthority(
		session.ctx, session.request.Repository, session.runtime, session.optimizerPlan, session.inputDirectory,
		prepared.data, prepared.stream, session.resumed,
		session.objective, prepared.preference, prepared.evaluators,
	)
	if err != nil {
		return Result{}, fmt.Errorf("training workflow: compile authority: %w", err)
	}
	result, state, err := session.execute(prepared, plan)
	if err != nil {
		return Result{}, err
	}
	result.Optimizer = plan.Optimizer()
	result.Plan = plan
	result.Checkpoint, err = publishCheckpoint(
		session.inputDirectory, session.request.OutputDirectory, session.model, authority, state,
	)
	result.authority = authority
	// The checkpoint records where the run's evidence goes, and a trained
	// GGUF registers there as the candidate the server, validate and the
	// scorecard open.
	if err == nil && session.request.Observations != nil {
		var candidate modelintake.Candidate
		candidate, err = RecordCheckpoint(session.ctx, session.request.Observations, session.request.OutputDirectory, result.Checkpoint)
		result.Candidate = candidate.Definition.ID
	}
	return result, err
}

func (session denseSession) prepare() (densePrepared, error) {
	if session.objective == trainingprogram.ObjectiveTokenPrediction {
		batches, units, stream, data, err := tokenBatchesResume(
			session.ctx, session.raw, session.request.Steps, session.request.MaximumSequence,
			session.encode, session.resumeStream,
		)
		return densePrepared{units: units, data: data, stream: stream, tokens: batches}, err
	}
	if session.objective == trainingprogram.ObjectiveGRPO {
		groups, units, stream, data, evaluators, err := groupedRolloutBatchesResume(
			session.ctx, session.raw, session.request.Steps, session.encode, session.resumeStream,
		)
		return densePrepared{units: units, data: data, stream: stream, groups: groups, evaluators: evaluators}, err
	}
	reference, err := densecausal.Load(session.request.ReferenceDirectory)
	if err != nil {
		return densePrepared{}, fmt.Errorf("training workflow: load reference: %w", err)
	}
	_, membership, err := recipeObjective(session.ctx, session.request.Repository, session.runtime.Definition())
	if err != nil {
		return densePrepared{}, err
	}
	referenceID, err := identifyModel(session.request.ReferenceDirectory)
	if err != nil {
		return densePrepared{}, fmt.Errorf("training workflow: identify reference: %w", err)
	}
	if session.request.PreferenceMode == trainingprogram.PreferenceKTO {
		if session.request.PreferenceMargin != 0 {
			return densePrepared{}, errors.New("training workflow: KTO has no preference margin")
		}
		read, err := unpairedBatchesResume(session.ctx, session.raw, session.request.Steps, session.encode, session.resumeStream, membership)
		if err != nil {
			return densePrepared{}, err
		}
		policy, err := trainingprogram.KTOPolicy(referenceID, session.request.ObjectiveScale, read.desirable, read.undesirable)
		return densePrepared{
			units: read.units, data: read.authority, stream: read.stream, reference: reference, unpaired: read.batches, preference: &policy,
		}, err
	}
	batches, units, stream, data, err := preferenceBatchesResume(
		session.ctx, session.raw, session.request.Steps, session.encode, session.resumeStream, membership,
	)
	if err != nil {
		return densePrepared{}, err
	}
	return densePrepared{
		units: units, data: data, stream: stream, reference: reference, pairs: batches,
		preference: &trainingprogram.PreferencePolicy{
			Reference: referenceID, Scale: session.request.ObjectiveScale,
			Mode: session.request.PreferenceMode, Margin: session.request.PreferenceMargin,
		},
	}, nil
}

// recipeObjective loads the objective a recipe binds and the committed
// training membership it splits its dataset by: nil for an objective that
// trains on all of it, or a recipe that binds none.
func recipeObjective(ctx context.Context, reader artifact.Reader, definition recipe.Definition) (trainingprogram.ObjectiveDocument, *dataset.Membership, error) {
	id, bound := definition.PrimaryDependency(recipe.DependencyObjective)
	if !bound {
		return trainingprogram.ObjectiveDocument{}, nil, nil
	}
	objective, err := trainingprogram.LoadObjective(ctx, reader, id)
	if err != nil {
		return trainingprogram.ObjectiveDocument{}, nil, err
	}
	membership, found, err := dataset.LoadMembership(ctx, reader, objective.Split)
	if err != nil || !found {
		return objective, nil, err
	}
	return objective, &membership, nil
}

func (session denseSession) execute(prepared densePrepared, plan trainingprogram.TrainingSessionPlan) (Result, densecausal.TrainState, error) {
	if session.objective == trainingprogram.ObjectiveGRPO {
		var resume *densecausal.RLState
		if session.resumed.ID().Valid() {
			resume = &densecausal.RLState{Optimizer: session.resumed.Optimizer, Stream: *session.resumeStream}
		}
		observations, state, err := session.model.TrainGRPOGroupsResume(
			prepared.groups, session.optimizerConfig.BaseLearningRate, session.optimizerConfig.Momentum,
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
		if prepared.preference.Mode == trainingprogram.PreferenceKTO {
			observations, state, err := session.model.TrainKTOBatchesResume(
				prepared.reference, prepared.unpaired, session.optimizerConfig.BaseLearningRate, session.optimizerConfig.Momentum,
				*prepared.preference, resume,
			)
			return Result{Backend: "host", Objective: session.objective, KTO: observations, StreamPosition: state.Stream.Position}, state.Optimizer, err
		}
		observations, state, err := session.model.TrainDPOBatchesResume(
			prepared.reference, prepared.pairs, session.optimizerConfig.BaseLearningRate, session.optimizerConfig.Momentum,
			*prepared.preference, resume, session.request.ObserveDPO,
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
	observe := func(step int, loss float64, stepWall time.Duration) error {
		session.stepSample()
		if session.request.Progress != nil {
			fmt.Fprintf(session.request.Progress, "step %d/%d loss %.6f wall %s backend %s\n",
				step+1, session.request.Steps, loss, stepWall.Round(time.Millisecond), lane)
		}
		return plan.AdmitStepWall(stepWall)
	}
	var collector *routerObservationCollector
	var observeRouter densecausal.MoERouterObserver
	if layers := session.model.MoERouterLayers(); session.request.Observations != nil && session.request.Host && len(layers) != 0 {
		firstStep := 0
		if resume != nil {
			firstStep = resume.Step
		}
		var err error
		collector, err = newRouterObservationCollector(firstStep, len(prepared.tokens), layers)
		if err != nil {
			return Result{}, densecausal.TrainState{}, err
		}
		observeRouter = collector.observe
	}
	losses, backend, state, err := runTrainingState(
		session.model, prepared.tokens, session.optimizerConfig.BaseLearningRate, session.optimizerConfig.Momentum,
		session.request.Device, !session.request.Host, session.request.FreezeLexical, resume, observe, observeRouter,
	)
	result := Result{
		Backend: backend, Objective: session.objective, Losses: losses,
		StreamPosition: prepared.stream.Position,
	}
	if err == nil && collector != nil {
		result.router, result.routerExpectation, err = collector.complete()
	}
	return result, state, err
}

// publishCheckpoint publishes the trained model in its source's layout
// with the checkpoint that resumes it: a GGUF source's trained GGUF, which
// the server opens, or a safetensors directory with its config and
// tokenizer.
func publishCheckpoint(source, target string, model *densecausal.Model, authority compiledAuthority, state densecausal.TrainState) (trainingprogram.Checkpoint, error) {
	spec, err := authority.checkpointSpec(state)
	if err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	checkpoint, err := trainingprogram.PublishCheckpoint(target, spec, func(stage string) error {
		if isGGUFModelInput(source) {
			return WriteTrainedModel(source, filepath.Join(stage, trainingprogram.CheckpointModel), len(model.Weights), func(name string) ([]float32, bool) {
				trained, named := tensorcatalog.HFDenseTensorName(name)
				values, found := model.Weights[trained]
				return values, named && found
			})
		}
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
	path, err := IdentifyModelWeights(directory)
	if err != nil {
		return artifact.ID{}, err
	}
	file, err := os.Open(path)
	if err != nil {
		return artifact.ID{}, err
	}
	id, _, identifyErr := artifact.Identify(artifact.KindModel, file)
	return id, errors.Join(identifyErr, file.Close())
}

// IdentifyModelWeights resolves the one weights file whose hash is the
// model identity — the same file a recorded model location names. A GGUF
// input is its own identity; a directory identifies by model.safetensors,
// by its single safetensors file under any recorded name, or — for a
// sharded checkpoint — by the first shard in order, the file intake
// recorded as the model location. Two unordered safetensors files are
// ambiguous and refuse.
func IdentifyModelWeights(directory string) (string, error) {
	if isGGUFModelInput(directory) {
		return directory, nil
	}
	path := filepath.Join(directory, trainingprogram.CheckpointWeights)
	if _, err := os.Stat(path); err == nil {
		return path, nil
	}
	shards, err := filepath.Glob(filepath.Join(directory, "model-*-of-*.safetensors"))
	if err != nil {
		return "", err
	}
	if len(shards) > 0 {
		slices.Sort(shards)
		return shards[0], nil
	}
	candidates, err := filepath.Glob(filepath.Join(directory, "*.safetensors"))
	if err != nil {
		return "", err
	}
	switch len(candidates) {
	case 0:
		return "", fmt.Errorf("training workflow: %s holds no safetensors weights", directory)
	case 1:
		return candidates[0], nil
	}
	return "", fmt.Errorf(
		"training workflow: %d unordered safetensors files in %s; identity is ambiguous", len(candidates), directory)
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
	muonPlan hostoptimizer.Plan,
	modelDirectory string,
	data batchAuthority,
	stream trainingdata.StreamState,
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
		reference, ok := ProgramReference(runtime)
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
		if err := trainingprogram.ValidateResume(runPlan, resumed, trainingprogram.ResumeAuthority{
			Model: modelID, Stream: resumed.Stream, OptimizerPlan: muonPlan.Identity(),
		}); err != nil {
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

// ProgramReference resolves the recipe-declared reference model.
func ProgramReference(program recipe.Program) (artifact.ID, bool) {
	definition := program.Definition()
	for _, stage := range program.Stages() {
		if stage.Module.ID == workflowrecipe.ModuleScoreReference {
			return definition.Dependency(recipe.DependencyModel, stage.Node.ModelSlot)
		}
	}
	return artifact.ID{}, false
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
		Optimizer: state, ParameterCount: state.Parameters(), RNG: rng,
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
