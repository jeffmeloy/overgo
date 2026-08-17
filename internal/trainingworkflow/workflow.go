// Package trainingworkflow executes compiled native training workflows.
package trainingworkflow

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/densecausal"
	artifactexport "overgo/internal/export"
	"overgo/internal/hfbpe"
	"overgo/internal/recipecontract"
	"overgo/internal/safetensors"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

type Request struct {
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
	DPOScale           float64
	Host               bool
	FreezeLexical      bool
	ObserveDPO         func(trainingprogram.DPOObservation)
}

type Result struct {
	Backend        string
	Objective      trainingprogram.ObjectiveKind
	Losses         []float64
	DPO            []trainingprogram.DPOObservation
	StreamPosition uint64
	Checkpoint     trainingprogram.Checkpoint
}

func Execute(ctx context.Context, request Request) (Result, error) {
	if ctx == nil || request.ModelDirectory == "" && request.ResumeDirectory == "" ||
		request.DatasetPath == "" || request.OutputDirectory == "" || request.Steps <= 0 {
		return Result{}, errors.New("training workflow: model or resume, dataset, output, and positive steps required")
	}
	if request.Host && request.FreezeLexical {
		return Result{}, errors.New("training workflow: host and frozen lexical execution are incompatible")
	}
	if request.ReferenceDirectory == "" && request.DPOScale != 0 {
		return Result{}, errors.New("training workflow: DPO scale requires reference model")
	}
	if request.ReferenceDirectory != "" && (request.DPOScale <= 0 || request.FreezeLexical) {
		return Result{}, errors.New("training workflow: reference requires positive DPO scale and trainable lexical weights")
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
	if request.ReferenceDirectory != "" {
		return executeDPO(ctx, request, inputDirectory, model, tokenizer.Encode, raw, resumed, resumeStream)
	}
	return executeTokenPrediction(ctx, request, inputDirectory, model, tokenizer.Encode, raw, resumed, resumeStream)
}

func executeTokenPrediction(
	ctx context.Context,
	request Request,
	inputDirectory string,
	model *densecausal.Model,
	encode func(string) ([]int, error),
	raw []byte,
	resumed trainingprogram.Checkpoint,
	resumeStream *trainingdata.StreamState,
) (Result, error) {
	batches, stream, data, err := tokenBatchesResume(ctx, raw, request.Steps, request.MaximumSequence, encode, resumeStream)
	if err != nil {
		return Result{}, err
	}
	authority, err := compileAuthority(model, inputDirectory, data, stream, request.LearningRate, resumed, request.Recipe, trainingprogram.ObjectiveTokenPrediction, nil)
	if err != nil {
		return Result{}, fmt.Errorf("training workflow: compile authority: %w", err)
	}
	var resume *densecausal.TrainState
	if resumed.ID().Valid() {
		state := densecausal.TrainState(resumed.Optimizer)
		resume = &state
	}
	losses, backend, state, err := runTrainingState(model, batches, request.LearningRate, request.Momentum, !request.Host, request.FreezeLexical, resume)
	if err != nil {
		return Result{}, fmt.Errorf("training workflow: train: %w", err)
	}
	checkpoint, err := publishCheckpoint(inputDirectory, request.OutputDirectory, model, authority, state)
	if err != nil {
		return Result{}, err
	}
	return Result{Backend: backend, Objective: trainingprogram.ObjectiveTokenPrediction, Losses: losses, StreamPosition: stream.Position, Checkpoint: checkpoint}, nil
}

func executeDPO(
	ctx context.Context,
	request Request,
	inputDirectory string,
	model *densecausal.Model,
	encode func(string) ([]int, error),
	raw []byte,
	resumed trainingprogram.Checkpoint,
	resumeStream *trainingdata.StreamState,
) (Result, error) {
	reference, err := densecausal.Load(request.ReferenceDirectory)
	if err != nil {
		return Result{}, fmt.Errorf("training workflow: load reference: %w", err)
	}
	batches, stream, data, err := preferenceBatchesResume(ctx, raw, request.Steps, encode, resumeStream)
	if err != nil {
		return Result{}, err
	}
	referenceID, err := identifyModel(request.ReferenceDirectory)
	if err != nil {
		return Result{}, fmt.Errorf("training workflow: identify reference: %w", err)
	}
	preference := &trainingprogram.PreferencePolicy{Reference: referenceID, Scale: request.DPOScale}
	authority, err := compileAuthority(model, inputDirectory, data, stream, request.LearningRate, resumed, request.Recipe, trainingprogram.ObjectiveDPO, preference)
	if err != nil {
		return Result{}, fmt.Errorf("training workflow: compile DPO authority: %w", err)
	}
	var resume *densecausal.DPOState
	if resumed.ID().Valid() {
		resume = &densecausal.DPOState{Optimizer: resumed.Optimizer, Stream: *resumeStream}
	}
	observations, state, err := model.TrainDPOBatchesResume(reference, batches, request.LearningRate, request.Momentum, request.DPOScale, resume, request.ObserveDPO)
	if err != nil {
		return Result{}, fmt.Errorf("training workflow: DPO train: %w", err)
	}
	checkpoint, err := publishCheckpoint(inputDirectory, request.OutputDirectory, model, authority, state.Optimizer)
	if err != nil {
		return Result{}, err
	}
	return Result{Backend: "host", Objective: trainingprogram.ObjectiveDPO, DPO: observations, StreamPosition: state.Stream.Position, Checkpoint: checkpoint}, nil
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
	file, err := os.Open(filepath.Join(directory, trainingprogram.CheckpointWeights))
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
	model *densecausal.Model,
	modelDirectory string,
	data batchAuthority,
	stream trainingdata.StreamState,
	learningRate float64,
	resumed trainingprogram.Checkpoint,
	recipeID artifact.ID,
	objective trainingprogram.ObjectiveKind,
	preference *trainingprogram.PreferencePolicy,
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
	}
	program, err := trainingprogram.CompileTrainingProgram(trainingprogram.ProgramSpec{
		Objective: objective, Operators: operators, Parameters: parameters, Optimizer: muonPlan, Preference: preference,
	})
	if err != nil {
		return compiledAuthority{}, err
	}
	profile := func(name string) artifact.ID {
		id, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("overgo/densecausal/"+name+"/v1"))
		return id
	}
	if !recipeID.Valid() {
		recipeID, _ = artifact.IdentifyBytes(artifact.KindRecipe, []byte("overgo/densecausal/training/"+string(objective)+"/v1"))
	} else if recipeID.Kind() != artifact.KindRecipe {
		return compiledAuthority{}, errors.New("training workflow: recipe identity differs")
	}
	initial := trainingprogram.InitialStateSpec{Model: modelID}
	if resumed.ID().Valid() {
		initial = trainingprogram.InitialStateSpec{Checkpoint: resumed.ID()}
	}
	runPlan, err := trainingprogram.CompileTrainingRunPlan(trainingprogram.RunSpec{
		Recipe: recipeID, Initial: initial, Dataset: data.Dataset, Split: data.Split,
		Signature:  recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{recipecontract.ModalityText}, Outputs: []recipecontract.Modality{recipecontract.ModalityText}},
		Processors: []artifact.ID{data.Processor},
		Policies: trainingprogram.PolicySpec{
			Objective: profile(string(objective)), Precision: profile("fp32-bf16"), Placement: profile("platform-resident"),
			Memory: profile("derived-memory"), Checkpoint: profile("exact-checkpoint"), Evaluation: profile("loss-trajectory"),
			Promotion: profile("heldout-promotion"),
		},
		Program: program,
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
	if resumed.ID().Valid() {
		lineage = append(lineage, trainingprogram.LineageParent{Artifact: resumed.ID(), Relation: artifact.RelationDerivedFrom})
	}
	return compiledAuthority{
		model: modelID, program: program.ID(), runPlan: runPlan.ID(), optimizerPlan: muonPlan.Identity(),
		data: data, stream: stream, rng: rng, lineage: lineage, projectors: resumed.Projectors, codecs: resumed.Codecs,
	}, nil
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
