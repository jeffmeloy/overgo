// Command train runs dense causal Muon training and publishes exact-resume
// checkpoints.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/densecausal"
	artifactexport "overgo/internal/export"
	"overgo/internal/hfbpe"
	"overgo/internal/recipecontract"
	"overgo/internal/safetensors"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

func main() {
	clioptions.MainNamed("train", run)
}

func run() error {
	modelDir := flag.String("model", "", "model directory (safetensors + config.json + tokenizer.json)")
	datasetPath := flag.String("dataset", "", "UTF-8 training dataset")
	outDir := flag.String("out", "", "output directory for the trained checkpoint")
	resumeDir := flag.String("resume", "", "resume checkpoint directory")
	referenceDir := flag.String("reference", "", "frozen reference model directory for DPO")
	dpoScale := flag.Float64("dpo-scale", 0, "DPO scale; required with -reference")
	steps := flag.Int("steps", 1, "number of Muon update steps")
	maxSeq := flag.Int("seq", 512, "cap the token sequence to this length (attention is O(seq^2)); <=0 keeps all")
	baseLR := flag.Float64("lr", 0, "base learning rate; <=0 derives n_params^-1/2")
	mu := flag.Float64("momentum", 0.9, "Muon momentum")
	host := flag.Bool("host", false, "force the host path even when CUDA is available")
	freezeLexical := flag.Bool("freeze-lexical", false, "freeze tied embedding/head; requires CUDA resident training")
	flag.Parse()

	if (*modelDir == "" && *resumeDir == "") || *datasetPath == "" || *outDir == "" {
		flag.Usage()
		return fmt.Errorf("-model, -dataset, and -out are required")
	}
	if *steps < 1 {
		return fmt.Errorf("-steps must be >= 1")
	}

	inputDir := *modelDir
	var resumed trainingprogram.Checkpoint
	var resumeState *densecausal.TrainState
	var resumeStream *trainingdata.StreamState
	if *resumeDir != "" {
		var err error
		resumed, err = trainingprogram.LoadCheckpoint(*resumeDir)
		if err != nil {
			return fmt.Errorf("load checkpoint: %w", err)
		}
		inputDir = *resumeDir
		state := densecausal.TrainState(resumed.Optimizer)
		resumeState = &state
		stream := trainingdata.StreamState{Identity: resumed.Stream.Identity, Position: resumed.Stream.Position}
		resumeStream = &stream
	}
	model, err := densecausal.Load(inputDir)
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	tok, err := hfbpe.Load(inputDir)
	if err != nil {
		return fmt.Errorf("load tokenizer: %w", err)
	}
	raw, err := os.ReadFile(*datasetPath)
	if err != nil {
		return fmt.Errorf("read dataset: %w", err)
	}
	if *host && *freezeLexical {
		return fmt.Errorf("-host and -freeze-lexical are mutually exclusive")
	}
	if *referenceDir != "" {
		if *dpoScale <= 0 || *freezeLexical {
			return fmt.Errorf("-reference requires positive -dpo-scale and does not support -freeze-lexical")
		}
		return runDPO(model, tok.Encode, inputDir, *referenceDir, *outDir, raw, *steps, *baseLR, *mu, *dpoScale, resumed, resumeStream)
	}
	if *dpoScale != 0 {
		return fmt.Errorf("-dpo-scale requires -reference")
	}
	batches, streamState, dataAuthority, err := tokenBatchesResume(context.Background(), raw, *steps, *maxSeq, tok.Encode, resumeStream)
	if err != nil {
		return err
	}
	authority, err := compileTrainingAuthority(model, inputDir, dataAuthority, streamState, *baseLR, resumed, trainingprogram.ObjectiveTokenPrediction, nil)
	if err != nil {
		return fmt.Errorf("compile training authority: %w", err)
	}
	traj, backend, trainState, err := runTrainingState(model, batches, *baseLR, *mu, !*host, *freezeLexical, resumeState)
	if err != nil {
		return fmt.Errorf("train: %w", err)
	}
	checkpointSpec, err := authority.checkpointSpec(trainState)
	if err != nil {
		return err
	}
	checkpoint, err := saveCheckpoint(inputDir, *outDir, model, checkpointSpec)
	if err != nil {
		return fmt.Errorf("save checkpoint: %w", err)
	}

	fmt.Printf("backend=%s batches=%d stream_position=%d steps=%d lr=%s momentum=%g freeze_lexical=%v\n", backend, len(batches), streamState.Position, *steps, lrLabel(*baseLR), *mu, *freezeLexical)
	for i, loss := range traj {
		fmt.Printf("step %d: loss %.6f\n", i, loss)
	}
	fmt.Printf("checkpoint=%s written to %s\n", checkpoint.ID(), *outDir)
	return nil
}

func runDPO(
	model *densecausal.Model,
	encode func(string) ([]int, error),
	inputDir, referenceDir, outDir string,
	raw []byte,
	steps int,
	baseLR, momentum, scale float64,
	resumed trainingprogram.Checkpoint,
	resumeStream *trainingdata.StreamState,
) error {
	reference, err := densecausal.Load(referenceDir)
	if err != nil {
		return fmt.Errorf("load DPO reference: %w", err)
	}
	batches, streamState, dataAuthority, err := preferenceBatchesResume(context.Background(), raw, steps, encode, resumeStream)
	if err != nil {
		return err
	}
	referenceID, err := identifyModel(referenceDir)
	if err != nil {
		return fmt.Errorf("resolve DPO reference identity: %w", err)
	}
	preference := &trainingprogram.PreferencePolicy{Reference: referenceID, Scale: scale}
	authority, err := compileTrainingAuthority(
		model, inputDir, dataAuthority, streamState, baseLR, resumed, trainingprogram.ObjectiveDPO, preference,
	)
	if err != nil {
		return fmt.Errorf("compile DPO authority: %w", err)
	}
	var resume *densecausal.DPOState
	if resumed.ID().Valid() {
		resume = &densecausal.DPOState{Optimizer: resumed.Optimizer, Stream: *resumeStream}
	}
	losses, state, err := model.TrainDPOBatchesResume(reference, batches, baseLR, momentum, scale, resume)
	if err != nil {
		return fmt.Errorf("DPO train: %w", err)
	}
	checkpointSpec, err := authority.checkpointSpec(state.Optimizer)
	if err != nil {
		return err
	}
	checkpoint, err := saveCheckpoint(inputDir, outDir, model, checkpointSpec)
	if err != nil {
		return fmt.Errorf("save DPO checkpoint: %w", err)
	}
	fmt.Printf("backend=host objective=dpo batches=%d stream_position=%d steps=%d lr=%s momentum=%g scale=%g\n", len(batches), state.Stream.Position, steps, lrLabel(baseLR), momentum, scale)
	for index, loss := range losses {
		fmt.Printf("step %d: loss %.6f\n", index, loss)
	}
	fmt.Printf("checkpoint=%s written to %s\n", checkpoint.ID(), outDir)
	return nil
}

func identifyModel(directory string) (artifact.ID, error) {
	file, err := os.Open(filepath.Join(directory, trainingprogram.CheckpointWeights))
	if err != nil {
		return artifact.ID{}, err
	}
	id, _, identifyErr := artifact.Identify(artifact.KindModel, file)
	return id, errors.Join(identifyErr, file.Close())
}

func lrLabel(lr float64) string {
	if lr <= 0 {
		return "derived(n^-1/2)"
	}
	return fmt.Sprintf("%g", lr)
}

func runHostTrainingState(m *densecausal.Model, batches [][]int, baseLR, mu float64, resume *densecausal.TrainState) ([]float64, densecausal.TrainState, error) {
	return m.TrainBatchesResume(batches, baseLR, mu, resume)
}

func tokenBatches(
	ctx context.Context,
	raw []byte,
	steps, maxSeq int,
	encode func(string) ([]int, error),
) ([][]int, trainingdata.StreamState, error) {
	batches, state, _, err := tokenBatchesResume(ctx, raw, steps, maxSeq, encode, nil)
	return batches, state, err
}

type batchAuthority struct {
	Dataset   artifact.ID
	Split     artifact.ID
	Processor artifact.ID
}

func tokenBatchesResume(
	ctx context.Context,
	raw []byte,
	steps, maxSeq int,
	encode func(string) ([]int, error),
	resume *trainingdata.StreamState,
) ([][]int, trainingdata.StreamState, batchAuthority, error) {
	if len(raw) == 0 || steps <= 0 || encode == nil {
		return nil, trainingdata.StreamState{}, batchAuthority{}, errors.New("train: invalid dataset stream input")
	}
	datasetID, err := artifact.IdentifyBytes(artifact.KindDataset, raw)
	if err != nil {
		return nil, trainingdata.StreamState{}, batchAuthority{}, err
	}
	splitID, err := artifact.IdentifyBytes(artifact.KindDatasetShard, append([]byte("all\x00"), raw...))
	if err != nil {
		return nil, trainingdata.StreamState{}, batchAuthority{}, err
	}
	processorID, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("overgo/training/text-utf8/v1"))
	if err != nil {
		return nil, trainingdata.StreamState{}, batchAuthority{}, err
	}
	authority := trainingdata.Authority{
		Dataset: datasetID, Split: splitID, Processors: []artifact.ID{processorID},
		Signature: recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{recipecontract.ModalityText}, Outputs: []recipecontract.Modality{recipecontract.ModalityText}},
	}
	materialized, err := trainingdata.MaterializeDocuments(authority, processorID, []string{string(raw)}, trainingdata.ProcessorBinding{
		Artifact:   processorID,
		Modalities: []recipecontract.Modality{recipecontract.ModalityText},
		Process:    trainingdata.Passthrough(trainingdata.RoleInput, recipecontract.ModalityText, "utf-8"),
	})
	if err != nil {
		return nil, trainingdata.StreamState{}, batchAuthority{}, err
	}
	defer materialized.Close()
	stream, err := trainingdata.NewStream(materialized, resume)
	if err != nil {
		return nil, trainingdata.StreamState{}, batchAuthority{}, err
	}
	batcher, err := trainingdata.NewBatcher(stream, trainingdata.BatchPolicy{Examples: 1, MicrobatchExamples: 1, DecodeWorkers: 1})
	if err != nil {
		return nil, trainingdata.StreamState{}, batchAuthority{}, err
	}
	batches := make([][]int, steps)
	for step := range steps {
		batch, err := batcher.Next(ctx)
		if err != nil {
			return nil, trainingdata.StreamState{}, batchAuthority{}, err
		}
		tokens, err := encode(string(batch.Examples[0].Values[0].Data))
		if err != nil {
			return nil, trainingdata.StreamState{}, batchAuthority{}, fmt.Errorf("train: encode dataset record: %w", err)
		}
		if maxSeq > 0 && len(tokens) > maxSeq {
			tokens = tokens[:maxSeq]
		}
		if len(tokens) < 2 {
			return nil, trainingdata.StreamState{}, batchAuthority{}, fmt.Errorf("train: record needs at least two tokens, got %d", len(tokens))
		}
		batches[step] = tokens
	}
	return batches, stream.Snapshot(), batchAuthority{Dataset: datasetID, Split: splitID, Processor: processorID}, nil
}

// saveCheckpoint stages one loader-compatible, exact-resume directory.
func saveCheckpoint(srcDir, outDir string, m *densecausal.Model, spec trainingprogram.CheckpointSpec) (trainingprogram.Checkpoint, error) {
	return trainingprogram.PublishCheckpoint(outDir, spec, func(stage string) error {
		meta := map[string]string{"format": "pt", "trainer": "overgo"}
		if err := safetensors.Save(filepath.Join(stage, trainingprogram.CheckpointWeights), m.Weights, m.Shapes, meta); err != nil {
			return err
		}
		return artifactexport.CopyFiles(srcDir, stage, []string{"config.json", "tokenizer.json"})
	})
}

type trainingAuthority struct {
	model, program, runPlan artifact.ID
	optimizerPlan           string
	data                    batchAuthority
	stream                  trainingdata.StreamState
	rng                     []trainingprogram.RNGState
	lineage                 []trainingprogram.LineageParent
	projectors, codecs      []artifact.ID
}

func compileTrainingAuthority(
	model *densecausal.Model,
	modelDir string,
	data batchAuthority,
	stream trainingdata.StreamState,
	baseLR float64,
	resumed trainingprogram.Checkpoint,
	objective trainingprogram.ObjectiveKind,
	preference *trainingprogram.PreferencePolicy,
) (trainingAuthority, error) {
	modelID := resumed.Model
	var err error
	if !modelID.Valid() {
		modelID, err = identifyModel(modelDir)
		if err != nil {
			return trainingAuthority{}, err
		}
	}
	muonPlan, _, err := model.TrainingPlan(baseLR)
	if err != nil {
		return trainingAuthority{}, err
	}
	parameters := make([]trainingprogram.ParameterSpec, muonPlan.GroupCount())
	for index := range parameters {
		group, _ := muonPlan.Group(index)
		parameters[index] = trainingprogram.ParameterSpec{Name: group.Name, Rows: group.Rows, Cols: group.Cols, Trainable: !group.Frozen}
	}
	operators := []trainingprogram.OperatorSpec{
		{ID: "dense-forward", Phase: trainingprogram.PhaseForward},
		{ID: "dense-backward", Phase: trainingprogram.PhaseBackward},
		{ID: "muon", Phase: trainingprogram.PhaseOptimize},
	}
	if objective == trainingprogram.ObjectiveDPO {
		operators = []trainingprogram.OperatorSpec{
			{ID: "policy-score", Phase: trainingprogram.PhaseForward},
			{ID: "reference-score", Phase: trainingprogram.PhaseForward},
			{ID: "dpo", Phase: trainingprogram.PhaseLoss},
			{ID: "dense-backward", Phase: trainingprogram.PhaseBackward},
			{ID: "muon", Phase: trainingprogram.PhaseOptimize},
		}
	}
	program, err := trainingprogram.CompileTrainingProgram(trainingprogram.ProgramSpec{
		Objective:  objective,
		Operators:  operators,
		Parameters: parameters, Optimizer: muonPlan,
		Preference: preference,
	})
	if err != nil {
		return trainingAuthority{}, err
	}
	profile := func(name string) artifact.ID {
		id, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("overgo/densecausal/"+name+"/v1"))
		return id
	}
	recipeID, _ := artifact.IdentifyBytes(artifact.KindRecipe, []byte("overgo/densecausal/training/"+string(objective)+"/v1"))
	initial := trainingprogram.InitialStateSpec{Model: modelID}
	if resumed.ID().Valid() {
		initial = trainingprogram.InitialStateSpec{Checkpoint: resumed.ID()}
	}
	runPlan, err := trainingprogram.CompileTrainingRunPlan(trainingprogram.RunSpec{
		Recipe: recipeID, Initial: initial, Dataset: data.Dataset, Split: data.Split,
		Signature: recipecontract.ModalitySignature{
			Inputs:  []recipecontract.Modality{recipecontract.ModalityText},
			Outputs: []recipecontract.Modality{recipecontract.ModalityText},
		},
		Processors: []artifact.ID{data.Processor},
		Policies: trainingprogram.PolicySpec{
			Objective: profile(string(objective)), Precision: profile("fp32-bf16"),
			Placement: profile("platform-resident"), Memory: profile("derived-memory"),
			Checkpoint: profile("exact-checkpoint"), Evaluation: profile("loss-trajectory"),
			Promotion: profile("heldout-promotion"),
		},
		Program: program,
	})
	if err != nil {
		return trainingAuthority{}, err
	}
	if resumed.ID().Valid() {
		if err := trainingprogram.ValidateResume(runPlan, resumed, stream.Identity); err != nil {
			return trainingAuthority{}, err
		}
	}
	rngAlgorithm, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("overgo/counter-rng/v1"))
	rng := append([]trainingprogram.RNGState(nil), resumed.RNG...)
	if len(rng) == 0 {
		rng = []trainingprogram.RNGState{
			{Name: "augmentation", Algorithm: rngAlgorithm},
			{Name: "data", Algorithm: rngAlgorithm},
		}
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
	return trainingAuthority{
		model: modelID, program: program.ID(), runPlan: runPlan.ID(), optimizerPlan: muonPlan.Identity(),
		data: data, stream: stream, rng: rng, lineage: lineage,
		projectors: resumed.Projectors, codecs: resumed.Codecs,
	}, nil
}

func (authority trainingAuthority) checkpointSpec(state densecausal.TrainState) (trainingprogram.CheckpointSpec, error) {
	if state.PlanIdentity != authority.optimizerPlan {
		return trainingprogram.CheckpointSpec{}, errors.New("checkpoint Muon plan differs from compiled training authority")
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
		Processors: []artifact.ID{authority.data.Processor}, Projectors: authority.projectors,
		Codecs: authority.codecs, Lineage: authority.lineage,
	}, nil
}
