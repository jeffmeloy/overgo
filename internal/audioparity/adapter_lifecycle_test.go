package audioparity

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"overgo/internal/adaptertrain"
	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/hfbpe"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
	"overgo/internal/workflowrecipe"
)

// These integration budgets are test admission ceilings, not process-peak or
// quality claims. The fixture is one independently pinned training row.
const adapterAcceptanceMemory = 4 << 30

type adapterLifecycle struct {
	audioPublication
	fixture    ctcTrainingFixture
	storePath  string
	base       recipe.Definition
	binding    adaptertrain.InputProjectionBinding
	transform  trainingdata.TextTransform
	data       *trainingdata.Dataset
	spec       trainingprogram.RunSpec
	plan       trainingprogram.TrainingRunPlan
	adapter    *adaptertrain.LinearCTC
	hidden     []float32
	frames     int
	policy     dataset.AudioInspectionPolicy
	origin     dataset.AudioPayloadOrigin
	rngProfile artifact.ID
}

func (l *adapterLifecycle) content(t *testing.T, content artifact.Content, err error) artifact.ID {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	l.commit(t, artifact.Batch{Key: content.Descriptor.ID.String(), Contents: []artifact.Content{content}}, nil)
	return content.Descriptor.ID
}

func newAdapterLifecycle(t *testing.T) *adapterLifecycle {
	t.Helper()
	l := &adapterLifecycle{fixture: loadCTCTrainingFixture(t)}
	var err error
	l.storePath = filepath.Join(t.TempDir(), "store")
	l.store, err = overgodb.Open(l.storePath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { l.store.Close() })
	inventory, err := modelartifact.FromHFPath(l.fixture.modelRoot)
	if err != nil || inventory.Manifest.ID != l.fixture.election.Model.ID {
		t.Fatalf("base inventory: %v", err)
	}
	batch, err := inventory.Batch("adapter/base")
	l.commit(t, batch, err)
	profile, err := speechrecognition.NewExecutionProfile(l.fixture.frontend, l.fixture.grouping, l.fixture.declaration, l.fixture.blank, "en")
	if err != nil {
		t.Fatal(err)
	}
	batch, err = profile.Batch("adapter/profile")
	l.commit(t, batch, err)
	contract, err := modelrecipe.NewAudioContract(recipecontract.AudioFormat{SampleRate: uint64(l.fixture.frontend.SampleRate), Channels: 1, Encoding: "pcm-f32le"}, l.fixture.frontend.Geometry, artifact.ID{}, artifact.ID{})
	if err != nil {
		t.Fatal(err)
	}
	batch, err = contract.Batch("adapter/contract")
	l.commit(t, batch, err)
	var tokenizerID artifact.ID
	for _, component := range inventory.Manifest.Components {
		if component.Role == artifact.ComponentTokenizer && component.Name == "tokenizer.json" {
			tokenizerID = component.Artifact
		}
	}
	l.base, err = modelrecipe.TranscriptionDefinition(inventory.Manifest.ID, contract.ID, profile.ID, tokenizerID, inventory.TensorInventory.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(t.Context(), l.store, "adapter/base-recipe", l.base); err != nil {
		t.Fatal(err)
	}
	l.transform, err = trainingdata.NewTextTransform(true)
	if err != nil {
		t.Fatal(err)
	}
	content, err := l.transform.Content()
	l.content(t, content, err)
	l.binding = adaptertrain.InputProjectionBinding{BaseRecipe: l.base.ID, TargetTransform: l.transform.ID}

	// Content identities bind the selected source coordinates and unmodified
	// transcript, not a fabricated dataset name or copied audio file.
	shard, err := artifact.ParseID("file:sha256:" + l.fixture.record.ShardSHA256)
	if err != nil {
		t.Fatal(err)
	}
	row := l.fixture.record.Row
	l.origin = dataset.AudioPayloadOrigin{Container: shard, Column: "audio.bytes", Row: &row}
	audio, err := artifact.IdentifyBytes(artifact.KindFile, l.fixture.payload)
	if err != nil {
		t.Fatal(err)
	}
	pair := dataset.SpeechRecord{Audio: audio, Origin: l.origin, Target: l.fixture.record.Text}
	encoded, err := json.Marshal(pair)
	if err != nil {
		t.Fatal(err)
	}
	content, err = artifact.JSONContent(artifact.JSONContract(artifact.KindDataset, "overgo/asr-selected-training-row/v1"), pair)
	datasetID := l.content(t, content, err)
	content, err = artifact.JSONContent(artifact.JSONContract(artifact.KindDatasetShard, "overgo/asr-selected-training-split/v1"), pair)
	splitID := l.content(t, content, err)
	path := filepath.Join(filepath.Dir(l.fixture.storeRoot), "datasets", "librispeech_asr-clean-xet", "clean", l.fixture.record.Split, l.fixture.record.Shard)
	location, err := artifact.CanonicalLocalLocation(shard, artifact.LocationFile, path)
	if err != nil {
		t.Fatal(err)
	}
	l.commit(t, artifact.Batch{Key: "adapter/source", Artifacts: []artifact.Descriptor{{ID: shard, Size: l.fixture.record.ShardBytes}}, Locations: []artifact.LocationEvent{{Location: location, Action: artifact.LocationAdd}}}, nil)
	policyBytes, err := os.ReadFile("../dataset/testdata/audio_inspection_policy.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(policyBytes, &l.policy); err != nil {
		t.Fatal(err)
	}
	l.policy.MaximumEncodedBytes = uint64(len(l.fixture.payload))
	l.policy.MaximumSamples = l.fixture.record.Samples
	source, err := dataset.NewAudioPayloadReader(adapterAcceptanceMemory)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { source.Close() })
	processor, err := trainingdata.AudioTextProcessor(l.store, source, l.policy)
	if err != nil {
		t.Fatal(err)
	}
	signature := recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{recipecontract.ModalityAudio}, Outputs: []recipecontract.Modality{recipecontract.ModalityText}}
	authority := trainingdata.Authority{Dataset: datasetID, Split: splitID, Processors: []artifact.ID{profile.ID}, Signature: signature}
	l.data, err = trainingdata.MaterializeRecords(authority, profile.ID, []trainingdata.RawRecord{{ID: l.fixture.record.RowID, Group: l.fixture.record.Split, Data: encoded}}, trainingdata.ProcessorBinding{Artifact: profile.ID, Modalities: []recipecontract.Modality{recipecontract.ModalityAudio, recipecontract.ModalityText}, Process: processor})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { l.data.Close() })
	// No augmentation and no shuffle: the stream cursor is the only changing
	// sampling state in this real single-record proof. Multi-record sampler
	// permutation coverage belongs to the trainingdata unit suite.
	content, err = artifact.JSONContent(artifact.JSONContract(artifact.KindProfile, "overgo/asr-deterministic-stream/v1"), struct{ Shuffle, Augment bool }{})
	l.rngProfile = l.content(t, content, err)
	var workspace speechrecognition.Workspace
	l.adapter, err = l.fixture.encoder.NewOutputAdapter(t.Context(), &workspace, l.fixture.frames, len(l.fixture.targets), l.fixture.blank, trainingprogram.BuiltinOptimizerPolicy())
	if err != nil {
		t.Fatal(err)
	}
	l.hidden, l.frames, err = l.fixture.encoder.Encode(t.Context(), l.fixture.features, l.fixture.frames, &workspace, nil)
	if err != nil {
		t.Fatal(err)
	}
	l.hidden = slices.Clone(l.hidden)
	l.spec = l.trainingSpec(t, datasetID, splitID, []artifact.ID{profile.ID, l.transform.ID}, signature)
	l.plan, err = trainingprogram.CompileTrainingRunPlanFromRepository(t.Context(), l.store, l.spec)
	if err != nil {
		t.Fatal(err)
	}
	return l
}

func (l *adapterLifecycle) trainingSpec(t *testing.T, datasetID, splitID artifact.ID, processors []artifact.ID, signature recipecontract.ModalitySignature) trainingprogram.RunSpec {
	t.Helper()
	profile := func(role, value string) artifact.ID {
		content, err := artifact.JSONContent(artifact.JSONContract(artifact.KindProfile, "overgo/asr-adapter-acceptance-policy/v1"), struct{ Role, Value string }{role, value})
		return l.content(t, content, err)
	}
	policies := trainingprogram.PolicySpec{Precision: profile("precision", "host-f32-weights-f64-muon"), Placement: profile("placement", "before-final-output"),
		Memory: profile("memory", "test-admission-4GiB-not-process-peak"), Optimizer: trainingprogram.BuiltinOptimizerPolicy().ID,
		Checkpoint: profile("checkpoint", "update-boundary-full-state"), Evaluation: profile("evaluation", "exact-replay-only-no-quality-claim"), Promotion: profile("promotion", "candidate-only")}
	content, err := trainingprogram.BuiltinOptimizerPolicy().Content()
	l.content(t, content, err)
	evidenceBytes, err := os.ReadFile("testdata/ctc_training_record.json")
	if err != nil {
		t.Fatal(err)
	}
	evidenceID, err := artifact.IdentifyBytes(artifact.KindEvidence, evidenceBytes)
	if err != nil {
		t.Fatal(err)
	}
	l.content(t, artifact.Content{Descriptor: artifact.Descriptor{ID: evidenceID, Size: uint64(len(evidenceBytes)), MediaType: artifact.JSONMediaType}, Data: evidenceBytes}, nil)
	objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{Name: "CTC adapter lifecycle acceptance", Kind: trainingprogram.ObjectiveCTC,
		Signature: signature, Dataset: datasetID, Split: splitID, Processors: processors, Loss: profile("loss", "ctc-sum"), Evaluation: policies.Evaluation,
		Metric: trainingprogram.MetricTokenAccuracy, Evidence: []artifact.ID{evidenceID}, Authority: trainingprogram.ObjectiveDeclared})
	if err != nil {
		t.Fatal(err)
	}
	content, err = objective.Content()
	policies.Objective = l.content(t, content, err)
	dependencies := []recipe.Dependency{{Role: recipe.DependencyModel, Artifact: l.base.Model}, {Role: recipe.DependencyObjective, Artifact: policies.Objective},
		{Role: recipe.DependencyPrecision, Artifact: policies.Precision}, {Role: recipe.DependencyPlacement, Artifact: policies.Placement},
		{Role: recipe.DependencyMemory, Artifact: policies.Memory}, {Role: recipe.DependencyOptimizer, Artifact: policies.Optimizer},
		{Role: recipe.DependencyCheckpointPolicy, Artifact: policies.Checkpoint}, {Role: recipe.DependencyEvaluation, Artifact: policies.Evaluation}, {Role: recipe.DependencyPromotion, Artifact: policies.Promotion}}
	definition, err := recipe.NewDefinitionWithDependencies(recipe.TaskTraining, dependencies,
		[]recipe.Node{{ID: "batch", Module: workflowrecipe.ModuleBatchDataset, Placement: recipe.PlacementHost}, {ID: "forward", Module: workflowrecipe.ModuleTrainingForward, Placement: recipe.PlacementHost},
			{ID: "backward", Module: workflowrecipe.ModuleBackward, Placement: recipe.PlacementHost}, {ID: "optimize", Module: workflowrecipe.ModuleOptimize, Placement: recipe.PlacementHost}},
		[]recipe.Edge{{From: recipe.Endpoint{Node: "batch", Port: "batch"}, To: recipe.Endpoint{Node: "forward", Port: "batch"}},
			{From: recipe.Endpoint{Node: "forward", Port: "loss"}, To: recipe.Endpoint{Node: "backward", Port: "loss"}},
			{From: recipe.Endpoint{Node: "backward", Port: "gradients"}, To: recipe.Endpoint{Node: "optimize", Port: "gradients"}}}, nil,
		[]recipe.Output{{Name: "checkpoint", Data: recipe.DataCheckpoint, Source: recipe.Endpoint{Node: "optimize", Port: "checkpoint"}}})
	if err != nil {
		t.Fatal(err)
	}
	content, err = definition.ArtifactContent()
	l.content(t, content, err)
	return trainingprogram.RunSpec{Recipe: definition.ID, Initial: trainingprogram.InitialStateSpec{Model: l.base.Model}, Dataset: datasetID, Split: splitID,
		Signature: signature, Processors: processors, Policies: policies, Program: l.adapter.Program()}
}

func (l *adapterLifecycle) rng(position uint64) []trainingprogram.RNGState {
	return []trainingprogram.RNGState{{Name: "augmentation", Algorithm: l.rngProfile}, {Name: "data", Algorithm: l.rngProfile, Counter: position}}
}

func (l *adapterLifecycle) batcher(t *testing.T, state *trainingdata.StreamState) *trainingdata.Batcher {
	t.Helper()
	stream, err := trainingdata.NewStream(l.data, state)
	if err != nil {
		t.Fatal(err)
	}
	batcher, err := trainingdata.NewBatcher(stream, trainingdata.BatchPolicy{Examples: 1, DecodeWorkers: 1, MaxBytes: adapterAcceptanceMemory})
	if err != nil {
		t.Fatal(err)
	}
	return batcher
}

type adapterUpdateReceipt struct {
	ID        string
	RawTarget string
	Loss      float64
	Update    optimizer.StepResult
	Stream    trainingdata.StreamState
}

func (l *adapterLifecycle) update(t *testing.T, batcher *trainingdata.Batcher) adapterUpdateReceipt {
	t.Helper()
	batch, err := batcher.Next(t.Context())
	if err != nil || len(batch.Examples) != 1 {
		t.Fatalf("batch: %v", err)
	}
	example := batch.Examples[0]
	if example.ID != l.fixture.record.RowID || len(example.Values) != 2 {
		t.Fatal("source identity differs")
	}
	raw := string(example.Values[1].Data)
	if raw != l.fixture.record.Text {
		t.Fatal("raw target was modified")
	}
	target, err := l.transform.Apply(raw)
	if err != nil {
		t.Fatal(err)
	}
	tokenizer, err := hfbpe.Load(l.fixture.modelRoot)
	if err != nil {
		t.Fatal(err)
	}
	tokens, err := tokenizer.Encode(target)
	if err != nil || !slices.Equal(tokens, l.fixture.targets) {
		t.Fatalf("target: %v", err)
	}
	// Features are immutable and reused only after the source owner has admitted
	// the same pinned audio. No reference transcript enters inference.
	training := adaptertrain.LinearCTCExample{Hidden: l.hidden, Frames: l.frames, Targets: tokens}
	execution, err := l.adapter.Bind(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if err := execution.Run(&training); err != nil {
		t.Fatal(err)
	}
	return adapterUpdateReceipt{ID: example.ID, RawTarget: raw, Loss: training.Loss, Update: training.Update, Stream: batch.State}
}

func (l *adapterLifecycle) publish(t *testing.T, directory string, state trainingdata.StreamState) trainingprogram.Checkpoint {
	t.Helper()
	spec := trainingprogram.CheckpointSpec{RunPlan: l.plan.ID(), Program: l.adapter.Program().ID(), Model: l.base.Model, Dataset: l.spec.Dataset, Split: l.spec.Split,
		Stream: trainingprogram.DatasetState{Identity: state.Identity, Position: state.Position}, RNG: l.rng(state.Position), Processors: l.plan.Processors(),
		Lineage: []trainingprogram.LineageParent{{Artifact: l.base.Model, Relation: artifact.RelationDependsOn}, {Artifact: l.spec.Dataset, Relation: artifact.RelationDependsOn}, {Artifact: l.spec.Split, Relation: artifact.RelationDependsOn}}}
	checkpoint, err := trainingworkflow.PublishInputProjection(directory, l.adapter, l.binding, spec)
	if err != nil {
		t.Fatal(err)
	}
	return checkpoint
}

func TestASRAdapterPublicationAcceptance(t *testing.T) {
	l := newAdapterLifecycle(t)
	step := l.update(t, l.batcher(t, nil))
	directory := filepath.Join(t.TempDir(), "checkpoint")
	checkpoint := l.publish(t, directory, step.Stream)
	content, err := checkpoint.Content()
	if err != nil || content.Descriptor.ID != checkpoint.ID() {
		t.Fatalf("canonical checkpoint: %v", err)
	}
	batch, err := checkpoint.Batch("adapter/checkpoint", directory)
	l.commit(t, batch, err)
	entries, err := os.ReadDir(directory)
	if err != nil || len(entries) != 2 {
		t.Fatalf("adapter publication must contain only weights and checkpoint state: %v", err)
	}
	loaded, err := l.fixture.encoder.WithOutputProjection(t.Context(), directory, l.binding, checkpoint.Weights)
	if err != nil {
		t.Fatal(err)
	}
	var workspace speechrecognition.Workspace
	hidden, frames, err := loaded.Encode(t.Context(), l.fixture.features, l.fixture.frames, &workspace, nil)
	if err != nil || !slices.Equal(hidden, l.hidden) || frames != l.frames {
		t.Fatalf("frozen encoder changed: %v", err)
	}
	logits, err := loaded.Project(t.Context(), hidden, frames, &workspace)
	if err != nil {
		t.Fatal(err)
	}
	execution, err := l.adapter.Bind(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	forward, err := execution.Select(trainingprogram.PhaseForward)
	if err != nil {
		t.Fatal(err)
	}
	example := adaptertrain.LinearCTCExample{Hidden: l.hidden, Frames: l.frames, Targets: l.fixture.targets}
	if err := forward.Run(&example); err != nil || !slices.Equal(example.Logits, logits) {
		t.Fatalf("loaded adapter logits differ: %v", err)
	}
	tokens, err := speechrecognition.GreedyCTC(t.Context(), make([]int, frames), make([]int, frames), example.Logits, frames, l.fixture.encoder.VocabularySize(), l.fixture.blank)
	if err != nil {
		t.Fatal(err)
	}
	tokenizer, err := hfbpe.Load(l.fixture.modelRoot)
	if err != nil {
		t.Fatal(err)
	}
	expectedText, err := tokenizer.DecodeStrict(tokens)
	if err != nil {
		t.Fatal(err)
	}
	definition, err := modelrecipe.AdaptedTranscriptionDefinition(l.base, checkpoint.ID())
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(t.Context(), l.store, "adapter/recipe", definition); err != nil {
		t.Fatal(err)
	}
	if err := l.store.Close(); err != nil {
		t.Fatal(err)
	}
	l.store, err = overgodb.Open(l.storePath)
	if err != nil {
		t.Fatal(err)
	}
	transcriber, err := speechrecognition.LoadTranscriber(t.Context(), l.store, definition.ID, adapterAcceptanceMemory)
	if err != nil {
		t.Fatal(err)
	}
	environment, err := runrecord.CurrentEnvironment("cpu", "go-host")
	if err != nil {
		t.Fatal(err)
	}
	content, err = environment.Content()
	l.content(t, content, err)
	commit := strings.TrimSpace(baselineCommand(t, l.fixture.root, "git", "rev-parse", "HEAD"))
	var tw speechrecognition.TranscriptionWorkspace
	transcript, run, err := transcriber.Transcribe(t.Context(), l.fixture.payload, l.origin, l.policy, &tw, speechrecognition.RunBinding{Key: "adapter/reload", CodeCommit: commit, Environment: environment.ID, Dataset: l.spec.Dataset, Split: l.spec.Split})
	if err != nil || run.Outcome != runrecord.OutcomeSucceeded {
		t.Fatalf("adapted transcription: %v", err)
	}
	if transcript.Text != expectedText {
		t.Fatalf("served transcript %q differs from trained adapter %q", transcript.Text, expectedText)
	}
	t.Logf("adapter publication: parameters=%d weights=%s checkpoint=%s; real CPU logits and frozen hidden features exact; transcript=%+v; raw training target preserved; no base copy, activation, held-out WER, HTTP or GPU claim", checkpoint.ParameterCount, checkpoint.Weights, checkpoint.ID(), transcript)
}

func TestASRExactResumeAcceptance(t *testing.T) {
	l := newAdapterLifecycle(t)
	if directory := os.Getenv("OVERGO_ASR_RESUME_CHILD"); directory != "" {
		checkpoint, err := trainingprogram.LoadCheckpoint(directory)
		if err != nil {
			t.Fatal(err)
		}
		l.spec.Initial = trainingprogram.InitialStateSpec{Checkpoint: checkpoint.ID()}
		l.plan, err = trainingprogram.CompileTrainingRunPlanFromRepository(t.Context(), l.store, l.spec)
		if err != nil {
			t.Fatal(err)
		}
		authority := trainingprogram.ResumeAuthority{Model: l.base.Model, Stream: trainingprogram.DatasetState{Identity: l.data.Identity(), Position: 1}, OptimizerPlan: l.adapter.Program().OptimizerIdentity()}
		restored, err := trainingworkflow.RestoreInputProjection(directory, l.adapter, l.binding, l.plan, authority, l.rng(1))
		if err != nil {
			t.Fatal(err)
		}
		batcher := l.batcher(t, &trainingdata.StreamState{Identity: restored.Stream.Identity, Position: restored.Stream.Position})
		receipts := []adapterUpdateReceipt{l.update(t, batcher), l.update(t, batcher)}
		l.publish(t, directory+"-resumed", receipts[len(receipts)-1].Stream)
		encoded, err := json.Marshal(receipts)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(directory+"-resumed", "receipts.json"), encoded, 0600); err != nil {
			t.Fatal(err)
		}
		return
	}
	batcher := l.batcher(t, nil)
	first := l.update(t, batcher)
	directory := filepath.Join(t.TempDir(), "checkpoint")
	l.publish(t, directory, first.Stream)
	uninterrupted := []adapterUpdateReceipt{l.update(t, batcher), l.update(t, batcher)}
	command := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestASRExactResumeAcceptance$", "-test.count=1", "-test.v")
	command.Env = append(os.Environ(), "OVERGO_ASR_RESUME_CHILD="+directory)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("fresh process resume: %v\n%s", err, output)
	}
	encoded, err := os.ReadFile(filepath.Join(directory+"-resumed", "receipts.json"))
	if err != nil {
		t.Fatal(err)
	}
	var resumed []adapterUpdateReceipt
	if err := json.Unmarshal(encoded, &resumed); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(uninterrupted, resumed) {
		t.Fatal("fresh-process next batches, raw targets, losses or updates differ")
	}
	final, err := trainingprogram.LoadCheckpoint(directory + "-resumed")
	if err != nil {
		t.Fatal(err)
	}
	state, err := l.adapter.OptimizerSnapshot()
	if err != nil || !reflect.DeepEqual(state, final.Optimizer) || !reflect.DeepEqual(l.rng(3), final.RNG) {
		t.Fatalf("optimizer/scheduler/RNG differs: %v", err)
	}
	weights, err := adaptertrain.LoadInputProjection(directory+"-resumed", l.binding, final.Weights, l.adapter.Program().Parameters()[0].Rows, l.adapter.StorageBytes())
	if err != nil || !slices.Equal(weights, l.adapter.WeightSnapshot()) {
		t.Fatalf("later parameters differ: %v", err)
	}
	t.Logf("fresh-process exact resume: checkpoint after update 1; updates 2 and 3 match next-batch identity, raw targets, losses, parameters, Muon momentum, constant scheduler, RNG and stream cursor; real train.100 rows=1; no shuffle, stochastic augmentation, quality or GPU claim")
}
