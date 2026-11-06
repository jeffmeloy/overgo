package modelrecipe

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestAudioTaskDefinitionsDeriveContractsFromArtifacts(t *testing.T) {
	store := overgodbtest.Open(t, t.TempDir())

	contextContent := audioParentContent(t, artifact.KindProfile, "context")
	stateContent := audioParentContent(t, artifact.KindCheckpoint, "state")
	parents := testcheck.Must(artifact.NewDocumentBatch(
		"fixture/audio-contract-parents", []artifact.Content{contextContent, stateContent}, nil, nil,
	))(t)
	if _, err := artifact.CommitBatch(t.Context(), store, parents); err != nil {
		t.Fatal(err)
	}
	contract := testcheck.Must(audioContractCodec.NewInitial(AudioContractDocument{
		Format:  recipecontract.AudioFormat{SampleRate: 16_000, Channels: 1, Encoding: "pcm-f32le"},
		Frame:   recipecontract.AudioFrameGeometry{WindowSamples: 400, HopSamples: 160, FeatureBins: 80},
		Context: contextContent.Descriptor.ID,
		State:   stateContent.Descriptor.ID,
	}))(t)
	batch := testcheck.Must(contract.Batch("fixture/audio-contract"))(t)
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}

	loaded, err := audioContractCodec.RequireExactLineage(
		t.Context(), store, contract.ID, AudioContractDocument.Lineage,
	)
	if err != nil || loaded.ID != contract.ID || loaded.Context != contextContent.Descriptor.ID ||
		loaded.State != stateContent.Descriptor.ID {
		t.Fatalf("loaded contract = %+v, %v", loaded, err)
	}

	tasks := map[recipe.Task]recipe.DataKind{
		recipe.TaskTranscription:     recipe.DataTranscription,
		recipe.TaskAlignment:         recipe.DataTimestampedAlignment,
		recipe.TaskDiarization:       recipe.DataSpeechTurns,
		recipe.TaskActivityDetection: recipe.DataActivitySegments,
		recipe.TaskAudioConversion:   recipe.DataConvertedAudio,
		recipe.TaskAudioGeneration:   recipe.DataGeneratedAudio,
	}
	for task, output := range tasks {
		t.Run(string(task), func(t *testing.T) {
			spec := audioTaskSpecs[task]
			module, found := catalog.Module(spec.module)
			if !found || !slices.Contains(module.Tasks, task) ||
				len(module.Outputs) != 1 || module.Outputs[0].Data != output {
				t.Fatalf("audio module = %+v, %t", module, found)
			}
		})
	}
}

func TestTranscriptionRecipeBindsArtifactLineage(t *testing.T) {
	store := overgodbtest.Open(t, t.TempDir())
	contract, err := NewAudioContract(
		recipecontract.AudioFormat{SampleRate: 16_000, Channels: 1, Encoding: "pcm-f32le"},
		recipecontract.AudioFrameGeometry{WindowSamples: 400, HopSamples: 160, FeatureBins: 80},
		artifact.ID{}, artifact.ID{},
	)
	testcheck.NoError(t, err)
	contractContent := testcheck.Must(contract.Content())(t)
	model := audioParentContent(t, artifact.KindModel, "transcription-model")
	processor := audioParentContent(t, artifact.KindProfile, "transcription-processor")
	tokenizer := audioParentContent(t, artifact.KindTokenizer, "transcription-tokenizer")
	inventory := audioParentContent(t, artifact.KindTensorInventory, "transcription-tensors")
	dependencies := testcheck.Must(artifact.NewDocumentBatch("fixture/transcription/dependencies",
		[]artifact.Content{model, contractContent, processor, tokenizer, inventory}, nil, nil))(t)
	if _, err = artifact.CommitBatch(t.Context(), store, dependencies); err != nil {
		t.Fatal(err)
	}
	definition := testcheck.Must(TranscriptionDefinition(model.Descriptor.ID, contract.ID, processor.Descriptor.ID, tokenizer.Descriptor.ID, inventory.Descriptor.ID))(t)
	program := testcheck.Must(CompileCapability(definition))(t)
	if program.Definition().ID != definition.ID || len(definition.Nodes) != 1 || definition.Nodes[0].Module != ModuleTranscribeAudio ||
		definition.Nodes[0].Placement != recipe.PlacementHost || definition.Nodes[0].Session != recipe.SessionCapacity ||
		definition.Nodes[0].Residency != recipe.ResidencyHostCache {
		t.Fatalf("compiled transcription program = %+v", program)
	}
	if _, _, err = PublishCandidate(t.Context(), store, "fixture/transcription/candidate", definition); err != nil {
		t.Fatal(err)
	}
	parents := testcheck.Must(store.Parents(t.Context(), definition.ID))(t)
	for _, dependency := range definition.Dependencies {
		edge := artifact.Lineage{Child: definition.ID, Parent: dependency.Artifact, Relation: artifact.RelationDependsOn}
		if !slices.Contains(parents, edge) {
			t.Fatalf("recipe lineage lacks %+v: %v", edge, parents)
		}
	}
	loaded, err := RequireAudioContract(t.Context(), store, contract.ID)
	if err != nil || loaded.ID != contract.ID {
		t.Fatalf("loaded audio contract = %+v, %v", loaded, err)
	}
	checkpoint := audioParentContent(t, artifact.KindCheckpoint, "adapter")
	adapted := testcheck.Must(AdaptedTranscriptionDefinition(definition, checkpoint.Descriptor.ID))(t)
	if _, err := CompileCapability(adapted); err != nil {
		t.Fatal(err)
	}
	if actual, ok := adapted.PrimaryDependency(recipe.DependencyCheckpoint); !ok || actual != checkpoint.Descriptor.ID {
		t.Fatal("adapted recipe lost checkpoint authority")
	}
	if _, err := AdaptedTranscriptionDefinition(adapted, checkpoint.Descriptor.ID); err == nil {
		t.Fatal("already adapted base accepted")
	}
	if _, err := AdaptedTranscriptionDefinition(definition, artifact.ID{}); err == nil {
		t.Fatal("absent checkpoint accepted")
	}
	altered := definition
	altered.Nodes = slices.Clone(definition.Nodes)
	altered.Nodes[0].Placement = recipe.PlacementDevice
	if _, err := AdaptedTranscriptionDefinition(altered, checkpoint.Descriptor.ID); err == nil {
		t.Fatal("mutated base topology accepted")
	}
	resident := testcheck.Must(DeviceTranscriptionDefinition(definition))(t)
	if _, err := CompileCapability(resident); err != nil {
		t.Fatal(err)
	}
	if resident.ID == definition.ID || resident.Nodes[0].Placement != recipe.PlacementHybrid || definition.Nodes[0].Placement != recipe.PlacementHost {
		t.Fatal("device binding lost separate host authority")
	}
	for _, invalid := range []recipe.Definition{adapted, altered, resident} {
		if _, err := DeviceTranscriptionDefinition(invalid); err == nil {
			t.Fatal("noncanonical device base accepted")
		}
	}
}

func TestAudioTaskDefinitionRejectsUnprovedContractLineage(t *testing.T) {
	store := overgodbtest.Open(t, t.TempDir())
	contextID := testutil.ArtifactID(t, artifact.KindProfile, "missing-context")
	contract := testcheck.Must(audioContractCodec.NewInitial(AudioContractDocument{
		Format:  recipecontract.AudioFormat{SampleRate: 16_000, Channels: 1, Encoding: "pcm-f32le"},
		Frame:   recipecontract.AudioFrameGeometry{WindowSamples: 400, HopSamples: 160},
		Context: contextID,
	}))(t)
	content := testcheck.Must(contract.Content())(t)
	unbound := testcheck.Must(artifact.NewDocumentBatch("fixture/unbound-audio-contract", []artifact.Content{content}, nil, nil))(t)
	if _, err := artifact.CommitBatch(t.Context(), store, unbound); err != nil {
		t.Fatal(err)
	}
	if _, err := audioContractCodec.RequireExactLineage(
		t.Context(), store, contract.ID, AudioContractDocument.Lineage,
	); err == nil {
		t.Fatal("audio contract accepted without stored artifact lineage")
	}
}

func audioParentContent(t *testing.T, kind artifact.Kind, name string) artifact.Content {
	t.Helper()
	contract := artifact.DocumentContract{
		Kind: kind, MediaType: "application/vnd.overgo.test-audio-parent+json", Schema: "overgo/test-audio-parent/v1",
	}
	data := []byte(`{"name":"` + name + `"}`)
	id := testcheck.Must(contract.Identify(data))(t)
	content := testcheck.Must(contract.Content(id, data))(t)
	return content
}
