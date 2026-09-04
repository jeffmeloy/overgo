package modelrecipe

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
)

func TestAudioTaskDefinitionsDeriveContractsFromArtifacts(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	contextContent := audioParentContent(t, artifact.KindProfile, "context")
	stateContent := audioParentContent(t, artifact.KindCheckpoint, "state")
	parents, err := artifact.NewDocumentBatch(
		"fixture/audio-contract-parents", []artifact.Content{contextContent, stateContent}, nil, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, parents); err != nil {
		t.Fatal(err)
	}
	contract, err := audioContractCodec.NewInitial(AudioContractDocument{
		Format:  recipecontract.AudioFormat{SampleRate: 16_000, Channels: 1, Encoding: "pcm-f32le"},
		Frame:   recipecontract.AudioFrameGeometry{WindowSamples: 400, HopSamples: 160, FeatureBins: 80},
		Context: contextContent.Descriptor.ID,
		State:   stateContent.Descriptor.ID,
	})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := contract.Batch("fixture/audio-contract")
	if err != nil {
		t.Fatal(err)
	}
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
			if !found || len(module.Tasks) != 1 || module.Tasks[0] != task ||
				len(module.Outputs) != 1 || module.Outputs[0].Data != output {
				t.Fatalf("audio module = %+v, %t", module, found)
			}
		})
	}
}

func TestAudioTaskDefinitionRejectsUnprovedContractLineage(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	contextID := testutil.ArtifactID(t, artifact.KindProfile, "missing-context")
	contract, err := audioContractCodec.NewInitial(AudioContractDocument{
		Format:  recipecontract.AudioFormat{SampleRate: 16_000, Channels: 1, Encoding: "pcm-f32le"},
		Frame:   recipecontract.AudioFrameGeometry{WindowSamples: 400, HopSamples: 160},
		Context: contextID,
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := contract.Content()
	if err != nil {
		t.Fatal(err)
	}
	unbound, err := artifact.NewDocumentBatch("fixture/unbound-audio-contract", []artifact.Content{content}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
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
	id, err := contract.Identify(data)
	if err != nil {
		t.Fatal(err)
	}
	content, err := contract.Content(id, data)
	if err != nil {
		t.Fatal(err)
	}
	return content
}
