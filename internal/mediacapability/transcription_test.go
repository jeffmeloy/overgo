package mediacapability

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
)

// TestTranscriptionCapabilityCompilesTheSpeechRecipe holds the transcription
// capability to registering a reviewed speech checkpoint as the recipe the
// speech runtime loads, never the bare capability topology: the recipe binds
// the checkpoint's Hugging Face model, tokenizer and tensor inventory, the
// execution profile its reviewed declarations give and the audio contract
// that profile reads, so the runtime's topology check accepts it; once the
// registration commits it, recipe verify plans its execution and evaluate
// publishes its task model definition.
func TestTranscriptionCapabilityCompilesTheSpeechRecipe(t *testing.T) {
	t.Parallel()
	if _, err := modelrecipe.CapabilityDefinition(recipe.TaskTranscription, testutil.ArtifactID(t, artifact.KindModel, "speech")); err == nil {
		t.Fatal("the bare capability topology still compiles transcription")
	}
	for _, modelType := range []string{"granite_speech5_ctc", "nemotron3_5_asr"} {
		t.Run(modelType, func(t *testing.T) {
			t.Parallel()
			ctx := t.Context()
			directory := speechCheckpoint(t, modelType)
			source, err := Catalog[recipe.TaskTranscription].Resolve(directory)
			if err != nil {
				t.Fatal(err)
			}
			definition, facts, err := source.Define(source.Inventory.Manifest.ID)
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := modelrecipe.SpeechComponents(definition); err != nil {
				t.Fatalf("the speech runtime refuses the compiled recipe: %v", err)
			}
			model, _ := definition.PrimaryDependency(recipe.DependencyModel)
			inventory, _ := definition.PrimaryDependency(recipe.DependencyTensorInventory)
			if model != source.Inventory.Manifest.ID || inventory != source.Inventory.TensorInventory.ID {
				t.Fatalf("recipe binds model %s and inventory %s, want the checkpoint's Hugging Face %s and %s",
					model, inventory, source.Inventory.Manifest.ID, source.Inventory.TensorInventory.ID)
			}
			store, err := overgodb.Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			registration, err := source.Inventory.Batch("transcription/registration")
			if err != nil {
				t.Fatal(err)
			}
			registration.Contents = append(registration.Contents, facts...)
			if _, err := artifact.Publish(ctx, store, registration); err != nil {
				t.Fatal(err)
			}
			if _, _, err := modelrecipe.PublishCandidate(ctx, store, "transcription/candidate", definition); err != nil {
				t.Fatal(err)
			}
			program, err := modelrecipe.CompileCapability(definition)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := modelrecipe.CompileCandidateExecution(ctx, store, program); err != nil {
				t.Fatalf("recipe verify cannot plan the candidate's execution: %v", err)
			}
			if _, err := modelrecipe.PublishTaskModelDefinition(ctx, store, definition.ID); err != nil {
				t.Fatalf("evaluate cannot publish the recipe's task model definition: %v", err)
			}
		})
	}
}

// TestWholeClipTranscriptionServesTheResidentRecipe holds the transcription
// capability to registering the resident device recipe wherever it loses no
// kind of request. A model has one active transcription recipe, and the
// server loads that one recipe for whole clips and for streams alike. A
// recognizer with no recurrent decoder transcribes whole clips only, so it
// registers the resident recipe; a recurrent recognizer streams only on its
// CPU recipe, so it keeps that one. A recognizer judging another model's
// speech runs on the CPU, and is named by its CPU recipe either way.
func TestWholeClipTranscriptionServesTheResidentRecipe(t *testing.T) {
	t.Parallel()
	for modelType, placement := range map[string]recipe.Placement{
		"granite_speech5_ctc": recipe.PlacementHybrid,
		"nemotron3_5_asr":     recipe.PlacementHost,
	} {
		t.Run(modelType, func(t *testing.T) {
			t.Parallel()
			directory := speechCheckpoint(t, modelType)
			source, err := Catalog[recipe.TaskTranscription].Resolve(directory)
			if err != nil {
				t.Fatal(err)
			}
			definition, _, err := source.Define(source.Inventory.Manifest.ID)
			if err != nil {
				t.Fatal(err)
			}
			var dependencies [5]artifact.ID
			for index, role := range []recipe.DependencyRole{recipe.DependencyModel, recipe.DependencyProfile, recipe.DependencyProcessorProfile, recipe.DependencyTokenizer, recipe.DependencyTensorInventory} {
				dependencies[index], _ = definition.PrimaryDependency(role)
			}
			host, err := modelrecipe.TranscriptionDefinition(dependencies[0], dependencies[1], dependencies[2], dependencies[3], dependencies[4])
			if err != nil {
				t.Fatal(err)
			}
			served := host
			if placement == recipe.PlacementHybrid {
				if served, err = modelrecipe.DeviceTranscriptionDefinition(host); err != nil {
					t.Fatal(err)
				}
			}
			if definition.ID != served.ID || definition.Nodes[0].Placement != placement {
				t.Fatalf("%s registers recipe %s placed %s, want %s placed %s", modelType, definition.ID, definition.Nodes[0].Placement, served.ID, placement)
			}
			if judge, err := RecognizerRecipe(directory); err != nil || judge != host.ID {
				t.Fatalf("a judging %s, which runs on the CPU, is named %s (%v), want its CPU recipe %s", modelType, judge, err, host.ID)
			}
		})
	}
}

// speechCheckpoint writes a speech checkpoint of the model type the reviewed
// declarations answer for: one tensor, its configuration and a tokenizer.
func speechCheckpoint(t *testing.T, modelType string) string {
	t.Helper()
	directory := t.TempDir()
	if err := safetensors.Save(filepath.Join(directory, "model.safetensors"), map[string][]float32{"weight": {1}}, map[string][]int{"weight": {1}}, nil); err != nil {
		t.Fatal(err)
	}
	for name, body := range map[string]string{
		"config.json":    `{"model_type":"` + modelType + `"}`,
		"tokenizer.json": `{"added_tokens":[],"model":{"type":"BPE","vocab":{"a":0},"merges":[],"byte_fallback":false}}`,
	} {
		if err := os.WriteFile(filepath.Join(directory, name), []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return directory
}
