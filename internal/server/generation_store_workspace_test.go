package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/latentimage"
	"overgo/internal/media"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// activatedImageModel publishes a model with bytes on disk and a verified
// oscillator image activation: the smallest generation recipe the store
// can declare, with no profile and host placement.
func activatedImageModel(t *testing.T, store *overgodb.Store) artifact.ID {
	t.Helper()
	location := filepath.Join(t.TempDir(), "model.safetensors")
	if err := os.WriteFile(location, []byte("oscillator-weights"), 0o600); err != nil {
		t.Fatal(err)
	}
	return activateModelAt(t, store, location, oscillatorImageDefinition)
}

func oscillatorImageDefinition(modelID artifact.ID) (recipe.Definition, error) {
	return modelrecipe.GenerationDefinition(modelrecipe.ModuleOscillatorImagePrepare, modelID, artifact.ID{})
}

// activateModel publishes a model with bytes on disk and a verified
// activation of the definition built over it.
func activateModel(t *testing.T, store *overgodb.Store, define func(artifact.ID) (recipe.Definition, error)) artifact.ID {
	t.Helper()
	location := filepath.Join(t.TempDir(), "model.safetensors")
	if err := os.WriteFile(location, []byte("oscillator-weights"), 0o600); err != nil {
		t.Fatal(err)
	}
	return activateModelAt(t, store, location, define)
}

// activateModelAt publishes the weights file at location as a model and a
// verified activation of the definition built over it.
func activateModelAt(t *testing.T, store *overgodb.Store, location string, define func(artifact.ID) (recipe.Definition, error)) artifact.ID {
	t.Helper()
	ctx := t.Context()
	payload, err := os.ReadFile(location)
	if err != nil {
		t.Fatal(err)
	}
	weights := testutil.ArtifactBytesID(t, artifact.KindTensorSet, payload)
	manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: weights,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "fixture/generation/model",
		Artifacts: []artifact.Descriptor{{ID: weights, Size: uint64(len(payload))}},
		Manifests: []artifact.Manifest{manifest},
		Locations: []artifact.LocationEvent{{Location: artifact.Location{
			Artifact: weights, Kind: artifact.LocationFile, Value: location,
		}, Action: artifact.LocationAdd}},
	}); err != nil {
		t.Fatal(err)
	}
	definition, err := define(manifest.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/generation/candidate", definition); err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.Transition(ctx, store, "fixture/generation/validated", definition, recipe.StatusValidated, nil, nil); err != nil {
		t.Fatal(err)
	}
	verification, err := modelrecipetest.PublishVerification(ctx, store, "fixture/generation/evidence", definition.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.ActivateVerified(ctx, store, "fixture/generation/active", definition, verification,
		recipe.EvidenceVerified, "generation workspace fixture", nil, nil); err != nil {
		t.Fatal(err)
	}
	return manifest.ID
}

// tinyPNG is a valid one-pixel RGB PNG as an executor would encode it.
func tinyPNG(t *testing.T) latentimage.EncodedImage {
	t.Helper()
	canvas := image.NewRGBA(image.Rect(0, 0, 2, 2))
	canvas.Set(0, 0, color.RGBA{R: 255, A: 255})
	var buffer bytes.Buffer
	if err := png.Encode(&buffer, canvas); err != nil {
		t.Fatal(err)
	}
	return latentimage.EncodedImage{Data: buffer.Bytes(), MediaType: media.PNGMediaType, Channels: media.RGBChannels, Height: 2, Width: 2, Maximum: 1}
}

type recordingReporter struct{ published bool }

func (r *recordingReporter) OperationID() artifact.ID { return artifact.ID{} }
func (r *recordingReporter) Progress(uint64, *uint64) {}
func (r *recordingReporter) Metric(operation.Metric)  {}
func (r *recordingReporter) Attempt(artifact.ID)      {}
func (r *recordingReporter) Text(string)              {}
func (r *recordingReporter) Publishing()              { r.published = true }

// TestGenerationWorkspaceIdentityRefresh requires repeated listings to
// retain the digest memo while still rejecting changed model bytes.
func TestGenerationWorkspaceIdentityRefresh(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	activatedImageModel(t, store)
	workspace := NewStoreGenerationWorkspace(store, BindGenerationCatalog(mediacapability.Catalog, mediacapability.Controls, mediacapability.OutputContent), 16)
	capabilities, err := workspace.WorkflowCapabilities(t.Context(), workflowGeneration)
	if err != nil || len(capabilities) != 1 {
		t.Fatalf("initial capabilities = %+v, %v", capabilities, err)
	}
	memo := workspace.memo
	if memo == nil {
		t.Fatal("identity memo was not retained")
	}
	t.Run("concurrent readers", func(t *testing.T) {
		for range 4 {
			t.Run("listing", func(t *testing.T) {
				t.Parallel()
				listed, err := workspace.WorkflowCapabilities(t.Context(), workflowGeneration)
				if err != nil || len(listed) != 1 {
					t.Fatalf("repeat capabilities = %+v, %v", listed, err)
				}
			})
		}
	})
	if workspace.memo != memo {
		t.Fatal("repeated listing discarded verified identities")
	}
	if err := os.WriteFile(capabilities[0].Location, []byte("changed oscillator weights"), 0o600); err != nil {
		t.Fatal(err)
	}
	if listed, err := workspace.WorkflowCapabilities(t.Context(), workflowGeneration); err != nil || len(listed) != 0 {
		t.Fatalf("changed bytes must not remain present: %+v, %v", listed, err)
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(nil)
	if _, err := workspace.WorkflowCapabilities(ctx, workflowGeneration); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled listing = %v", err)
	}
}

// TestGenerationWorkspaceListsAndRunsStoreActivations pins the workspace
// over a store: the activation lists with the controls its request type
// declares, the run reaches the catalog's executor with the model bytes,
// and the output comes back as a PNG artifact behind a run record.
func TestGenerationWorkspaceListsAndRunsStoreActivations(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := activatedImageModel(t, store)
	var executedPath, executedInput string
	// The executor records the request it decoded, as the runtime does, and
	// the envelope carries it out for the run to cite.
	requestContract := artifact.JSONContract(artifact.KindFile, "overgo.test-image-gen-input.v1")
	catalog := map[recipe.Task]mediacapability.Capability{recipe.TaskImageGen: {
		Execute: func(_ context.Context, _ artifact.Repository, path string, _ modelrecipe.CapabilityEvidenceSelection, raw string) (any, error) {
			executedPath, executedInput = path, raw
			request, err := requestContract.ContentBytes([]byte(raw))
			if err != nil {
				return nil, err
			}
			return capabilityruntime.Measured{Output: tinyPNG(t), Input: request}, nil
		},
	}}
	workspace := NewStoreGenerationWorkspace(store, BindGenerationCatalog(catalog, mediacapability.Controls, mediacapability.OutputContent), 16)
	capabilities, err := workspace.WorkflowCapabilities(ctx, workflowGeneration)
	if err != nil || len(capabilities) != 1 {
		t.Fatalf("capabilities = %+v, %v", capabilities, err)
	}
	capability := capabilities[0]
	if capability.Task != recipe.TaskImageGen || capability.Model != modelID || capability.Refusal != "" || capability.Name != filepath.Base(filepath.Dir(capability.Location)) {
		t.Fatalf("capability = %+v", capability)
	}
	names := map[string]string{}
	for _, control := range capability.Controls {
		names[control.Name] = string(control.Type)
	}
	if names["class"] != "integer" || names["seed"] != "integer" {
		t.Fatalf("controls = %v", names)
	}
	if err := validateWorkflowCapabilities(capabilities); err != nil {
		t.Fatal(err)
	}
	if training, err := workspace.WorkflowCapabilities(ctx, WorkflowTraining); err != nil || training != nil {
		t.Fatalf("training kind answered %+v, %v", training, err)
	}

	reporter := &recordingReporter{}
	completion, err := workspace.ExecuteWorkflow(ctx, workflowGeneration, capability, json.RawMessage(`{"class":1,"seed":7}`), reporter)
	if err != nil {
		t.Fatal(err)
	}
	if !reporter.published || completion.Run.Kind() != artifact.KindRun || len(completion.Outputs) != 1 {
		t.Fatalf("completion = %+v published=%v", completion, reporter.published)
	}
	// Media executors read a repository directory: the one holding the file
	// the catalog resolved.
	if executedPath != filepath.Dir(capability.Location) || !strings.HasSuffix(capability.Location, "model.safetensors") || executedInput != `{"class":1,"seed":7}` {
		t.Fatalf("executor saw path %q input %q for %q", executedPath, executedInput, capability.Location)
	}
	descriptor, found, err := store.Artifact(ctx, completion.Outputs[0])
	if err != nil || !found || descriptor.MediaType != media.PNGMediaType {
		t.Fatalf("output artifact = %+v found=%v err=%v", descriptor, found, err)
	}
	// The run cites the request document the executor recorded as its one
	// input: the stored record a gallery opens and a page regenerates from.
	request, err := requireRecordedRequest(ctx, store, completion.Run)
	if err != nil || request.Schema != requestContract.Schema {
		t.Fatalf("request record = %+v, %v", request, err)
	}
	// A submission's sources (a prompt enhancement the page accepted) are
	// the run's inputs beside the request, so the original stays its source.
	enhancement, err := artifact.JSONContract(artifact.KindFile, promptEnhancementSchema).ContentBytes([]byte(`{"original":"a square","enhanced":"a red square"}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{Key: "fixture/generation/enhancement", Contents: []artifact.Content{enhancement}}); err != nil {
		t.Fatal(err)
	}
	sourced, err := workspace.ExecuteWorkflow(withWorkflowSources(ctx, []artifact.ID{enhancement.Descriptor.ID}), workflowGeneration, capability, json.RawMessage(`{"class":1,"seed":8}`), reporter)
	if err != nil {
		t.Fatal(err)
	}
	run, err := runrecord.RequireRun(ctx, store, sourced.Run)
	if err != nil || len(run.Inputs) != 2 || !slices.Contains(run.Inputs, enhancement.Descriptor.ID) || slices.Contains(run.Inputs, request.ID) {
		t.Fatalf("sourced run = %+v, %v", run, err)
	}
	if _, err := workspace.ExecuteWorkflow(ctx, workflowGeneration, WorkflowCapability{Task: recipe.TaskSpeech, Recipe: capability.Recipe}, json.RawMessage(`{}`), reporter); err == nil {
		t.Fatal("a recipe ran under a task it does not serve")
	}
}

// requireRecordedRequest reads the one input a generation run cites and
// requires it to be a stored JSON document: the request record.
func requireRecordedRequest(ctx context.Context, store *overgodb.Store, runID artifact.ID) (artifact.Descriptor, error) {
	run, err := runrecord.RequireRun(ctx, store, runID)
	if err != nil {
		return artifact.Descriptor{}, err
	}
	if len(run.Inputs) != 1 {
		return artifact.Descriptor{}, fmt.Errorf("run %s cites %d inputs", runID, len(run.Inputs))
	}
	descriptor, reader, found, err := store.OpenContent(ctx, run.Inputs[0])
	if err != nil || !found {
		return descriptor, errors.Join(errors.New("request record is not stored"), err)
	}
	var request map[string]any
	if err := json.NewDecoder(reader).Decode(&request); err != nil || descriptor.MediaType != artifact.JSONMediaType {
		return descriptor, fmt.Errorf("request record is not a JSON document: %v", err)
	}
	return descriptor, nil
}
