package server

import (
	"context"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognitiontest"
	"overgo/internal/testcheck"
)

// TestTranscriptionWorkspaceRecordsWhereItsRecipeRuns holds the runs a
// workspace serves for a host recipe to the host, not to a CPU label no
// other writer uses.
func TestTranscriptionWorkspaceRecordsWhereItsRecipeRuns(t *testing.T) {
	t.Parallel()
	if device := workspaceDevice(t, false); device != "host" {
		t.Fatalf("a host recipe's runs are recorded on device %q, want host", device)
	}
}

// TestTranscriptionWorkspaceRecordsResidentRunsOnTheDevice holds the runs a
// workspace serves for a resident recipe to the CUDA device its encoder binds.
func TestTranscriptionWorkspaceRecordsResidentRunsOnTheDevice(t *testing.T) {
	t.Parallel()
	cudatest.Require(t)
	if device := workspaceDevice(t, true); device != "cuda:0" {
		t.Fatalf("a resident recipe's runs are recorded on device %q, want cuda:0", device)
	}
}

// workspaceDevice activates the fixture encoder's host recipe, or its resident
// recipe, opens a workspace on it and reads the device its runs are recorded on.
func workspaceDevice(t *testing.T, resident bool) string {
	t.Helper()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	testcheck.NoError(t, err)
	defer store.Close()
	fixture := speechrecognitiontest.Publish(t, store, "../speechrecognition/testdata/encoder.json")
	definition, name := fixture.Definition, "host"
	if resident {
		name = "resident"
		if definition, err = modelrecipe.DeviceTranscriptionDefinition(definition); err != nil {
			t.Fatal(err)
		}
		if _, _, err := modelrecipe.PublishCandidate(ctx, store, "transcription/resident-candidate", definition); err != nil {
			t.Fatal(err)
		}
	}
	verification := testcheck.Must(modelrecipetest.PublishVerification(ctx, store, "transcription/"+name+"-verification", definition.ID))(t)
	testcheck.NoError(t, modelrecipe.ActivateCapability(ctx, store, definition, verification, recipe.EvidenceParity, "isolated environment fixture activation"))
	workspace := testcheck.Must(NewTranscriptionWorkspace(ctx, store, TranscriptionPolicy{Recipe: definition.ID, MemoryBytes: 1 << 20, Inspection: fixture.Policy}, transcriptionHTTPCommit))(t)
	defer func() {
		if err := workspace.Close(context.WithoutCancel(ctx)); err != nil {
			t.Error(err)
		}
	}()
	environment := testcheck.Must(runrecord.RequireEnvironment(ctx, store, workspace.environment))(t)
	return environment.Device
}
