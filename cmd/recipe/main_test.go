package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/mediacapability"
	"overgo/internal/modelartifact"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestPrepareCapabilityReusesPublishedFacts(t *testing.T) {
	path := testutil.TempGGUF(t, "facts.gguf", nil, []gguf.TensorData{{
		Name: "weight", Shape: []uint64{1}, Type: gguf.DTypeF32,
		Data: bytes.NewReader(make([]byte, binary.Size(float32(0)))),
	}})
	file, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	inventory, err := modelartifact.FromGGUF(file, artifact.KindModel)
	if err != nil {
		t.Fatal(err)
	}
	capability := mediacapability.Capability{Resolve: func(string) (mediacapability.Source, error) {
		return mediacapability.Source{Inventory: inventory}, nil
	}}
	want, err := modelrecipe.CapabilityDefinition(recipe.TaskSeq2Seq, inventory.Manifest.ID)
	if err != nil {
		t.Fatal(err)
	}
	directory := t.TempDir()
	store, err := overgodb.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	batch, err := inventory.Batch("fixture/another-publisher")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	head, sequence := store.Head()
	unchanged := func() {
		t.Helper()
		if gotHead, gotSequence := store.Head(); gotHead != head || gotSequence != sequence {
			t.Fatalf("publication changed head/sequence: %s/%d, want %s/%d", gotHead, gotSequence, head, sequence)
		}
	}
	for _, phase := range []string{"foreign publisher", "repeat", "reopen"} {
		t.Run(phase, func(t *testing.T) {
			if phase == "reopen" {
				if err := store.Close(); err != nil {
					t.Fatal(err)
				}
				store, err = overgodb.Open(directory)
				if err != nil {
					t.Fatal(err)
				}
			}
			model, definition, err := prepareCapability(t.Context(), store, path, recipe.TaskSeq2Seq, capability)
			if err != nil || model != inventory.Manifest.ID || definition.ID != want.ID {
				t.Fatalf("prepare = %s/%s, %v; want %s/%s", model, definition.ID, err, inventory.Manifest.ID, want.ID)
			}
			unchanged()
		})
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	if _, _, err := prepareCapability(ctx, store, path, recipe.TaskSeq2Seq, capability); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled preparation = %v", err)
	}
	unchanged()
	inventory.Components[0].MediaType = "application/octet-stream"
	if _, _, err := prepareCapability(t.Context(), store, path, recipe.TaskSeq2Seq, capability); !errors.Is(err, overgodb.ErrArtifactConflict) {
		t.Fatalf("conflicting facts = %v", err)
	}
	unchanged()
}

func TestCommandsRegisterExecutableCapabilities(t *testing.T) {
	for _, task := range []recipe.Task{
		recipe.TaskGeneration, recipe.TaskForecast, recipe.TaskTabular, recipe.TaskSeq2Seq,
		recipe.TaskSpeech, recipe.TaskImageGen, recipe.TaskVideoGen,
	} {
		t.Run(string(task), func(t *testing.T) {
			capability, ok := mediacapability.Catalog[task]
			if !ok || capability.Resolve == nil || capability.Execute == nil {
				t.Fatalf("capability = %+v", capability)
			}
		})
	}
	projection := mediacapability.Projection(t.Context(), nil, "")
	if projection.Resolve == nil || projection.Execute != nil {
		t.Fatalf("projection capability = %+v", projection)
	}
}

func TestCapabilityVerificationActivatesCandidate(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "recipe-verifier-model")
	testutil.PublishArtifact(t, store, modelID)
	definition, err := modelrecipe.CapabilityDefinition(recipe.TaskSeq2Seq, modelID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/recipe-verifier/candidate", definition); err != nil {
		t.Fatal(err)
	}
	verification, err := modelintake.PublishVerification(
		ctx, store, definition, "0123456789abcdef0123456789abcdef01234567", time.Millisecond,
		"host", "go", "fixture output validated",
	)
	if err != nil {
		t.Fatal(err)
	}
	verified, err := runrecord.VerifyGateRun(ctx, store, definition.ID, verification.Gate, verification.Run)
	if err != nil {
		t.Fatal(err)
	}
	if verified.Gate.Recipe != definition.ID || verified.Run.Recipe != definition.ID {
		t.Fatalf("verification subject = %s/%s, want %s", verified.Gate.Recipe, verified.Run.Recipe, definition.ID)
	}
	if err := modelrecipe.ActivateCapability(
		ctx, store, definition, verification, recipe.EvidenceVerified, "fixture candidate execution",
	); err != nil {
		t.Fatal(err)
	}
	activation, active, err := modelrecipe.ActiveRecord(ctx, store, modelID, recipe.TaskSeq2Seq)
	if err != nil {
		t.Fatal(err)
	}
	if !active || activation.Definition.ID != definition.ID {
		t.Fatalf("active = %t/%s, want true/%s", active, activation.Definition.ID, definition.ID)
	}
}
