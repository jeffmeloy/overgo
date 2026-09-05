package main

import (
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/mediacapability"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

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
	projection := mediacapability.Projection("")
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
