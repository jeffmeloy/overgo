//go:build windows

package server

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func publishMOSSProgressFixture(t *testing.T, ctx context.Context, store artifact.Repository, execution modelrecipe.CapabilityEvidenceSelection, running func() error) error {
	t.Helper()
	id := testutil.ArtifactID(t, artifact.KindEvidence, "moss-progress-operation")
	if _, err := artifact.Publish(ctx, store, artifact.Batch{Key: "fixture/moss-progress-operation", Artifacts: []artifact.Descriptor{{ID: id}}}); err != nil {
		return err
	}
	for _, stage := range execution.Program.Stages() {
		receipt := runrecord.StageReceipt{Operation: id, Recipe: execution.Program.Definition().ID, Node: stage.Node.ID, Attempt: 1, State: runrecord.StageAdmitted}
		for _, state := range []runrecord.StageState{runrecord.StageAdmitted, runrecord.StageRunning, runrecord.StageCompleted} {
			receipt.State = state
			if _, err := runrecord.PublishStageReceipt(ctx, store, receipt, nil, nil); err != nil {
				return err
			}
			if state == runrecord.StageRunning && running != nil {
				if err := running(); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

type generationTextReporter struct {
	recordingReporter
	texts []string
}

func (r *generationTextReporter) Text(text string) { r.texts = append(r.texts, text) }

func TestMOSSGenerationProgress(t *testing.T) {
	fixture := newMOSSGenerationFixture(t, func(ctx context.Context, store artifact.Repository, execution modelrecipe.CapabilityEvidenceSelection, _ latentimage.Request) error {
		return publishMOSSProgressFixture(t, ctx, store, execution, nil)
	})
	_, validate, err := fixture.handler.config.LibraryIntake.Validate(t.Context(), fixture.store, fixture.root, "")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = validate(t.Context(), &recordingReporter{}); err != nil {
		t.Fatal(err)
	}
	capabilities, err := fixture.workspace.WorkflowCapabilities(t.Context(), workflowGeneration)
	if err != nil || len(capabilities) != 1 {
		t.Fatal("active capability", capabilities, err)
	}
	reporter := &generationTextReporter{}
	_, err = fixture.workspace.ExecuteWorkflow(t.Context(), workflowGeneration, capabilities[0], json.RawMessage(`{"prompt":"cube","width":32,"height":48}`), reporter)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"Loading model", "Processing prompt", "Generating image", "Decoding image"}
	if !reflect.DeepEqual(reporter.texts, want) {
		t.Fatalf("progress=%q want=%q", reporter.texts, want)
	}
	t.Run("committed-recipe-only", func(t *testing.T) {
		id := testutil.ArtifactID(t, artifact.KindEvidence, "moss-progress-refusal")
		if _, err := artifact.Publish(t.Context(), fixture.store, artifact.Batch{Key: "fixture/moss-progress-refusal", Artifacts: []artifact.Descriptor{{ID: id}}}); err != nil {
			t.Fatal(err)
		}
		receipt := runrecord.StageReceipt{Operation: id, Recipe: capabilities[0].Recipe, Node: capabilities[0].Stages[0].Node.ID, Attempt: 1, State: runrecord.StageAdmitted}
		if _, err := runrecord.PublishStageReceipt(t.Context(), fixture.store, receipt, nil, nil); err != nil {
			t.Fatal(err)
		}
		receipt.State = runrecord.StageRunning
		_, batch, err := runrecord.PrepareStageReceipt(t.Context(), fixture.store, receipt, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		held := &generationTextReporter{}
		store := &generationProgressRepository{Store: fixture.store, recipe: capabilities[0].Recipe, phases: generationPhases(capabilities[0].Stages), reporter: held}
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		if _, err = store.Commit(ctx, batch); err == nil || len(held.texts) != 0 {
			t.Fatal("failed commit announced progress", held.texts, err)
		}
		store.recipe = testutil.ArtifactID(t, artifact.KindRecipe, "other-recipe")
		if _, err = store.Commit(t.Context(), batch); err != nil || len(held.texts) != 0 {
			t.Fatal("other recipe announced progress", held.texts, err)
		}
	})
}
