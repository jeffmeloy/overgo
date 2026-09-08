package trainingworkflow

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/trainingprogram"
)

func TestAudioTrainingContract(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	id, err := artifact.IdentifyBytes(artifact.KindRecipe, []byte("audio admission"))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name   string
		change func(*Request)
	}{
		{"repository", func(r *Request) { r.Repository = nil }},
		{"recipe", func(r *Request) { r.Recipe = artifact.ID{} }},
		{"base", func(r *Request) { r.Audio.BaseRecipe = artifact.ID{} }},
		{"case", func(r *Request) { r.Audio.Lowercase = nil }},
		{"memory", func(r *Request) { r.Audio.MemoryBytes = 0 }},
		{"encoded bound", func(r *Request) { r.Audio.Inspection.MaximumEncodedBytes = r.Audio.MemoryBytes + 1 }},
		{"decoded bound", func(r *Request) { r.Audio.Inspection.MaximumSamples = r.Audio.MemoryBytes }},
		{"signal admission", func(r *Request) { r.Audio.Inspection.Admission = recipecontract.AudioAdmissionPolicy{} }},
		{"model", func(r *Request) { r.ModelDirectory = "unused" }},
		{"dataset", func(r *Request) { r.DatasetPath = "unused" }},
		{"reference", func(r *Request) { r.ReferenceDirectory = "unused" }},
		{"objective scale", func(r *Request) { r.ObjectiveScale = 1 }},
		{"lexical", func(r *Request) { r.FreezeLexical = true }},
		{"token sequence", func(r *Request) { r.MaximumSequence = 1 }},
		{"negative steps", func(r *Request) { r.Steps = -1 }},
		{"existing output", func(r *Request) { r.OutputDirectory = t.TempDir() }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			lowercase := false
			r := Request{Repository: store, Recipe: id, OutputDirectory: filepath.Join(t.TempDir(), "checkpoint"), Audio: &AudioTrainingSpec{
				BaseRecipe: id, Lowercase: &lowercase, MemoryBytes: 1024,
				Inspection: dataset.AudioInspectionPolicy{MaximumEncodedBytes: 128, MaximumSamples: 16, ClipThreshold: 1,
					Admission: recipecontract.AudioAdmissionPolicy{MinimumChannels: 1, MaximumChannels: 1, SilenceRMSThreshold: 1.0 / 32768, MaximumAbsoluteDCOffset: 1}},
			}}
			head, sequence := store.Head()
			tc.change(&r)
			if _, err := Execute(t.Context(), r); err == nil {
				t.Fatal("invalid audio request admitted")
			}
			if after, count := store.Head(); after != head || count != sequence {
				t.Fatal("refused request changed store")
			}
		})
	}
	t.Run("cancel before work", func(t *testing.T) {
		ctx, cancel := context.WithCancelCause(t.Context())
		cancel(context.Canceled)
		output := filepath.Join(t.TempDir(), "checkpoint")
		head, sequence := store.Head()
		_, err := Execute(ctx, Request{Repository: store, Recipe: id, OutputDirectory: output, Audio: &AudioTrainingSpec{}})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancellation=%v", err)
		}
		if after, count := store.Head(); after != head || count != sequence {
			t.Fatal("cancellation changed store")
		}
		if _, err := os.Stat(output); !os.IsNotExist(err) {
			t.Fatalf("cancelled request created output: %v", err)
		}
	})
	t.Run("observation units and cancellation", func(t *testing.T) {
		model, err := artifact.IdentifyBytes(artifact.KindModel, []byte("audio observation"))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: "audio/observation", Artifacts: []artifact.Descriptor{{ID: model}, {ID: id}}}); err != nil {
			t.Fatal(err)
		}
		for _, objective := range []trainingprogram.ObjectiveKind{trainingprogram.ObjectiveCTC, trainingprogram.ObjectiveTokenPrediction} {
			observer, err := NewObserver(store, true)
			if err != nil {
				t.Fatal(err)
			}
			result := Result{Objective: objective, StreamPosition: 3}
			observationID, _, _, err := observer.finishTraining(t.Context(), model, id, fmt.Errorf("step: %w", context.Canceled), result)
			if err != nil {
				t.Fatal(err)
			}
			observation, err := runrecord.RequireServingObservation(t.Context(), store, observationID)
			wantTokens := result.StreamPosition
			if objective == trainingprogram.ObjectiveCTC {
				wantTokens = 0
			}
			if err != nil || observation.Outcome != runrecord.OutcomeCancelled || observation.Failure != "" || observation.Usage.InputTokens != wantTokens {
				t.Fatalf("observation=%+v err=%v", observation, err)
			}
		}
	})
}
