package loop

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// TestMediaExperimentResume holds the acquisition-to-report resume contract: a
// first run acquires every cell once; reopening the store and resuming reuses
// every retained cell without opening a model and returns the identical report;
// an interrupted acquisition checkpoints its completed cells so recovery
// re-acquires only the missing one; and a tampered experiment identity is
// refused.
func TestMediaExperimentResume(t *testing.T) {
	model := testutil.ArtifactID(t, artifact.KindEvidence, "media-model")
	data := testutil.ArtifactID(t, artifact.KindEvidence, "media-data")
	budget := testutil.ArtifactID(t, artifact.KindEvidence, "media-budget")
	const cells = 3

	build := func(t *testing.T, store *overgodb.Store) MediaExperiment {
		t.Helper()
		media := make([]MediaCell, cells)
		for index := range media {
			request := testutil.ArtifactID(t, artifact.KindEvidence, fmt.Sprintf("media-request-%d", index))
			testutil.PublishArtifact(t, store, request)
			media[index] = MediaCell{Index: index, Request: request}
		}
		experiment, err := NewMediaExperiment(model, data, budget, media)
		if err != nil {
			t.Fatal(err)
		}
		return experiment
	}
	acquirer := func(store *overgodb.Store, calls *int, failAt int) Acquirer {
		return func(_ context.Context, cell MediaCell) (artifact.ID, uint64, error) {
			if cell.Index == failAt {
				return artifact.ID{}, 0, errors.New("simulated interruption")
			}
			*calls++
			output := testutil.ArtifactID(t, artifact.KindEvidence, fmt.Sprintf("media-output-%d", cell.Index))
			testutil.PublishArtifact(t, store, output)
			return output, uint64(cell.Index + 1), nil
		}
	}
	poison := func(context.Context, MediaCell) (artifact.ID, uint64, error) {
		t.Error("acquirer opened a model for a fully retained experiment")
		return artifact.ID{}, 0, errors.New("model must not open on a retained experiment")
	}

	directory := t.TempDir()
	store, err := overgodb.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	experiment := build(t, store)

	calls := 0
	report, err := ResumeMediaExperiment(t.Context(), store, experiment, acquirer(store, &calls, -1))
	if err != nil {
		t.Fatal(err)
	}
	if calls != cells || report.Acquired != cells || report.Reused != 0 || len(report.Acquisitions) != cells {
		t.Fatalf("full run: calls=%d report=%+v", calls, report)
	}

	// Reopen the store and resume: retained cells are reused without any acquire.
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, err := overgodb.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	resumed, err := ResumeMediaExperiment(t.Context(), reopened, experiment, poison)
	if err != nil {
		t.Fatal(err)
	}
	if resumed.ID != report.ID || resumed.Reused != cells || resumed.Acquired != 0 {
		t.Fatalf("resume did not reuse every cell without mutation: %+v vs %+v", resumed, report)
	}

	// An interrupted acquisition checkpoints its completed cells; recovery
	// re-acquires only the missing cell.
	partialDirectory := t.TempDir()
	partialStore, err := overgodb.Open(partialDirectory)
	if err != nil {
		t.Fatal(err)
	}
	defer partialStore.Close()
	partial := build(t, partialStore)
	interrupted := 0
	if _, err := ResumeMediaExperiment(t.Context(), partialStore, partial, acquirer(partialStore, &interrupted, cells-1)); err == nil {
		t.Fatal("interrupted acquisition did not surface its error")
	}
	if interrupted != cells-1 {
		t.Fatalf("interrupted run acquired %d cells, want %d before the interruption", interrupted, cells-1)
	}
	recovered := 0
	report, err = ResumeMediaExperiment(t.Context(), partialStore, partial, acquirer(partialStore, &recovered, -1))
	if err != nil {
		t.Fatal(err)
	}
	if recovered != 1 || report.Acquired != 1 || report.Reused != cells-1 {
		t.Fatalf("recovery re-acquired %d cells, want only the missing one: %+v", recovered, report)
	}

	// A tampered experiment identity is refused before any acquisition.
	tampered := experiment
	tampered.ID = testutil.ArtifactID(t, artifact.KindEvidence, "not-the-real-experiment")
	if _, err := ResumeMediaExperiment(t.Context(), reopened, tampered, poison); err == nil {
		t.Fatal("tampered experiment identity was accepted")
	}
}
