package server

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
)

// TestGenerationRunPublishesServingObservation runs one image generation
// that succeeds and one that fails through the store generation workspace,
// and holds the scorecard's row for the model to what a user met: one
// failure in two attempts, a time to the whole output, and the device
// memory the successful run peaked at.
func TestGenerationRunPublishesServingObservation(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := activatedImageModel(t, store)
	const peak = 3 << 20
	catalog := map[recipe.Task]mediacapability.Capability{recipe.TaskImageGen: {
		Execute: func(_ context.Context, _ artifact.Repository, _ string, _ modelrecipe.CapabilityEvidenceSelection, raw string) (any, error) {
			if raw == `{"class":2}` {
				return nil, errors.New("the model refused the class")
			}
			return capabilityruntime.Measured{Output: tinyPNG(t), PeakDeviceBytes: peak}, nil
		},
	}}
	workspace := NewStoreGenerationWorkspace(store, BindGenerationCatalog(catalog, mediacapability.Controls, mediacapability.OutputContent), 16)
	capabilities, err := workspace.WorkflowCapabilities(ctx, workflowGeneration)
	if err != nil || len(capabilities) != 1 {
		t.Fatalf("capabilities = %+v, %v", capabilities, err)
	}
	reporter := &recordingReporter{}
	if _, err := workspace.ExecuteWorkflow(ctx, workflowGeneration, capabilities[0], json.RawMessage(`{"class":1}`), reporter); err != nil {
		t.Fatal(err)
	}
	if _, err := workspace.ExecuteWorkflow(ctx, workflowGeneration, capabilities[0], json.RawMessage(`{"class":2}`), reporter); err == nil {
		t.Fatal("the refused generation succeeded")
	}
	rows, err := scorecard.Build(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range rows {
		if row.Model != modelID || row.Task != recipe.TaskImageGen {
			continue
		}
		if cell := row.Cells[metric.Failures]; !cell.Measured() || cell.Value != 0.5 {
			t.Fatalf("failure cell = %+v, want one failure in two attempts", cell)
		}
		if cell := row.Cells[metric.FirstOutput]; !cell.Measured() || cell.Unit != "ms" {
			t.Fatalf("first-output cell = %+v, want the whole output's time", cell)
		}
		if cell := row.Cells[metric.PeakMemory]; !cell.Measured() || cell.Value != peak {
			t.Fatalf("peak-memory cell = %+v, want %d bytes", cell, peak)
		}
		return
	}
	t.Fatalf("the scorecard has no image row for %s: %+v", modelID, rows)
}
