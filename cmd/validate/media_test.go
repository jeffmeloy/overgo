package main

import (
	"bytes"
	"context"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestMediaReportRendersStoreTruth pins the rendered-report contract:
// an empty store renders the empty statement, media-task scope
// excludes text tasks, and a claim cell states tier, measurements, and
// commit -- or the dash when no verification stands.
func TestMediaReportRendersStoreTruth(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	store, err := overgodb.Create(root + "/store")
	testcheck.NoError(t, err)
	testcheck.NoError(t, store.Close())
	writeEmptyAcceptedCoverage(t, root)
	data, err := generateMediaReport(root, root+"/store")
	testcheck.NoError(t, err)
	text := string(data.Markdown)
	if !strings.Contains(text, "No media activations stand in the store.") ||
		!strings.Contains(text, "Rendered by `go run ./cmd/validate -media markdown`") {
		t.Fatalf("empty-store report:\n%s", text)
	}
	if !strings.Contains(text, "## Measured results") ||
		!strings.Contains(text, "No healthy activation carries a readable run record.") {
		t.Fatalf("measured-results section absent on the empty store:\n%s", text)
	}

	if slices.Contains(mediaTasks, recipe.TaskInference) || slices.Contains(mediaTasks, recipe.TaskTraining) || !slices.Contains(mediaTasks, recipe.TaskImageGen) || !slices.Contains(mediaTasks, recipe.TaskSpeech) {
		t.Fatal("media scope admits text tasks or drops media tasks")
	}

	claims := []runrecord.CapabilityClaim{{
		Capability: "image-gen", Tier: runrecord.TierRealArtifactSmoke,
		Commit: strings.Repeat("ab", 20), WallNS: 1500000000, PeakDeviceBytes: 42,
	}}
	cell := claimCell(claims, recipe.TaskImageGen)
	if !strings.Contains(cell, "real-artifact-smoke") || !strings.Contains(cell, "1.5s") ||
		!strings.Contains(cell, "42 B peak") || !strings.Contains(cell, "abababababab") {
		t.Fatalf("claim cell = %q", cell)
	}
	if claimCell(claims, recipe.TaskVideoGen) != "-" {
		t.Fatal("an absent claim must render the dash")
	}
	if staleCell("") != "-" || !strings.Contains(staleCell("no policy"), "stale: no policy") {
		t.Fatal("stale cell contract")
	}
}

func TestMediaMeasurementProjection(t *testing.T) {
	t.Parallel()
	store := overgodbtest.Open(t, t.TempDir())
	model := testutil.ArtifactID(t, artifact.KindModel, "media projection")
	testutil.PublishArtifact(t, store, model)
	definition, err := modelrecipe.GenerationDefinition(modelrecipe.ModuleOscillatorImagePrepare, model, artifact.ID{})
	testcheck.NoError(t, err)
	testcheck.NoError(t, modelrecipetest.PublishActivation(t.Context(), store, "projection/active", definition))
	missing := testutil.ArtifactID(t, artifact.KindModel, "missing projection activation")
	entries := []discovery.CatalogEntry{
		{Model: model, Capabilities: []discovery.Capability{
			{Task: recipe.TaskImageGen, Recipe: definition.ID},
			{Task: recipe.TaskVideoGen, Stale: "missing verifier"},
			{Task: recipe.TaskInference},
		}},
		{Model: missing, Capabilities: []discovery.Capability{{Task: recipe.TaskImageGen}}},
	}
	scope, err := resolveMediaReportScope("image-video")
	testcheck.NoError(t, err)
	reader := &mediaProjectionReader{Store: store, resolves: map[string]int{}}
	rows, err := loadMediaReportRows(t.Context(), reader, entries, scope, nil, nil)
	testcheck.NoError(t, err)
	if len(rows) != 3 {
		t.Fatalf("selected denominator = %d", len(rows))
	}
	var healthy reportRow
	for _, row := range rows {
		if row.verification.Run.ID.Valid() {
			healthy = row
		}
	}
	if healthy.verification.Run.Recipe != definition.ID || healthy.activation.Definition.ID != definition.ID {
		t.Fatal("missing or foreign verifier")
	}
	if reader.resolves[modelrecipe.ActiveAliasPrefix(recipe.TaskImageGen)+model.String()] != 1 ||
		reader.resolves[modelrecipe.ActiveAliasPrefix(recipe.TaskVideoGen)+model.String()] != 0 ||
		reader.resolves[modelrecipe.ActiveAliasPrefix(recipe.TaskInference)+model.String()] != 0 {
		t.Fatal("repeated activation read or out-of-scope lookup", reader.resolves)
	}
	var headline, inventories bytes.Buffer
	projection, err := projectMediaRows(rows, scope, false)
	testcheck.NoError(t, err)
	writeMediaVerdict(&headline, projection)
	writeMediaInventories(&inventories, rows)
	if !strings.Contains(headline.String(), "3 activations; 2 are stale and 1 healthy") || !strings.Contains(inventories.String(), shortArtifact(definition.ID)) {
		t.Fatal("headline and typed inventory disagree", headline.String())
	}
	if mediaRunResult(nil) != "- | - | - | - | -" {
		t.Fatal("missing measurement invented a zero")
	}
	t.Run("wrong selected recipe", func(t *testing.T) {
		changed := []discovery.CatalogEntry{{Model: model, Capabilities: []discovery.Capability{{Task: recipe.TaskImageGen, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "foreign")}}}}
		rows, err := loadMediaReportRows(t.Context(), store, changed, scope, nil, nil)
		if err != nil || len(rows) != 1 || rows[0].stale == "" || rows[0].verification.Run.ID.Valid() {
			t.Fatal("foreign recipe gained measurement credit", err)
		}
	})
	t.Run("exact bytes and source", func(t *testing.T) {
		verified := healthy.verification
		verified.Gate.Steps = []runrecord.GateStep{{Evidence: "peak_device_bytes=001048576"}}
		measurement, err := projectMediaVerifier(verified)
		testcheck.NoError(t, err)
		cell := mediaRunResult(measurement)
		if !strings.Contains(cell, "1048576 B") || !strings.Contains(cell, shortCommit(verified.Run.CodeCommit)) {
			t.Fatal("byte count or historical producer changed", cell)
		}
		for _, value := range []string{"", "peak_device_bytes=", "peak_device_bytes=-1", "peak_device_bytes=18446744073709551616", "peak_device_bytes=42.5", "peak_device_bytes=42MiB", "not_peak_device_bytes=42"} {
			if _, ok := stepPeakDeviceBytes(value); ok {
				t.Fatal("invalid or absent bytes accepted", value)
			}
		}
		if value, ok := stepPeakDeviceBytes("peak_device_bytes=0"); !ok || value != 0 {
			t.Fatal("observed zero treated as absent")
		}
	})
}

type mediaProjectionReader struct {
	*overgodb.Store
	resolves map[string]int
}

func (r *mediaProjectionReader) ResolveAlias(ctx context.Context, name string) (artifact.ID, bool, error) {
	r.resolves[name]++
	return r.Store.ResolveAlias(ctx, name)
}
