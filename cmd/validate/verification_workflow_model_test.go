//go:build modeltest

package main

import (
	"bytes"
	"cmp"
	"encoding/json"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestStructuredMediaReportProjection holds the media report to the live
// store's retained evidence and its protocol to the declared denominator.
// It reads the live store, so it runs in the improvement cycle: an
// activation change in the store never refuses a landing.
func TestStructuredMediaReportProjection(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	storePath := retainedReferenceStore(roots.Store)
	store, err := overgodb.OpenReadOnly(storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	t.Run("shared retained projection", func(t *testing.T) {
		assertStructuredMediaReport(t, root, storePath)
	})
	t.Run("protocol counterexamples", func(t *testing.T) {
		assertStructuredMediaProtocol(t, root, store)
	})
}

func assertStructuredMediaReport(t *testing.T, root, storePath string) {
	t.Helper()
	output, err := generateMediaReport(root, storePath)
	if err != nil {
		t.Fatal(err)
	}
	var value mediaProjection
	if err := json.Unmarshal(output.JSON, &value); err != nil {
		t.Fatal(err)
	}
	var index mediaValidationIndex
	if err := jsonfile.DecodeStrict(filepath.Join(root, mediaValidationPath), &index); err != nil {
		t.Fatal(err)
	}
	var protocol imageVideoProtocol
	if err := jsonfile.DecodeStrict(filepath.Join(root, imageVideoProtocolPath), &protocol); err != nil {
		t.Fatal(err)
	}
	if value.Validation == nil || value.Protocol == nil || len(value.Validation.Checks) != len(index.Checks) || len(value.Protocol.Cases) != len(protocol.Cases) || value.Counts.Activations != len(value.Rows) {
		t.Fatal("declared denominator omitted or totals disagree")
	}
	if value.Validation.Passed == 0 || value.Validation.Unresolved == 0 || value.Validation.Passed+value.Validation.Unresolved != len(index.Checks) {
		t.Fatal("historical successes or failures lost", value.Validation)
	}
	for i, check := range value.Validation.Checks {
		declared := index.Checks[i]
		if check.Evidence != declared.Evidence || check.PriorEvidence != declared.PriorEvidence || check.Scope != declared.Scope || check.Source != cmp.Or(declared.Source, index.Source) ||
			!maps.Equal(check.SourceFiles, declared.SourceFiles) || !maps.Equal(check.Harness, declared.Harness) || len(check.Assertions) == 0 || check.RepairOwner == "" {
			t.Fatal("lost assertion scope, provenance or repair ownership", check.Name)
		}
	}
	if value.Validation.PriorCapacity != index.PriorCapacity {
		t.Fatal("prior capacity checkpoint lost")
	}
	for i, cell := range value.Protocol.Cases {
		if cell.ID != protocol.Cases[i].ID || cell.Model != protocol.Cases[i].Model || cell.Task != protocol.Cases[i].Task || cell.Run != protocol.Cases[i].Run || !slices.Equal(cell.Outputs, protocol.Cases[i].Outputs) || cell.Gap == "" || cell.RepairOwner == "" {
			t.Fatal("lost protocol scope or inferred current acceptance", cell.ID)
		}
	}
	for _, row := range value.Rows {
		if row.GenerationGap == "" || row.RepairOwner == "" {
			t.Fatal("activation-only row gained generation acceptance")
		}
	}
	var rendered bytes.Buffer
	writeMediaValidation(&rendered, value.Validation)
	writeMediaProtocol(&rendered, value.Protocol)
	if !bytes.Contains(output.Markdown, rendered.Bytes()) {
		t.Fatal("human report differs from typed evidence projection")
	}
	t.Logf("%d activations; %d declared checks (%d historical passes, %d unresolved); %d protocol cases; no acquisition or current-source promotion", len(value.Rows), len(index.Checks), value.Validation.Passed, value.Validation.Unresolved, len(protocol.Cases))
}

func assertStructuredMediaProtocol(t *testing.T, root string, reader artifact.Reader) {
	t.Helper()
	var protocol imageVideoProtocol
	if err := jsonfile.DecodeStrict(filepath.Join(root, imageVideoProtocolPath), &protocol); err != nil {
		t.Fatal(err)
	}
	write := func(value imageVideoProtocol) string {
		t.Helper()
		fixture := t.TempDir()
		path := filepath.Join(fixture, imageVideoProtocolPath)
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			t.Fatal(err)
		}
		raw, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, raw, 0600); err != nil {
			t.Fatal(err)
		}
		return fixture
	}
	for name, mutate := range map[string]func(*imageVideoProtocol){
		"empty denominator":  func(p *imageVideoProtocol) { p.Cases = nil },
		"duplicate case":     func(p *imageVideoProtocol) { p.Cases = append(p.Cases, p.Cases[0]) },
		"missing assertions": func(p *imageVideoProtocol) { p.RequiredChecks = nil },
		"duplicate assertion": func(p *imageVideoProtocol) {
			p.RequiredChecks = append(slices.Clone(p.RequiredChecks), p.RequiredChecks[0])
		},
		"task scope":          func(p *imageVideoProtocol) { p.Cases[0].Task = recipe.TaskInference },
		"negative width":      func(p *imageVideoProtocol) { p.Cases[0].Observations[0].Width = -1 },
		"zero height":         func(p *imageVideoProtocol) { p.Cases[0].Observations[0].Height = 0 },
		"empty frames":        func(p *imageVideoProtocol) { p.Cases[0].Observations[0].Frames = 0 },
		"negative frame rate": func(p *imageVideoProtocol) { p.Cases[0].Observations[0].FPS = -1 },
		"negative delay":      func(p *imageVideoProtocol) { p.Cases[0].Observations[0].Delay = -1 },
	} {
		t.Run(name, func(t *testing.T) {
			changed := protocol
			changed.Cases = slices.Clone(protocol.Cases)
			changed.Cases[0].Observations = slices.Clone(protocol.Cases[0].Observations)
			mutate(&changed)
			if _, err := projectMediaProtocol(t.Context(), write(changed), reader, nil); err == nil {
				t.Fatal("invalid protocol accepted")
			}
		})
	}
	first := protocol.Cases[0]
	other := testutil.ArtifactID(t, artifact.KindRecipe, "replacement active recipe")
	rows := []mediaProjectionRow{{Model: first.Model, Task: first.Task, Recipe: other}}
	projected, err := projectMediaProtocol(t.Context(), root, reader, rows)
	if err != nil || !projected.Cases[0].Retained || projected.Cases[0].CurrentRecipe != other || projected.Cases[0].Recipe != first.Recipe || !strings.Contains(projected.Cases[0].Gap, "different recipe") {
		t.Fatal("replacement activation overwrote historical authority", err)
	}
	changed := protocol
	changed.Cases = slices.Clone(protocol.Cases)
	changed.Cases[0].Run = testutil.ArtifactID(t, artifact.KindRun, "missing retained generation")
	projected, err = projectMediaProtocol(t.Context(), write(changed), reader, rows)
	if err != nil || projected.Cases[0].Retained || projected.Cases[0].Gap == "" || projected.Retained >= len(protocol.Cases) {
		t.Fatal("missing run gained evidence credit", err)
	}
}
