package main

import (
	"bytes"
	"encoding/json"
	"slices"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

func TestAcceptedProtocolCoverageProjection(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration)
	}
	// Frozen independent assignments and native-output checks remain lower bounds.
	t.Run("text-vision", TestAcceptedTextVisionEvidence)
	t.Run("image-video", TestAcceptedImageVideoEvidence)
	t.Run("specialized", TestAcceptedSpecializedTaskEvidence)
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	entries, truncated, err := discovery.RegisteredCatalog(t.Context(), store, mediaCatalogLimit, discovery.LoadMemo(t.Context(), store))
	if err != nil || truncated {
		t.Fatal("full denominator unavailable", err)
	}
	value, err := projectModalityCoverage(t.Context(), root, store)
	if err != nil {
		t.Fatal(err)
	}
	first := slices.IndexFunc(value.Models, func(row modalityModelProjection) bool { return len(row.Cells) > 0 && row.Cells[0].Accepted })
	gap := slices.IndexFunc(value.Models, func(row modalityModelProjection) bool { return len(row.Cells) > 0 && !row.Cells[0].Accepted })
	if first < 0 || gap < 0 {
		t.Fatal("accepted protocols or uncovered activations disappeared")
	}
	for name, mutate := range map[string]func(*modalityCoverageProjection){
		"omitted model":    func(p *modalityCoverageProjection) { p.Models = p.Models[1:] },
		"duplicated model": func(p *modalityCoverageProjection) { p.Models[1] = p.Models[0] },
		"omitted cell":     func(p *modalityCoverageProjection) { p.Models[first].Cells = nil },
		"duplicated cell": func(p *modalityCoverageProjection) {
			p.Models[first].Cells = append(p.Models[first].Cells, p.Models[first].Cells[0])
		},
		"unbound recipe":           func(p *modalityCoverageProjection) { p.Models[first].Cells[0].Recipe = p.Models[gap].Cells[0].Recipe },
		"omitted scope":            func(p *modalityCoverageProjection) { p.Models[first].Cells[0].Scope = "" },
		"missing proof":            func(p *modalityCoverageProjection) { p.Proofs = nil },
		"missing phase":            func(p *modalityCoverageProjection) { p.Models[first].Cells[0].AcceptedPhases-- },
		"activation-only accepted": func(p *modalityCoverageProjection) { p.Models[gap].Cells[0].Accepted = true },
		"hosted claim":             func(p *modalityCoverageProjection) { p.Models[first].Execution = "hosted" },
		"missing artifact":         func(p *modalityCoverageProjection) { p.Models[first].Present = false },
		"contradictory total":      func(p *modalityCoverageProjection) { p.AcceptedActivations++ },
		"gap without owner":        func(p *modalityCoverageProjection) { p.Models[gap].Cells[0].RepairOwner = "" },
	} {
		t.Run(name, func(t *testing.T) {
			data, err := json.Marshal(value)
			if err != nil {
				t.Fatal(err)
			}
			var changed modalityCoverageProjection
			if err := json.Unmarshal(data, &changed); err != nil {
				t.Fatal(err)
			}
			mutate(&changed)
			if checkModalityCoverageProjection(&changed, entries) == nil {
				t.Fatal("invalid coverage received credit")
			}
		})
	}
	output, err := generateMediaReport(root, retainedReferenceStore(roots.Store))
	if err != nil {
		t.Fatal(err)
	}
	var report mediaProjection
	if err := json.Unmarshal(output.JSON, &report); err != nil {
		t.Fatal(err)
	}
	if report.Coverage == nil {
		t.Fatal("combined coverage omitted")
	}
	if err := checkModalityCoverageProjection(report.Coverage, entries); err != nil {
		t.Fatal(err)
	}
	var rendered bytes.Buffer
	writeModalityCoverage(&rendered, report.Coverage)
	if !bytes.Contains(output.Markdown, rendered.Bytes()) {
		t.Fatal("human and structured reports disagree")
	}
	t.Logf("%d registrations; %d activations; %d accepted in original protocol scopes; no model acquisitions", value.Registered, value.Activations, value.AcceptedActivations)
}
