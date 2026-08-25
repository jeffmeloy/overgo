package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/dataset"
	"overgo/internal/model"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
)

func TestProfileCatalogQueryReportsExactCoverage(t *testing.T) {
	repository := t.TempDir()
	store, err := overgodb.Open(repository)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	var emptyOutput bytes.Buffer
	if err := run([]string{"-repo", repository, "-profiles", "-json"}, &emptyOutput); err != nil {
		t.Fatal(err)
	}
	var empty modelrecipe.ProfileCatalogCoverage
	if err := json.Unmarshal(emptyOutput.Bytes(), &empty); err != nil {
		t.Fatal(err)
	}
	if empty.Complete || empty.Published != 0 || empty.Registered != len(model.SupportedArchitectures()) {
		t.Fatalf("empty coverage = %+v", empty)
	}
	store, err = overgodb.Open(repository)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := modelrecipe.PublishArchitectureProfileCatalog(context.Background(), store); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := run([]string{"-repo", repository, "-profiles", "-json"}, &output); err != nil {
		t.Fatal(err)
	}
	var coverage modelrecipe.ProfileCatalogCoverage
	if err := json.Unmarshal(output.Bytes(), &coverage); err != nil {
		t.Fatal(err)
	}
	if !coverage.Complete || coverage.Registered != len(model.SupportedArchitectures()) ||
		coverage.Published != coverage.Registered || len(coverage.Entries) != coverage.Registered {
		t.Fatalf("coverage = %+v", coverage)
	}
}

func TestDatasetCatalogQueryReportsExactCoverage(t *testing.T) {
	root, path := t.TempDir(), filepath.Join(t.TempDir(), "fixture.txt")
	if err := os.WriteFile(path, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	inventory, err := dataset.NewInventory([]dataset.InventoryFile{{
		Path: filepath.Base(path), OriginalName: filepath.Base(path), Modality: "text", Format: "txt", Bytes: 1,
	}})
	if err != nil {
		t.Fatal(err)
	}
	version, err := dataset.NewVersion([]dataset.Asset{{Name: "inventory", Artifact: inventory.ID, Records: 1}})
	if err != nil {
		t.Fatal(err)
	}
	catalog, err := dataset.NewCatalog([]dataset.CatalogEntry{{
		Name: "fixture", Dataset: version.ID, Inventory: inventory.ID, StorageKind: "file", Source: "test",
		Modality: "text", Formats: []string{"txt"}, Files: 1, Bytes: 1,
	}})
	if err != nil {
		t.Fatal(err)
	}
	location, err := artifact.CanonicalLocalLocation(version.ID, artifact.LocationFile, path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := dataset.PublishCatalog(context.Background(), store, dataset.CompiledCatalog{
		Catalog: catalog, Datasets: []dataset.Document{version}, Inventories: []dataset.Inventory{inventory},
		Locations: []artifact.Location{location},
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := run([]string{"-repo", root, "-datasets", "-json"}, &output); err != nil {
		t.Fatal(err)
	}
	var coverage dataset.CatalogCoverage
	if err := json.Unmarshal(output.Bytes(), &coverage); err != nil {
		t.Fatal(err)
	}
	if !coverage.Complete || coverage.Published != coverage.Registered || coverage.Available != coverage.Registered {
		t.Fatalf("coverage = %+v", coverage)
	}
}

func TestMagicClosureQueries(t *testing.T) {
	root := t.TempDir()
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	source := []byte("package policy\nconst PolicyWindow = 3\n")
	sourceHash, callsiteHash := sha256.Sum256(source), sha256.Sum256([]byte("callsite"))
	owner, err := artifact.IdentifyBytes(artifact.KindFile, source)
	if err != nil {
		t.Fatal(err)
	}
	fixture, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("fixture"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key: "magic/fixture", Artifacts: []artifact.Descriptor{{ID: owner}, {ID: fixture}},
	}); err != nil {
		t.Fatal(err)
	}
	binding := closureledger.SourceBinding{
		Kind: closureledger.BindingConstant, Package: "internal/policy", File: "internal/policy/policy.go",
		Scope: "package", Name: "PolicyWindow", Line: 1, Expression: "3",
		SourceID: hex.EncodeToString(sourceHash[:]), CallsiteID: hex.EncodeToString(callsiteHash[:]), Owner: owner,
	}
	document, err := closureledger.New(
		binding.Name, json.RawMessage("3"), closureledger.TierDerivationBlocked, closureledger.StatusOpen,
		"Fixture policy remains unresolved.", []closureledger.SourceBinding{binding},
		"Derive from the policy owner.", "Policy owner change.", fixture,
	)
	if err != nil {
		t.Fatal(err)
	}
	documentBatch, err := document.Batch("magic/document", nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(context.Background(), documentBatch); err != nil {
		t.Fatal(err)
	}
	head, sequence := store.Head()
	first, err := closurescan.NewCensusEvidence(
		magicCensusFixture(hex.EncodeToString(sourceHash[:]), 1, 2), head, sequence,
		[]closureledger.Document{document}, []closurescan.BindingIssue{{Kind: closurescan.DriftSource, Name: binding.Name, File: binding.File}},
	)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := first.Batch(nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(context.Background(), batch); err != nil {
		t.Fatal(err)
	}
	head, sequence = store.Head()
	second, err := closurescan.NewCensusEvidence(
		magicCensusFixture(hex.EncodeToString(sourceHash[:]), 1, 3), head, sequence,
		[]closureledger.Document{document}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	batch, err = second.Batch(&first.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(context.Background(), batch); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	var encoded bytes.Buffer
	if err := run([]string{"-repo", root, "-magic-closures", "-json"}, &encoded); err != nil {
		t.Fatal(err)
	}
	var report magicClosureReport
	if err := json.Unmarshal(encoded.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if len(report.Runs) != len([]artifact.ID{first.ID, second.ID}) || report.Runs[0].ID != first.ID || report.Runs[1].ID != second.ID {
		t.Fatalf("runs = %+v", report.Runs)
	}
	if len(report.Runs[0].Evidence.Unresolved) != first.Pressure.OpenDocuments ||
		len(report.Runs[0].Evidence.Stale) != first.Pressure.StaleBindings ||
		report.Runs[1].Delta.Counts.InlineLiterals != second.Counts.InlineLiterals-first.Counts.InlineLiterals ||
		report.Runs[1].Delta.Pressure.StaleBindings != second.Pressure.StaleBindings-first.Pressure.StaleBindings {
		t.Fatalf("report = %+v", report)
	}
	var text bytes.Buffer
	if err := run([]string{"-repo", root, "-magic-closures"}, &text); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"open=PolicyWindow", "stale=source", "delta named="} {
		if !strings.Contains(text.String(), want) {
			t.Fatalf("text output missing %q: %s", want, text.String())
		}
	}
}

func magicCensusFixture(source string, named, inline int) closurescan.Census {
	return closurescan.Census{
		Schema: closurescan.CensusSchema, Source: source,
		Counts: closurescan.CensusCounts{
			ProductionFiles: 1, NamedConstants: named, InlineLiterals: inline, Unknown: inline,
		},
		Owners: []closurescan.OwnerPressure{{
			Package: "internal/policy", DecisionSurfaces: named + inline,
			NamedConstants: named, InlineLiterals: inline,
		}},
		Files: []closurescan.FilePressure{{
			File: "internal/policy/policy.go", Package: "internal/policy", DecisionSurfaces: named + inline,
			NamedConstants: named, InlineLiterals: inline,
		}},
	}
}

func TestRunQueriesSeededStore(t *testing.T) {
	root := t.TempDir()
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	id, _ := artifact.IdentifyBytes(artifact.KindDataset, []byte("dataset"))
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key:       "fixture/query-cli",
		Artifacts: []artifact.Descriptor{{ID: id, Size: 7}},
		Aliases:   []artifact.AliasBinding{{Name: "dataset/current", Target: id}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	var text bytes.Buffer
	if err := run([]string{"-repo", root, "-kind", "dataset"}, &text); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(text.String(), id.String()) || !strings.Contains(text.String(), "dataset/current") {
		t.Fatalf("text output = %q", text.String())
	}
	var encoded bytes.Buffer
	if err := run([]string{"-repo", root, "-alias", "dataset/current", "-json"}, &encoded); err != nil {
		t.Fatal(err)
	}
	var result overgodb.QueryResult
	if err := json.Unmarshal(encoded.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Artifacts) != 1 || result.Artifacts[0].ID != id {
		t.Fatalf("JSON result = %+v", result)
	}
}

func TestRunRejectsUnboundedAndInvalidFilters(t *testing.T) {
	if err := run(nil, &bytes.Buffer{}); err == nil {
		t.Fatal("missing repository accepted")
	}
	if err := run([]string{"-repo", "x", "-limit", "0"}, &bytes.Buffer{}); err == nil {
		t.Fatal("zero result bound accepted")
	}
	if err := run([]string{"-repo", "x", "-magic-closures", "-limit", "0"}, &bytes.Buffer{}); err == nil {
		t.Fatal("specialized query accepted zero result bound")
	}
	if _, err := parseFollow("sideways"); err == nil {
		t.Fatal("invalid follow direction accepted")
	}
}
