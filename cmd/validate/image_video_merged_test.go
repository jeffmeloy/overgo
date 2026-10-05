package main

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/testevidence"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

const imageVideoMergedSHA256 = "f6073ad6651b6c1cb705cf249ac9b0b69e8e8f110a60fd1922ba27eab85cb373"

type mediaMergedEvidence struct {
	Version       uint16             `json:"version"`
	Local         string             `json:"local_parent"`
	Incoming      string             `json:"incoming_parent"`
	Merged        string             `json:"merged_commit"`
	Scope         string             `json:"scope"`
	RuntimePaths  []string           `json:"runtime_paths"`
	RuntimeSHA256 string             `json:"runtime_sha256"`
	Inventory     artifact.ID        `json:"inventory_census"`
	Checks        []mediaSharedCheck `json:"checks"`
}

func TestImageVideoMergedCapabilitiesAcceptance(t *testing.T) {
	t.Parallel()
	testskip.Short(t, "merged media evidence requires the private store")
	root := testutil.RepoRoot(t)
	path := filepath.Join(root, "docs/image_video_merged.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkMediaProtocolIdentity(raw, imageVideoMergedSHA256); err != nil {
		t.Fatal(err)
	}
	var value mediaMergedEvidence
	if err := jsonfile.DecodeStrict(path, &value); err != nil {
		t.Fatal(err)
	}
	if value.Version != artifact.InitialDocumentVersion || !gitauthority.ValidObjectID(value.Local) || !gitauthority.ValidObjectID(value.Incoming) || !gitauthority.ValidObjectID(value.Merged) || value.Scope == "" || len(value.Checks) == 0 {
		t.Fatal("incomplete merged media evidence")
	}
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	for _, check := range value.Checks {
		content, found, err := artifact.ReadContent(t.Context(), store, check.Evidence)
		if err != nil || !found {
			t.Fatalf("missing %s: %v", check.Name, err)
		}
		report, err := testevidence.GoTestJSONReport(string(content.Data))
		if err != nil {
			t.Fatal(err)
		}
		if err := testevidence.RequireComplete(report); err != nil {
			t.Fatal(err)
		}
		if check.Name == "" || check.Command == "" || report.PassedPackages == 0 || report.PassedTests == 0 {
			t.Fatal("empty merged-source check")
		}
	}
	inventory, err := loadFrozenMediaInventory(root)
	if err != nil {
		t.Fatal(err)
	}
	if inventory.Census != value.Inventory {
		t.Fatal("changed frozen media denominator")
	}
	census, err := capabilityCensusCodec.Require(t.Context(), store, inventory.Census)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkImageVideoCoverage(inventory, census, inventory.Tasks); err != nil {
		t.Fatal(err)
	}
	if err := checkImageVideoDefinitions(t.Context(), store, inventory); err != nil {
		t.Fatal(err)
	}
	scope, err := resolveMediaReportScope("image-video")
	if err != nil {
		t.Fatal(err)
	}
	entries, truncated, err := discovery.CapabilityCatalogForTasks(t.Context(), store, catalogLimit, discovery.LoadMemo(t.Context(), store), scope.Tasks...)
	if err != nil || truncated {
		t.Fatalf("current media catalog incomplete: %v", err)
	}
	if _, err := loadMediaSampleSelection(t.Context(), root, store, scope, entries); err != nil {
		t.Fatal(err)
	}
	cells := 0
	for _, entry := range entries {
		cells += len(entry.Capabilities)
	}
	if cells != len(inventory.Cells) {
		t.Fatal("current active media coverage changed")
	}
	bad := inventory
	bad.Cells = slices.Clone(inventory.Cells[1:])
	if err := checkImageVideoCoverage(bad, census, inventory.Tasks); err == nil {
		t.Fatal("accepted omitted media cell")
	}
	t.Logf("master %s merged at %s; %d active media cells retain their recipe/input bindings and unchanged generation source scope", value.Incoming, value.Merged, cells)
}
