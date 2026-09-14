package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/binaryschema"
	"overgo/internal/dataroot"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/processcontrol"
	"overgo/internal/testevidence"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// The completed acquisition is frozen before the decoder migration.
const imageVideoBaselineSHA256 = "d652708aa03884476c4cdc8a7e37d56780394bc001db90eeb5150397a5a0e124"

type mediaLoaderBaseline struct {
	Version         uint16                   `json:"version"`
	Source          string                   `json:"source"`
	Model           artifact.ID              `json:"model"`
	Recipe          artifact.ID              `json:"recipe"`
	Quality         string                   `json:"quality_protocol_sha256"`
	Resources       string                   `json:"resource_protocol_sha256"`
	Sources         map[string]string        `json:"source_sha256"`
	Harness         artifact.ID              `json:"harness"`
	Acceptance      artifact.ID              `json:"acceptance"`
	Environment     artifact.ID              `json:"environment"`
	QualityEvidence artifact.ID              `json:"quality_evidence"`
	Prior           []artifact.ID            `json:"prior_attempts"`
	Scope           string                   `json:"scope"`
	Limitations     string                   `json:"limitations"`
	Acquisitions    []mediaLoaderAcquisition `json:"acquisitions"`
}

type mediaLoaderAcquisition struct {
	Consumer     string      `json:"consumer"`
	Observations artifact.ID `json:"observations"`
	Evidence     artifact.ID `json:"evidence"`
	Command      string      `json:"command"`
}

type mediaLoaderObservation struct {
	Consumer  string `json:"consumer"`
	State     string `json:"load_state"`
	Wall      int64  `json:"wall_ns"`
	Allocated uint64 `json:"total_alloc_bytes"`
	Calls     uint64 `json:"allocation_calls"`
	Peak      uint64 `json:"process_lifetime_peak_host_bytes"`
	Live      uint64 `json:"live_heap_bytes"`
	Released  uint64 `json:"heap_bytes_after_release_gc"`
	Source    uint64 `json:"source_bytes"`
	Elements  uint64 `json:"elements"`
	Tensors   uint64 `json:"tensors"`
	Values    string `json:"values_sha256"`
	Upload    uint64 `json:"host_to_device_bytes"`
	Download  uint64 `json:"device_to_host_bytes"`
	Device    uint64 `json:"device_allocated_bytes"`
	Scope     string `json:"scope"`
}

func checkMediaLoaderBaseline(b mediaLoaderBaseline, inventory imageVideoInventory) error {
	consumers := []string{"denoiser", "projection"}
	if b.Version != artifact.InitialDocumentVersion || !gitauthority.ValidObjectID(b.Source) || b.Quality != imageVideoProtocolSHA256 || b.Resources != imageVideoResourceProtocolSHA256 || b.Scope == "" || b.Limitations == "" || len(b.Acquisitions) != len(consumers) {
		return errors.New("loader baseline identity, scope or coverage differs")
	}
	if !slices.ContainsFunc(inventory.Cells, func(c imageVideoInventoryCell) bool {
		return c.Name == "Wan2.1-T2V-1.3B" && c.Model == b.Model && c.Recipe == b.Recipe
	}) {
		return errors.New("loader baseline model/recipe differs")
	}
	seen := map[string]bool{}
	for _, a := range b.Acquisitions {
		if !slices.Contains(consumers, a.Consumer) || seen[a.Consumer] || a.Command == "" || a.Observations.Kind() != artifact.KindEvidence || a.Evidence.Kind() != artifact.KindEvidence {
			return errors.New("loader baseline omits or duplicates a real consumer")
		}
		seen[a.Consumer] = true
	}
	for _, path := range []string{"internal/latentvideo/denoiser.go", "internal/latentvideo/textcond.go", "internal/safetensors/convert.go"} {
		if b.Sources[path] == "" {
			return errors.New("loader baseline source owner is absent")
		}
	}
	if len(b.Prior) == 0 {
		return errors.New("loader baseline omitted prior attempts")
	}
	return nil
}

func checkMediaLoaderObservations(consumer string, rows []mediaLoaderObservation) error {
	states := []string{"first-load", "repeat-load"}
	if len(rows) != len(states) {
		return errors.New("loader baseline omits a load state")
	}
	for index, row := range rows {
		if row.Consumer != consumer || row.State != states[index] || row.Wall <= 0 || row.Allocated == 0 || row.Calls == 0 || row.Peak == 0 || row.Live == 0 || row.Released == 0 || row.Released >= row.Live || row.Elements == 0 || row.Tensors == 0 || row.Source != row.Elements*binaryschema.Uint32Bytes || len(row.Values) != sha256.Size*2 || row.Scope == "" {
			return errors.New("loader baseline lacks observed cost, output or release data")
		}
		if row.Upload != 0 || row.Download != 0 || row.Device != 0 {
			return errors.New("CPU-only loader observation contains device work")
		}
		first := rows[0]
		if row.Values != first.Values || row.Source != first.Source || row.Elements != first.Elements || row.Tensors != first.Tensors || row.Peak < first.Peak {
			return errors.New("repeated loader output or cumulative peak differs")
		}
	}
	return nil
}

func TestImageVideoOptimizationBaselineAcceptance(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration + ": loader baseline reads retained acquisitions")
	}
	root := testutil.RepoRoot(t)
	document, err := plan.Load(filepath.Join(root, plan.Path))
	if err != nil {
		t.Fatal(err)
	}
	if document.Lane != "image_video_gen" || os.Getenv(dataroot.Env) == "" {
		t.Skip("integration: media baseline requires its explicit data root")
	}
	path := filepath.Join(root, "docs/image_video_baseline.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkMediaProtocolIdentity(data, imageVideoBaselineSHA256); err != nil {
		t.Fatal(err)
	}
	var baseline mediaLoaderBaseline
	if err := jsonfile.DecodeStrict(path, &baseline); err != nil {
		t.Fatal(err)
	}
	var inventory imageVideoInventory
	if err := jsonfile.DecodeStrict(filepath.Join(root, "docs/image_video_inventory.json"), &inventory); err != nil {
		t.Fatal(err)
	}
	if err := checkMediaLoaderBaseline(baseline, inventory); err != nil {
		t.Fatal(err)
	}
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	read := func(id artifact.ID) []byte {
		t.Helper()
		if id.Kind() != artifact.KindEvidence {
			t.Fatal("non-evidence baseline input")
		}
		content, found, err := artifact.ReadContent(t.Context(), store, id)
		if err != nil || !found {
			t.Fatalf("missing baseline input %s: %v", id, err)
		}
		return content.Data
	}
	requirePass := func(id artifact.ID) {
		t.Helper()
		report, err := testevidence.GoTestJSONReport(string(read(id)))
		if err != nil {
			t.Fatal(err)
		}
		if err := testevidence.RequireComplete(report); err != nil {
			t.Fatal(err)
		}
		if report.PassedTests == 0 || report.PassedPackages == 0 {
			t.Fatal("empty baseline check")
		}
	}
	for path, digest := range baseline.Sources {
		var output bytes.Buffer
		receipt, err := processcontrol.Run(t.Context(), processcontrol.Command{Path: "git", Args: []string{"show", baseline.Source + ":" + path}, Dir: root, Env: gitauthority.ReaderEnvironment(), Stdout: &output})
		if err != nil || receipt.ExitCode != 0 || fmt.Sprintf("%x", sha256.Sum256(output.Bytes())) != digest {
			t.Fatalf("baseline source differs: %s: %v", path, err)
		}
	}
	for _, id := range append([]artifact.ID{baseline.Harness, baseline.Acceptance, baseline.Environment}, baseline.Prior...) {
		if len(read(id)) == 0 {
			t.Fatal("empty retained baseline input")
		}
	}
	requirePass(baseline.QualityEvidence)
	for _, a := range baseline.Acquisitions {
		requirePass(a.Evidence)
		var rows []mediaLoaderObservation
		if err := json.Unmarshal(read(a.Observations), &rows); err != nil {
			t.Fatal(err)
		}
		if err := checkMediaLoaderObservations(a.Consumer, rows); err != nil {
			t.Fatal(err)
		}
		bad := slices.Clone(rows)
		bad[0].Peak = 0
		if checkMediaLoaderObservations(a.Consumer, bad) == nil {
			t.Fatal("accepted unavailable peak")
		}
		bad = slices.Clone(rows)
		bad[len(bad)-1].Values = "changed"
		if checkMediaLoaderObservations(a.Consumer, bad) == nil {
			t.Fatal("accepted changed repeated values")
		}
	}
	bad := baseline
	bad.Acquisitions = slices.Clone(baseline.Acquisitions)
	bad.Acquisitions[len(bad.Acquisitions)-1] = bad.Acquisitions[0]
	if checkMediaLoaderBaseline(bad, inventory) == nil {
		t.Fatal("accepted duplicate instead of missing consumer")
	}
	bad = baseline
	bad.Quality = "changed"
	if checkMediaLoaderBaseline(bad, inventory) == nil {
		t.Fatal("accepted substituted quality protocol")
	}
	var changed map[string]any
	if err := json.Unmarshal(data, &changed); err != nil {
		t.Fatal(err)
	}
	changed["source"] = "changed"
	changedData, err := json.Marshal(changed)
	if err != nil {
		t.Fatal(err)
	}
	if checkMediaProtocolIdentity(changedData, imageVideoBaselineSHA256) == nil {
		t.Fatal("accepted substituted acquisition identity")
	}
	t.Log("retained both complete loader baselines, exact repeated values, source identities, fixed acceptance and prior attempts; no model loaded")
}
