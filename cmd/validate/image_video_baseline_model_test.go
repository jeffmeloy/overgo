//go:build modeltest

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
	"overgo/internal/dataroot"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
	"overgo/internal/testevidence"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// The completed acquisition is frozen before the decoder migration.
const imageVideoBaselineSHA256 = "d652708aa03884476c4cdc8a7e37d56780394bc001db90eeb5150397a5a0e124"

// imageVideoAcquiredResourceProtocolSHA256 is the identity of the resource protocol the loader baseline and the cost profile were acquired under.
//   - a record of an acquisition names the protocol it was acquired under, and a later correction of the protocol's text never moves it
const imageVideoAcquiredResourceProtocolSHA256 = "2e7b7bfb8e8d308a498ea5b54572b0dedf3dccfa9ec12731f31758775cc7cc02"

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

func checkMediaLoaderBaseline(b mediaLoaderBaseline, inventory imageVideoInventory) error {
	consumers := []string{"denoiser", "projection"}
	if b.Version != artifact.InitialDocumentVersion || !gitauthority.ValidObjectID(b.Source) || b.Quality != imageVideoProtocolSHA256 || b.Resources != imageVideoAcquiredResourceProtocolSHA256 || b.Scope == "" || b.Limitations == "" || len(b.Acquisitions) != len(consumers) {
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

func TestImageVideoOptimizationBaselineAcceptance(t *testing.T) {
	t.Parallel()
	testskip.Short(t, "loader baseline reads retained acquisitions")
	root := testutil.RepoRoot(t)
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
	store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
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
