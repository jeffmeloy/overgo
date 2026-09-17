package main

import (
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
	"overgo/internal/testevidence"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

const imageVideoSharedComponentsSHA256 = "dfbcd24816c86297a5cd2afb7a8a112e669b24365a176d2284c3d643a0b2472c"

type mediaSharedComponents struct {
	Version        uint16                   `json:"version"`
	Baseline       string                   `json:"baseline_sha256"`
	Source         string                   `json:"source_base"`
	Sources        map[string]string        `json:"source_sha256"`
	SourceEvidence map[string]artifact.ID   `json:"source_evidence"`
	Model          artifact.ID              `json:"model"`
	Recipe         artifact.ID              `json:"recipe"`
	Harness        artifact.ID              `json:"harness"`
	Acceptance     artifact.ID              `json:"acceptance"`
	Environment    artifact.ID              `json:"environment"`
	Quality        []mediaSharedCheck       `json:"quality"`
	Acquisitions   []mediaLoaderAcquisition `json:"acquisitions"`
	Prior          []artifact.ID            `json:"prior_attempts"`
	Decision       string                   `json:"decision"`
	Changes        string                   `json:"changes"`
	Limitations    string                   `json:"limitations"`
}

type mediaSharedCheck struct {
	Name     string      `json:"name"`
	Evidence artifact.ID `json:"evidence"`
	Command  string      `json:"command"`
}

func compareMediaLoaderObservations(consumer string, before, after []mediaLoaderObservation) error {
	if err := checkMediaLoaderObservations(consumer, before); err != nil {
		return err
	}
	if err := checkMediaLoaderObservations(consumer, after); err != nil {
		return err
	}
	for index, b := range before {
		a := after[index]
		if a.Values != b.Values || a.Elements != b.Elements || a.Tensors != b.Tensors || a.Source != b.Source {
			return errors.New("shared decoder changed loaded values or extents")
		}
		if a.Allocated >= b.Allocated || a.Peak > b.Peak || a.Wall > b.Wall {
			return errors.New("shared decoder failed frozen allocation, peak or wall comparison")
		}
	}
	return nil
}

func TestImageVideoSharedComponentsAcceptance(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration + ": shared decoder reads retained acquisitions")
	}
	root := testutil.RepoRoot(t)
	t.Run("frozen-baseline", TestImageVideoOptimizationBaselineAcceptance)
	var baseline mediaLoaderBaseline
	if err := jsonfile.DecodeStrict(filepath.Join(root, "docs/image_video_baseline.json"), &baseline); err != nil {
		t.Fatal(err)
	}
	candidateData, err := os.ReadFile(filepath.Join(root, "docs/image_video_shared_components.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := checkMediaProtocolIdentity(candidateData, imageVideoSharedComponentsSHA256); err != nil {
		t.Fatal(err)
	}
	var candidate mediaSharedComponents
	if err := jsonfile.DecodeStrict(filepath.Join(root, "docs/image_video_shared_components.json"), &candidate); err != nil {
		t.Fatal(err)
	}
	if candidate.Version != artifact.InitialDocumentVersion || candidate.Baseline != imageVideoBaselineSHA256 || !gitauthority.ValidObjectID(candidate.Source) || candidate.Model != baseline.Model || candidate.Recipe != baseline.Recipe || candidate.Harness != baseline.Harness || candidate.Acceptance != baseline.Acceptance || candidate.Decision != "adopt" || candidate.Changes == "" || candidate.Limitations == "" || len(candidate.Prior) == 0 {
		t.Fatal("candidate changed frozen inputs or omitted its decision")
	}
	if len(candidate.Acquisitions) != len(baseline.Acquisitions) || len(candidate.Sources) != len(baseline.Sources) || len(candidate.SourceEvidence) != len(baseline.Sources) {
		t.Fatal("candidate coverage differs")
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
			t.Fatal("non-evidence comparison input")
		}
		content, found, err := artifact.ReadContent(t.Context(), store, id)
		if err != nil || !found || len(content.Data) == 0 {
			t.Fatalf("missing comparison input %s: %v", id, err)
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
			t.Fatal("empty candidate acquisition")
		}
	}
	read(candidate.Environment)
	for _, id := range candidate.Prior {
		read(id)
	}
	for path := range baseline.Sources {
		if candidate.Sources[path] == "" || fmt.Sprintf("%x", sha256.Sum256(read(candidate.SourceEvidence[path]))) != candidate.Sources[path] {
			t.Fatalf("candidate source snapshot differs: %s", path)
		}
	}
	names := []string{"shared-decoder-contracts", "wan-references"}
	if len(candidate.Quality) != len(names) {
		t.Fatal("candidate quality coverage differs")
	}
	for index, check := range candidate.Quality {
		if check.Name != names[index] || check.Command == "" {
			t.Fatal("candidate omitted owning component or real consumer verification")
		}
		requirePass(check.Evidence)
	}
	for index, b := range baseline.Acquisitions {
		a := candidate.Acquisitions[index]
		if a.Consumer != b.Consumer || a.Command == "" {
			t.Fatal("candidate omitted or duplicated consumer")
		}
		requirePass(a.Evidence)
		var before, after []mediaLoaderObservation
		if err := json.Unmarshal(read(b.Observations), &before); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(read(a.Observations), &after); err != nil {
			t.Fatal(err)
		}
		if err := compareMediaLoaderObservations(b.Consumer, before, after); err != nil {
			t.Fatal(err)
		}
		for _, mutate := range []func([]mediaLoaderObservation){
			func(rows []mediaLoaderObservation) { rows[0].Values = "changed" },
			func(rows []mediaLoaderObservation) { rows[0].Allocated = before[0].Allocated },
			func(rows []mediaLoaderObservation) { rows[0].Peak = before[0].Peak + 1 },
			func(rows []mediaLoaderObservation) { rows[0].Wall = before[0].Wall + 1 },
		} {
			bad := slices.Clone(after)
			mutate(bad)
			if compareMediaLoaderObservations(b.Consumer, before, bad) == nil {
				t.Fatal("accepted changed values or conflicting efficiency")
			}
		}
		if compareMediaLoaderObservations(b.Consumer, before, after[:len(after)-1]) == nil {
			t.Fatal("accepted omitted load state")
		}
	}
	t.Log("both real loaders preserve exact values and meet the frozen allocation, process peak and complete-loader wall comparisons; retained owning-component and native references pass")
}
