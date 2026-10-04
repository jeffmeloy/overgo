//go:build modeltest

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/testevidence"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

const imageVideoProcessingSHA256 = "d0bd66db0cf42b56093f12d55c2423bcdf57bfb0c5e84025d623a5a927a52b02"

type mediaProcessingLimits struct {
	ExtraBytes uint64 `json:"extra_allocated_bytes"`
	ExtraCalls uint64 `json:"extra_allocation_calls"`
	Available  uint64 `json:"available_host_bytes"`
	Harness    string `json:"harness_sha256"`
}

type mediaProcessingComparison struct {
	Version      uint16                                  `json:"version"`
	Decision     string                                  `json:"decision"`
	Rationale    string                                  `json:"rationale"`
	Protocol     artifact.ID                             `json:"protocol"`
	Harness      artifact.ID                             `json:"harness"`
	Sources      map[string]artifact.ID                  `json:"sources"`
	Acquisitions map[string]artifact.ID                  `json:"acquisitions"`
	Observations map[string][]mediaProcessingObservation `json:"observations"`
	Quality      []mediaSharedCheck                      `json:"quality"`
}

func compareMediaProcessing(before, after []mediaProcessingObservation, limits mediaProcessingLimits) (bool, error) {
	if len(before) != 2 || len(after) != len(before) || limits.Available == 0 {
		return false, errors.New("missing processing observations or capacity")
	}
	accepted := true
	for i, state := range []string{"first", "repeat"} {
		b, a := before[i], after[i]
		if b.State != state || a.State != state || b.Wall == 0 || a.Wall == 0 || b.Allocated == 0 || a.Allocated == 0 || b.Calls == 0 || a.Calls == 0 || b.Peak == 0 || a.Peak == 0 || b.Head == (artifact.CommitID{}) || b.Sequence == 0 || b.Report == "" || len(b.Samples) == 0 {
			return false, errors.New("incomplete processing observation")
		}
		if b.Head != a.Head || b.Sequence != a.Sequence || b.Report != a.Report || !maps.Equal(b.Samples, a.Samples) {
			return false, errors.New("processing changed report, media or store identity")
		}
		if b.Head != before[0].Head || b.Report != before[0].Report || !maps.Equal(b.Samples, before[0].Samples) {
			return false, errors.New("processing changed inputs or outputs between lifetimes")
		}
		accepted = accepted && a.Wall < b.Wall && a.Allocated <= b.Allocated+limits.ExtraBytes && a.Calls <= b.Calls+limits.ExtraCalls && a.Peak <= limits.Available && b.Peak <= limits.Available
	}
	return accepted, nil
}

func mediaProcessingRecordedRows(raw []byte) ([]mediaProcessingObservation, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	var output strings.Builder
	for {
		var event struct{ Test, Action, Output string }
		err := decoder.Decode(&event)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, err
		}
		if event.Test == "TestAcquireMediaProcessing" && event.Action == "output" {
			output.WriteString(event.Output)
		}
	}
	var rows []mediaProcessingObservation
	for line := range strings.SplitSeq(output.String(), "\n") {
		_, data, found := strings.Cut(line, "MEDIA_PROCESSING ")
		if !found {
			continue
		}
		var row mediaProcessingObservation
		if err := json.Unmarshal([]byte(data), &row); err != nil {
			return nil, err
		}
		rows = append(rows, row)
	}
	return rows, nil
}

func TestImageVideoProcessingOptimizationAcceptance(t *testing.T) {
	t.Parallel()
	testskip.Short(t, "retained media processing comparison")
	root := testutil.RepoRoot(t)
	path := filepath.Join(root, "docs/image_video_processing.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkMediaProtocolIdentity(raw, imageVideoProcessingSHA256); err != nil {
		t.Fatal(err)
	}
	var value mediaProcessingComparison
	if err := jsonfile.DecodeStrict(path, &value); err != nil {
		t.Fatal(err)
	}
	if value.Version != artifact.InitialDocumentVersion || value.Rationale == "" || len(value.Quality) == 0 || len(value.Sources) == 0 {
		t.Fatal("incomplete processing comparison")
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
		content, found, err := artifact.ReadContent(t.Context(), store, id)
		if err != nil || !found || id.Kind() != artifact.KindEvidence {
			t.Fatalf("missing processing evidence %s: %v", id, err)
		}
		return content.Data
	}
	protocol := read(value.Protocol)
	var limits mediaProcessingLimits
	if err := json.Unmarshal(protocol, &limits); err != nil {
		t.Fatal(err)
	}
	current, err := os.ReadFile(filepath.Join(root, "docs/image_video_processing_protocol.json"))
	if err != nil {
		t.Fatal(err)
	}
	var frozen any
	if err := json.Unmarshal(protocol, &frozen); err != nil {
		t.Fatal(err)
	}
	canonical, err := json.Marshal(frozen)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkMediaProtocolIdentity(current, fmt.Sprintf("%x", sha256.Sum256(canonical))); err != nil {
		t.Fatalf("processing protocol changed after acquisition: %v", err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(read(value.Harness))) != limits.Harness {
		t.Fatal("processing harness changed")
	}
	for _, id := range value.Sources {
		read(id)
	}
	for _, check := range value.Quality {
		report, err := testevidence.GoTestJSONReport(string(read(check.Evidence)))
		if err != nil {
			t.Fatal(err)
		}
		if err := testevidence.RequireComplete(report); err != nil {
			t.Fatal(err)
		}
		if report.PassedTests == 0 || report.PassedPackages == 0 || check.Name == "" || check.Command == "" {
			t.Fatal("empty processing quality check")
		}
	}
	for _, variant := range []string{"baseline", "candidate", "ablation"} {
		report, err := testevidence.GoTestJSONReport(string(read(value.Acquisitions[variant])))
		if err != nil {
			t.Fatal(err)
		}
		if err := testevidence.RequireComplete(report); err != nil {
			t.Fatal(err)
		}
		if report.PassedTests == 0 {
			t.Fatal("empty processing acquisition")
		}
		rows := value.Observations[variant]
		if len(rows) != 2 {
			t.Fatal("missing processing lifetime")
		}
		recorded, err := mediaProcessingRecordedRows(read(value.Acquisitions[variant]))
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(recorded, rows) {
			t.Fatal("processing observations differ from retained acquisition")
		}
		for _, row := range rows {
			if row.Variant != variant {
				t.Fatal("mislabelled processing variant")
			}
		}
	}
	baseline := value.Observations["baseline"]
	candidate := value.Observations["candidate"]
	accepted, err := compareMediaProcessing(baseline, candidate, limits)
	if err != nil {
		t.Fatal(err)
	}
	if value.Decision != "accepted" && value.Decision != "rejected" {
		t.Fatal("missing processing decision")
	}
	if accepted != (value.Decision == "accepted") {
		t.Fatal("processing decision contradicts frozen comparison")
	}
	t.Run("current-owners-keep-the-memo", TestMediaReportOwnersKeepTheMemo)
	t.Run("current-report-contract", TestImageVideoReportContract)
	t.Run("current-report-truth", TestMediaReportRendersStoreTruth)
	if _, err := compareMediaProcessing(baseline, value.Observations["ablation"], limits); err != nil {
		t.Fatal(err)
	}
	bad := slices.Clone(candidate)
	bad[0].Report = "changed"
	if _, err := compareMediaProcessing(baseline, bad, limits); err == nil {
		t.Fatal("accepted changed report bytes")
	}
	bad = slices.Clone(candidate)
	bad[0].Wall = baseline[0].Wall
	if accepted, err := compareMediaProcessing(baseline, bad, limits); err != nil || accepted {
		t.Fatal("accepted lost latency benefit")
	}
	t.Logf("original processing candidate %s; original report and sample identities retained for both lifetimes; memo optimisation still in force in both owners and current report behavior pass; original costs remain historical", value.Decision)
}
