package main

import (
	"cmp"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"overgo/internal/artifact"
	"overgo/internal/cuda/driver"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

const imageVideoLatentImagesSHA256 = "8d600c3ab06f7531bef85d5d45a077e079efb8e31f550b7081f6db3268a4a77f"

type mediaLatentBundleCase struct {
	Case        string      `json:"case"`
	Run         artifact.ID `json:"run"`
	Acquisition artifact.ID `json:"acquisition"`
	Review      artifact.ID `json:"review"`
	Test        artifact.ID `json:"test_evidence"`
}
type mediaLatentImageBundle struct {
	Version         uint16                  `json:"version"`
	Source          string                  `json:"source_base"`
	Protocol        artifact.ID             `json:"protocol"`
	Cases           []mediaLatentBundleCase `json:"cases"`
	Numerical       []mediaSharedCheck      `json:"numerical"`
	Instrumentation map[string]artifact.ID  `json:"instrumentation"`
	Failed          []artifact.ID           `json:"failed_attempts"`
	Scope           string                  `json:"scope"`
}

type mediaLatentObservation struct {
	Case        string                 `json:"case"`
	Source      string                 `json:"source_base"`
	Model       artifact.ID            `json:"model"`
	Recipe      artifact.ID            `json:"recipe"`
	Input       artifact.ID            `json:"input"`
	Run         artifact.ID            `json:"validation_run"`
	Output      artifact.ID            `json:"output"`
	Environment artifact.ID            `json:"environment"`
	Width       int                    `json:"width"`
	Height      int                    `json:"height"`
	Minimum     float64                `json:"minimum"`
	Maximum     float64                `json:"maximum"`
	Finite      bool                   `json:"prequantization_finite"`
	Wall        uint64                 `json:"request_ns"`
	Allocated   uint64                 `json:"allocated_bytes"`
	Peak        uint64                 `json:"peak_host_bytes"`
	Before      *driver.ExecutionStats `json:"before_counters"`
	Load        *driver.ExecutionStats `json:"load_counters"`
	After       driver.ExecutionStats  `json:"after_counters"`
	Memory      driver.MemoryStats     `json:"after_memory"`
	Closed      *driver.MemoryStats    `json:"closed_memory"`
}

func mediaLatentRows(raw []byte) ([]mediaLatentObservation, *driver.MemoryStats, error) {
	var group struct {
		Rows   []mediaLatentObservation `json:"observations"`
		Closed *driver.MemoryStats      `json:"closed_memory"`
	}
	if err := json.Unmarshal(raw, &group); err != nil {
		return nil, nil, err
	}
	if len(group.Rows) == 0 {
		var row mediaLatentObservation
		if err := json.Unmarshal(raw, &row); err != nil {
			return nil, nil, err
		}
		group.Rows = append(group.Rows, row)
	}
	return group.Rows, group.Closed, nil
}

func checkMediaLatentObservation(row mediaLatentObservation, run runrecord.Run, source string, closed *driver.MemoryStats) error {
	if row.Case == "" || row.Source != source || run.CodeCommit != source || row.Run != run.ID || row.Recipe != run.Recipe || row.Environment != run.Environment || !slices.Equal(run.Inputs, []artifact.ID{row.Input}) || !slices.Equal(run.Outputs, []artifact.ID{row.Output}) || run.Outcome != runrecord.OutcomeSucceeded {
		return errors.New("image acquisition and source-bound run differ")
	}
	if !row.Finite || math.IsNaN(row.Minimum) || math.IsInf(row.Minimum, 0) || math.IsNaN(row.Maximum) || math.IsInf(row.Maximum, 0) || row.Minimum > row.Maximum || row.Wall == 0 || row.Wall != run.MeasuredNS || row.Allocated == 0 || row.Peak == 0 || row.Memory.PeakBytes == 0 || row.Width <= 0 || row.Height <= 0 || len(run.Phases) == 0 {
		return errors.New("image acquisition lacks finite output, phases or resource measurements")
	}
	before := cmp.Or(row.Before, row.Load)
	if before == nil || row.After.KernelLaunches <= before.KernelLaunches || row.After.DeviceToHostBytes <= before.DeviceToHostBytes || row.After.HostToDeviceBytes < before.HostToDeviceBytes {
		return errors.New("image acquisition lacks executed GPU work and output transfer")
	}
	if row.Closed != nil {
		closed = row.Closed
	}
	if closed == nil || closed.CurrentBytes != 0 {
		return errors.New("image acquisition lacks completed device cleanup")
	}
	return nil
}

func checkMediaLatentCoverage(cases []mediaLatentBundleCase, expected []string) error {
	if len(cases) != len(expected) {
		return errors.New("image bundle omits a required case")
	}
	seen := map[string]bool{}
	for _, c := range cases {
		if !slices.Contains(expected, c.Case) || seen[c.Case] {
			return errors.New("image bundle changes or duplicates coverage")
		}
		seen[c.Case] = true
	}
	return nil
}

func TestImageVideoLatentImageAcceptance(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration + ": retained image case bundle")
	}
	root := testutil.RepoRoot(t)
	document, err := plan.Load(filepath.Join(root, plan.Path))
	if err != nil {
		t.Fatal(err)
	}
	if document.Lane != "image_video_gen" || os.Getenv(dataroot.Env) == "" {
		t.Skip("integration: explicit media data root required")
	}
	var bundle mediaLatentImageBundle
	if err := jsonfile.DecodeStrict(filepath.Join(root, "docs/image_video_latent_images.json"), &bundle); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(root, "docs/image_video_latent_images.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := checkMediaProtocolIdentity(raw, imageVideoLatentImagesSHA256); err != nil {
		t.Fatal(err)
	}
	if bundle.Version != artifact.InitialDocumentVersion || bundle.Source == "" || len(bundle.Numerical) != 5 || bundle.Scope == "" || len(bundle.Failed) != 3 {
		t.Fatal("incomplete image bundle")
	}
	var merged mediaMergedEvidence
	if err := jsonfile.DecodeStrict(filepath.Join(root, "docs/image_video_merged.json"), &merged); err != nil {
		t.Fatal(err)
	}
	for _, revision := range []string{bundle.Source, "HEAD"} {
		if err := checkMediaRuntimeAtRevision(root, revision, merged.RuntimePaths, merged.RuntimeSHA256); err != nil {
			t.Fatal(err)
		}
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
		content, found, err := artifact.ReadContent(t.Context(), store, id)
		if err != nil || !found || len(content.Data) == 0 {
			t.Fatal("missing retained image evidence", id, err)
		}
		return content.Data
	}
	checkTest := func(id artifact.ID) {
		t.Helper()
		report, err := testevidence.GoTestJSONReport(string(read(id)))
		if err != nil {
			t.Fatal(err)
		}
		if err := testevidence.RequireComplete(report); err != nil {
			t.Fatal(err)
		}
		if report.PassedTests == 0 || report.PassedPackages == 0 {
			t.Fatal("empty image check")
		}
	}
	primary := read(bundle.Protocol)
	var header struct {
		Source     string `json:"source_base"`
		Quality    string `json:"quality_protocol_sha256"`
		Harness    string `json:"krea_harness_sha256"`
		Acceptance string `json:"initial_acceptance_sha256"`
	}
	if err := json.Unmarshal(primary, &header); err != nil {
		t.Fatal(err)
	}
	if header.Source != bundle.Source || header.Quality != imageVideoProtocolSHA256 {
		t.Fatal("image acquisition protocol differs from frozen source or quality criteria")
	}
	for path, digest := range map[string]string{"media_krea_acquisition_test.go": header.Harness, "media_latent_image_acceptance_initial_test.go": header.Acceptance} {
		if fmt.Sprintf("%x", sha256.Sum256(read(bundle.Instrumentation[path]))) != digest {
			t.Fatal("changed acquisition instrumentation", path)
		}
	}
	var instrumentation struct {
		Source   string `json:"source_base"`
		Protocol string `json:"protocol_sha256"`
		Harness  string `json:"harness_sha256"`
	}
	if err := json.Unmarshal(read(bundle.Instrumentation["media-diffusion-acquisition-instrumentation-third.json"]), &instrumentation); err != nil {
		t.Fatal(err)
	}
	if instrumentation.Source != bundle.Source || instrumentation.Protocol != fmt.Sprintf("%x", sha256.Sum256(primary)) || instrumentation.Harness != fmt.Sprintf("%x", sha256.Sum256(read(bundle.Instrumentation["media_diffusion_acquisition_test.go"]))) {
		t.Fatal("changed diffusion instrumentation")
	}
	for _, id := range bundle.Failed {
		report, err := testevidence.GoTestJSONReport(string(read(id)))
		if err != nil {
			t.Fatal(err)
		}
		if len(report.Failed) == 0 {
			t.Fatal("lost failed acquisition evidence")
		}
	}
	quality, err := os.ReadFile(filepath.Join(root, "docs/image_video_protocol.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := checkImageVideoProtocolIdentity(quality); err != nil {
		t.Fatal(err)
	}
	var protocol struct {
		Cases []struct {
			ID     string        `json:"id"`
			Model  artifact.ID   `json:"model"`
			Recipe artifact.ID   `json:"recipe"`
			Inputs []artifact.ID `json:"inputs"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(quality, &protocol); err != nil {
		t.Fatal(err)
	}
	var expected []string
	for _, c := range protocol.Cases {
		if strings.HasPrefix(c.ID, "Krea-") || strings.HasPrefix(c.ID, "SimpleDiffusion-") {
			expected = append(expected, c.ID)
		}
	}
	if err := checkMediaLatentCoverage(bundle.Cases, expected); err != nil {
		t.Fatal(err)
	}
	if err := checkMediaLatentCoverage(bundle.Cases[1:], expected); err == nil {
		t.Fatal("accepted omitted image case")
	}
	for _, check := range bundle.Numerical {
		if check.Name == "" || check.Command == "" {
			t.Fatal("unnamed numerical scope")
		}
		checkTest(check.Evidence)
	}
	for _, c := range bundle.Cases {
		checkTest(c.Test)
		run, err := runrecord.RequireExactRun(t.Context(), store, c.Run)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := runrecord.RequireEnvironment(t.Context(), store, run.Environment); err != nil {
			t.Fatal(err)
		}
		acquisition := read(c.Acquisition)
		var identity struct {
			Protocol string `json:"protocol_sha256"`
		}
		if err := json.Unmarshal(acquisition, &identity); err != nil {
			t.Fatal(err)
		}
		if identity.Protocol != fmt.Sprintf("%x", sha256.Sum256(primary)) {
			t.Fatal("image acquisition used a different protocol")
		}
		rows, closed, err := mediaLatentRows(acquisition)
		if err != nil {
			t.Fatal(err)
		}
		index := slices.IndexFunc(rows, func(row mediaLatentObservation) bool { return row.Case == c.Case })
		if index < 0 {
			t.Fatal("case missing from raw acquisition")
		}
		row := rows[index]
		if err := checkMediaLatentObservation(row, run, bundle.Source, closed); err != nil {
			t.Fatal(err)
		}
		found := false
		for _, frozen := range protocol.Cases {
			if frozen.ID == c.Case {
				found = true
				if row.Model != frozen.Model || row.Recipe != frozen.Recipe || !slices.Equal(run.Inputs, frozen.Inputs) {
					t.Fatal("case differs from frozen request and recipe")
				}
			}
		}
		if !found {
			t.Fatal("unexpected image case")
		}
		output, found, err := artifact.ReadContent(t.Context(), store, row.Output)
		if err != nil || !found {
			t.Fatal("missing generated image", err)
		}
		if err := checkImageVideoObservation(imageVideoObservation{Artifact: row.Output, Width: row.Width, Height: row.Height, Frames: 1}, output); err != nil {
			t.Fatal(err)
		}
		var review struct {
			Output       artifact.ID `json:"output"`
			Subject      string      `json:"subject"`
			Preservation string      `json:"source_preservation"`
			Defects      string      `json:"visible_defects"`
			Video        string      `json:"video"`
		}
		if err := json.Unmarshal(read(c.Review), &review); err != nil {
			t.Fatal(err)
		}
		if review.Output != row.Output || review.Subject == "" || review.Preservation == "" || review.Defects == "" || review.Video == "" {
			t.Fatal("image review omits a question or refers to other bytes")
		}
		bad := row
		bad.Finite = false
		if err := checkMediaLatentObservation(bad, run, bundle.Source, closed); err == nil {
			t.Fatal("accepted failed finiteness")
		}
		bad = row
		bad.Output = artifact.ID{}
		if err := checkMediaLatentObservation(bad, run, bundle.Source, closed); err == nil {
			t.Fatal("accepted changed output identity")
		}
		bad = row
		bad.Recipe = row.Model
		if err := checkMediaLatentObservation(bad, run, bundle.Source, closed); err == nil {
			t.Fatal("accepted mismatched recipe")
		}
	}
}
