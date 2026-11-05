package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"image/gif"
	"math"
	"os"
	"overgo/internal/artifact"
	"overgo/internal/cuda/driver"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/latentvideo"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testevidence"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
)

const imageVideoConditionedSHA256 = "35dfb03e6fc433e4c3b04ddbedcc85cfe46e4513d21f4e0e1dc74e81c3086589"

type mediaConditionedCase struct {
	mediaRoutedCase
	RawFrames   []artifact.ID `json:"raw_frames"`
	BaselineRun artifact.ID   `json:"baseline_run,omitzero"`
}
type mediaConditionedBundle struct {
	Version         uint16                 `json:"version"`
	Source          string                 `json:"source_base"`
	Protocol        artifact.ID            `json:"protocol"`
	Cases           []mediaConditionedCase `json:"cases"`
	Numerical       []mediaSharedCheck     `json:"numerical"`
	Instrumentation map[string]artifact.ID `json:"instrumentation"`
	Inputs          artifact.ID            `json:"input_identities"`
	Inspection      []artifact.ID          `json:"inspection"`
	Scope           string                 `json:"scope"`
	Correction      struct {
		Source         string      `json:"source_base"`
		Path           string      `json:"path"`
		Before         string      `json:"before_git_sha256"`
		After          string      `json:"after_git_sha256"`
		Protocol       artifact.ID `json:"protocol"`
		Baseline       artifact.ID `json:"baseline_acquisition"`
		BaselineTest   artifact.ID `json:"baseline_test"`
		NativeFrameSHA string      `json:"native_frame_aggregate_sha256"`
		Scope          string      `json:"scope"`
	} `json:"pixel_range_correction"`
}
type mediaConditionedObservation struct {
	Case           string                           `json:"case"`
	Source         string                           `json:"source_base"`
	Inputs         []artifact.ID                    `json:"inputs"`
	Recipe         artifact.ID                      `json:"recipe"`
	Run            artifact.ID                      `json:"validation_run"`
	Workflow       artifact.ID                      `json:"workflow_run"`
	Output         artifact.ID                      `json:"output"`
	Environment    artifact.ID                      `json:"environment"`
	Wall           uint64                           `json:"request_ns"`
	Allocated      uint64                           `json:"allocated_bytes"`
	Peak           uint64                           `json:"peak_host_bytes"`
	Width          int                              `json:"width"`
	Height         int                              `json:"height"`
	Frames         int                              `json:"frames"`
	FPS            int                              `json:"fps"`
	Delay          int                              `json:"delay_centiseconds"`
	Finite         bool                             `json:"prequantization_finite"`
	Same           bool                             `json:"identical_except_delay"`
	Minimum        float64                          `json:"minimum"`
	Maximum        float64                          `json:"maximum"`
	FrameSHA       []string                         `json:"frame_f32_sha256"`
	NativeFrameSHA string                           `json:"frame_aggregate_sha256"`
	OldMatch       bool                             `json:"old_range_matches_historical_except_delay"`
	CorrectMatch   bool                             `json:"corrected_encoding_matches_raw_frames"`
	Load           uint64                           `json:"load_ns"`
	Close          uint64                           `json:"close_ns"`
	Before         map[string]driver.ExecutionStats `json:"before_counters"`
	After          map[string]driver.ExecutionStats `json:"after_counters"`
	Memory         map[string]driver.MemoryStats    `json:"after_memory"`
	Closed         map[string]driver.MemoryStats    `json:"closed_memory"`
}

func TestImageVideoConditionedVideoAcceptance(t *testing.T) {
	t.Parallel()
	testskip.Short(t, "retained source-conditioned video evidence")
	root := testutil.RepoRoot(t)
	var bundle mediaConditionedBundle
	path := filepath.Join(root, "docs/image_video_conditioned_videos.json")
	testcheck.NoError(t, jsonfile.DecodeStrict(path, &bundle))
	raw, err := os.ReadFile(path)
	testcheck.NoError(t, err)
	testcheck.NoError(t, checkMediaProtocolIdentity(raw, imageVideoConditionedSHA256))
	if bundle.Version != artifact.InitialDocumentVersion || bundle.Source == "" || bundle.Scope == "" || len(bundle.Cases) != 4 || len(bundle.Numerical) != 3 || len(bundle.Inspection) == 0 {
		t.Fatal("incomplete conditioned video bundle")
	}
	roots, err := dataroot.Resolve(root)
	testcheck.NoError(t, err)
	store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
	testcheck.NoError(t, err)
	defer store.Close()
	read := func(id artifact.ID) []byte {
		t.Helper()
		content, found, err := artifact.ReadContent(t.Context(), store, id)
		if err != nil || !found {
			t.Fatal("missing conditioned video evidence", id, err)
		}
		return content.Data
	}
	decode := func(id artifact.ID, value any) {
		t.Helper()
		testcheck.NoError(t, json.Unmarshal(read(id), value))
	}
	sample := func(id artifact.ID) []byte {
		t.Helper()
		content, found, err := artifact.ReadContent(t.Context(), store, id)
		if err != nil || !found {
			t.Fatal("missing media", err)
		}
		name, data, ok := sampleFile(content)
		if !ok || filepath.Ext(name) != ".gif" {
			t.Fatal("not an exportable GIF")
		}
		return data
	}
	checkTest := func(id artifact.ID) {
		t.Helper()
		report, err := testevidence.GoTestJSONReport(string(read(id)))
		testcheck.NoError(t, err)
		testcheck.NoError(t, testevidence.RequireComplete(report))
		if report.PassedTests == 0 || report.PassedPackages == 0 {
			t.Fatal("empty passing evidence")
		}
	}
	var header struct {
		Source  string `json:"source_base"`
		Quality string `json:"quality_protocol_sha256"`
	}
	decode(bundle.Protocol, &header)
	if header.Source != bundle.Source || header.Quality != imageVideoProtocolSHA256 {
		t.Fatal("original quality protocol changed")
	}
	fix := bundle.Correction
	if fix.Source != bundle.Source || fix.Path != "internal/latentvideo/production_cuda_windows.go" || fix.Scope == "" || fix.NativeFrameSHA == "" {
		t.Fatal("missing encoding correction scope")
	}
	// The recorded correction is the lifecycle patch's first state of the file.
	lifecycle, err := readMediaLifecycleBundle(root)
	if err != nil || lifecycle.Changes[fix.Path].Before != fix.After {
		t.Fatal("lifecycle patch does not preserve the recorded pixel-range correction", err)
	}
	checkTest(fix.BaselineTest)
	read(fix.Baseline)
	read(fix.Protocol)
	for _, spec := range []struct{ Name, Harness, Owner string }{{"media-un0-video-instrumentation.json", "media_un0_video_acquisition_test.go", "media_un0_video_audited.go"}, {"media-liveedit-corrected-instrumentation.json", "media_liveedit_corrected_acquisition_test.go", "media_liveedit_corrected_audited_owner.go"}} {
		var instrument struct {
			Source     string `json:"source_base"`
			Protocol   string `json:"protocol_sha256"`
			Harness    string `json:"harness_sha256"`
			Owner      string `json:"audited_owner_sha256"`
			Correction string `json:"correction_protocol_sha256"`
		}
		decode(bundle.Instrumentation[spec.Name], &instrument)
		if instrument.Source != bundle.Source || instrument.Protocol != fmt.Sprintf("%x", sha256.Sum256(read(bundle.Protocol))) || instrument.Harness != fmt.Sprintf("%x", sha256.Sum256(read(bundle.Instrumentation[spec.Harness]))) || instrument.Owner != fmt.Sprintf("%x", sha256.Sum256(read(bundle.Instrumentation[spec.Owner]))) {
			t.Fatal("instrumentation differs", spec.Name)
		}
		if instrument.Correction != "" && instrument.Correction != fmt.Sprintf("%x", sha256.Sum256(read(fix.Protocol))) {
			t.Fatal("correction protocol differs")
		}
	}
	failed, err := testevidence.GoTestJSONReport(string(read(bundle.Instrumentation["media-un0-video-acquisition-failed-1.jsonl"])))
	if err != nil || len(failed.Failed) == 0 {
		t.Fatal("lost failed reference-reading attempt", err)
	}
	raw, err = os.ReadFile(filepath.Join(root, "docs/image_video_protocol.json"))
	testcheck.NoError(t, err)
	testcheck.NoError(t, checkImageVideoProtocolIdentity(raw))
	var protocol imageVideoProtocol
	testcheck.NoError(t, json.Unmarshal(raw, &protocol))
	expected, seen := 0, map[string]bool{}
	for _, c := range protocol.Cases {
		if !strings.Contains(c.ID, "/video-gen/") || strings.HasPrefix(c.ID, "Wan") {
			continue
		}
		expected++
		index := slices.IndexFunc(bundle.Cases, func(v mediaConditionedCase) bool { return v.Case == c.ID })
		if index < 0 || seen[c.ID] {
			t.Fatal("missing conditioned case")
		}
		seen[c.ID] = true
		entry := bundle.Cases[index]
		run, err := runrecord.RequireExactRun(t.Context(), store, entry.Run)
		testcheck.NoError(t, err)
		if run.CodeCommit != bundle.Source || run.Recipe != c.Recipe || !slices.Equal(run.Inputs, c.Inputs) || !slices.Equal(run.Outputs, []artifact.ID{entry.Output}) || run.Outcome != runrecord.OutcomeSucceeded || run.MeasuredNS == 0 || len(run.Phases) == 0 {
			t.Fatal("conditioned run identity differs")
		}
		if _, err := runrecord.RequireEnvironment(t.Context(), store, run.Environment); err != nil {
			t.Fatal(err)
		}
		var group struct {
			Loads int                           `json:"model_loads"`
			Rows  []mediaConditionedObservation `json:"observations"`
		}
		decode(entry.Acquisition, &group)
		rowIndex := slices.IndexFunc(group.Rows, func(v mediaConditionedObservation) bool { return v.Case == c.ID })
		if rowIndex < 0 || len(group.Rows) != 2 {
			t.Fatal("missing actual observation")
		}
		row := group.Rows[rowIndex]
		if row.Source != bundle.Source || row.Run != run.ID || row.Recipe != c.Recipe || !slices.Equal(row.Inputs, c.Inputs) || row.Output != entry.Output || row.Environment != run.Environment || row.Wall != run.MeasuredNS || row.Allocated == 0 || row.Peak == 0 || !row.Finite || row.Frames != c.Observations[0].Frames || row.Width != c.Observations[0].Width || row.Height != c.Observations[0].Height || row.FPS != c.Observations[0].FPS {
			t.Fatal("incomplete or inconsistent video acquisition")
		}
		workflow, err := runrecord.RequireExactRun(t.Context(), store, row.Workflow)
		if err != nil || workflow.Recipe != c.Recipe || !slices.Equal(workflow.Inputs, c.Inputs) || !slices.Equal(workflow.Outputs, run.Outputs) {
			t.Fatal("compiled recipe lineage differs", err)
		}
		current, historical := sample(entry.Output), sample(c.Outputs[0])
		checkTest(entry.Test)
		var review struct {
			Case         string `json:"case"`
			SHA          string `json:"gif_sha256"`
			Subject      string `json:"subject"`
			Preservation string `json:"source_preservation"`
			Defects      string `json:"visible_defects"`
			Video        string `json:"video"`
			Scope        string `json:"review_scope"`
		}
		decode(entry.Review, &review)
		if review.Case != c.ID || review.SHA != fmt.Sprintf("%x", sha256.Sum256(current)) || review.Subject == "" || review.Preservation == "" || review.Defects == "" || review.Video == "" || review.Scope == "" {
			t.Fatal("missing content-specific visual review")
		}
		switch entry.Mode {
		case "acquired-host":
			if group.Loads != 1 || len(row.Before) != 0 || len(row.Memory) != 0 || !row.Same || len(row.FrameSHA) != row.Frames || math.IsNaN(row.Minimum) || math.IsInf(row.Minimum, 0) || math.IsNaN(row.Maximum) || math.IsInf(row.Maximum, 0) || row.Minimum > row.Maximum {
				t.Fatal("CPU video audit differs")
			}
			testcheck.NoError(t, checkMediaWanGIF(current, historical, c.Observations[0]))
		case "acquired-cuda-corrected":
			if group.Loads != 2 || row.Load == 0 || row.Close == 0 || len(row.Before) != 3 || len(row.After) != 3 || len(row.Closed) != 3 || row.Same || !row.OldMatch || !row.CorrectMatch || row.NativeFrameSHA != fix.NativeFrameSHA || len(entry.RawFrames) != row.Frames {
				t.Fatal("missing corrected CUDA evidence")
			}
			for name, before := range row.Before {
				after := row.After[name]
				if after.KernelLaunches+after.GraphLaunches <= before.KernelLaunches+before.GraphLaunches || row.Memory[name].PeakBytes == 0 || row.Closed[name].CurrentBytes != 0 {
					t.Fatal("missing GPU work or cleanup", name)
				}
			}
			if row.After["decoder"].DeviceToHostBytes <= row.Before["decoder"].DeviceToHostBytes {
				t.Fatal("missing output transfer")
			}
			baseline, err := runrecord.RequireExactRun(t.Context(), store, entry.BaselineRun)
			if err != nil || baseline.CodeCommit != bundle.Source || !slices.Equal(baseline.Inputs, c.Inputs) {
				t.Fatal("missing original encoding baseline", err)
			}
			testcheck.NoError(t, checkMediaWanGIF(sample(baseline.Outputs[0]), historical, c.Observations[0]))
			oldEncoder, err := latentvideo.NewGIFEncoder(row.FPS, latentvideo.UnitPixels)
			testcheck.NoError(t, err)
			fixedEncoder, err := latentvideo.NewGIFEncoder(row.FPS, latentvideo.SignedUnitPixels)
			testcheck.NoError(t, err)
			aggregate := sha256.New()
			for i, id := range entry.RawFrames {
				raw := read(id)
				if len(raw) != 3*row.Width*row.Height*4 {
					t.Fatal("raw frame geometry differs")
				}
				values := make([]float32, len(raw)/4)
				for j := range values {
					values[j] = math.Float32frombits(binary.LittleEndian.Uint32(raw[j*4:]))
					if math.IsNaN(float64(values[j])) || math.IsInf(float64(values[j]), 0) {
						t.Fatal("nonfinite raw frame")
					}
				}
				digest := sha256.Sum256(raw)
				aggregate.Write(digest[:])
				testcheck.NoError(t, oldEncoder.Add(i, values, row.Height, row.Width))
				testcheck.NoError(t, fixedEncoder.Add(i, values, row.Height, row.Width))
			}
			if hex.EncodeToString(aggregate.Sum(nil)) != fix.NativeFrameSHA {
				t.Fatal("native float frame sequence changed")
			}
			oldEncoded, err := oldEncoder.Finish()
			testcheck.NoError(t, err)
			fixedEncoded, err := fixedEncoder.Finish()
			testcheck.NoError(t, err)
			testcheck.NoError(t, checkMediaWanGIF(oldEncoded.Data, historical, c.Observations[0]))
			if !bytes.Equal(fixedEncoded.Data, current) || bytes.Equal(oldEncoded.Data, current) {
				t.Fatal("corrected encoding does not follow signed float range")
			}
			testcheck.NoError(t, checkMediaWanGIF(current, current, c.Observations[0]))
			oldGIF, err := gif.DecodeAll(bytes.NewReader(oldEncoded.Data))
			testcheck.NoError(t, err)
			newGIF, err := gif.DecodeAll(bytes.NewReader(current))
			testcheck.NoError(t, err)
			if reflect.DeepEqual(oldGIF, newGIF) {
				t.Fatal("encoding defect not corrected")
			}
		default:
			t.Fatal("unknown conditioned acquisition mode")
		}
	}
	if expected != len(bundle.Cases) || len(seen) != expected {
		t.Fatal("conditioned video coverage differs")
	}
	for _, check := range bundle.Numerical {
		checkTest(check.Evidence)
	}
	read(bundle.Inputs)
	for _, id := range bundle.Inspection {
		read(id)
	}
}
