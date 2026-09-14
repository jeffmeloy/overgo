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
	"overgo/internal/cuda/driver"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/testevidence"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// Filled from the retained comparison only after all declared acquisitions.
const imageVideoTransferSHA256 = "331e33f5a9c6c72afe576eae8ac138deeb3c297ce423a0273404cf92a083f3b0"

type mediaTransferComparison struct {
	Version    uint16                        `json:"version"`
	Decision   string                        `json:"decision"`
	Rationale  string                        `json:"rationale"`
	Acceptance artifact.ID                   `json:"acceptance"`
	Evidence   []artifact.ID                 `json:"evidence"`
	Sources    map[string]artifact.ID        `json:"sources"`
	Cases      []mediaTransferComparisonCase `json:"cases"`
	Quality    []mediaSharedCheck            `json:"quality"`
}

type mediaTransferComparisonCase struct {
	Name           string      `json:"name"`
	Steps          uint64      `json:"steps"`
	StepInputBytes uint64      `json:"step_input_bytes"`
	Baseline       artifact.ID `json:"baseline"`
	Candidate      artifact.ID `json:"candidate"`
	Ablation       artifact.ID `json:"ablation,omitempty"`
}

type mediaTransferRecord struct {
	Source       string                                         `json:"source_base"`
	Sources      map[string]string                              `json:"source_sha256"`
	Input        string                                         `json:"input_sha256"`
	Model        artifact.ID                                    `json:"model"`
	Recipe       artifact.ID                                    `json:"recipe"`
	Harness      string                                         `json:"harness_sha256"`
	Acceptance   string                                         `json:"acceptance_sha256"`
	CopyOwner    string                                         `json:"copy_owner_sha256"`
	CopyProbe    string                                         `json:"copy_probe_sha256"`
	CopyTiming   bool                                           `json:"copy_timing_hooks"`
	Completed    string                                         `json:"completed_at"`
	Observations []mediaTransferObservation                     `json:"observations"`
	AfterClose   struct{ Denoiser, Decoder driver.MemoryStats } `json:"after_close"`
}

type mediaTransferObservation struct {
	State          string                `json:"state"`
	Wall           int64                 `json:"wall_ns"`
	Allocated      uint64                `json:"total_alloc_bytes"`
	HostPeak       uint64                `json:"process_lifetime_peak_host_bytes"`
	DenoiserBefore driver.ExecutionStats `json:"denoiser_before"`
	DenoiserAfter  driver.ExecutionStats `json:"denoiser_after"`
	DecoderBefore  driver.ExecutionStats `json:"decoder_before"`
	DecoderAfter   driver.ExecutionStats `json:"decoder_after"`
	DenoiserMemory driver.MemoryStats    `json:"denoiser_memory"`
	DecoderMemory  driver.MemoryStats    `json:"decoder_memory"`
	CopyBefore     map[string]int64      `json:"copy_api_before_ns"`
	CopyAfter      map[string]int64      `json:"copy_api_after_ns"`
	Latent         string                `json:"latent_sha256"`
	Frames         []string              `json:"frame_sha256"`
	GIF            string                `json:"gif_sha256"`
	FrameCount     int                   `json:"frames"`
	Width          int                   `json:"width"`
	Height         int                   `json:"height"`
	FPS            int                   `json:"fps"`
	Delay          int                   `json:"delay_centiseconds"`
}

func checkMediaTransferRecord(record mediaTransferRecord) error {
	if record.Completed == "" || !record.CopyTiming || len(record.Observations) != 2 || record.AfterClose.Denoiser.CurrentBytes != 0 || record.AfterClose.Decoder.CurrentBytes != 0 {
		return errors.New("incomplete profiled acquisition or unreleased device ownership")
	}
	for index, state := range []string{"first-request", "repeat-request"} {
		row := record.Observations[index]
		if row.State != state || row.Wall <= 0 || row.Allocated == 0 || row.HostPeak == 0 || row.FrameCount != len(row.Frames) || row.FrameCount == 0 || row.Latent == "" || row.GIF == "" {
			return errors.New("missing request state, cost or output")
		}
		for _, direction := range []string{"host_to_device", "device_to_host", "device_to_device"} {
			before, haveBefore := row.CopyBefore[direction]
			after, haveAfter := row.CopyAfter[direction]
			if !haveBefore || !haveAfter || before < 0 || after < before {
				return errors.New("missing or decreasing copy API timer")
			}
		}
		first := record.Observations[0]
		if row.Latent != first.Latent || row.GIF != first.GIF || !slices.Equal(row.Frames, first.Frames) {
			return errors.New("repeat request changed output")
		}
	}
	return nil
}

// Quality and the predicted transfer effect are mandatory even when a candidate
// is rejected on complete-request cost. The decision cannot hide either result.
func compareMediaTransfers(before, after mediaTransferRecord, steps, stepBytes uint64) (bool, error) {
	if err := checkMediaTransferRecord(before); err != nil {
		return false, err
	}
	if err := checkMediaTransferRecord(after); err != nil {
		return false, err
	}
	if before.Source != after.Source || before.Input != after.Input || before.Model != after.Model || before.Recipe != after.Recipe || before.Harness != after.Harness || before.Acceptance != after.Acceptance || before.CopyOwner != after.CopyOwner || before.CopyProbe != after.CopyProbe || steps == 0 || stepBytes == 0 {
		return false, errors.New("comparison changed its workload, instrumentation or base")
	}
	efficient := true
	for index, b := range before.Observations {
		a := after.Observations[index]
		if a.Latent != b.Latent || a.GIF != b.GIF || !slices.Equal(a.Frames, b.Frames) || a.FrameCount != b.FrameCount || a.Width != b.Width || a.Height != b.Height || a.FPS != b.FPS || a.Delay != b.Delay {
			return false, errors.New("transfer change altered output")
		}
		bh, ah := b.DenoiserAfter.HostToDeviceBytes-b.DenoiserBefore.HostToDeviceBytes, a.DenoiserAfter.HostToDeviceBytes-a.DenoiserBefore.HostToDeviceBytes
		bc, ac := b.DenoiserAfter.HostToDeviceCopies-b.DenoiserBefore.HostToDeviceCopies, a.DenoiserAfter.HostToDeviceCopies-a.DenoiserBefore.HostToDeviceCopies
		if bh < ah || bh-ah != steps*stepBytes || bc < ac || bc-ac != steps*3 {
			return false, errors.New("measured transfer effect differs from the three graph inputs")
		}
		for _, pair := range [][4]driver.ExecutionStats{{b.DenoiserBefore, b.DenoiserAfter, a.DenoiserBefore, a.DenoiserAfter}, {b.DecoderBefore, b.DecoderAfter, a.DecoderBefore, a.DecoderAfter}} {
			if pair[1].DeviceToHostBytes-pair[0].DeviceToHostBytes != pair[3].DeviceToHostBytes-pair[2].DeviceToHostBytes || pair[1].DeviceToHostCopies-pair[0].DeviceToHostCopies != pair[3].DeviceToHostCopies-pair[2].DeviceToHostCopies || pair[1].DeviceToDeviceBytes-pair[0].DeviceToDeviceBytes != pair[3].DeviceToDeviceBytes-pair[2].DeviceToDeviceBytes || pair[1].DeviceToDeviceCopies-pair[0].DeviceToDeviceCopies != pair[3].DeviceToDeviceCopies-pair[2].DeviceToDeviceCopies {
				return false, errors.New("candidate changed required output or device copies")
			}
		}
		efficient = efficient && a.Wall <= b.Wall && a.Allocated <= b.Allocated && a.HostPeak <= b.HostPeak && a.DenoiserMemory.PeakBytes <= b.DenoiserMemory.PeakBytes && a.DecoderMemory.PeakBytes <= b.DecoderMemory.PeakBytes
	}
	return efficient, nil
}

func TestImageVideoTransferOptimizationAcceptance(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration + ": transfer comparison reads retained acquisitions")
	}
	root := testutil.RepoRoot(t)
	document, err := plan.Load(filepath.Join(root, plan.Path))
	if err != nil {
		t.Fatal(err)
	}
	if document.Lane != "image_video_gen" || os.Getenv(dataroot.Env) == "" {
		t.Skip("integration: media comparison requires its explicit data root")
	}
	path := filepath.Join(root, "docs/image_video_transfers.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkMediaProtocolIdentity(raw, imageVideoTransferSHA256); err != nil {
		t.Fatal(err)
	}
	var comparison mediaTransferComparison
	if err := jsonfile.DecodeStrict(path, &comparison); err != nil {
		t.Fatal(err)
	}
	if comparison.Version != artifact.InitialDocumentVersion || comparison.Rationale == "" || len(comparison.Cases) != 2 || len(comparison.Quality) == 0 || len(comparison.Evidence) == 0 || len(comparison.Sources) == 0 {
		t.Fatal("incomplete comparison")
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
			t.Fatal("comparison input is not retained evidence")
		}
		content, found, err := artifact.ReadContent(t.Context(), store, id)
		if err != nil || !found || len(content.Data) == 0 {
			t.Fatalf("missing %s: %v", id, err)
		}
		return content.Data
	}
	acceptance := fmt.Sprintf("%x", sha256.Sum256(read(comparison.Acceptance)))
	for _, id := range comparison.Evidence {
		read(id)
	}
	for _, id := range comparison.Sources {
		read(id)
	}
	for _, check := range comparison.Quality {
		report, err := testevidence.GoTestJSONReport(string(read(check.Evidence)))
		if err != nil {
			t.Fatal(err)
		}
		if err := testevidence.RequireComplete(report); err != nil {
			t.Fatal(err)
		}
		if report.PassedTests == 0 || report.PassedPackages == 0 || check.Name == "" || check.Command == "" {
			t.Fatal("empty quality acquisition")
		}
	}
	efficient := true
	for index, name := range []string{"g4", "full"} {
		entry := comparison.Cases[index]
		if entry.Name != name {
			t.Fatal("missing or duplicate workload")
		}
		var before, after mediaTransferRecord
		if err := json.Unmarshal(read(entry.Baseline), &before); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(read(entry.Candidate), &after); err != nil {
			t.Fatal(err)
		}
		if before.Acceptance != acceptance {
			t.Fatal("acquisition changed frozen acceptance")
		}
		accepted, err := compareMediaTransfers(before, after, entry.Steps, entry.StepInputBytes)
		if err != nil {
			t.Fatal(err)
		}
		if name == "full" {
			efficient = accepted
		}
		for _, record := range []struct {
			prefix string
			value  mediaTransferRecord
		}{{"baseline", before}, {"candidate", after}} {
			for path, digest := range record.value.Sources {
				if fmt.Sprintf("%x", sha256.Sum256(read(comparison.Sources[record.prefix+"/"+path]))) != digest {
					t.Fatalf("%s acquisition source differs: %s", record.prefix, path)
				}
			}
		}
		bad := after
		bad.Observations = slices.Clone(after.Observations)
		bad.Observations[0].FrameCount++
		if _, err := compareMediaTransfers(before, bad, entry.Steps, entry.StepInputBytes); err == nil {
			t.Fatal("accepted altered frame count")
		}
		bad.Observations = slices.Clone(after.Observations)
		bad.Observations[0].DenoiserAfter.HostToDeviceCopies++
		if _, err := compareMediaTransfers(before, bad, entry.Steps, entry.StepInputBytes); err == nil {
			t.Fatal("accepted a different transfer effect")
		}
		bad.Observations = slices.Clone(after.Observations)
		bad.Observations[0].Wall = before.Observations[0].Wall + 1
		if accepted, err := compareMediaTransfers(before, bad, entry.Steps, entry.StepInputBytes); err != nil || accepted {
			t.Fatal("failed to reject complete-request time regression")
		}
		if name == "g4" {
			var ablation mediaTransferRecord
			if err := json.Unmarshal(read(entry.Ablation), &ablation); err != nil {
				t.Fatal(err)
			}
			if _, err := compareMediaTransfers(ablation, after, entry.Steps, entry.StepInputBytes); err != nil {
				t.Fatalf("causal ablation: %v", err)
			}
		}
	}
	if (comparison.Decision == "adopt") != efficient || (comparison.Decision != "adopt" && comparison.Decision != "reject") {
		t.Fatal("decision contradicts frozen comparisons")
	}
	t.Logf("retained complete-request comparison: %s", comparison.Decision)
}
