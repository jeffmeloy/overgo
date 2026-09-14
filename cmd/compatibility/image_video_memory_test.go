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

const imageVideoMemorySHA256 = "5c62c46d0f730731a5f1bf199491da557fe3b5295891188a429381fb060006f6"

type mediaMemoryComparison struct {
	Version         uint16                 `json:"version"`
	Decision        string                 `json:"decision"`
	Rationale       string                 `json:"rationale"`
	Acceptance      artifact.ID            `json:"acceptance"`
	Sources         map[string]artifact.ID `json:"sources"`
	Evidence        []artifact.ID          `json:"evidence"`
	Quality         []mediaSharedCheck     `json:"quality"`
	Cases           []mediaMemoryCase      `json:"cases"`
	LayoutBaseline  artifact.ID            `json:"layout_baseline"`
	LayoutCandidate artifact.ID            `json:"layout_candidate"`
	LayoutAblation  artifact.ID            `json:"layout_ablation"`
}

type mediaMemoryCase struct {
	Name      string      `json:"name"`
	Steps     uint64      `json:"steps"`
	Baseline  artifact.ID `json:"baseline"`
	Candidate artifact.ID `json:"candidate"`
}

type mediaLayoutLifetime struct {
	Trace     bool   `json:"trace"`
	Allocated uint64 `json:"allocated_bytes"`
	Calls     uint64 `json:"allocation_calls"`
	Bytes     uint64 `json:"layout_bytes"`
	Steps     uint64 `json:"steps"`
	Output    string `json:"output_sha256"`
}

type mediaMemoryStorage struct {
	Pool struct {
		Total     uint64 `json:"total_bytes"`
		Free      uint64 `json:"free_bytes"`
		Live      uint64 `json:"live_bytes"`
		Count     uint64 `json:"allocation_count"`
		FreeCount uint64 `json:"free_count"`
	} `json:"denoiser_pool"`
	Arena struct {
		Required  uint64 `json:"arenaRequiredBytes"`
		Committed uint64 `json:"arenaCommittedBytes"`
		Unused    uint64 `json:"arenaUnusedBytes"`
	} `json:"denoiser_arena"`
	DecoderBytes uint64 `json:"decoder_workspace_bytes"`
	DecoderCount uint64 `json:"decoder_workspace_count"`
	Patch        uint64 `json:"patch_logical_bytes"`
	Branch       uint64 `json:"branch_output_logical_bytes"`
}

func checkMediaMemoryStorage(data []byte) (string, error) {
	var record struct {
		PoolProbe    string `json:"pool_probe_sha256"`
		Observations []struct {
			Storage mediaMemoryStorage `json:"storage"`
		} `json:"observations"`
	}
	if err := json.Unmarshal(data, &record); err != nil {
		return "", err
	}
	if record.PoolProbe == "" || len(record.Observations) != 2 {
		return "", errors.New("missing pool ownership observations")
	}
	for _, row := range record.Observations {
		s := row.Storage
		if s.Pool.Total == 0 || s.Pool.Free+s.Pool.Live != s.Pool.Total || s.Pool.FreeCount > s.Pool.Count || s.Arena.Required == 0 || s.Arena.Committed < s.Arena.Required || s.Arena.Unused != s.Arena.Committed-s.Arena.Required || s.DecoderBytes == 0 || s.DecoderCount == 0 || s.Patch == 0 || s.Branch == 0 {
			return "", errors.New("incomplete or inconsistent memory ownership scopes")
		}
	}
	return record.PoolProbe, nil
}

func mediaTransferCountersEqual(before0, before1, after0, after1 driver.ExecutionStats) bool {
	return before1.HostToDeviceBytes-before0.HostToDeviceBytes == after1.HostToDeviceBytes-after0.HostToDeviceBytes &&
		before1.HostToDeviceCopies-before0.HostToDeviceCopies == after1.HostToDeviceCopies-after0.HostToDeviceCopies &&
		before1.DeviceToHostBytes-before0.DeviceToHostBytes == after1.DeviceToHostBytes-after0.DeviceToHostBytes &&
		before1.DeviceToHostCopies-before0.DeviceToHostCopies == after1.DeviceToHostCopies-after0.DeviceToHostCopies &&
		before1.DeviceToDeviceBytes-before0.DeviceToDeviceBytes == after1.DeviceToDeviceBytes-after0.DeviceToDeviceBytes &&
		before1.DeviceToDeviceCopies-before0.DeviceToDeviceCopies == after1.DeviceToDeviceCopies-after0.DeviceToDeviceCopies &&
		before1.StreamSynchronizations-before0.StreamSynchronizations == after1.StreamSynchronizations-after0.StreamSynchronizations &&
		before1.ContextSynchronizations-before0.ContextSynchronizations == after1.ContextSynchronizations-after0.ContextSynchronizations
}

func compareMediaMemory(before, after mediaTransferRecord) (bool, error) {
	if err := checkMediaTransferRecord(before); err != nil {
		return false, err
	}
	if err := checkMediaTransferRecord(after); err != nil {
		return false, err
	}
	if before.Source != after.Source || before.Input != after.Input || before.Model != after.Model || before.Recipe != after.Recipe || before.Harness != after.Harness || before.Acceptance != after.Acceptance || before.CopyOwner != after.CopyOwner || before.CopyProbe != after.CopyProbe {
		return false, errors.New("memory comparison changed its input or instrumentation")
	}
	accepted := true
	for i, b := range before.Observations {
		a := after.Observations[i]
		if a.Latent != b.Latent || a.GIF != b.GIF || !slices.Equal(a.Frames, b.Frames) || a.FrameCount != b.FrameCount || a.Width != b.Width || a.Height != b.Height || a.FPS != b.FPS || a.Delay != b.Delay {
			return false, errors.New("layout reuse changed generated output")
		}
		if !mediaTransferCountersEqual(b.DenoiserBefore, b.DenoiserAfter, a.DenoiserBefore, a.DenoiserAfter) || !mediaTransferCountersEqual(b.DecoderBefore, b.DecoderAfter, a.DecoderBefore, a.DecoderAfter) {
			return false, errors.New("layout reuse changed GPU transfers or synchronization")
		}
		accepted = accepted && a.Allocated < b.Allocated && a.Wall <= b.Wall && a.HostPeak <= b.HostPeak && a.DenoiserMemory.PeakBytes <= b.DenoiserMemory.PeakBytes && a.DecoderMemory.PeakBytes <= b.DecoderMemory.PeakBytes
	}
	return accepted, nil
}

func TestImageVideoMemoryOptimizationAcceptance(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration + ": memory comparison reads retained acquisitions")
	}
	root := testutil.RepoRoot(t)
	document, err := plan.Load(filepath.Join(root, plan.Path))
	if err != nil {
		t.Fatal(err)
	}
	if document.Lane != "image_video_gen" || os.Getenv(dataroot.Env) == "" {
		t.Skip("integration: media comparison requires its explicit data root")
	}
	path := filepath.Join(root, "docs/image_video_memory.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkMediaProtocolIdentity(raw, imageVideoMemorySHA256); err != nil {
		t.Fatal(err)
	}
	var comparison mediaMemoryComparison
	if err := jsonfile.DecodeStrict(path, &comparison); err != nil {
		t.Fatal(err)
	}
	if comparison.Version != artifact.InitialDocumentVersion || comparison.Rationale == "" || len(comparison.Cases) != 2 || len(comparison.Quality) == 0 || len(comparison.Sources) == 0 {
		t.Fatal("incomplete memory comparison")
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
	accepted := false
	for index, name := range []string{"g4", "full"} {
		entry := comparison.Cases[index]
		if entry.Name != name || entry.Steps == 0 {
			t.Fatal("missing or duplicated workload")
		}
		beforeData, afterData := read(entry.Baseline), read(entry.Candidate)
		beforeProbe, err := checkMediaMemoryStorage(beforeData)
		if err != nil {
			t.Fatal(err)
		}
		afterProbe, err := checkMediaMemoryStorage(afterData)
		if err != nil {
			t.Fatal(err)
		}
		if beforeProbe != afterProbe {
			t.Fatal("memory comparison changed its pool instrumentation")
		}
		var before, after mediaTransferRecord
		if err := json.Unmarshal(beforeData, &before); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(afterData, &after); err != nil {
			t.Fatal(err)
		}
		if before.Acceptance != acceptance {
			t.Fatal("changed frozen memory acceptance")
		}
		result, err := compareMediaMemory(before, after)
		if err != nil {
			t.Fatal(err)
		}
		if name == "full" {
			accepted = result
		}
		for _, record := range []struct {
			prefix string
			value  mediaTransferRecord
		}{{"baseline", before}, {"candidate", after}} {
			for path, digest := range record.value.Sources {
				if fmt.Sprintf("%x", sha256.Sum256(read(comparison.Sources[record.prefix+"/"+path]))) != digest {
					t.Fatalf("%s source differs: %s", record.prefix, path)
				}
			}
		}
		bad := after
		bad.Observations = slices.Clone(after.Observations)
		bad.Observations[0].DenoiserAfter.HostToDeviceCopies++
		if _, err := compareMediaMemory(before, bad); err == nil {
			t.Fatal("accepted shifted GPU transfer cost")
		}
		bad.Observations = slices.Clone(after.Observations)
		bad.Observations[0].Allocated = before.Observations[0].Allocated
		if accepted, err := compareMediaMemory(before, bad); err != nil || accepted {
			t.Fatal("accepted absent host allocation reduction")
		}
	}
	var baseline, candidate, ablation []mediaLayoutLifetime
	if err := json.Unmarshal(read(comparison.LayoutBaseline), &baseline); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(read(comparison.LayoutCandidate), &candidate); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(read(comparison.LayoutAblation), &ablation); err != nil {
		t.Fatal(err)
	}
	if len(baseline) != 2 || len(candidate) != 2 || len(ablation) != 2 {
		t.Fatal("missing trace-state allocation ablation")
	}
	for i, trace := range []bool{false, true} {
		b, a, c := baseline[i], candidate[i], ablation[i]
		buffers := uint64(3)
		if trace {
			buffers = 1
		}
		calls := (b.Steps - 1) * buffers
		if b.Trace != trace || a.Trace != trace || c.Trace != trace || b.Steps < 2 || b.Bytes == 0 || a.Steps != b.Steps || c.Steps != b.Steps || a.Bytes != b.Bytes || c.Bytes != b.Bytes || b.Output == "" || a.Output != b.Output || c.Output != b.Output || b.Allocated != c.Allocated || b.Calls != c.Calls || b.Allocated < a.Allocated || b.Allocated-a.Allocated != calls*b.Bytes || b.Calls < a.Calls || b.Calls-a.Calls != calls {
			t.Fatal("allocation lifetime ablation changed output or failed its predicted effect")
		}
	}
	if (comparison.Decision == "adopt") != accepted || (comparison.Decision != "adopt" && comparison.Decision != "reject") {
		t.Fatal("memory decision contradicts frozen comparisons")
	}
	t.Logf("complete-request memory comparison: %s; synthetic allocation ablation preserves traced and untraced output", comparison.Decision)
}
