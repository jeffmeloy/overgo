package main

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/cuda/driver"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/plan"
	"overgo/internal/processcontrol"
	"overgo/internal/processmeasure"
	"overgo/internal/runrecord"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// The resource criteria remain fixed while later steps collect measurements.
const imageVideoResourceProtocolSHA256 = "2e7b7bfb8e8d308a498ea5b54572b0dedf3dccfa9ec12731f31758775cc7cc02"

type imageVideoResourceProtocol struct {
	Version               uint16                     `json:"version"`
	QualityProtocolSHA256 string                     `json:"quality_protocol_sha256"`
	Admission             []string                   `json:"admission"`
	Measurement           []string                   `json:"measurement"`
	CommonMetrics         []runrecord.ResourceMetric `json:"required_common_metrics"`
	DriverCounters        []string                   `json:"required_driver_counters"`
	MemoryScopes          []string                   `json:"required_memory_scopes"`
	AcceptanceOwners      []string                   `json:"acceptance_owners"`
	KnownGaps             []string                   `json:"known_gaps"`
}

func TestImageVideoResourceProtocolAcceptance(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration + ": media resource protocol discovers current hardware")
	}
	root := testutil.RepoRoot(t)
	document, err := plan.Load(filepath.Join(root, plan.Path))
	if err != nil {
		t.Fatal(err)
	}
	if document.Lane != "image_video_gen" || os.Getenv(dataroot.Env) == "" {
		t.Skip("integration: image_video_gen resources require their explicit data root")
	}
	var protocol imageVideoResourceProtocol
	protocolPath := filepath.Join(root, "docs/image_video_resources.json")
	data, err := os.ReadFile(protocolPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkMediaProtocolIdentity(data, imageVideoResourceProtocolSHA256); err != nil {
		t.Fatal(err)
	}
	if err := jsonfile.DecodeStrict(protocolPath, &protocol); err != nil {
		t.Fatal(err)
	}
	if protocol.Version != artifact.InitialDocumentVersion || protocol.QualityProtocolSHA256 != imageVideoProtocolSHA256 {
		t.Fatal("resource and quality protocol binding differs")
	}
	want := []runrecord.ResourceMetric{runrecord.ResourceWallNS, runrecord.ResourcePeakHostBytes, runrecord.ResourcePeakDeviceBytes, runrecord.ResourceHostToDeviceBytes, runrecord.ResourceDeviceToHostBytes}
	if !slices.Equal(protocol.CommonMetrics, want) {
		t.Fatal("required comparison metrics differ")
	}
	for _, metric := range protocol.CommonMetrics {
		if !metric.Valid() {
			t.Fatalf("unsupported common metric %s", metric)
		}
	}
	encoded, err := json.Marshal(driver.ExecutionStats{})
	if err != nil {
		t.Fatal(err)
	}
	var counters map[string]uint64
	if err := json.Unmarshal(encoded, &counters); err != nil {
		t.Fatal(err)
	}
	for _, name := range protocol.DriverCounters {
		if _, found := counters[name]; !found {
			t.Fatalf("counter %s absent from driver owner", name)
		}
	}
	if len(protocol.DriverCounters) == 0 || len(protocol.MemoryScopes) == 0 || len(protocol.Admission) == 0 || len(protocol.Measurement) == 0 || len(protocol.KnownGaps) == 0 {
		t.Fatal("resource protocol omits measurement scopes or requirements")
	}
	for _, field := range []string{"required_common_metrics", "required_driver_counters", "required_memory_scopes", "admission", "measurement"} {
		t.Run("omitted "+field, func(t *testing.T) {
			var changed map[string]any
			if err := json.Unmarshal(data, &changed); err != nil {
				t.Fatal(err)
			}
			delete(changed, field)
			encoded, err := json.Marshal(changed)
			if err != nil {
				t.Fatal(err)
			}
			if checkMediaProtocolIdentity(encoded, imageVideoResourceProtocolSHA256) == nil {
				t.Fatal("incomplete resource protocol accepted")
			}
		})
	}

	// Query metadata only. Discovery holds no device reservation and does not
	// imply that any model's live ranges fit the values observed here.
	if runtime.GOOS != "windows" {
		t.Fatal("UNAVAILABLE: this campaign's current hardware discovery requires Windows")
	}
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	command := processcontrol.Command{
		Path: "powershell.exe", Args: []string{"-NoProfile", "-NonInteractive", "-Command", `$mediaOS=Get-CimInstance Win32_OperatingSystem; $mediaCPU=Get-CimInstance Win32_Processor; $mediaDisk=Get-CimInstance Win32_LogicalDisk | Where-Object DeviceID -eq $env:OVERGO_MEDIA_DISK_ROOT; [ordered]@{host_available_bytes=([uint64]$mediaOS.FreePhysicalMemory*1024);host_total_bytes=([uint64]$mediaOS.TotalVisibleMemorySize*1024);cpu_logical_processors=(($mediaCPU|Measure-Object NumberOfLogicalProcessors -Sum).Sum);cpu_load_percent=(($mediaCPU|Measure-Object LoadPercentage -Average).Average);disk_available_bytes=[uint64]$mediaDisk.FreeSpace}|ConvertTo-Json -Compress`},
		Env: append(os.Environ(), "OVERGO_MEDIA_DISK_ROOT="+filepath.VolumeName(roots.Store)), Stdout: &stdout, Stderr: &stderr,
	}
	receipt, err := processcontrol.Run(t.Context(), command)
	if err != nil || receipt.ExitCode != 0 {
		t.Fatalf("UNAVAILABLE: host capacity discovery: %v: %s", err, stderr.String())
	}
	var host struct {
		Available uint64  `json:"host_available_bytes"`
		Total     uint64  `json:"host_total_bytes"`
		CPU       float64 `json:"cpu_logical_processors"`
		Load      float64 `json:"cpu_load_percent"`
		Disk      uint64  `json:"disk_available_bytes"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &host); err != nil {
		t.Fatal(err)
	}
	if host.Available == 0 || host.Available > host.Total || host.CPU <= 0 || host.Disk == 0 {
		t.Fatal("UNAVAILABLE: incomplete host capacity")
	}
	t.Logf("current host discovery=%s; wall_ns=%d; no model admitted", strings.TrimSpace(stdout.String()), receipt.WallNS)
	stdout.Reset()
	stderr.Reset()
	receipt, err = processcontrol.Run(t.Context(), processcontrol.Command{Path: "nvidia-smi", Args: []string{"--query-gpu=uuid,driver_version,memory.total,memory.free,utilization.gpu", "--format=csv,noheader,nounits"}, Stdout: &stdout, Stderr: &stderr})
	if err != nil || receipt.ExitCode != 0 {
		t.Fatalf("UNAVAILABLE: GPU capacity discovery: %v: %s", err, stderr.String())
	}
	rows, err := csv.NewReader(strings.NewReader(stdout.String())).ReadAll()
	if err != nil || len(rows) == 0 {
		t.Fatalf("UNAVAILABLE: GPU records: %v", err)
	}
	for _, row := range rows {
		if len(row) != 5 || !strings.HasPrefix(strings.TrimSpace(row[0]), "GPU-") || strings.TrimSpace(row[1]) == "" {
			t.Fatalf("UNAVAILABLE: incomplete GPU record %v", row)
		}
		total, e1 := strconv.ParseUint(strings.TrimSpace(row[2]), 10, 64)
		free, e2 := strconv.ParseUint(strings.TrimSpace(row[3]), 10, 64)
		if e1 != nil || e2 != nil || total == 0 || free > total {
			t.Fatalf("UNAVAILABLE: invalid GPU capacity %v", row)
		}
		t.Logf("current GPU metadata=%v; memory unit=MiB; no allocation reservation", row)
	}

	// Exercise the existing admission and metric owners with boundary fixtures.
	// OS-level exclusion and process-death checks run as separate named owners.
	now := time.Now()
	capacity := plan.Resources{CPUThreads: 2, HostRAMGiB: 2, VRAMGiB: 2}
	lease := plan.WorkLease{Task: "media/resource-fixture", Worktree: "media-fixture", ExpiresAt: now.Add(time.Hour).Format(time.RFC3339Nano), Resources: capacity}
	if !plan.AssessResources(now, capacity, []plan.WorkLease{lease}).Fits {
		t.Fatal("exact capacity refused")
	}
	for _, name := range []string{"cpu", "host", "device"} {
		reduced := capacity
		switch name {
		case "cpu":
			reduced.CPUThreads--
		case "host":
			reduced.HostRAMGiB--
		case "device":
			reduced.VRAMGiB--
		}
		if plan.AssessResources(now, reduced, []plan.WorkLease{lease}).Fits {
			t.Fatalf("unsupported reduced %s capacity admitted", name)
		}
	}
	id := func(kind artifact.Kind, label string) artifact.ID {
		return testutil.ArtifactID(t, kind, "media resources "+label)
	}
	fitness, err := runrecord.NewResourceFitness(runrecord.ResourceFitness{Scope: runrecord.ResourceScope{Surface: runrecord.SurfaceEvaluation, Workload: id(artifact.KindProfile, "workload"), Attempt: id(artifact.KindRun, "attempt")}, Measures: []runrecord.ResourceMeasure{{Metric: runrecord.ResourceWallNS, Value: 1}, {Metric: runrecord.ResourceHostToDeviceBytes, Value: 0}}})
	if err != nil {
		t.Fatal(err)
	}
	if value, known := fitness.Measure(runrecord.ResourceHostToDeviceBytes); !known || value != 0 {
		t.Fatal("observed zero transfer lost")
	}
	if _, known := fitness.Measure(runrecord.ResourcePeakHostBytes); known {
		t.Fatal("missing host peak became an observation")
	}
	grant, err := plan.NewExplorationGrant(id(artifact.KindEvidence, "proposer"), id(artifact.KindEvidence, "authority"), 2)
	if err != nil {
		t.Fatal(err)
	}
	charge, err := plan.NewExplorationCharge(grant.ID, id(artifact.KindEvidence, "experiment"), 1)
	if err != nil {
		t.Fatal(err)
	}
	content, err := grant.Content()
	if err != nil {
		t.Fatal(err)
	}
	replayed, err := plan.ParseExplorationGrant(content.Data)
	if err != nil {
		t.Fatal(err)
	}
	remaining, err := plan.ExplorationBalance(replayed, []plan.ExplorationCharge{charge})
	if err != nil || remaining != 1 {
		t.Fatalf("replay reset charged budget: remaining=%d err=%v", remaining, err)
	}
	decision, err := plan.AdmitExploration(replayed, []plan.ExplorationCharge{charge}, 2, false)
	if err != nil || decision.Admitted {
		t.Fatalf("over-budget continuation accepted: %+v %v", decision, err)
	}
	peak, err := processmeasure.SelfPeakWorkingSet()
	if err != nil || peak == 0 {
		t.Fatalf("process peak unavailable: %d %v", peak, err)
	}
	t.Logf("acceptance process lifetime peak working set=%d bytes; no generation or optimization acceptance", peak)
}
