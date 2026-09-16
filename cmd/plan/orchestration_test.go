package main

import (
	"bytes"
	"encoding/json"
	"os"
	"overgo/internal/worklease"
	"path/filepath"
	"testing"
)

func writeWorkLeaseFixture(t *testing.T, root string) {
	t.Helper()
	data, err := json.Marshal(worklease.Lease{
		Version: 1,
		Task:    "automation/do", Worktree: "C:/repo/automation", Branch: "codex/automation", Role: "developer",
		TargetHead: "0123456789abcdef0123456789abcdef01234567", ConflictsWith: []string{},
		Resources: worklease.Resources{CPUThreads: 8, HostRAMGiB: 16, VRAMGiB: 12}, ExpiresAt: "2099-01-01T00:00:00Z",
	})
	if err != nil {
		t.Fatal(err)
	}
	input := filepath.Join(root, "lease.json")
	if err := os.WriteFile(input, data, 0o644); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := recordWorkLease(root, input, &output); err != nil {
		t.Fatal(err)
	}
	if output.Len() == 0 {
		t.Fatal("record work lease produced no identity")
	}
}

func TestWorkLeaseRecord(t *testing.T) {
	root := t.TempDir()
	writeWorkLeaseFixture(t, root)
	var output bytes.Buffer
	if err := printLeaseReport(root, worklease.Resources{CPUThreads: 16}, &output); err != nil {
		t.Fatal(err)
	}
	var report leaseReport
	if err := json.Unmarshal(output.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if len(report.Leases) != 1 || report.Advisory.Reserved.CPUThreads != 8 || !report.Advisory.Fits {
		t.Fatalf("lease report = %+v", report)
	}
}
