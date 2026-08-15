package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/repodb"
)

func TestWorkLeaseRecord(t *testing.T) {
	root := t.TempDir()
	lease, err := plan.NewWorkLease(plan.WorkLease{
		Task: "automation/do", Worktree: "C:/repo/automation", Branch: "codex/automation", Role: "developer",
		TargetHead: "0123456789abcdef0123456789abcdef01234567", DependsOn: []string{}, ConflictsWith: []string{},
		Resources: plan.ResourceRequest{CPUThreads: 8, HostRAMGiB: 16, VRAMGiB: 12}, EvidenceLanes: []string{"go-test"}, ExpiresAt: "2099-01-01T00:00:00Z",
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := lease.Content()
	if err != nil {
		t.Fatal(err)
	}
	input := filepath.Join(root, "lease.json")
	if err := os.WriteFile(input, content.Data, 0o644); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := recordWorkLease(root, input, &output); err != nil {
		t.Fatal(err)
	}
	store, err := repodb.OpenReadOnly(filepath.Join(root, "repodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	current, ok, err := store.ResolveAlias(context.Background(), plan.WorkLeaseAlias(lease.Worktree))
	store.Close()
	if err != nil || !ok || current != lease.ID || output.Len() == 0 {
		t.Fatalf("recorded lease = (%s, %t, %v), output=%q", current, ok, err, output.String())
	}
}

func TestResourceAdvisoryReport(t *testing.T) {
	root := t.TempDir()
	lease, _ := plan.NewWorkLease(plan.WorkLease{
		Task: "automation/do", Worktree: "C:/repo/automation", Branch: "codex/automation", Role: "developer",
		TargetHead: "0123456789abcdef0123456789abcdef01234567", DependsOn: []string{}, ConflictsWith: []string{},
		Resources: plan.ResourceRequest{CPUThreads: 8, HostRAMGiB: 16}, EvidenceLanes: []string{"go-test"}, ExpiresAt: "2099-01-01T00:00:00Z",
	})
	batch, _ := plan.WorkLeaseBatch(lease, nil)
	store, err := repodb.Open(filepath.Join(root, "repodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(context.Background(), batch); err != nil {
		t.Fatal(err)
	}
	store.Close()
	var output bytes.Buffer
	if err := printLeaseReport(root, plan.ResourceCapacity{CPUThreads: 16}, &output); err != nil {
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
