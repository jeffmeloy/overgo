package main

import (
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/loop"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/processcontrol"
	"overgo/internal/worklease"
)

func TestCampaignActiveWorkerStop(t *testing.T) {
	const helperRoot = "OVERGO_CAMPAIGN_STOP_TEST_ROOT"
	if root := os.Getenv(helperRoot); root != "" {
		writer, err := overgodb.Open(filepath.Join(root, "overgodb-store"))
		if err != nil {
			t.Fatal(err)
		}
		defer writer.Close()
		_, err = plan.RecordControlEvent(t.Context(), writer, plan.ControlEvent{Kind: plan.ControlStop, Lane: worklease.UnassignedRole, Worker: "operator", Worktree: filepath.ToSlash(root), Mode: plan.ExecutionAll, CodeCommit: strings.Repeat("a", 40), ReasonCode: "user-stop", Detail: "active worker fixture"})
		if err != nil {
			t.Fatal(err)
		}
		// Stay alive in an OS wait: only supervisor cancellation ends this child.
		listener, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		defer listener.Close()
		_, _ = listener.Accept()
		return
	}
	root := t.TempDir()
	t.Chdir(root)
	writer, err := overgodb.Open(filepath.Join(root, "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	reader, err := overgodb.OpenReadOnly(filepath.Join(root, "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	campaign, err := processcontrol.BeginCampaign(root, "worker")
	if err != nil {
		t.Fatal(err)
	}
	defer campaign.Close()
	t.Setenv(helperRoot, root)
	world := &execWorld{stopStore: reader, campaign: campaign, config: config{Worker: []string{os.Args[0], "-test.run=^TestCampaignActiveWorkerStop$"}, WorkerTimeoutMinutes: 1}}
	_, err = world.RunWorker(loop.Step{Item: "fixture", ID: "do"}, "fixture", "")
	if err != nil || !world.Paused() {
		t.Fatalf("active worker stop: %v", err)
	}
}
