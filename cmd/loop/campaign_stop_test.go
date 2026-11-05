package main

import (
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/loop"
	"overgo/internal/plan"
	"overgo/internal/processcontrol"
	"overgo/internal/testcheck"
	"overgo/internal/worklease"
)

func TestCampaignActiveWorkerStop(t *testing.T) {
	const helperRoot = "OVERGO_CAMPAIGN_STOP_TEST_ROOT"
	if root := os.Getenv(helperRoot); root != "" {
		_, err := plan.RecordStop(t.Context(), plan.ControlEvent{Kind: plan.ControlStop, Lane: worklease.UnassignedRole, Worker: "operator", Worktree: filepath.ToSlash(root), Mode: plan.ExecutionAll, CodeCommit: strings.Repeat("a", 40), ReasonCode: "requested", Detail: "active worker fixture"})
		testcheck.NoError(t, err)
		// Stay alive in an OS wait: only supervisor cancellation ends this child.
		listener, err := net.Listen("tcp", "127.0.0.1:0")
		testcheck.NoError(t, err)
		defer listener.Close()
		_, _ = listener.Accept()
		return
	}
	root := t.TempDir()
	t.Chdir(root)
	campaign, err := processcontrol.BeginCampaign(root, "worker")
	testcheck.NoError(t, err)
	defer campaign.Close()
	t.Setenv(helperRoot, root)
	world := &execWorld{campaign: campaign, config: config{Worker: []string{os.Args[0], "-test.run=^TestCampaignActiveWorkerStop$"}, WorkerTimeoutMinutes: 1}}
	_, err = world.RunWorker(loop.Step{Item: "fixture", ID: "do"}, "fixture", "")
	if err != nil || !world.Paused() {
		t.Fatalf("active worker stop: %v", err)
	}
}
