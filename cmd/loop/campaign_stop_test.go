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
		testcheck.Must(plan.RecordStop(t.Context(), plan.ControlEvent{Kind: plan.ControlStop, Lane: worklease.UnassignedRole, Worker: "operator", Worktree: filepath.ToSlash(root), Mode: plan.ExecutionAll, CodeCommit: strings.Repeat("a", 40), ReasonCode: "requested", Detail: "active worker fixture"}))(t)
		// Stay alive in an OS wait: only supervisor cancellation ends this child.
		listener := testcheck.Must(net.Listen("tcp", "127.0.0.1:0"))(t)
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
	world := &execWorld{campaign: campaign, config: config{Worker: []string{os.Args[0], "-test.run=^TestCampaignActiveWorkerStop$"}}}
	_, err = world.RunWorker(loop.Step{Item: "fixture", ID: "do"}, "fixture", "")
	if err != nil || !world.Paused() {
		t.Fatalf("active worker stop: %v", err)
	}
}
