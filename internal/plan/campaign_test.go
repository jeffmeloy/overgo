package plan

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
)

func TestCampaignDispatchAdmission(t *testing.T) {
	document := Plan{Items: []Item{{ID: "row", Status: StatusOpen, Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}}}}}
	fixture := newCompletionFixture(t, document, "row", "do")
	owner, err := processcontrol.BeginCampaign(fixture.repository, "worker")
	if err != nil {
		t.Fatal(err)
	}
	defer owner.Close()
	t.Setenv(processcontrol.CampaignEnvironment, "forged")
	_, err = ResolveDispatch(t.Context(), fixture.repository, DispatchRequest{Acquire: true, Worker: "worker"})
	if err == nil || !strings.Contains(err.Error(), "campaign:") {
		t.Fatalf("dispatch admitted outside supervisor: %v", err)
	}
}

func TestCampaignCommitAdmission(t *testing.T) {
	root := t.TempDir()
	store, err := overgodb.Open(filepath.Join(root, "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	owner, err := processcontrol.BeginCampaign(root, "worker")
	if err != nil {
		t.Fatal(err)
	}
	defer owner.Close()
	t.Setenv(AutomationWorkerEnvironment, "worker")
	t.Setenv(processcontrol.CampaignEnvironment, "forged")
	_, _, _, err = RequireDispatch(t.Context(), store, Plan{}, CompletionAuthority{}, root, "", "row/do")
	if err == nil || !strings.Contains(err.Error(), "campaign:") {
		t.Fatalf("commit admitted outside supervisor: %v", err)
	}
	// Changing execution mode cannot turn off the worktree's supervision.
	t.Setenv(AutomationModeEnvironment, ExecutionInteractive)
	_, _, _, err = RequireDispatch(t.Context(), store, Plan{}, CompletionAuthority{}, root, "", "row/do")
	if err == nil || !strings.Contains(err.Error(), "campaign:") {
		t.Fatalf("interactive mode bypass: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "tmp/loop_supervisor.json"), []byte("{invalid"), 0o600); err != nil {
		t.Fatal(err)
	}
	_, _, _, err = RequireDispatch(t.Context(), store, Plan{}, CompletionAuthority{}, root, "", "row/do")
	if err == nil || !strings.Contains(err.Error(), "campaign:") {
		t.Fatalf("corrupt owner admitted: %v", err)
	}
}
