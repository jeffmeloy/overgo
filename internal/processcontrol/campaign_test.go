package processcontrol

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"testing"

	"overgo/internal/processlock"
)

func TestCampaignProcessLifecycle(t *testing.T) {
	t.Parallel()
	const helperRoot = "OVERGO_CAMPAIGN_TEST_ROOT"
	if root := os.Getenv(helperRoot); root != "" {
		owner, err := BeginCampaign(root, "process-worker")
		if err != nil {
			t.Fatal(err)
		}
		defer owner.Close()
		fmt.Println(owner.Environment())
		_, _ = io.Copy(io.Discard, os.Stdin)
		return
	}
	root := t.TempDir()
	input, inputWriter := io.Pipe()
	defer inputWriter.Close()
	output, outputWriter := io.Pipe()
	defer output.Close()
	defer outputWriter.Close()
	child, err := Start(t.Context(), Command{Path: os.Args[0], Args: []string{"-test.run=^TestCampaignProcessLifecycle$"}, Env: append(os.Environ(), helperRoot+"="+root), Stdin: input, Stdout: outputWriter, Stderr: os.Stderr})
	if err != nil {
		t.Fatal(err)
	}
	defer child.Terminate()
	if _, err := bufio.NewReader(output).ReadString('\n'); err != nil {
		t.Fatal(err)
	}
	if running, err := CampaignRunning(root); err != nil || !running {
		t.Fatalf("live supervisor running = %t, %v", running, err)
	}
	if err := child.Terminate(); err != nil {
		t.Fatal(err)
	}
	_ = inputWriter.Close()
	if _, err := child.Wait(t.Context()); err != nil {
		t.Fatal(err)
	}
	if running, err := CampaignRunning(root); err != nil || running {
		t.Fatalf("dead supervisor running = %t, %v", running, err)
	}
}

func TestCampaignRunningFollowsOwnerLock(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	running := func() bool {
		t.Helper()
		live, err := CampaignRunning(root)
		if err != nil {
			t.Fatal(err)
		}
		return live
	}
	if running() {
		t.Fatal("worktree without a supervisor reported a campaign")
	}
	owner, err := BeginCampaign(root, "worker")
	if err != nil {
		t.Fatal(err)
	}
	defer owner.Close()
	if duplicate, err := BeginCampaign(root, "worker"); !errors.Is(err, processlock.ErrBusy) {
		if duplicate != nil {
			_ = duplicate.Close()
		}
		t.Fatalf("duplicate supervisor: %v", err)
	}
	if !running() {
		t.Fatal("live supervisor not reported")
	}
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	if running() {
		t.Fatal("supervisor reported after lock release")
	}
	replacement, err := BeginCampaign(root, "worker")
	if err != nil {
		t.Fatal(err)
	}
	defer replacement.Close()
	if !running() {
		t.Fatal("replacement supervisor not reported")
	}
}
