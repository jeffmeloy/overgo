package processcontrol

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/processlock"
)

func TestCampaignProcessLifecycle(t *testing.T) {
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
	line, err := bufio.NewReader(output).ReadString('\n')
	if err != nil {
		t.Fatal(err)
	}
	_, token, _ := strings.Cut(strings.TrimSpace(line), "=")
	t.Setenv(CampaignEnvironment, token)
	if err := RequireCampaign(root, "process-worker"); err != nil {
		t.Fatal(err)
	}
	if err := child.Terminate(); err != nil {
		t.Fatal(err)
	}
	_ = inputWriter.Close()
	if _, err := child.Wait(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := RequireCampaign(root, "process-worker"); err == nil {
		t.Fatal("dead supervisor admitted its former session")
	}
}

func TestCampaignAdmission(t *testing.T) {
	root := t.TempDir()
	if err := RequireCampaign(root, "worker"); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "docs/loop.json"), []byte("{}"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := RequireCampaign(root, "worker"); err == nil {
		t.Fatal("configured campaign admitted without supervisor")
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
	t.Setenv(CampaignEnvironment, "forged-session")
	if err := RequireCampaign(root, "worker"); err == nil {
		t.Fatal("environment alone admitted")
	}
	_, token, _ := strings.Cut(owner.Environment(), "=")
	t.Setenv(CampaignEnvironment, token)
	if err := RequireCampaign(root, "other-worker"); err == nil {
		t.Fatal("foreign worker admitted")
	}
	if err := RequireCampaign(root, "worker"); err != nil {
		t.Fatal(err)
	}
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	if err := RequireCampaign(root, "worker"); err == nil {
		t.Fatal("stale session admitted after lock release")
	}
	replacement, err := BeginCampaign(root, "worker")
	if err != nil {
		t.Fatal(err)
	}
	defer replacement.Close()
	if err := RequireCampaign(root, "worker"); err == nil {
		t.Fatal("previous session admitted by new supervisor")
	}
	_, token, _ = strings.Cut(replacement.Environment(), "=")
	t.Setenv(CampaignEnvironment, token)
	if err := RequireCampaign(root, "worker"); err != nil {
		t.Fatal(err)
	}
}
