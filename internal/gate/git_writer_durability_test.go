package gate

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"overgo/internal/gitauthority"
)

func TestGateGitWriterCommandOverridesWeakRepositoryFsyncConfiguration(t *testing.T) {
	t.Parallel()
	repository := t.TempDir()
	runGitFixture(t, repository, "init", "-q")
	runGitFixture(t, repository, "config", "core.fsync", "none")
	runGitFixture(t, repository, "config", "core.fsyncMethod", "writeout-only")

	command := newGateGitWriterCommand(repository, nil, "config", "--get", "core.fsync")
	for _, entry := range command.Env {
		name, _, _ := strings.Cut(entry, "=")
		if strings.EqualFold(name, "GIT_OPTIONAL_LOCKS") {
			t.Fatalf("gate Git writer unexpectedly inherited reader-only policy: %q", entry)
		}
	}
	wantArguments := []string{
		"git", "--no-replace-objects",
		"-c", "core.fsync=all",
		"-c", "core.fsyncMethod=fsync",
		"config", "--get", "core.fsync",
	}
	if !reflect.DeepEqual(command.Args, wantArguments) {
		t.Fatalf("gate Git writer child arguments = %q, want %q", command.Args, wantArguments)
	}
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("inspect effective gate writer fsync: %v: %s", err, output)
	}
	if got := strings.TrimSpace(string(output)); got != "all" {
		t.Fatalf("effective gate writer core.fsync = %q, want all", got)
	}

	method := newGateGitWriterCommand(repository, nil, "config", "--get", "core.fsyncMethod")
	output, err = method.CombinedOutput()
	if err != nil {
		t.Fatalf("inspect effective gate writer fsync method: %v: %s", err, output)
	}
	if got := strings.TrimSpace(string(output)); got != "fsync" {
		t.Fatalf("effective gate writer core.fsyncMethod = %q, want fsync", got)
	}
}

func TestGateGitReaderStatusDoesNotRefreshCapturedIndex(t *testing.T) {
	t.Parallel()
	repository := t.TempDir()
	runGitFixture(t, repository, "init", "-q")
	runGitFixture(t, repository, "config", "user.email", "gate@example.invalid")
	runGitFixture(t, repository, "config", "user.name", "Gate Test")
	tracked := filepath.Join(repository, "tracked.txt")
	if err := os.WriteFile(tracked, []byte("unchanged\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repository, "add", "tracked.txt")
	runGitFixture(t, repository, "commit", "-q", "-m", "base")

	index := filepath.Join(repository, ".git", "index")
	initial, err := os.ReadFile(index)
	if err != nil {
		t.Fatal(err)
	}
	changedTime := time.Now().Add(24 * time.Hour).Truncate(time.Second)
	if err := os.Chtimes(tracked, changedTime, changedTime); err != nil {
		t.Fatal(err)
	}
	control := exec.Command("git", "status", "--porcelain=v1", "--untracked-files=no")
	control.Dir = repository
	control.Env = append(gitauthority.RepositoryEnvironment(), "GIT_OPTIONAL_LOCKS=1")
	if output, err := control.CombinedOutput(); err != nil {
		t.Fatalf("control git status: %v: %s", err, output)
	}
	refreshed, err := os.ReadFile(index)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(initial, refreshed) {
		t.Fatal("control git status did not refresh the captured index")
	}

	changedTime = changedTime.Add(24 * time.Hour)
	if err := os.Chtimes(tracked, changedTime, changedTime); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(index)
	if err != nil {
		t.Fatal(err)
	}
	beforeInfo, err := os.Stat(index)
	if err != nil {
		t.Fatal(err)
	}
	output, err := command(repository, "git", "status", "--porcelain=v1", "--untracked-files=no")
	if err != nil {
		t.Fatal(err)
	}
	if output != "" {
		t.Fatalf("gate reader status reported a content change: %q", output)
	}
	after, err := os.ReadFile(index)
	if err != nil {
		t.Fatal(err)
	}
	afterInfo, err := os.Stat(index)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(after, before) || afterInfo.Mode() != beforeInfo.Mode() ||
		!afterInfo.ModTime().Equal(beforeInfo.ModTime()) {
		t.Fatal("gate reader status refreshed the captured index")
	}
}

func TestGateGitWriterRequirementCoversOnlyWriterEntryModes(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name                                                      string
		reconcile, recordFailure, recoverInterrupted, admitReview bool
		watchdog, inspectPlan, merge, want                        bool
	}{
		{name: "default commit", want: true},
		{name: "merge", merge: true, want: true},
		{name: "reconcile", reconcile: true, want: true},
		{name: "recover interrupted", recoverInterrupted: true, want: true},
		{name: "inspect plan", inspectPlan: true, want: true},
		{name: "record failure", recordFailure: true, want: false},
		{name: "admit review", admitReview: true, want: false},
		{name: "watchdog", watchdog: true, want: false},
	} {
		t.Run(test.name, func(t *testing.T) {
			got := gateWritesGit(
				test.reconcile,
				test.recordFailure,
				test.recoverInterrupted,
				test.admitReview,
				test.watchdog,
				test.inspectPlan,
				test.merge,
			)
			if got != test.want {
				t.Fatalf("requires Git writer = %t, want %t", got, test.want)
			}
		})
	}
}
