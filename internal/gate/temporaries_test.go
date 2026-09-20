package gate

import (
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"
)

// TestGateSweepsOrphanedTemporaries holds the sweep to removing what a killed
// gate left and nothing else: the root of a process that has exited goes,
// with the candidate inside it; the root of this live process stays, and so
// does a neighbour whose name is no process id.
func TestGateSweepsOrphanedTemporaries(t *testing.T) {
	temporary := t.TempDir()
	for _, name := range []string{"TMP", "TEMP", "TMPDIR"} {
		t.Setenv(name, temporary)
	}
	repository := t.TempDir()
	runGitFixture(t, repository, "init", "-q")

	exited := exec.Command(os.Args[0], "-test.run=^$")
	if err := exited.Run(); err != nil {
		t.Fatal(err)
	}
	orphan := filepath.Join(gateTemporaryRoot(exited.Process.Pid), "acceptance-1", "candidate")
	foreign := filepath.Join(filepath.Dir(gateTemporaryRoot(os.Getpid())), "not-a-process")
	for _, directory := range []string{orphan, foreign} {
		if err := os.MkdirAll(directory, gatePrivateDirectoryMode); err != nil {
			t.Fatal(err)
		}
	}
	live, err := gateTemporary("index-*")
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(filepath.Dir(live)) != strconv.Itoa(os.Getpid()) {
		t.Fatalf("temporary %s is not under this process's root", live)
	}

	if err := sweepOrphanedTemporaries(repository); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(gateTemporaryRoot(exited.Process.Pid)); !os.IsNotExist(err) {
		t.Fatalf("the exited process's root survived the sweep: %v", err)
	}
	for _, kept := range []string{live, foreign} {
		if _, err := os.Stat(kept); err != nil {
			t.Fatalf("the sweep removed %s: %v", kept, err)
		}
	}
}
