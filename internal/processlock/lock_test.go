package processlock

import (
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestAcquireIsExclusiveAndReleasedByClose(t *testing.T) {
	path := filepath.Join(t.TempDir(), "process.lock")
	first, err := Acquire(path, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	if second, err := Acquire(path, 0o600); !errors.Is(err, ErrBusy) {
		_ = second.Close()
		t.Fatalf("competing acquisition = %v; want OS-proven contention", err)
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	second, err := Acquire(path, 0o600)
	if err != nil {
		t.Fatalf("reacquire after close: %v", err)
	}
	if err := second.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestAcquireRejectsFalseContention(t *testing.T) {
	path := filepath.Join(t.TempDir(), "process.lock")
	if err := os.WriteFile(path, []byte("stale owner metadata"), 0o600); err != nil {
		t.Fatal(err)
	}
	lock, err := Acquire(path, 0o600)
	if err != nil {
		t.Fatalf("unowned lock file blocked acquisition: %v", err)
	}
	if err := lock.Close(); err != nil {
		t.Fatal(err)
	}
	if lock, err := Acquire(t.TempDir(), 0o600); err == nil || errors.Is(err, ErrBusy) {
		_ = lock.Close()
		t.Fatalf("invalid lock path classified as contention: %v", err)
	}
}

func TestAcquireReleasedAfterProcessDeath(t *testing.T) {
	const helperPath = "OVERGO_TEST_LOCK_PROCESS_PATH"
	if path := os.Getenv(helperPath); path != "" {
		lock, err := Acquire(path, 0o600)
		if err != nil {
			t.Fatal(err)
		}
		defer lock.Close()
		fmt.Println("locked")
		_, _ = io.Copy(io.Discard, os.Stdin)
		return
	}
	path := filepath.Join(t.TempDir(), "process.lock")
	child := exec.Command(os.Args[0], "-test.run=^TestAcquireReleasedAfterProcessDeath$", "-test.count=1")
	child.Env = append(os.Environ(), helperPath+"="+path)
	input, err := child.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	defer input.Close()
	output, err := child.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	child.Stderr = os.Stderr
	if err := child.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = child.Process.Kill(); _ = child.Wait() })
	var ready string
	if _, err := fmt.Fscan(output, &ready); err != nil || ready != "locked" {
		t.Fatalf("child readiness = %q, %v", ready, err)
	}
	if lock, err := Acquire(path, 0o600); !errors.Is(err, ErrBusy) {
		_ = lock.Close()
		t.Fatalf("live writer contention = %v", err)
	}
	if err := child.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	_ = child.Wait()
	// The file remains. Only OS ownership determines whether reuse is safe.
	if _, err := os.Stat(path); err != nil {
		t.Fatal(err)
	}
	lock, err := Acquire(path, 0o600)
	if err != nil {
		t.Fatalf("dead writer blocked acquisition: %v", err)
	}
	if err := lock.Close(); err != nil {
		t.Fatal(err)
	}
}
