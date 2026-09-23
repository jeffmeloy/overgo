//go:build windows

package fsatomic

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
)

const mappedReaderEnvironment = "OVERGO_FSATOMIC_MAPPED_READER"

// mapForReading maps path read-only through a handle that shares every
// access, the way a background git status maps a worktree file it hashes,
// and returns the release.
func mapForReading(t *testing.T, path string) func() {
	t.Helper()
	name, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		t.Fatal(err)
	}
	handle, err := syscall.CreateFile(name, syscall.GENERIC_READ,
		syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE, nil, syscall.OPEN_EXISTING, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	mapping, err := syscall.CreateFileMapping(handle, nil, syscall.PAGE_READONLY, 0, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	view, err := syscall.MapViewOfFile(mapping, syscall.FILE_MAP_READ, 0, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	return func() {
		_ = syscall.UnmapViewOfFile(view)
		_ = syscall.CloseHandle(mapping)
		_ = syscall.CloseHandle(handle)
	}
}

// TestWriteWaitsOnMappedReader replaces a destination another process has
// mapped: the replace finds that process through the Restart Manager, waits
// on it rather than on a timer, and completes once it exits. A holder the
// writer cannot wait on -- the writer's own process -- is refused by name.
func TestWriteWaitsOnMappedReader(t *testing.T) {
	if target := os.Getenv(mappedReaderEnvironment); target != "" {
		release := mapForReading(t, target)
		fmt.Println("mapped")
		_, _ = io.Copy(io.Discard, os.Stdin)
		release()
		return
	}
	directory := t.TempDir()
	target := filepath.Join(directory, "compatibility.json")
	if err := os.WriteFile(target, []byte("before"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Run("another process", func(t *testing.T) {
		source := filepath.Join(directory, "next.json")
		if err := os.WriteFile(source, []byte("after"), 0o600); err != nil {
			t.Fatal(err)
		}
		child := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestWriteWaitsOnMappedReader$")
		child.Env = append(os.Environ(), mappedReaderEnvironment+"="+target)
		stdin, err := child.StdinPipe()
		if err != nil {
			t.Fatal(err)
		}
		stdout, err := child.StdoutPipe()
		if err != nil {
			t.Fatal(err)
		}
		if err := child.Start(); err != nil {
			t.Fatal(err)
		}
		if line, err := bufio.NewReader(stdout).ReadString('\n'); err != nil || strings.TrimSpace(line) != "mapped" {
			t.Fatalf("the reader did not map the file: %q %v", line, err)
		}
		waiting := make(chan uint32, 1)
		waitingOnHolder = func(holder uint32) { waiting <- holder }
		defer func() { waitingOnHolder = func(uint32) {} }()
		done := make(chan error, 1)
		go func() { done <- Replace(source, target) }()
		select {
		case holder := <-waiting:
			if holder != uint32(child.Process.Pid) {
				t.Errorf("waited on process %d, not the reader %d", holder, child.Process.Pid)
			}
		case err := <-done:
			t.Fatalf("the replace finished while the reader held the file: %v", err)
		}
		if err := stdin.Close(); err != nil {
			t.Fatal(err)
		}
		if err := <-done; err != nil {
			t.Fatalf("the replace failed after the reader released the file: %v", err)
		}
		if err := child.Wait(); err != nil {
			t.Fatal(err)
		}
		if data, err := os.ReadFile(target); err != nil || string(data) != "after" {
			t.Fatalf("destination = %q %v, want the replacement", data, err)
		}
	})
	t.Run("this process", func(t *testing.T) {
		source := filepath.Join(directory, "again.json")
		if err := os.WriteFile(source, []byte("again"), 0o600); err != nil {
			t.Fatal(err)
		}
		release := mapForReading(t, target)
		defer release()
		err := Replace(source, target)
		if err == nil || !strings.Contains(err.Error(), fmt.Sprintf("process %d", os.Getpid())) {
			t.Fatalf("a holder that cannot be waited on was not named: %v", err)
		}
	})
}
