package authoritylock

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/processlock"
)

func TestAcquireSerializesPlanAndGateMutation(t *testing.T) {
	repository := t.TempDir()
	first, err := Acquire(repository)
	if err != nil {
		t.Fatal(err)
	}
	if second, err := Acquire(repository); err == nil {
		_ = second.Close()
		t.Fatal("concurrent authority mutation acquired the shared lock")
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	second, err := Acquire(repository)
	if err != nil {
		t.Fatalf("reacquire authority lock: %v", err)
	}
	if err := second.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestAcquireDoesNotInventActiveOwner(t *testing.T) {
	repository := t.TempDir()
	if err := os.MkdirAll(filepath.Join(repository, relativePath), authorityDirectoryMode); err != nil {
		t.Fatal(err)
	}
	lock, err := Acquire(repository)
	if lock != nil {
		_ = lock.Close()
	}
	if err == nil || errors.Is(err, processlock.ErrBusy) || strings.Contains(err.Error(), "mutation is active") {
		t.Fatalf("invalid lock path reported an active owner: %v", err)
	}
}
