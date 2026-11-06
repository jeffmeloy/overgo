package authoritylock

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

func TestAcquireSerializesPlanAndGateMutation(t *testing.T) {
	repository := t.TempDir()
	first := testcheck.Must(Acquire(repository))(t)
	if second, err := Acquire(repository); err == nil {
		_ = second.Close()
		t.Fatal("concurrent authority mutation acquired the shared lock")
	}
	testcheck.NoError(t, first.Close())
	second, err := Acquire(repository)
	if err != nil {
		t.Fatalf("reacquire authority lock: %v", err)
	}
	testcheck.NoError(t, second.Close())
}

func TestAcquireDoesNotInventActiveOwner(t *testing.T) {
	repository := t.TempDir()
	testcheck.NoError(t, os.MkdirAll(filepath.Join(repository, relativePath), 0o700))
	lock, err := Acquire(repository)
	if lock != nil {
		_ = lock.Close()
	}
	if err == nil || errors.Is(err, processlock.ErrBusy) || strings.Contains(err.Error(), "mutation is active") {
		t.Fatalf("invalid lock path reported an active owner: %v", err)
	}
}
