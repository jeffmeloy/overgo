package processlock

import (
	"path/filepath"
	"testing"
)

func TestAcquireIsExclusiveAndReleasedByClose(t *testing.T) {
	path := filepath.Join(t.TempDir(), "process.lock")
	first, err := Acquire(path, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	if second, err := Acquire(path, 0o600); err == nil {
		_ = second.Close()
		t.Fatal("second process lock acquisition succeeded")
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
