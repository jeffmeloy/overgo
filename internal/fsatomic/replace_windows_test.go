//go:build windows

package fsatomic

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"

	"overgo/internal/testcheck"
)

func TestReplaceRequestsWriteThrough(t *testing.T) {
	sentinel := errors.New("move stopped")
	var flags uint32
	err := replace("source", "target", func(_, _ *uint16, observed uint32) error {
		flags = observed
		return sentinel
	})
	if !errors.Is(err, sentinel) {
		t.Fatalf("replace error = %v, want sentinel", err)
	}
	if flags&moveFileReplaceExisting == 0 || flags&moveFileWriteThrough == 0 {
		t.Fatalf("MoveFileEx flags = %#x, want replace + write-through", flags)
	}
}

func TestReplaceSupportsExtendedLengthPath(t *testing.T) {
	directory := t.TempDir()
	for range 3 {
		directory = filepath.Join(directory, strings.Repeat("segment", 14))
	}
	testcheck.NoError(t, os.MkdirAll(directory, 0o700))
	source := filepath.Join(directory, "source")
	target := filepath.Join(directory, "target")
	testcheck.NoError(t, os.WriteFile(source, []byte("after"), 0o600))
	testcheck.NoError(t, Replace(source, target))
	got := testcheck.Must(os.ReadFile(target))(t)
	if string(got) != "after" {
		t.Fatalf("target = %q, want after", got)
	}
}

func TestReplaceRejectsEmbeddedNUL(t *testing.T) {
	err := Replace("source\x00suffix", "target")
	if err == nil || !errors.Is(err, syscall.EINVAL) {
		t.Fatalf("embedded-NUL error = %v", err)
	}
}

func TestRemoveCleansInterruptedTombstone(t *testing.T) {
	path := filepath.Join(t.TempDir(), "document")
	tombstone := removalTombstone(path)
	testcheck.NoError(t, os.WriteFile(tombstone, []byte("retired"), 0o600))
	testcheck.NoError(t, Remove(path))
	if _, err := os.Stat(tombstone); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("tombstone state = %v, want absent", err)
	}
}

func TestRemoveReusesStaleTombstone(t *testing.T) {
	path := filepath.Join(t.TempDir(), "document")
	tombstone := removalTombstone(path)
	testcheck.NoError(t, os.WriteFile(tombstone, []byte("prior"), 0o600))
	testcheck.NoError(t, os.WriteFile(path, []byte("current"), 0o600))
	testcheck.NoError(t, Remove(path))
	for _, candidate := range []string{path, tombstone} {
		if _, err := os.Stat(candidate); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("removed state for %s = %v, want absent", candidate, err)
		}
	}
}
