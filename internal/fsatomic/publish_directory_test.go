package fsatomic

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
)

// TestPublishDirectoryLeavesOnlyWhatCompleted holds a published directory to
// what its populate wrote, a name already taken to a refusal that leaves the
// holder's files alone, and a failed write to no directory and no stage.
func TestPublishDirectoryLeavesOnlyWhatCompleted(t *testing.T) {
	t.Parallel()
	parent := t.TempDir()
	write := func(stage string) error { return os.WriteFile(filepath.Join(stage, "file"), []byte("first"), 0o600) }

	target, err := PublishDirectory(parent, "taken", 0o755, write)
	if err != nil || target != filepath.Join(parent, "taken") {
		t.Fatalf("publish = %q, %v", target, err)
	}
	if _, err := PublishDirectory(parent, "taken", 0o755, func(stage string) error {
		return os.WriteFile(filepath.Join(stage, "file"), []byte("second"), 0o600)
	}); !errors.Is(err, fs.ErrExist) {
		t.Fatalf("a taken name was published again: %v", err)
	}
	if held, err := os.ReadFile(filepath.Join(target, "file")); err != nil || string(held) != "first" {
		t.Fatalf("the holder's file = %q, %v", held, err)
	}

	failed := errors.New("write failed")
	if _, err := PublishDirectory(parent, "failed", 0o755, func(stage string) error {
		return errors.Join(write(stage), failed)
	}); !errors.Is(err, failed) {
		t.Fatalf("a failed write = %v, want %v", err, failed)
	}
	entries, err := os.ReadDir(parent)
	if err != nil || len(entries) != 1 || entries[0].Name() != "taken" {
		t.Fatalf("the parent holds %v, want only the published directory: %v", entries, err)
	}
}

// TestPathComponentRefusesWhatIsNotOneName holds a user's name to one safe path
// component, which PublishDirectory and the training workspace share: no empty
// name, dot names, separators, or characters that split or hide a path.
func TestPathComponentRefusesWhatIsNotOneName(t *testing.T) {
	t.Parallel()
	for _, name := range []string{"", ".", "..", "a/b", "../escape", "a\x00b", "a\nb", "a\rb"} {
		if PathComponent(name) == nil {
			t.Errorf("%q was taken for one path component", name)
		}
	}
	for _, name := range []string{"run-1", "model.f16", "a b"} {
		if err := PathComponent(name); err != nil {
			t.Errorf("%q was refused: %v", name, err)
		}
	}
	if _, err := PublishDirectory(t.TempDir(), "../escape", 0o755, func(string) error { return nil }); err == nil {
		t.Error("a published name escaped its parent")
	}
}
