//go:build windows

package heldfile

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAFileIsHeldWhileAnOwnerHoldsItsDirectory holds the set to its owners (held-release): a file named with no owner is
// hashed and kept by no one, so a writer replaces it; named while a session owns its directory it is held, a writer
// refused, and named again it opens nothing; a request owning every file holds one outside the session's directory
// until it is served, the session's own staying held; and the session's close lets its files go, which are then read
// again under their new bytes. adapter-hold held every file it named until a session under its directory closed, so a
// request that opened none, and a test that read a model's adapter, left it held for the process's life (SQA-454).
func TestAFileIsHeldWhileAnOwnerHoldsItsDirectory(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	adapter, outside := filepath.Join(root, "model", "adapters", "a.safetensors"), filepath.Join(root, "other", "b.safetensors")
	testutil.WriteTextFile(t, root, "model/adapters/a.safetensors", "the adapter's weights", 0o600)
	testutil.WriteTextFile(t, root, "other/b.safetensors", "another model's weights", 0o600)
	replace := func(file, bytes string) error {
		testutil.WriteTextFile(t, filepath.Dir(file), "written", bytes, 0o600)
		return os.Rename(filepath.Join(filepath.Dir(file), "written"), file)
	}
	opens := 0
	set := &Set{Open: func(path string) (*os.File, error) {
		opens++
		return open(path)
	}}
	unowned := testcheck.Must(set.Identity(adapter))(t)
	if err := replace(adapter, "weights written with no owner"); err != nil {
		t.Fatalf("a file named with no owner could not be replaced: %v; want it held by no one", err)
	}
	session := testcheck.Must(set.Own(filepath.Join(root, "model")))(t)
	held := testcheck.Must(set.Identity(adapter))(t)
	if held == unowned {
		t.Error("a file replaced after a naming with no owner is named as before; want its new bytes read")
	}
	if replace(adapter, "weights written while a session owns them") == nil {
		t.Fatal("a file its session's directory holds was replaced; want the writer refused")
	}
	before := opens
	if again := testcheck.Must(set.Identity(adapter))(t); again != held || opens != before {
		t.Errorf("a held file named again opened %d files more and is named %v; want nothing opened and %v", opens-before, again, held)
	}
	request := testcheck.Must(set.Own(""))(t)
	testcheck.Must(set.Identity(outside))(t)
	if replace(outside, "weights written while a request is served") == nil {
		t.Error("a file a request named was replaced while it was served; want the writer refused")
	}
	testcheck.NoError(t, request())
	if err := replace(outside, "weights written once the request was served"); err != nil {
		t.Errorf("a file a served request named stayed held: %v; want it let go", err)
	}
	if replace(adapter, "weights written beside the session after the request") == nil {
		t.Error("the session's file was replaced once a request beside it was served; want it held while the session lives")
	}
	testcheck.NoError(t, session())
	if err := replace(adapter, "weights written after the session closed"); err != nil {
		t.Fatalf("the session's file stayed held after its close: %v; want it let go", err)
	}
	if testcheck.Must(set.Identity(adapter))(t) == held {
		t.Error("a file replaced after its session closed is named as before; want its new bytes read")
	}
}
