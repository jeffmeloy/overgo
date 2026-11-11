package main

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/fsatomic"
	"overgo/internal/testcheck"
)

// TestRemovingASessionFolderLeavesLinkedDataWhole holds a removal of a finished entry, a session folder among them, to
// names and never data held elsewhere (session-leftovers/linked-data; the owner's word of 2026-11-09: be careful not to
// delete links to data in other places like models). A folder holding a link a recursive delete could follow, to a
// models folder, is not finished. A folder holding second names of models, one of them read-only, is finished, and its
// removal leaves each model's own name whole, its bytes and its read-only attribute: on Windows the attribute belongs
// to the file, not the name, and a removal that cleared it to delete the name would leave the model writable, which Go's
// removal, read here on 2026-11-11, does not.
func TestRemovingASessionFolderLeavesLinkedDataWhole(t *testing.T) {
	t.Parallel()
	models, sessions := t.TempDir(), t.TempDir()
	model, weights := filepath.Join(models, "model.gguf"), filepath.Join(models, "weights.bin")
	testcheck.NoError(t, os.WriteFile(model, []byte("read-only model"), 0o600))
	testcheck.NoError(t, os.WriteFile(weights, []byte("writable weights"), 0o600))
	testcheck.NoError(t, os.Chmod(model, 0o400))
	t.Cleanup(func() { _ = os.Chmod(model, 0o600) })
	weigh := func(folder string) leftover {
		t.Helper()
		return weighed(leftover{path: folder, action: actionDelete, state: stateFinished}, true)
	}

	linked := filepath.Join(sessions, "linked")
	testcheck.NoError(t, os.MkdirAll(linked, 0o700))
	if fixtureLink(models, filepath.Join(linked, "models")) {
		if entry := weigh(linked); entry.state == stateFinished {
			t.Errorf("a session folder holding a link to the models folder is finished; want it held by the link")
		}
	} else {
		t.Log("this machine makes no link: the link case is not read")
	}

	served := filepath.Join(sessions, "served", "run")
	testcheck.NoError(t, os.MkdirAll(served, 0o700))
	testcheck.NoError(t, os.Link(model, filepath.Join(served, "model.gguf")))
	testcheck.NoError(t, os.Link(weights, filepath.Join(served, "weights.bin")))
	entry := weigh(filepath.Dir(served))
	if entry.state != stateFinished {
		t.Fatalf("a session folder holding second names of models is %s by %q; want it finished", entry.state, entry.rule)
	}
	testcheck.NoError(t, removeEntry(hygiene{removeTree: fsatomic.RemoveTree}, entry.path))
	if _, err := os.Stat(entry.path); !os.IsNotExist(err) {
		t.Fatalf("the finished session folder is still there after its removal (%v)", err)
	}
	for name, want := range map[string]string{model: "read-only model", weights: "writable weights"} {
		if content, err := os.ReadFile(name); err != nil || string(content) != want {
			t.Errorf("after the folder's removal %s reads %q (%v); want it whole", name, content, err)
		}
	}
	if info, err := os.Stat(model); err != nil || info.Mode().Perm()&0o200 != 0 {
		t.Errorf("the read-only model is %v (%v) after its second name was removed; want it still read-only", info.Mode(), err)
	}
}
