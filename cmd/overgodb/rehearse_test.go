package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// TestARehearsalCopyOfTheStoreEndsWithItsRow holds the command's half of session-leftovers/rehearsal-copy: overgodb
// rehearse copies the store into the scratch folder of a row the working plan holds, tmp/<row>/store, as a store that
// opens at the same head, carries no backup seal and is not the store's own name, so the leftovers' walk does not take
// it for a backup; a row the plan does not hold is refused, as its copy would be finished as soon as it was made. The
// listing's half is the test of this name in cmd/loop.
func TestARehearsalCopyOfTheStoreEndsWithItsRow(t *testing.T) {
	t.Parallel()
	base := t.TempDir()
	checkout, live := filepath.Join(base, "checkout"), filepath.Join(base, "store")
	testcheck.NoError(t, os.MkdirAll(filepath.Join(checkout, "docs"), 0o755))
	testcheck.NoError(t, plan.Save(filepath.Join(checkout, filepath.FromSlash(plan.Path)), plan.Plan{Campaign: "fixture", Items: []plan.Item{{
		ID: "rehearsed-row", Title: "a row of the fixture", Status: plan.StatusOpen, Steps: []plan.Step{{ID: "do", Title: "its step", Status: plan.StatusOpen, Verify: "go version"}},
	}}}))
	store := testcheck.Must(overgodb.Create(live))(t)
	contents := []artifact.Content{textContent(t, "inline"), textContent(t, strings.Repeat("held in a blob ", 1<<10))}
	testcheck.Must(store.Commit(t.Context(), artifact.Batch{Key: "rehearsal/fixture", Contents: contents}))(t)
	head, _ := store.Head()
	testcheck.NoError(t, store.Close())

	err := runSubcommand([]string{"rehearse", "-repo", live, "-row", "rehearsed-row", "-checkout", checkout})
	if err != nil {
		t.Fatalf("a rehearsal copy for an open row was refused: %v", err)
	}
	copied := filepath.Join(checkout, "tmp", "rehearsed-row", "store")
	opened := testcheck.Must(overgodb.OpenReadOnly(copied))(t)
	if got, _ := opened.Head(); got != head {
		t.Errorf("the rehearsal copy opens at %s; want the store's head %s", got, head)
	}
	testcheck.NoError(t, opened.Close())
	if _, err := overgodb.ReadBackupSeal(copied); err == nil {
		t.Error("the rehearsal copy carries a backup seal, which the leftovers' walk never deletes")
	}
	if err := runSubcommand([]string{"rehearse", "-repo", live, "-row", "no-such-row", "-checkout", checkout}); err == nil {
		t.Error("a rehearsal copy for a row the plan does not hold was made")
	}
	if _, err := os.Stat(filepath.Join(checkout, "tmp", "no-such-row")); !os.IsNotExist(err) {
		t.Errorf("the refused copy left its folder (%v)", err)
	}
}
