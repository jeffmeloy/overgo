package main

import (
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestARehearsalCopyOfTheStoreEndsWithItsRow holds the listing's half of session-leftovers/rehearsal-copy: a rehearsal
// copy of the store, a clone in its row's scratch folder as overgodb rehearse makes it, is held while the working plan
// holds the row and finished once the row leaves, and the walk does not hold it as a store; a sealed backup in another
// finished scratch folder is never deleted. The command's half is the test of this name in cmd/overgodb.
func TestARehearsalCopyOfTheStoreEndsWithItsRow(t *testing.T) {
	t.Parallel()
	root := testcheck.Must(filepath.EvalSymlinks(t.TempDir()))(t)
	at := func(slash string) string { return filepath.Join(root, filepath.FromSlash(slash)) }
	repo := madeFolder(t, at("repo"))
	gitFixture(t, repo, "init", "-q", "-b", "master")
	gitFixture(t, repo, "config", "core.autocrlf", "false")
	testutil.WriteTextFile(t, root, "repo/.gitignore", "/tmp/\n")
	savePlan := func(rows ...string) {
		t.Helper()
		document := plan.Plan{Campaign: "hygiene", Lane: "master-lead"}
		for _, row := range rows {
			document.Items = append(document.Items, plan.Item{ID: row, Title: "a row of the fixture", Status: plan.StatusOpen, Steps: []plan.Step{{ID: "do", Title: "its step", Status: plan.StatusOpen, Verify: "go version"}}})
		}
		testcheck.NoError(t, plan.Save(filepath.Join(repo, filepath.FromSlash(plan.Path)), document))
	}
	madeFolder(t, at("repo/docs"))
	savePlan("other-row")
	gitFixture(t, repo, "add", "-A")
	gitFixture(t, repo, "commit", "-q", "-m", "base")
	savePlan("other-row", "rehearsed-row")

	store := testcheck.Must(overgodb.Create(at("data/overgodb-store")))(t)
	data := []byte(strings.Repeat("held in a blob ", 1<<10))
	id := testcheck.Must(artifact.IdentifyBytes(artifact.KindEvidence, data))(t)
	content := artifact.Content{Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(data)), MediaType: "text/plain"}, Data: data}
	testcheck.Must(store.Commit(t.Context(), artifact.Batch{Key: "rehearsal/fixture", Contents: []artifact.Content{content}}))(t)
	testcheck.Must(store.Backup(t.Context(), at("repo/tmp/sealed-row/backup")))(t)
	testcheck.NoError(t, store.Close())
	madeFolder(t, at("repo/tmp/rehearsed-row"))
	testcheck.Must(overgodb.CloneStore(at("data/overgodb-store"), at("repo/tmp/rehearsed-row/store")))(t)

	h := hygiene{
		checkout: repo, data: at("data"), sessions: madeFolder(t, at("sessions")), temporary: madeFolder(t, at("temporary")), cache: madeFolder(t, at("cache")),
		executable: at("temporary/loop.exe"),
		stores:     func() ([]leftover, error) { return nil, nil },
		registered: func() ([]string, error) { return nil, nil },
		processes:  func() ([]string, error) { return nil, nil },
	}
	scratch := func() map[string]string {
		t.Helper()
		states := map[string]string{}
		for _, entry := range testcheck.Must(listLeftovers(h, true))(t) {
			if entry.place == "tmp" {
				states[filepath.Base(entry.path)] = entry.state
			}
		}
		return states
	}
	if states := scratch(); states["rehearsed-row"] != stateHeld || states["sealed-row"] == stateFinished {
		t.Fatalf("with the row open the scratch folders are %v; want the rehearsal copy's held and the sealed backup's not finished", states)
	}
	savePlan("other-row")
	if states := scratch(); states["rehearsed-row"] != stateFinished || states["sealed-row"] == stateFinished {
		t.Fatalf("with the row gone the scratch folders are %v; want the rehearsal copy's finished and the sealed backup's never", states)
	}
}
