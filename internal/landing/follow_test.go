package landing

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/processlock"
)

// TestTrunkCheckoutFollowsWithItsOwnChangesKept holds the trunk's own
// checkout to following a commit made from another worktree with what the
// lead has uncommitted there kept.
//
// The lead has three things uncommitted: the working plan, in which it has
// retitled a row and reworded the reason of the first row's count, the line
// beside the count the commit changes, where a merge of the plan as text
// does not merge; a file the commit does not touch; and a file of its own
// that git does not hold. A worktree's commit, of the first of a row's two
// steps, lands. The checkout is then on it: it holds the commit's file, the
// lead's file and the lead's own file as they were, and its working plan is
// the lead's with the commit's change made to it, the step gone, the row's
// count less what the commit added, and the retitled row still retitled,
// where the plan the commit published has the row as it was. What is
// uncommitted there is the lead's three things and nothing else, and nothing
// is left beside the plan. The lead then commits from the checkout as it
// always has, its retitling and its files with its row.
//
// What cannot be kept refuses the commit before the trunk moves, by name,
// with the checkout as it was: an uncommitted change of the lead's to a file
// the commit changes, and a working plan the commit's row cannot be taken
// from, the lead having removed it.
func TestTrunkCheckoutFollowsWithItsOwnChangesKept(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	const packaged = "// Package p is the fixture.\npackage p\n"
	fixture := newRowFixture(t, map[string]string{"a.txt": "base\n", "docs/notes.txt": "base\n", "internal/p/p.go": packaged})
	fixture.git("branch", "-M", "trunk")
	// The repository's files are the bytes git holds, as this repository's are: a checkout rewrites none.
	fixture.git("config", "core.autocrlf", "false")
	grown := packageNodes(t, oneFunction)
	const spare = 3
	row := func(id string, steps ...string) plan.Item {
		item := plan.Item{ID: id, Title: id, Status: plan.StatusOpen, Budget: plan.Budget{Nodes: grown + spare, Reason: "the fixture's growth"}}
		for _, step := range steps {
			item.Steps = append(item.Steps, plan.Step{ID: step, Title: step, Status: plan.StatusOpen, Verify: "true"})
		}
		return item
	}
	fixture.commitPlan(plan.Plan{Campaign: "fixture", Items: []plan.Item{row("first", "do", "later"), row("second", "do"), row("third", "do")}})
	linked := &sliceFixture{t: t, repo: filepath.Join(t.TempDir(), "linked")}
	fixture.git("worktree", "add", "-q", "--detach", linked.repo, "HEAD")
	message := []byte("Change the fixture\n\nCause: the fixture needs a change.\nPredicted effect: none.\n")
	land := func(checkout *sliceFixture, reference string) (string, error) {
		return Run(ctx, Options{Repo: checkout.repo, Ref: reference, Message: message, Log: io.Discard})
	}
	holds := func(checkout *sliceFixture, name string) string {
		data, err := os.ReadFile(filepath.Join(checkout.repo, filepath.FromSlash(name)))
		if err != nil {
			return ""
		}
		return string(data)
	}
	// working edits the lead's working plan in place, uncommitted.
	working := func(edit func(*plan.Plan)) {
		t.Helper()
		document, err := plan.Load(fixture.planPath())
		if err != nil {
			t.Fatal(err)
		}
		edit(&document)
		if err := plan.Save(fixture.planPath(), document); err != nil {
			t.Fatal(err)
		}
	}
	trunk := func() string { return fixture.git("rev-parse", "refs/heads/trunk") }
	const fileOne, retitled, leads, notes = "internal/p/a.go", "second, as the lead has it", "the lead's\n", "docs/own.txt"

	// The lead's three things, uncommitted.
	working(func(document *plan.Plan) {
		document.Items[0].Budget.Reason, document.Items[1].Title = retitled, retitled
	})
	fixture.write("a.txt", leads)
	fixture.write(notes, leads)

	linked.write(fileOne, oneFunction)
	first, err := land(linked, "first/do")
	if err != nil || trunk() != first || fixture.git("rev-parse", "HEAD") != first {
		t.Fatalf("the worktree's commit = %q with the trunk at %s (%v), want it landed with the lead's changes uncommitted beside it", first, trunk(), err)
	}
	followed, err := plan.Load(fixture.planPath())
	if err != nil || len(followed.Items) != 3 || len(followed.Items[0].Steps) != 1 || followed.Items[0].Steps[0].ID != "later" || followed.Items[0].Budget.Nodes != spare || followed.Items[0].Budget.Reason != retitled || followed.Items[1].Title != retitled {
		t.Fatalf("the lead's working plan after the commit is %+v (%v); want the first row's step gone and its count %d, and the second row as the lead retitled it", followed.Items, err, spare)
	}
	if published := fixture.git("show", first+":"+plan.Path); strings.Contains(published, retitled) || !strings.Contains(published, `"later"`) {
		t.Fatalf("the commit published the plan %s, want the row as the trunk had it and not the lead's retitling", published)
	}
	if holds(fixture, fileOne) != oneFunction || holds(fixture, "a.txt") != leads || holds(fixture, notes) != leads {
		t.Fatalf("the checkout holds %q, %q and %q; want the commit's file and the lead's two as they were", holds(fixture, fileOne), holds(fixture, "a.txt"), holds(fixture, notes))
	}
	if uncommitted := fixture.git("status", "--porcelain", "--untracked-files=all"); uncommitted != "M a.txt\n M docs/plan.json\n?? "+notes {
		t.Fatalf("the checkout has uncommitted %q, want the lead's file, its plan and its own file and nothing else", uncommitted)
	}
	if _, err := os.Stat(filepath.Join(fixture.repo, processlock.StateDirectory, planAside)); !os.IsNotExist(err) {
		t.Fatalf("the lead's plan is still kept beside the checkout's (%v)", err)
	}
	// The lead commits from the checkout as it always has.
	third, err := land(fixture, "third/do")
	if err != nil || trunk() != third || fixture.git("status", "--porcelain") != "" || !strings.Contains(fixture.git("show", third+":"+plan.Path), retitled) {
		t.Fatalf("the lead's own commit = %q (%v) with uncommitted %q; want it landed with its retitling and its files", third, err, fixture.git("status", "--porcelain"))
	}

	// What cannot be kept refuses the commit before the trunk moves.
	fixture.write("internal/p/p.go", packaged+"\n// the lead's line\n")
	linked.write("internal/p/p.go", packaged+"\n// the worktree's line\n")
	if _, err := land(linked, "second/do"); err == nil || !strings.Contains(err.Error(), "uncommitted changes of its own to files second/do also changes: internal/p/p.go") {
		t.Fatalf("a commit that changes a file the lead has changed = %v, want it refused naming the file", err)
	}
	if trunk() != third || !strings.Contains(holds(fixture, "internal/p/p.go"), "the lead's line") || !strings.Contains(holds(linked, "internal/p/p.go"), "the worktree's line") {
		t.Fatalf("after the refusal the trunk is at %s, the lead's file %q and the worktree's %q; want the trunk unmoved and both as they were", trunk(), holds(fixture, "internal/p/p.go"), holds(linked, "internal/p/p.go"))
	}
	fixture.write("internal/p/p.go", packaged)
	working(func(document *plan.Plan) {
		document.Items = append(document.Items[:1:1], document.Items[len(document.Items):]...)
	})
	removed := holds(fixture, plan.Path)
	if _, err := land(linked, "second/do"); err == nil || !strings.Contains(err.Error(), "holds an edited plan that second/do cannot be applied to") {
		t.Fatalf("a commit of a row the lead's working plan no longer holds = %v, want it refused for the plan", err)
	}
	if trunk() != third || holds(fixture, plan.Path) != removed {
		t.Fatalf("after the refusal the trunk is at %s and the lead's plan changed %t; want the trunk unmoved and the plan as the lead left it", trunk(), holds(fixture, plan.Path) != removed)
	}
}
