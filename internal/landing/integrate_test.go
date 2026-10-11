package landing

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/plan"
)

// TestWorktreeCandidateIsIntegratedOntoTheTrunk holds a commit from a linked
// worktree to the trunk as it stands when the commit's turn comes.
//
// Two worktrees begin at one commit of the trunk, each with a row that may
// add one function's nodes and no more. The first lands its file: the trunk
// is its commit, the worktree is on it, and the trunk's own checkout has
// followed, holding the file and a plan without the row, with nothing left
// uncommitted. The lead then commits, on the trunk, another check for the
// second row: one that passes only on a tree that holds the first's file.
//
// The second, still on the commit it began at, lands its own file. It is
// brought up to the trunk first and says so; it is checked on the tree the
// first left and by the check that tree's plan holds, which fails on the
// tree it was written on and under the plan it was written against; it is
// charged its own file and not the first's, which its count could not pay
// for; its commit's parent is the lead's; and both checkouts hold both files
// after it, the trunk's with nothing uncommitted.
//
// A change the trunk cannot take is refused with nothing moved. The first
// worktree, now behind, writes the second's file otherwise: the commit is
// refused naming the file, before any check, with the worktree where it was.
// And with a change of the lead's own uncommitted in the trunk's checkout to
// a file a commit changes, the commit is refused naming the checkout, the
// trunk unmoved and the lead's change as it was; put aside, the commit lands
// and the trunk's checkout follows it. A worktree on a commit of its own,
// which the trunk does not descend from, is refused: nothing says what its
// change is against.
func TestWorktreeCandidateIsIntegratedOntoTheTrunk(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	const packaged = "// Package p is the fixture.\npackage p\n"
	fixture := newRowFixture(t, map[string]string{"a.txt": "base\n", "internal/p/p.go": packaged})
	fixture.git("branch", "-M", "trunk")
	// The repository's files are the bytes git holds, as this repository's are: a checkout rewrites none.
	fixture.git("config", "core.autocrlf", "false")
	grown := packageNodes(t, oneFunction)
	row := func(id, check string) plan.Item {
		return plan.Item{ID: id, Title: id, Status: plan.StatusOpen, Budget: plan.Budget{Nodes: grown, Reason: "the fixture's growth"},
			Steps: []plan.Step{{ID: "do", Title: id, Status: plan.StatusOpen, Verify: check}}}
	}
	fixture.commitPlan(plan.Plan{Campaign: "fixture", Doctrine: "fixture", Items: []plan.Item{row("first", "true"), row("second", "false"), row("third", "true")}})
	base := fixture.git("rev-parse", "HEAD")
	outside := t.TempDir()
	one, two := &sliceFixture{t: t, repo: filepath.Join(outside, "one")}, &sliceFixture{t: t, repo: filepath.Join(outside, "two")}
	for _, linked := range []*sliceFixture{one, two} {
		fixture.git("worktree", "add", "-q", "--detach", linked.repo, base)
	}
	message := []byte("Change the fixture\n\nCause: the fixture needs a change.\nPredicted effect: none.\n")
	// land commits a checkout's change for a row and returns the commit, what the commit said, and its refusal.
	land := func(checkout *sliceFixture, reference string) (string, string, error) {
		var said strings.Builder
		landed, err := Run(ctx, Options{Repo: checkout.repo, Ref: reference, Message: message, Log: &said})
		return landed, said.String(), err
	}
	holds := func(checkout *sliceFixture, name string) string {
		data, err := os.ReadFile(filepath.Join(checkout.repo, filepath.FromSlash(name)))
		if err != nil {
			return ""
		}
		return string(data)
	}
	rows := func(checkout *sliceFixture) (ids []string) {
		t.Helper()
		document, err := plan.Load(checkout.planPath())
		if err != nil {
			t.Fatal(err)
		}
		for _, item := range document.Items {
			ids = append(ids, item.ID)
		}
		return ids
	}
	const fileOne, fileTwo = "internal/p/a.go", "internal/p/b.go"
	sourceTwo := strings.ReplaceAll(oneFunction, "a()", "b()")

	// The first lands: the trunk is its commit and the trunk's checkout follows.
	one.write(fileOne, oneFunction)
	first, _, err := land(one, "first/do")
	if err != nil || first == "" || fixture.git("rev-parse", "refs/heads/trunk") != first || one.git("rev-parse", "HEAD") != first || fixture.git("rev-parse", first+"^") != base {
		t.Fatalf("the first worktree's commit = %q (%v), want it the trunk's tip, on the commit both began at", first, err)
	}
	if holds(fixture, fileOne) != oneFunction || strings.Join(rows(fixture), ",") != "second,third" || fixture.git("status", "--porcelain") != "" {
		t.Fatalf("after the first commit the trunk's checkout holds %q of the file, the rows %v and uncommitted %q; want the file, the row gone and nothing uncommitted", holds(fixture, fileOne), rows(fixture), fixture.git("status", "--porcelain"))
	}

	// The lead gives the second row a check only the first's tree passes.
	document, err := plan.Load(fixture.planPath())
	if err != nil {
		t.Fatal(err)
	}
	document.Items[0].Steps[0].Verify = "test -f " + fileOne
	fixture.commitPlan(document)
	leads := fixture.git("rev-parse", "HEAD")

	// The second lands from the commit it began at.
	two.write(fileTwo, sourceTwo)
	second, said, err := land(two, "second/do")
	if err != nil || second == "" || !strings.Contains(said, "brought the trunk") {
		t.Fatalf("the second worktree's commit = %q after saying %q (%v), want it brought up to the trunk and landed by the check and the count the trunk's plan holds", second, said, err)
	}
	if fixture.git("rev-parse", "refs/heads/trunk") != second || two.git("rev-parse", "HEAD") != second || fixture.git("rev-parse", second+"^") != leads {
		t.Fatalf("the second commit %s is on %s with the trunk at %s, want it the trunk's tip on the lead's %s", second, fixture.git("rev-parse", second+"^"), fixture.git("rev-parse", "refs/heads/trunk"), leads)
	}
	for name, checkout := range map[string]*sliceFixture{"the trunk's checkout": fixture, "the second worktree": two} {
		if holds(checkout, fileOne) != oneFunction || holds(checkout, fileTwo) != sourceTwo || strings.Join(rows(checkout), ",") != "third" || checkout.git("status", "--porcelain") != "" {
			t.Fatalf("after the second commit %s holds %q and %q, the rows %v and uncommitted %q; want both files, the third row alone and nothing uncommitted", name, holds(checkout, fileOne), holds(checkout, fileTwo), rows(checkout), checkout.git("status", "--porcelain"))
		}
	}
	if changed := fixture.git("diff-tree", "-r", "--name-only", leads, second); changed != "docs/plan.json\n"+fileTwo && changed != fileTwo+"\ndocs/plan.json" {
		t.Fatalf("the second commit changes %q, want its own file and its row's removal", changed)
	}

	// The first worktree, now behind, writes the second's file otherwise.
	one.write(fileTwo, strings.ReplaceAll(sourceTwo, `"a"`, `"other"`))
	mine := holds(one, fileTwo)
	if _, _, err := land(one, "third/do"); err == nil || !strings.Contains(err.Error(), fileTwo) || !strings.Contains(err.Error(), "also changes") {
		t.Fatalf("a change to a file the trunk changed otherwise = %v, want it refused naming %s", err, fileTwo)
	}
	if one.git("rev-parse", "HEAD") != first || holds(one, fileTwo) != mine || fixture.git("rev-parse", "refs/heads/trunk") != second {
		t.Fatalf("the refused worktree is on %s holding %q with the trunk at %s; want it where it was, on %s", one.git("rev-parse", "HEAD"), holds(one, fileTwo), fixture.git("rev-parse", "refs/heads/trunk"), first)
	}
	if err := os.Remove(filepath.Join(one.repo, filepath.FromSlash(fileTwo))); err != nil {
		t.Fatal(err)
	}

	// The lead has a change of its own, uncommitted, to a file the commit changes.
	fixture.write("a.txt", "the lead's\n")
	one.write("a.txt", "the first worktree's\n")
	if _, _, err := land(one, "third/do"); err == nil || !strings.Contains(err.Error(), "cannot follow") {
		t.Fatalf("a commit the trunk's checkout cannot follow = %v, want it refused for the lead's uncommitted change", err)
	}
	if fixture.git("rev-parse", "refs/heads/trunk") != second || holds(fixture, "a.txt") != "the lead's\n" || holds(one, "a.txt") != "the first worktree's\n" || one.git("rev-parse", "HEAD") != second {
		t.Fatalf("after the refusal the trunk is at %s, its checkout's file %q and the worktree's %q on %s; want the trunk unmoved, both files as they were, and the worktree brought up to %s", fixture.git("rev-parse", "refs/heads/trunk"), holds(fixture, "a.txt"), holds(one, "a.txt"), one.git("rev-parse", "HEAD"), second)
	}
	fixture.write("a.txt", "base\n")
	third, _, err := land(one, "third/do")
	if err != nil || fixture.git("rev-parse", "refs/heads/trunk") != third || holds(fixture, "a.txt") != "the first worktree's\n" || fixture.git("status", "--porcelain") != "" {
		t.Fatalf("with the lead's change put aside the commit = %q (%v) and the trunk's checkout holds %q with %q uncommitted; want it landed and followed", third, err, holds(fixture, "a.txt"), fixture.git("status", "--porcelain"))
	}

	// A worktree on a commit of its own, which the trunk does not descend from.
	two.write("a.txt", "committed apart\n")
	two.git("commit", "-q", "-a", "-m", "apart")
	two.write("a.txt", "a change on it\n")
	if _, _, err := land(two, "third/do"); err == nil || !strings.Contains(err.Error(), "does not descend from") || fixture.git("rev-parse", "refs/heads/trunk") != third {
		t.Fatalf("a commit from a worktree on a commit of its own = %v with the trunk at %s, want it refused and the trunk at %s", err, fixture.git("rev-parse", "refs/heads/trunk"), third)
	}
}
