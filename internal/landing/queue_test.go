package landing

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"overgo/internal/plan"
)

// TestTwoWorktreesLandThroughOneQueue holds two sessions, each in a worktree
// of its own begun at one commit of the trunk, to one queue onto it.
//
// Neither is charged with the other's files, and the second is checked on
// the tree the first left by the terms that tree's plan holds. The first
// lands its file. The lead then moves the second row's terms on the trunk:
// its count to nothing, and its check to one that passes only on the first's
// tree. The second, from the commit it began at, is brought up to the trunk
// and refused under those terms, and the refusal says which of them moved
// and what they were; nothing is published and its file is as it was. With
// the count given back on the trunk it lands: its commit changes its own
// file and its row's removal and nothing of the first's, which its count
// could not have paid for.
//
// A queue stopped and started again loses no candidate and removes no row
// twice. A commit stopped after the trunk moved and before its worktree's own
// HEAD did has its change on the trunk and still in its working tree; begun
// again it is that commit's: it says so, no second commit is made, the
// worktree is on the trunk with nothing uncommitted, and its row is gone
// once. And a commit stopped while it waits for its turn leaves its worktree
// as it was and the trunk where it was; begun again when the turn is free, it
// lands.
func TestTwoWorktreesLandThroughOneQueue(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	fixture := newRowFixture(t, map[string]string{"a.txt": "base\n", "internal/p/p.go": "// Package p is the fixture.\npackage p\n"})
	fixture.git("branch", "-M", "trunk")
	// The repository's files are the bytes git holds, as this repository's are: a checkout rewrites none.
	fixture.git("config", "core.autocrlf", "false")
	grown := packageNodes(t, oneFunction)
	row := func(id string) plan.Item {
		return plan.Item{ID: id, Title: id, Status: plan.StatusOpen, Budget: plan.Budget{Nodes: grown, Reason: "the fixture's growth"},
			Steps: []plan.Step{{ID: "do", Title: id, Status: plan.StatusOpen, Verify: "true"}}}
	}
	fixture.commitPlan(plan.Plan{Campaign: "fixture", Items: []plan.Item{row("first"), row("second"), row("third")}})
	base := fixture.git("rev-parse", "HEAD")
	outside := t.TempDir()
	one, two := &sliceFixture{t: t, repo: filepath.Join(outside, "one")}, &sliceFixture{t: t, repo: filepath.Join(outside, "two")}
	for _, linked := range []*sliceFixture{one, two} {
		fixture.git("worktree", "add", "-q", "--detach", linked.repo, base)
	}
	message := []byte("Change the fixture\n\nCause: the fixture needs a change.\nPredicted effect: none.\n")
	land := func(ctx context.Context, checkout *sliceFixture, reference string, log io.Writer) (string, string, error) {
		var said strings.Builder
		landed, err := Run(ctx, Options{Repo: checkout.repo, Ref: reference, Message: message, Log: io.MultiWriter(&said, log)})
		return landed, said.String(), err
	}
	holds := func(checkout *sliceFixture, name string) string {
		data, err := os.ReadFile(filepath.Join(checkout.repo, filepath.FromSlash(name)))
		if err != nil {
			return ""
		}
		return string(data)
	}
	trunk := func() string { return fixture.git("rev-parse", "refs/heads/trunk") }
	// moveSecond commits, on the trunk, the second row's count and check.
	moveSecond := func(count int, check string) string {
		t.Helper()
		document, err := plan.Load(fixture.planPath())
		if err != nil || document.Items[0].ID != "second" {
			t.Fatalf("the trunk's plan opens with %+v (%v), want the second row", document.Items, err)
		}
		document.Items[0].Budget.Nodes, document.Items[0].Steps[0].Verify = count, check
		fixture.commitPlan(document)
		return fixture.git("rev-parse", "HEAD")
	}
	const fileOne, fileTwo = "internal/p/a.go", "internal/p/b.go"
	sourceTwo := strings.ReplaceAll(oneFunction, "a()", "b()")

	// Two sessions, one base: the first lands, and the lead moves the second's terms.
	one.write(fileOne, oneFunction)
	if first, _, err := land(ctx, one, "first/do", io.Discard); err != nil || trunk() != first {
		t.Fatalf("the first session's commit = %q with the trunk at %s (%v), want it the trunk's tip", first, trunk(), err)
	}
	check := "test -f " + fileOne
	moved := moveSecond(0, check)
	two.write(fileTwo, sourceTwo)
	_, _, err := land(ctx, two, "second/do", io.Discard)
	for _, want := range []string{"may add 0", "moved its terms", fmt.Sprintf(`second/do was checked by "true" and could add %d`, grown), fmt.Sprintf("is checked by %q and may add 0", check)} {
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("the second session's commit under terms the trunk moved = %v, want it refused saying %q", err, want)
		}
	}
	if trunk() != moved || two.git("rev-parse", "HEAD") != moved || holds(two, fileTwo) != sourceTwo {
		t.Fatalf("after the refusal the trunk is at %s and the worktree on %s holding %q; want the trunk unmoved at %s, the worktree brought up to it, and its file as it was", trunk(), two.git("rev-parse", "HEAD"), holds(two, fileTwo), moved)
	}
	given := moveSecond(grown, check)
	second, said, err := land(ctx, two, "second/do", io.Discard)
	if err != nil || trunk() != second || fixture.git("rev-parse", second+"^") != given || strings.Contains(said, "moved its terms") {
		t.Fatalf("with the count given back the second session's commit = %q on %s (%v), saying %q; want it landed on the lead's %s", second, fixture.git("rev-parse", second+"^"), err, said, given)
	}
	if changed := fixture.git("diff-tree", "-r", "--name-only", given, second); changed != plan.Path+"\n"+fileTwo {
		t.Fatalf("the second session's commit changes %q, want its own file and its row's removal", changed)
	}

	// Stopped after the trunk moved and before the worktree's own HEAD did.
	commits := fixture.git("rev-list", "--count", "refs/heads/trunk")
	two.git("update-ref", "--no-deref", "HEAD", given)
	two.git("checkout", "-q", given, "--", plan.Path)
	two.git("reset", "-q")
	if left := two.git("status", "--porcelain"); !strings.Contains(left, fileTwo) {
		t.Fatalf("the stopped worktree has uncommitted %q, want its file still to commit", left)
	}
	again, said, err := land(ctx, two, "second/do", io.Discard)
	if err != nil || again != second || !strings.Contains(said, "had landed as") {
		t.Fatalf("begun again, the stopped commit = %q saying %q (%v), want the commit %s the trunk already holds", again, said, err, second)
	}
	if trunk() != second || fixture.git("rev-list", "--count", "refs/heads/trunk") != commits || two.git("rev-parse", "HEAD") != second || two.git("status", "--porcelain") != "" || holds(two, plan.Path) != holds(fixture, plan.Path) || holds(two, fileTwo) != sourceTwo {
		t.Fatalf("begun again, the trunk is at %s after %s commits, where it held %s, and the worktree is on %s with uncommitted %q; want no second commit, the worktree on the trunk with its file and the trunk's plan",
			trunk(), fixture.git("rev-list", "--count", "refs/heads/trunk"), commits, two.git("rev-parse", "HEAD"), two.git("status", "--porcelain"))
	}

	// Stopped while it waits for its turn.
	one.write("a.txt", "the first worktree's\n")
	holder := &landing{Options: Options{Repo: fixture.repo, Log: io.Discard}}
	held, err := holder.holdTurn(ctx)
	if err != nil {
		t.Fatal(err)
	}
	stopping, stop := context.WithCancel(ctx)
	defer stop()
	where := one.git("rev-parse", "HEAD")
	_, _, err = land(stopping, one, "third/do", firstWrite{sync.OnceFunc(stop)})
	released := held()
	if err == nil || released != nil || trunk() != second || one.git("rev-parse", "HEAD") != where || holds(one, "a.txt") != "the first worktree's\n" {
		t.Fatalf("a commit stopped while it waited = %v (the turn given back: %v) with the trunk at %s and the worktree on %s holding %q; want it refused with the trunk and the worktree where they were", err, released, trunk(), one.git("rev-parse", "HEAD"), holds(one, "a.txt"))
	}
	third, _, err := land(ctx, one, "third/do", io.Discard)
	if err != nil || trunk() != third || holds(fixture, "a.txt") != "the first worktree's\n" {
		t.Fatalf("begun again with the turn free, the commit = %q (%v) with the trunk at %s; want it landed and followed", third, err, trunk())
	}
}
