package landing

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/plan"
	"overgo/internal/planverify"
	"overgo/internal/runrecord"
)

// undoSection opens the message's account of the undo run.
const undoSection = "Checked by undoing:"

// productionGo says a changed path is production Go, the code a change's behaviour lives in: a Go file, not a test,
// not testdata.
func productionGo(path string) bool {
	return strings.HasSuffix(path, ".go") && !strings.HasSuffix(path, "_test.go") && !strings.Contains("/"+path, "/testdata/")
}

// undo runs the landed steps' checks on the change with the production Go it wrote restored to the commit it builds
// on, and its tests and the stages' own rewrites kept. A check that passes there proves nothing of the change and is
// refused; one that fails on a test's assertion caught it; one that fails before any test ran, its tests not building
// without the change, is no evidence and is said so. A consolidation, which must pass before and after, and a change to no production Go take no undo
// run (SQA-332), nor does a check that runs no go test, whose failure cannot be told from a build's. What it found goes
// into the message.
func (l *landing) undo(ctx context.Context) error {
	restored := slices.DeleteFunc(slices.Clone(l.written), func(path string) bool { return !productionGo(path) })
	named, _ := rows(l.Ref)
	consolidates := slices.ContainsFunc(l.terms.Items, func(item plan.Item) bool {
		return item.Class == plan.ClassConsolidation && slices.ContainsFunc(named, func(row ref) bool { return row.item == item.ID })
	})
	checks := slices.DeleteFunc(slices.Clone(l.verifiers), func(check string) bool {
		parsed, err := runrecord.ParseVerify(check)
		return err != nil || len(parsed.GoTests()) == 0
	})
	switch {
	case len(restored) == 0:
		l.undone = []string{"none run: the change touches no production Go"}
		return nil
	case consolidates:
		l.undone = []string{"none run: a consolidation passes before and after"}
		return nil
	case len(checks) == 0:
		l.undone = []string{"none run: no check runs a go test"}
		return nil
	}
	tree, release, err := l.treeAt(ctx, "undone")
	if err != nil {
		return err
	}
	defer release()
	// The change's other files, its tests among them, are kept; a file it deletes stays as it was.
	kept, err := l.git(ctx, append([]string{"ls-tree", "-r", "--name-only", l.candidate, "--"}, l.paths...)...)
	if err != nil {
		return err
	}
	var carried []string
	for path := range strings.Lines(kept) {
		if path = strings.TrimSpace(path); !slices.Contains(restored, path) {
			carried = append(carried, path)
		}
	}
	if len(carried) > 0 {
		if _, err := l.git(ctx, append([]string{"-C", tree, "checkout", l.candidate, "--"}, carried...)...); err != nil {
			return err
		}
	}
	said := fmt.Sprintf("%s restored to %.12s", strings.Join(restored, ", "), l.head)
	l.undone = nil
	for _, check := range checks {
		_, err := planverify.Execute(ctx, tree, check, nil)
		failure, _, _ := strings.Cut(fmt.Sprint(err), "\n")
		switch {
		case err == nil:
			return fmt.Errorf("land: the check %q passes with %s, so it proves nothing of the change; make it fail without the change", check, said)
		case errors.Is(err, planverify.ErrAssertion):
			l.undone = append(l.undone, "with "+said+", "+check+" fails: "+failure)
		default:
			l.undone = append(l.undone, "with "+said+", "+check+" fails before a test asserts anything, so it is no evidence: "+failure)
		}
	}
	return nil
}

// treeAt checks the commit the change builds on out in a worktree of its own, named name in the scratch, and returns
// it and what removes it.
func (l *landing) treeAt(ctx context.Context, name string) (string, func(), error) {
	tree := filepath.Join(l.scratch, name)
	if _, err := l.git(ctx, "worktree", "add", "--detach", tree, l.head); err != nil {
		return "", nil, err
	}
	return tree, func() {
		if _, removed := l.git(context.WithoutCancel(ctx), "worktree", "remove", "--force", tree); removed != nil {
			fmt.Fprintf(l.Log, "land: the worktree %s a check ran in is left: %v\n", tree, removed)
		}
	}, nil
}
