package landing

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
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
	restored, checks, none := l.undoable()
	if none != "" {
		l.undone = []string{none}
		return nil
	}
	said := restoredTo(restored, l.head)
	l.undone, l.unevidenced = nil, nil
	// A check the form stage found not building without the change is not run again there: how it fails is known.
	run := slices.DeleteFunc(slices.Clone(checks), func(check string) bool {
		failure, known := l.unbuilt[check]
		if known {
			l.undone = append(l.undone, "with "+said+", "+check+" fails before a test asserts anything, so it is no evidence: "+failure)
			l.unevidenced = append(l.unevidenced, check)
		}
		return known
	})
	if len(run) > 0 {
		tree, _, release, err := l.undoTree(ctx, restored)
		if err != nil {
			return err
		}
		defer release()
		for _, check := range run {
			_, err := planverify.Execute(ctx, tree, check, nil)
			failure, _, _ := strings.Cut(fmt.Sprint(err), "\n")
			switch {
			case err == nil:
				return fmt.Errorf("land: the check %q passes with %s, so it proves nothing of the change; make it fail without the change", check, said)
			case errors.Is(err, planverify.ErrAssertion):
				l.undone = append(l.undone, "with "+said+", "+check+" fails: "+failure)
			default:
				l.undone = append(l.undone, "with "+said+", "+check+" fails before a test asserts anything, so it is no evidence: "+failure)
				l.unevidenced = append(l.unevidenced, check)
			}
		}
	}
	if len(l.unevidenced) > 0 && l.Log != nil {
		fmt.Fprintf(l.Log, "land: %s has no undo evidence: %s do not build without the change, so no assertion is shown to catch its absence\n",
			l.Ref, strings.Join(l.unevidenced, "; "))
	}
	// A change none of whose checks asserts without it lands only with a test through a boundary that builds either
	// way, or with its step's reason why none is practical (the owner's review, 2026-11-09: a component test can hold
	// its local rule while the assembled path breaks the guarantee).
	if len(l.unevidenced) == len(checks) {
		if len(l.boundaryReasons) == 0 {
			return l.noBoundary(said, l.unevidenced)
		}
		l.undone = append(l.undone, "no boundary test, as the step says: "+strings.Join(l.boundaryReasons, "; "))
	}
	return nil
}

// undoable is what the undo run restores, the production Go the change wrote, and the checks it runs, those that run
// a go test; or, where it runs none, what the message says instead.
func (l *landing) undoable() (restored, checks []string, none string) {
	restored = slices.DeleteFunc(slices.Clone(l.written), func(path string) bool { return !productionGo(path) })
	checks = slices.DeleteFunc(slices.Clone(l.verifiers), func(check string) bool {
		parsed, err := runrecord.ParseVerify(check)
		return err != nil || len(parsed.GoTests()) == 0
	})
	switch {
	case len(restored) == 0:
		none = "none run: the change touches no production Go"
	case l.consolidates():
		none = "none run: a consolidation passes before and after"
	case len(checks) == 0:
		none = "none run: no check runs a go test"
	}
	return restored, checks, none
}

// restoredTo says which production Go the undo run restored, and to which commit.
func restoredTo(restored []string, head string) string {
	return fmt.Sprintf("%s restored to %.12s", strings.Join(restored, ", "), head)
}

// noBoundary refuses a landing none of whose checks, unevidenced, fails on an assertion with the change's production
// Go restored, as said, and whose step does not say why no test through a boundary is practical.
func (l *landing) noBoundary(said string, unevidenced []string) error {
	// A landing is judged by the check of the commit it builds on, so a check widened in the same change is never run
	// here; the note is the remedy that works (finding bc29104a, 2026-11-11).
	return fmt.Errorf("land: no check of %s fails on an assertion with %s, since %s do not build without the change: say in a line of the step's rationale opening %q why the check has no test through a boundary that builds without the change, naming any such test the change adds (go run ./cmd/plan -note)",
		l.Ref, said, strings.Join(unevidenced, "; "), noBoundaryLine)
}

// undoTree checks out the commit the change builds on with the change's files kept but restored, the production Go
// it wrote, and returns it, what says so, and what removes it.
func (l *landing) undoTree(ctx context.Context, restored []string) (tree, said string, release func(), err error) {
	tree, release, err = l.treeAt(ctx, "undone")
	if err != nil {
		return "", "", nil, err
	}
	// The change's other files, its tests among them, are kept, and so is its deletion of a file other than production
	// Go, which stays as it was: a test the change moves to another file would otherwise be declared twice, and its
	// package would build in neither tree (test-reuse, 2026-11-11).
	carry := func() error {
		kept, err := l.git(ctx, append([]string{"ls-tree", "-r", "--name-only", l.candidate, "--"}, l.paths...)...)
		if err != nil {
			return err
		}
		var carried, present []string
		for path := range strings.Lines(kept) {
			path = strings.TrimSpace(path)
			present = append(present, path)
			if !slices.Contains(restored, path) {
				carried = append(carried, path)
			}
		}
		if len(carried) > 0 {
			if _, err := l.git(ctx, append([]string{"-C", tree, "checkout", l.candidate, "--"}, carried...)...); err != nil {
				return err
			}
		}
		for _, path := range l.paths {
			if productionGo(path) || slices.Contains(present, path) {
				continue
			}
			if err := os.Remove(filepath.Join(tree, filepath.FromSlash(path))); err != nil && !errors.Is(err, fs.ErrNotExist) {
				return err
			}
		}
		return nil
	}
	if err := carry(); err != nil {
		release()
		return "", "", nil, err
	}
	return tree, restoredTo(restored, l.head), release, nil
}

// noBoundaryLine opens the line of a step's rationale that says why no test through an existing boundary is practical,
// which admits a landing whose checks do not build without its change.
const noBoundaryLine = "No boundary test:"

// noBoundaryReason is the reason a rationale's noBoundaryLine gives, empty where it has none: the line opens with it,
// or with the "Note:" go run ./cmd/plan -note writes before it.
func noBoundaryReason(rationale string) string {
	for line := range strings.Lines(rationale) {
		line = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(line), "Note:"))
		if reason, stated := strings.CutPrefix(line, noBoundaryLine); stated && strings.TrimSpace(reason) != "" {
			return strings.TrimSpace(reason)
		}
	}
	return ""
}

// undoEvidenceTrailer is the trailer of a landing whose undo run is no evidence, so git log counts such landings
// (git log --grep="Undo-Evidence: none").
const undoEvidenceTrailer = "Undo-Evidence"

// unevidencedNote is the line that opens a message's Change section when its undo run is no evidence: what the
// undo section says at its end, said where a reader starts.
func (l *landing) unevidencedNote() string {
	return "- No undo evidence: " + strings.Join(l.unevidenced, "; ") +
		" do not build with the change's production Go restored, so no assertion here is shown to catch its absence."
}

// consolidates says a landed step is a consolidation's, whose change must shrink production code and pass before and
// after.
func (l *landing) consolidates() bool {
	named, _ := rows(l.Ref)
	return slices.ContainsFunc(l.terms.Items, func(item plan.Item) bool {
		return item.Class == plan.ClassConsolidation && slices.ContainsFunc(named, func(row ref) bool { return row.item == item.ID })
	})
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
