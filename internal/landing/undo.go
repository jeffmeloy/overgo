package landing

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"maps"
	"os"
	pathpkg "path"
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

// production says which of the change's written and named paths are production: production Go, and a file a package
// of the build embeds, as the kernels' PTX and a package's own records are. An undo run that restored the Go alone ran
// the old Go on a change's new kernels, and the check of their fixed-order sums passed there (fixed-order/do). A file
// is embedded only by a package in a directory above it, so those directories alone are listed.
func (l *landing) production(ctx context.Context) (map[string]bool, error) {
	if l.produced != nil {
		return l.produced, nil
	}
	produced, above := map[string]bool{}, map[string]bool{}
	for _, path := range slices.Concat(l.written, l.paths) {
		if productionGo(path) {
			produced[path] = true
			continue
		}
		for directory := pathpkg.Dir(path); directory != "."; directory = pathpkg.Dir(directory) {
			above["./"+directory] = true
		}
	}
	if len(above) > 0 {
		root, err := filepath.Abs(l.stageRoot())
		if err != nil {
			return nil, err
		}
		listed, err := l.runIn(ctx, root, nil, "go", append([]string{"list", "-e", "-json=Dir,EmbedFiles"}, slices.Sorted(maps.Keys(above))...)...)
		if err != nil {
			return nil, err
		}
		decoder := json.NewDecoder(strings.NewReader(listed))
		for {
			var listing struct {
				Dir        string
				EmbedFiles []string
			}
			if err := decoder.Decode(&listing); errors.Is(err, io.EOF) {
				break
			} else if err != nil {
				return nil, err
			}
			for _, file := range listing.EmbedFiles {
				relative, err := filepath.Rel(root, filepath.Join(listing.Dir, filepath.FromSlash(file)))
				if err != nil {
					return nil, err
				}
				produced[filepath.ToSlash(relative)] = true
			}
		}
	}
	l.produced = produced
	return produced, nil
}

// undo runs the landed steps' checks on the change with the production Go it wrote restored to the commit it builds
// on, and its tests and the stages' own rewrites kept. A check that passes there proves nothing of the change and is
// refused; one that fails on a test's assertion caught it; one that fails before any test ran, its tests not building
// without the change, is no evidence and is said so. A consolidation, which must pass before and after, and a change to no production Go take no undo
// run (SQA-332), nor does a check that runs no go test, whose failure cannot be told from a build's. What it found goes
// into the message.
func (l *landing) undo(ctx context.Context) error {
	restored, checks, none, err := l.undoable(ctx)
	if err != nil {
		return err
	}
	if none != "" {
		l.undone = []string{none}
		return nil
	}
	l.undone, l.unevidenced = nil, nil
	tree, said, release, err := l.undoTree(ctx, restored)
	if err != nil {
		return err
	}
	defer release()
	for _, check := range checks {
		_, err := planverify.Execute(ctx, tree, check, nil)
		failure, _, _ := strings.Cut(fmt.Sprint(err), "\n")
		switch {
		case err == nil:
			// The reason comes first, the check after it: a refusal's record keeps its first line cut to causeBytes.
			return fmt.Errorf("land: a check passes with the change's production Go restored, so it proves nothing of the change; make it fail without the change: %q, with %s", check, said)
		case errors.Is(err, planverify.ErrAssertion):
			l.undone = append(l.undone, "with "+said+", "+check+" fails: "+failure)
		default:
			l.undone = append(l.undone, "with "+said+", "+check+" fails before a test asserts anything, so it is no evidence: "+failure)
			l.unevidenced = append(l.unevidenced, check)
		}
	}
	// A check that does not build without the change is no evidence, and the landing says so where it is read: in its
	// log, its message's Change section and its trailer (Undo-Evidence: none), which git log counts. It is not refused:
	// a refusal asking each such step for a line of reason refused 10 of 61 recorded landings, and the reasons were one
	// sentence repeated (the owner's review, 2026-11-14).
	if len(l.unevidenced) > 0 && l.Log != nil {
		fmt.Fprintf(l.Log, "land: %s has no undo evidence: %s do not build without the change, so no assertion is shown to catch its absence\n",
			l.Ref, strings.Join(l.unevidenced, "; "))
	}
	return nil
}

// undoable is what the undo run restores, the production the change wrote, and the checks it runs, those that run
// a go test; or, where it runs none, what the message says instead.
func (l *landing) undoable(ctx context.Context) (restored, checks []string, none string, err error) {
	produced, err := l.production(ctx)
	if err != nil {
		return nil, nil, "", err
	}
	restored = slices.DeleteFunc(slices.Clone(l.written), func(path string) bool { return !produced[path] })
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
	return restored, checks, none, nil
}

// restoredTo says which production Go the undo run restored, and to which commit.
func restoredTo(restored []string, head string) string {
	return fmt.Sprintf("%s restored to %.12s", strings.Join(restored, ", "), head)
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
			if l.produced[path] || slices.Contains(present, path) {
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
