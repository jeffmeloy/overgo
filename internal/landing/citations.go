package landing

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/gitauthority"
	"overgo/internal/plan"
	"overgo/internal/repoanalysis"
)

// citingDocuments are the documents whose lines cite the tests that hold them: skill.md's rules and README.md's claims.
var citingDocuments = []string{"skill.md", "README.md"}

// followRenames rewrites each citation of a test the change renamed in the documents that cite tests, and takes a
// document it rewrote into the commit, as the generators' outputs are: on 2026-11-08 landing-scope/scoped-commit was
// refused because skill.md cited a test its own change had renamed. A test file whose change removes one test and adds
// one renames it (renames). A removed test no rename names is cited no longer by any test the change holds, and the
// landing prints each line citing it, with the tests the change added beside it; so it does for a document the checkout
// holds changed outside the commit, which is not rewritten.
func (l *landing) followRenames(ctx context.Context) error {
	renamed, unread := l.renames(ctx)
	for _, document := range citingDocuments {
		file := filepath.Join(l.Repo, document)
		data, err := os.ReadFile(file)
		if errors.Is(err, fs.ErrNotExist) {
			continue
		}
		if err != nil {
			return err
		}
		held, _ := gitauthority.Query(ctx, l.Repo, "cat-file", "blob", l.builtOn()+":"+document)
		writable := bytes.Equal(held, data) || l.scope == nil || plan.Lists(l.scope, document)
		text, rewrote, printed := string(data), 0, maps.Clone(unread)
		for _, old := range slices.Sorted(maps.Keys(renamed)) {
			if !writable {
				printed[old] = []string{renamed[old]}
				continue
			}
			var count int
			text, count = repoanalysis.FollowRename(text, old, renamed[old])
			rewrote += count
		}
		for _, gone := range slices.Sorted(maps.Keys(printed)) {
			if lines := repoanalysis.CitingLines(text, gone); len(lines) > 0 {
				fmt.Fprintf(l.Log, "land: %s cites %s, which the change removed beside adding %s, and is left as it stands: %s\n",
					document, gone, namedOrNone(printed[gone]), strings.Join(lines, " | "))
			}
		}
		if rewrote == 0 {
			continue
		}
		if err := os.WriteFile(file, []byte(text), clioptions.OutputFileMode); err != nil {
			return err
		}
		fmt.Fprintf(l.Log, "land: %s cites the renamed tests by their new names, %d citations rewritten\n", document, rewrote)
		if l.scope != nil && !plan.Lists(l.scope, document) {
			l.scope = append(l.scope, document)
		}
	}
	return nil
}

// namedOrNone lists names, or says there are none.
func namedOrNone(names []string) string {
	if len(names) == 0 {
		return "no test"
	}
	return strings.Join(names, ", ")
}

// renames are the tests the change renamed, each old name to its new one, and the tests it removed that no rename
// names, each with the tests the change added beside it. A test file whose change removes exactly one test and adds
// exactly one renames it; a test that only moved to another file is neither removed nor added.
func (l *landing) renames(ctx context.Context) (renamed map[string]string, unread map[string][]string) {
	type change struct{ gone, added []string }
	var changes []change
	allGone, allAdded := map[string]bool{}, map[string]bool{}
	for _, path := range l.paths {
		if !strings.HasSuffix(path, "_test.go") {
			continue
		}
		was, _ := gitauthority.Query(ctx, l.Repo, "cat-file", "blob", l.builtOn()+":"+path)
		now, _ := os.ReadFile(filepath.Join(l.Repo, filepath.FromSlash(path)))
		before, after := testNames(was), testNames(now)
		var file change
		for _, name := range before {
			if !slices.Contains(after, name) {
				file.gone, allGone[name] = append(file.gone, name), true
			}
		}
		for _, name := range after {
			if !slices.Contains(before, name) {
				file.added, allAdded[name] = append(file.added, name), true
			}
		}
		changes = append(changes, file)
	}
	renamed, unread = map[string]string{}, map[string][]string{}
	for _, file := range changes {
		gone := slices.DeleteFunc(file.gone, func(name string) bool { return allAdded[name] })
		added := slices.DeleteFunc(file.added, func(name string) bool { return allGone[name] })
		if len(gone) == 1 && len(added) == 1 {
			renamed[gone[0]] = added[0]
			continue
		}
		for _, name := range gone {
			unread[name] = added
		}
	}
	return renamed, unread
}
