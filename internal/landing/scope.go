package landing

import (
	"context"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/plan"
	"overgo/internal/repoanalysis"
)

// commitScope is what a landing commits: the files its named steps list, the plan, the findings register a review
// left and what the repository's generators derive, which the landing writes itself. A listed directory holds
// everything under it, tests and fixtures among them. The lists are the trunk's plan's and the change's plan's: plan
// -setfiles widens a list with its reason noted, which is how a landing that left out a path its change needs is
// widened (leftOut); seven steps were retired and filed again on 2026-11-08 for a short list. A step the change itself
// adds is read from the plan it holds. A step that lists nothing, a merge, or a plan that cannot be read leaves the
// scope open: the whole tree, as every landing took before. On 2026-11-08 another session's untracked document in the
// lead's checkout stopped every landing for forty minutes, since the landing took the whole folder though each step
// names what it may change.
func (l *landing) commitScope(ctx context.Context) ([]string, error) {
	named, err := rows(l.Ref)
	if err != nil || len(l.merging) > 0 {
		return nil, err
	}
	var terms, document plan.Plan
	if held, err := gitauthority.Query(ctx, l.Repo, "cat-file", "blob", l.builtOn()+":"+plan.Path); err == nil {
		terms, _ = plan.Parse(held)
	}
	if written, err := os.ReadFile(filepath.Join(l.Repo, filepath.FromSlash(plan.Path))); err == nil {
		document, _ = plan.Parse(written)
	}
	var listed []string
	for _, row := range named {
		step, found := plan.ExactStep(terms, row.item, row.step)
		written, held := plan.ExactStep(document, row.item, row.step)
		if !found {
			step, found = written, held
		}
		if !found || len(step.Files) == 0 {
			return nil, nil
		}
		listed = append(listed, step.Files...)
		if held {
			listed = append(listed, written.Files...)
		}
	}
	listed = append(listed, plan.Path, findingsRegister)
	for _, generator := range l.generators() {
		listed = append(listed, generator.writes...)
	}
	return listed, nil
}

// inside splits the changed paths of the checkout into those the scope holds and those it leaves as they stand; an
// open scope holds them all.
func inside(scope []string, entries []repoanalysis.DirtyPath) (held []repoanalysis.DirtyPath, left []string) {
	for _, entry := range entries {
		if scope == nil || plan.Lists(scope, entry.Path) && (entry.OriginalPath == "" || plan.Lists(scope, entry.OriginalPath)) {
			held = append(held, entry)
			continue
		}
		left = append(left, entry.Path)
	}
	return held, left
}

// nameLeft says, once, which changed paths a landing leaves as they stand outside its commit, and keeps the latest for
// a refusal to name (leftOut).
func (l *landing) nameLeft(left []string) {
	l.left = left
	if len(left) > 0 && !l.leftNamed {
		fmt.Fprintf(l.Log, "land: left as it stands, outside the commit, since %s lists none of it: %s\n", l.Ref, strings.Join(left, ", "))
		l.leftNamed = true
	}
}

// leftOut is what a refused landing adds when its commit left changed paths out, since a check that fails on a path the
// commit lacks names none: the paths, and the plan -setfiles command that adds their directories to the first step's
// list, after which landing again takes them (commitScope). A landing that left nothing out adds nothing.
func (l *landing) leftOut() string {
	named, err := rows(l.Ref)
	if err != nil || len(l.left) == 0 || l.scope == nil {
		return ""
	}
	said := fmt.Sprintf("land: the commit left out %s, which %s does not list", strings.Join(l.left, ", "), l.Ref)
	document, err := plan.Load(filepath.Join(l.Repo, filepath.FromSlash(plan.Path)))
	step, found := plan.ExactStep(document, named[0].item, named[0].step)
	if err != nil || !found {
		return said
	}
	files, added := slices.Clone(step.Files), []string(nil)
	for _, left := range l.left {
		if plan.Lists(files, left) {
			continue
		}
		entry := path.Dir(left)
		if entry == "." {
			entry = left
		}
		files, added = append(files, entry), append(added, entry)
	}
	return fmt.Sprintf("%s; if the change needs them, list them and land again: go run ./cmd/plan -setfiles -files %s -reason %q %s %s",
		said, strings.Join(files, ","), "the change needs "+strings.Join(added, ", "), named[0].item, named[0].step)
}
