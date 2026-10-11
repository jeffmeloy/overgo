package landing

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/authoritylock"
	"overgo/internal/clioptions"
	"overgo/internal/gitauthority"
	"overgo/internal/plan"
	"overgo/internal/processlock"
)

// planAside is where the lead's edited plan is kept, in the process state of
// the trunk's checkout, for the moment its file holds the tip's.
const planAside = "plan.following"

// trunkFollows prepares the trunk's own checkout to follow a commit made
// from another worktree, landed on the tip the checkout stands on, and
// returns what completes it: called with true once the trunk has moved, it
// writes the checkout forward; called with false, it gives up what it holds.
//
// What the lead has uncommitted in the checkout is kept. Its files, as git
// add -A stages them over the old tip, are merged onto the commit three ways,
// and a file the lead and the commit both changed otherwise refuses the
// commit here, by name, before the trunk moves. The plan is not merged as
// text, where a row removed beside a row edited would not merge: every commit
// changes it, removing its rows, and the lead edits it in place between
// commits. So the commit's own change to it is made again to the lead's
// working plan, its rows removed and what it took from their counts taken.
// A working plan those rows cannot be taken from, or a commit that edits the
// plan itself while the lead's is edited, is refused. The checkout's plan
// lock is held from here until the checkout is written, so the plan command
// and the follow do not write the plan at once; while the plan's file holds
// the tip's for the merge, the lead's is kept beside it (planAside).
func (l *landing) trunkFollows(ctx context.Context, landed string) (func(moved bool) error, error) {
	lock, err := authoritylock.AcquireContext(ctx, l.trunk)
	if err != nil {
		return nil, err
	}
	refuse := func(err error) (func(bool) error, error) { return nil, errors.Join(err, lock.Close()) }
	environment := append(gitauthority.RepositoryEnvironment(), "GIT_INDEX_FILE="+filepath.Join(l.scratch, "trunk-index"))
	there := func(arguments ...string) (string, error) {
		output, err := l.run(ctx, environment, "git", gitauthority.WriterArguments(append([]string{"-C", l.trunk}, arguments...)...)...)
		return strings.TrimSpace(output), err
	}
	planPath := filepath.Join(l.trunk, filepath.FromSlash(plan.Path))
	working, err := os.ReadFile(planPath)
	if err != nil {
		return refuse(err)
	}
	held, err := gitauthority.Query(ctx, l.Repo, "cat-file", "blob", l.head+":"+plan.Path)
	if err != nil {
		return refuse(err)
	}
	var edited []byte
	if !bytes.Equal(working, held) {
		document, err := plan.Parse(working)
		named, _ := rows(l.Ref)
		for _, row := range named {
			if err == nil {
				document, err = plan.Advance(document, row.item, row.step)
			}
		}
		if err == nil && !bytes.Equal(l.planned, held) {
			err = errors.New("the commit edits the plan itself")
		}
		if err != nil {
			return refuse(fmt.Errorf("land: the trunk's checkout %s holds an edited plan that %s cannot be applied to; it is committed or put back there first: %w", l.trunk, l.Ref, err))
		}
		for at, item := range document.Items {
			was := slices.IndexFunc(l.terms.Items, func(other plan.Item) bool { return other.ID == item.ID })
			is := slices.IndexFunc(l.advanced.Items, func(other plan.Item) bool { return other.ID == item.ID })
			if was >= 0 && is >= 0 {
				document.Items[at].Budget.Nodes += l.advanced.Items[is].Budget.Nodes - l.terms.Items[was].Budget.Nodes
			}
		}
		written := filepath.Join(l.scratch, "trunk-plan.json")
		if err := plan.Save(written, document); err != nil {
			return refuse(err)
		}
		if edited, err = os.ReadFile(written); err != nil {
			return refuse(err)
		}
	}
	// own is the checkout's tree with its plan as the tip holds it; merged is what it becomes.
	blob, err := there("rev-parse", l.head+":"+plan.Path)
	staging := [][]string{{"read-tree", l.head}, {"add", "-A"}}
	if edited != nil {
		staging = append(staging, []string{"update-index", "--cacheinfo", regularFile + "," + blob + "," + plan.Path})
	}
	for _, staged := range staging {
		if err == nil {
			_, err = there(staged...)
		}
	}
	if err != nil {
		return refuse(err)
	}
	own, err := there("write-tree")
	if err != nil {
		return refuse(err)
	}
	candidate, err := there("commit-tree", own, "-p", l.head, "-m", "the trunk's checkout before "+l.Ref)
	if err != nil {
		return refuse(err)
	}
	said, err := l.run(ctx, gitauthority.RepositoryEnvironment(), "git", "merge-tree", "--write-tree", "--name-only", landed, candidate)
	merged, contested, _ := strings.Cut(strings.TrimSpace(said), "\n")
	if err != nil && contested == "" {
		return refuse(err)
	}
	if err != nil {
		contested, _, _ = strings.Cut(contested, "\n\n")
		return refuse(fmt.Errorf("land: the trunk's checkout %s has uncommitted changes of its own to files %s also changes: %s; they are committed or put aside there first", l.trunk, l.Ref, strings.ReplaceAll(contested, "\n", ", ")))
	}
	return func(moved bool) (err error) {
		defer func() { err = errors.Join(err, lock.Close()) }()
		if !moved {
			return nil
		}
		aside := filepath.Join(l.trunk, processlock.StateDirectory, planAside)
		if edited != nil {
			// The merge reads the plan's file as the tip's; the lead's waits beside it.
			if err := errors.Join(os.WriteFile(aside, working, clioptions.PrivateFileMode), os.WriteFile(planPath, held, clioptions.PrivateFileMode)); err != nil {
				return err
			}
			_, _ = there("update-index", "-q", "--refresh")
		}
		if _, err := there("read-tree", "-m", "-u", own, merged); err != nil {
			if edited != nil {
				err = errors.Join(err, os.WriteFile(planPath, working, clioptions.PrivateFileMode))
			}
			return err
		}
		if edited != nil {
			if err := errors.Join(os.WriteFile(planPath, edited, clioptions.PrivateFileMode), os.Remove(aside)); err != nil {
				return err
			}
		}
		_, err = l.git(ctx, "-C", l.trunk, "reset", "-q")
		return err
	}, nil
}
