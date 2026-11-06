package repoanalysis

import (
	"go/scanner"
	"go/token"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// lockResources are the lock files the production code names, each with the one resource it guards, as one-home's
// inventory of them measured: a lock named by its file, or one a source builds by suffixing ".lock" to a path, named by
// that source. Locks the code takes by a resource's name, through processcontrol, share the one file named by
// resource_shared_windows.go.
var lockResources = map[string]string{
	"gate.lock":                         "a checkout's plan and its publication to the trunk",
	"landing_turn.lock":                 "the trunk, one landing's commit at a time across the worktrees",
	"stop.lock":                         "a checkout's stop record",
	"loop_supervisor.lock":              "a worktree's one loop supervisor and its stop endpoint",
	"removal.lock":                      "the removal of leftovers and its receipt",
	"overgodb.lock":                     "nothing any longer: an older store's lock file, which a clone leaves behind",
	"cmd/loop/improve.go:.lock":         "the measurement cycle's checkout and its state",
	"internal/hfhub/download.go:.lock":  "one file a download writes",
	"internal/overgodb/backup.go:.lock": "a store backup's destination",
	"internal/overgodb/store.go:.lock":  "one store's journal, segments and checkpoints",
	"internal/processcontrol/resource_shared_windows.go:.lock": "a machine-wide resource by its name: the card, the quiet machine",
}

// TestEachResourceHasOneLock holds the production code's locks to lockResources: every lock file a source names is
// declared there with the resource it guards, no declared lock is named by nothing, and no two locks guard one
// resource, so a second lock for a resource another already guards, or a lock nothing takes any more, is refused
// rather than waited on.
func TestEachResourceHasOneLock(t *testing.T) {
	t.Parallel()
	named := map[string][]string{}
	for _, tree := range []string{"cmd", "internal"} {
		testcheck.NoError(t, filepath.WalkDir(filepath.Join(checkoutRoot, tree), func(file string, entry fs.DirEntry, err error) error {
			if err != nil || entry.IsDir() || !strings.HasSuffix(file, ".go") || strings.HasSuffix(file, "_test.go") {
				return err
			}
			relative := filepath.ToSlash(testcheck.Must(filepath.Rel(checkoutRoot, file))(t))
			source := testcheck.Must(os.ReadFile(file))(t)
			var words scanner.Scanner
			words.Init(token.NewFileSet().AddFile(file, -1, len(source)), source, nil, 0)
			for _, kind, literal := words.Scan(); kind != token.EOF; _, kind, literal = words.Scan() {
				text, _ := strconv.Unquote(literal)
				if kind != token.STRING || !strings.HasSuffix(text, ".lock") {
					continue
				}
				lock := path.Base(text)
				if lock == ".lock" {
					lock = relative + ":.lock"
				}
				named[lock] = append(named[lock], relative)
			}
			return nil
		}))
	}
	for lock, sources := range named {
		if _, declared := lockResources[lock]; !declared {
			t.Errorf("%s names lock %s, which lockResources does not declare with the resource it guards", strings.Join(slices.Compact(sources), ", "), lock)
		}
	}
	guards := map[string]string{}
	for lock, resource := range lockResources {
		if _, held := named[lock]; !held {
			t.Errorf("lockResources declares %s, which no production source names", lock)
		}
		if other, taken := guards[resource]; taken {
			t.Errorf("%s and %s both guard %s; hold the resource by one", lock, other, resource)
		}
		guards[resource] = lock
	}
}
