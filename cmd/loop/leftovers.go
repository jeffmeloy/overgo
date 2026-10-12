package main

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"

	"overgo/internal/artifact"
	"overgo/internal/authoritylock"
	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/fsatomic"
	"overgo/internal/gitauthority"
	"overgo/internal/guard"
	"overgo/internal/landing"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/pathidentity"
	"overgo/internal/plan"
	"overgo/internal/processlock"
	"overgo/internal/protection"
	"overgo/internal/repoanalysis"
	"overgo/internal/scorecard"
	"overgo/internal/trainingprogram"
)

// A listing gives every entry a state: finished; held, when something it
// established still uses the entry; unknown, when it could not establish the
// state, so the entry is kept; kept, when the entry's place is never finished;
// undeclared, when no line of the table names it; unowned, when nothing ties
// it to a row of the plan; misplaced, when its kind has another place. A
// line's action is what may be done with a finished entry of it.
const (
	stateFinished, stateHeld, stateUnknown, stateKept, stateUndeclared = "finished", "held", "unknown", "kept", "undeclared"
	stateUnowned, stateMisplaced                                       = "unowned", "misplaced"
	actionDelete, actionReport, actionNever                            = "delete", "report", "never"

	// agentWorktrees is where the harness's agents make their helper copies.
	agentWorktrees = protection.HarnessConfigDirectory + "worktrees/"
	// noTrunk is why nothing reads as on the trunk when the main worktree is detached.
	noTrunk = "the main worktree is on no branch, so nothing says what the trunk is"
	// receiptPrefix opens the name of a listing's receipt, which ends with the trunk's commit:
	// the listings of two commits are two files, and no literal is a whole file name.
	receiptPrefix = "leftovers-"
	// registrationPage bounds one read of the store's catalog, which refuses a read with no
	// bound; it is the page cmd/overgodb's own walk of the catalog defaults to.
	registrationPage = 1000
	// pathRunes are the characters a path written in a line of text continues through, in lower case.
	// runEnd closes a run for cited: only the end of a run names a path given with runEnd after it.
	pathRunes, runEnd = "abcdefghijklmnopqrstuvwxyz0123456789_-./", "/\x00/"
	// noAnswer is the exit status with which git says no: a commit is not an ancestor, no line matches.
	noAnswer = toolExit(1)
	// removalReceipt is the file, in the process state of the trunk's checkout, that every removal
	// appends its lines to. removalLock is the file there that one removing command locks at a
	// time: nothing else keeps two from running at once, each on a listing the other changes.
	removalReceipt, removalLock = receiptPrefix + "removed.txt", receiptPrefix + "removal.lock"
	// tokenDigits is how many hex digits of the digest of a listing's finished entries its token shows.
	tokenDigits = 16
	// withdrawalKey opens the key of a withdrawal's one commit, which ends with the head of the
	// store it was made at: no two withdrawals share a key, so none is taken for another's repeat.
	withdrawalKey = "loop/leftovers/withdrawn/"
	// gateLock is the lock a commit holds in the process state of the checkout it runs in (internal/authoritylock).
	gateLock = "gate.lock"
	// commitRecord is the record of its waits that every run of a commit's checks adds to as it
	// ends (internal/landing): the one file there that says a commit ran. The ignore rule and
	// the gate's lock do not: a command that only takes that lock leaves both.
	commitRecord = "landing_waits.json"
)

// commitFiles are what a commit writes in the process state of the checkout it runs in: the
// ignore rule (internal/processlock), the gate's lock, and the records of its waits and of its
// refusals (internal/landing). Any other entry there is a session's or a loop's, and holds the copy.
var commitFiles = []string{".gitignore", gateLock, commitRecord, "landing_refusals.json"}

// ownersData are the places whose entries are the owner's data: only the owner or the lead
// removes them, by hand, so a worker's environment is refused them as it is refused a push.
var ownersData = []string{"branch", "experiment"}

// place is one line of the table: somewhere files accumulate, what it holds,
// the state that makes an entry finished, the action, the function that lists
// its entries, and, on a line that deletes and on no other, the function that
// removes one finished entry. Its at is one literal, a root word and a colon
// before a slash path, and never a bare path: a directory written bare would
// make this package a reader of every document beneath it (internal/repoanalysis
// TrackedDocumentCensus). The roots checkout, data, sessions, temporary and
// cache are directories of the fact source; git, store and elsewhere are not.
type place struct {
	name, at, holds, finished, action string
	list                              func(f *facts, line place) []leftover
	remove                            func(h hygiene, path string) error
}

// leftover is one entry of a place, with the state the listing established,
// the rule that decided it, and its bytes when they were measured.
type leftover struct {
	place, action, path, state, rule string
	bytes                            int64
	measured                         bool
}

// hygiene is where a listing reads the machine, so a test supplies each fact:
// the trunk's checkout, the checkout this command runs in, the directory that
// holds the store, the store itself, which only a withdrawal opens, the
// harness's folder of this checkout's sessions, the temporary directory, the
// go build cache, this command's own executable, and the directories the
// data-root contract and the scorecards name. stores is
// what overgodb precheck -leftovers lists; registered, the file and directory
// locations the store registers; processes, the go commands and test binaries
// that run, but for this command's parent. removeTree is how a removal takes
// a directory away, which a test replaces with one that fails.
type hygiene struct {
	checkout, running, data, store, sessions, temporary, cache, executable string
	kept                                                                   []string
	stores                                                                 func() ([]leftover, error)
	registered, processes                                                  func() ([]string, error)
	removeTree                                                             func(path string) error
}

// facts is the fact source with what one listing reads once: git's registered
// worktrees, the trunk's checkout first; the checkout's untracked paths, a
// directory's ending in a slash; the store's registrations, or why they could
// not be read, once a line has asked for them; and the directories a line or
// the data-root contract names.
type facts struct {
	hygiene
	trees                     []landing.Worktree
	untracked, located, named []string
	unlocated                 error
	asked                     bool
}

// leftoverPlaces is the table. A line that only data tells from another
// shares its list function with it. The last line reads what the lines before
// it named.
var leftoverPlaces = []place{
	{"helper-copy", "checkout:" + agentWorktrees, "checkouts a session or an agent made to build or test in, here and under the temporary directory", "a commit ran in it and is on the trunk, no session holds it, git does not hold it locked and no commit runs in it, and it holds nothing uncommitted, nothing ignored and nothing in its process state but what a commit writes", actionDelete, worktrees, removeCopy},
	{"worktree", "git:worktrees", "every other registered worktree but the trunk's checkout and those inside process state", "its commit is on the trunk and it holds nothing uncommitted and nothing ignored", actionReport, worktrees, nil},
	{"branch", "git:refs/heads", "local branches but the trunk", "wholly in the trunk and checked out in no worktree", actionDelete, branches, removeBranch},
	{"stash", "git:stash", "stashed changes", "none: a stash may hold the only copy of work", actionReport, stashes, nil},
	{"store-backup", "store:", "superseded stores and sealed backups, as overgodb precheck -leftovers lists them", "none: a backup is never deleted", actionNever, storeBackups, nil},
	{"experiment", "data:experiments/", "what the runs of each experiment wrote, an entry a child of the experiment's directory", "a directory whose experiment a committed record names, which no committed record names in a store field and no store registration names, and which holds no store, no store backup and no run that is still going, in it or in its experiment's directory", actionDelete, outputs, removeEntry},
	{"checkpoint", "data:checkpoints/", "trained models", "none: a registered model is kept", actionReport, outputs, nil},
	{"scratch", "data:scratch/", "outputs of runs that left no record", "none until a maker is recorded", actionReport, outputs, nil},
	{"data-build", "data:build/", "build outputs kept outside the module", "none: the data root is where build outputs belong", actionReport, wholeDirectory, nil},
	{"session-scratch", "sessions:", "the harness's scratch folder of each session of this checkout", "none this command can read", actionReport, sessionFolders, nil},
	{"temporary-build", "temporary:go-build", "the folders a go command builds in", "no go command and no test binary runs, and it does not hold this command's own executable", actionDelete, temporaryBuilds, removeEntry},
	{"build-cache", "cache:", "the go build cache", "none: clearing it takes the owner's approval", actionReport, wholeDirectory, nil},
	{"bin", "checkout:bin/", "built executables", "no tracked file names it", actionReport, executables, nil},
	{"tmp", "checkout:tmp/", "short-lived probes, each in the folder named for the plan row that made it", "a folder named for a row that neither the working plan nor the plan of the trunk's commit holds, a step of which a commit on the trunk landed, and which no committed line and no store registration names", actionDelete, scratchEntries, removeEntry},
	{"build", "checkout:build/", "generated build inputs", "none here: go run ./cmd/buildscrub decides", actionReport, wholeDirectory, nil},
	{"release", "checkout:dist/", "release bundles", "none: the release command's output", actionReport, wholeDirectory, nil},
	{"python-place", "checkout:scripts/python_golden_references/", "the scripts that produce golden references", "none: the one place for Python", actionNever, wholeDirectory, nil},
	{"checkout-store", "checkout:" + gitauthority.CanonicalOvergoDBDirectory + "/", "a store inside the checkout", "none: a store is never deleted", actionNever, wholeDirectory, nil},
	{"docs-untracked", "checkout:docs/", "files under docs that git does not track", "none: listed, never deleted", actionNever, untrackedUnder, nil},
	{"media-samples", "checkout:docs/media_samples/", "media samples, each named by the digest of its bytes", "nothing reads it: no test and no read document carries its digest", actionReport, samples, nil},
	{"verification", "checkout:docs/verification/", "committed records of runs", "none: a record is kept, and one that names what the tree no longer has is history", actionNever, records, nil},
	{"process-state", "checkout:" + processlock.StateDirectory + "/", "locks, markers and the cycle checkout", "none: the processes that write it own it", actionNever, wholeDirectory, nil},
	{"undeclared", "elsewhere:", "directories under the checkout or the data root that no line above names", "none: give it a line, or remove it by hand", actionReport, undeclared, nil},
}

// listLeftovers reads every line's entries, each with its state, the rule
// that decided it and its bytes. It removes nothing. A line's directory is
// named, and so is what a line that never deletes lists: a superseded store
// beside the store is a backup, not an undeclared directory. With places
// named it lists those lines alone, for a removal: it reads only the facts
// they read, so a removal of helper copies never opens the store, and it
// walks only the entries a removal would take.
func listLeftovers(h hygiene, only ...string) ([]leftover, error) {
	trees, err := landing.Worktrees(context.Background(), h.checkout)
	if err != nil {
		return nil, err
	}
	f := &facts{hygiene: h, trees: trees, named: slices.Clone(h.kept)}
	if len(only) == 0 {
		untracked, err := gitRead(h.checkout, "ls-files", "--others", "--directory", "-z")
		if err != nil {
			return nil, err
		}
		f.untracked = strings.Split(untracked, "\x00")
	}
	var found []leftover
	for _, line := range leftoverPlaces {
		if len(only) != 0 && !slices.Contains(only, line.name) {
			continue
		}
		f.named = append(f.named, f.directory(line))
		for _, entry := range line.list(f, line) {
			entry.place, entry.action = line.name, line.action
			if line.action == actionNever {
				f.named = append(f.named, entry.path)
			}
			if len(only) == 0 || entry.state == stateFinished {
				entry = weighed(entry)
			}
			found = append(found, entry)
		}
	}
	return found, nil
}

// weighed walks an entry of a line that deletes or reports, summing its
// regular files; a branch, a stash and an entry that is never deleted are not
// measured. The walk descends no symbolic link and no junction, which Go
// reports as irregular: a recursive delete could follow one out of the entry,
// so on a line that deletes, a finished entry that holds one, or that cannot
// be walked, is not finished. Neither is one that holds, at any depth, a
// store, a store backup or the lock of a run that is still going: those are
// never deleted, whatever line the entry is on. On a line that reports
// nothing is deleted, so its entries are only measured.
func weighed(entry leftover) leftover {
	deletable, state := entry.action == actionDelete && entry.state == stateFinished, stateUnknown
	if !filepath.IsAbs(entry.path) || entry.action == actionNever {
		return entry
	}
	err := filepath.WalkDir(entry.path, func(path string, item fs.DirEntry, err error) error {
		switch {
		case err != nil:
		case deletable && item.Type()&(fs.ModeSymlink|fs.ModeIrregular) != 0:
			err = errors.New("it holds a link a recursive delete could follow: " + path)
		case deletable && item.IsDir() && holdsStore(path):
			state, err = stateHeld, errors.New("it holds a store or a store backup, which is never deleted: "+path)
		case deletable && item.Name() == processlock.OutputLock && runHolds(path):
			state, err = stateHeld, errors.New("a run holds it: its lock "+path+" is taken, or cannot be")
		case item.Type().IsRegular():
			var info fs.FileInfo
			if info, err = item.Info(); err == nil {
				entry.bytes += info.Size()
			}
		}
		return err
	})
	if deletable && err != nil {
		entry.state, entry.rule = state, err.Error()
	}
	entry.measured = err == nil
	return entry
}

// holdsStore reports whether a directory is a store or a backup of one: by
// the name a store has, or by the seal a backup carries. A seal that is there
// and cannot be read is still a backup's.
func holdsStore(directory string) bool {
	_, err := overgodb.ReadBackupSeal(directory)
	return filepath.Base(directory) == gitauthority.CanonicalOvergoDBDirectory || !errors.Is(err, fs.ErrNotExist)
}

// runHolds reports whether a run holds the lock file it left in its output
// directory, by taking the lock and letting it go at once. A lock that cannot
// be taken for any reason is held: only a lock taken says the run has ended.
func runHolds(lock string) bool {
	taken, err := processlock.Acquire(lock, clioptions.PrivateFileMode)
	return err != nil || taken.Close() != nil
}

// directory is the directory a line names: its root word's, joined with the
// slash path after it. A root that is no directory gives none.
func (f *facts) directory(line place) string {
	word, rest, _ := strings.Cut(line.at, ":")
	base := map[string]string{"checkout": f.checkout, "data": f.data, "sessions": f.sessions, "temporary": f.temporary, "cache": f.cache}[word]
	if base == "" {
		return ""
	}
	return filepath.Join(base, rest)
}

// under reports whether path is root or lies beneath it, by file identity: a
// short name, a link or another letter case is the same place, and a path
// that does not exist lies where it would. A root that cannot be read holds
// nothing.
func under(root, path string) bool {
	inside, err := pathidentity.Contains(root, path)
	return path != "" && err == nil && inside
}

// failed is the one entry of a place whose source failed: a failed source is
// never silence, so it reads unknown with the failure as its rule.
func failed(path string, err error) []leftover {
	return []leftover{{path: path, state: stateUnknown, rule: err.Error()}}
}

// each lists the children of directory with the state and rule decide gives
// each; a child it gives no state is not the line's. An absent directory
// holds nothing.
func each(directory string, decide func(child os.DirEntry, path string) (state, rule string)) []leftover {
	children, err := os.ReadDir(directory)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return failed(directory, err)
	}
	var found []leftover
	for _, child := range children {
		path := filepath.Join(directory, child.Name())
		if state, rule := decide(child, path); state != "" {
			found = append(found, leftover{path: path, state: state, rule: rule})
		}
	}
	return found
}

// worktrees lists the registered worktrees of one of two lines. The helper
// copies lie under the temporary directory or in the agents' folder, which is
// the directory their line names; the other line takes the rest but the
// trunk's checkout and those inside its process state. The helper line also
// lists what the agents' folder holds that git does not register.
func worktrees(f *facts, line place) []leftover {
	agents, helpers := filepath.Join(f.checkout, agentWorktrees), f.directory(line) != ""
	var found []leftover
	for _, tree := range f.trees[1:] {
		helper := under(f.temporary, tree.Root) || under(agents, tree.Root)
		if helper == helpers && !under(filepath.Join(f.checkout, processlock.StateDirectory), tree.Root) {
			state, rule := f.copyState(tree, helpers)
			found = append(found, leftover{path: tree.Root, state: state, rule: rule})
		}
	}
	if !helpers {
		return found
	}
	return append(found, each(agents, func(_ os.DirEntry, path string) (string, string) {
		if slices.ContainsFunc(f.trees, func(tree landing.Worktree) bool { return under(tree.Root, path) && under(path, tree.Root) }) {
			return "", ""
		}
		return stateUnknown, "no git link: its commit cannot be read"
	})...)
}

// copyState decides one registered worktree. It is held by a commit the trunk
// lacks, by the session its process state names, and by any path git's status
// reports, ignored ones among them; an unstaged deletion is a change unless
// every tracked file is missing, which is a copy emptied of its content. A
// worktree whose directory is gone cannot be read, so it is unknown.
//
// A helper copy is removed when it is finished, so more must hold of it. It
// is not the checkout this command runs in. Its process state holds only
// what a commit writes, which git's status then leaves out; any other entry
// there is a session's or a loop's and holds it. A commit ran in it, which
// only the record of a commit's run says: a copy that holds nothing and has
// no such record is one a session may have just made, to build in with
// nothing yet to show, and is unknown. Git does not hold it locked: a session
// locks a copy to keep it. And no commit runs in it now: the gate's lock,
// when the file is there, is taken and let go, last of all, so a copy kept
// for another reason is never probed.
func (f *facts) copyState(tree landing.Worktree, helper bool) (state, rule string) {
	trunk, written := f.trees[0], filepath.Join(tree.Root, processlock.StateDirectory)
	if trunk.Branch == "" {
		return stateUnknown, noTrunk
	}
	if _, err := gitRead(f.checkout, "merge-base", "--is-ancestor", tree.Head, trunk.Head); errors.Is(err, noAnswer) {
		return stateHeld, fmt.Sprintf("its commit %.*s is not on the trunk", commitAbbreviation, tree.Head)
	} else if err != nil {
		return stateUnknown, err.Error()
	}
	if holder, err := os.ReadFile(filepath.Join(written, guard.WorktreeHolder)); err == nil {
		return stateHeld, "the session " + strings.TrimSpace(string(holder)) + " holds it"
	}
	locked, ran := false, !helper
	if helper {
		if same, _ := pathidentity.Same(f.running, tree.Root); same {
			return stateHeld, "it is the checkout this command runs in"
		}
		entries, _ := os.ReadDir(written)
		for _, entry := range entries {
			if !slices.Contains(commitFiles, entry.Name()) {
				return stateHeld, "its process state holds " + entry.Name() + ", which no commit writes"
			}
			locked, ran = locked || entry.Name() == gateLock, ran || entry.Name() == commitRecord
		}
	}
	status, err := gitRead(tree.Root, "status", "--porcelain=v1", "-z", "--ignored")
	paths, parseErr := repoanalysis.ParseDirtyStatus([]byte(status))
	tracked, trackedErr := gitRead(tree.Root, "ls-files", "-z")
	if err = errors.Join(err, parseErr, trackedErr); err != nil {
		return stateUnknown, err.Error()
	}
	paths = slices.DeleteFunc(paths, func(path repoanalysis.DirtyPath) bool {
		return helper && strings.TrimSuffix(path.Path, "/") == processlock.StateDirectory
	})
	deleted := 0
	for _, path := range paths {
		if path.IndexStatus+path.WorktreeStatus == " D" {
			deleted++
		}
	}
	files := strings.Count(tracked, "\x00")
	if len(paths) != 0 && (deleted != len(paths) || deleted != files) {
		return stateHeld, fmt.Sprintf("it holds %d uncommitted or ignored paths, the first %s, %d of them deletions among its %d tracked files", len(paths), paths[0].Path, deleted, files)
	}
	if !ran {
		return stateUnknown, "it has no process state that holds the record of a commit's run, " + commitRecord + ", so nothing says a commit ran in it"
	}
	if helper {
		kept, err := gitRead(tree.Root, "rev-parse", "--path-format=absolute", "--git-path", "locked")
		if err != nil {
			return stateUnknown, err.Error()
		}
		// The file's text is the reason given; one that is there and cannot be read locks as well.
		if reason, err := os.ReadFile(strings.TrimSpace(kept)); !errors.Is(err, fs.ErrNotExist) {
			return stateHeld, "git holds it locked against removal: " + strings.TrimSpace(string(reason))
		}
	}
	if locked {
		lock, err := authoritylock.Acquire(tree.Root)
		if err != nil {
			return stateHeld, "a commit is running in it: " + err.Error()
		}
		defer lock.Close()
	}
	if len(paths) == 0 {
		return stateFinished, fmt.Sprintf("its commit %.*s is on the trunk and it holds nothing uncommitted and nothing ignored", commitAbbreviation, tree.Head)
	}
	return stateFinished, fmt.Sprintf("its commit is on the trunk and it is emptied: every one of its %d tracked files is missing from disk", deleted)
}

// branches lists the local branches but the trunk.
func branches(f *facts, line place) []leftover {
	trunk := f.trees[0]
	merged, mergedErr := gitRead(f.checkout, "for-each-ref", "--merged="+trunk.Head, "--format=%(refname)", "refs/heads")
	listed, err := gitRead(f.checkout, "for-each-ref", "--format=%(refname)%00%(worktreepath)", "refs/heads")
	if err = errors.Join(err, mergedErr); err != nil {
		return failed(line.at, err)
	}
	var found []leftover
	for row := range strings.Lines(listed) {
		name, tree, _ := strings.Cut(strings.TrimSpace(row), "\x00")
		entry := leftover{path: name, state: stateFinished, rule: line.finished}
		switch {
		case name == trunk.Branch:
			continue
		case trunk.Branch == "":
			entry.state, entry.rule = stateUnknown, noTrunk
		case tree != "":
			entry.state, entry.rule = stateHeld, "the worktree "+tree+" has it checked out"
		case !slices.Contains(strings.Fields(merged), name):
			entry.state, entry.rule = stateHeld, "it holds commits the trunk lacks"
		}
		found = append(found, entry)
	}
	return found
}

// stashes lists the stashes, each with what it says it holds.
func stashes(f *facts, line place) []leftover {
	listed, err := gitRead(f.checkout, "stash", "list")
	if err != nil {
		return failed(line.at, err)
	}
	var found []leftover
	for row := range strings.Lines(listed) {
		name, described, _ := strings.Cut(strings.TrimSpace(row), ": ")
		found = append(found, leftover{path: name, state: stateUnknown, rule: line.finished + ": " + described})
	}
	return found
}

// storeBackups lists what the store's own command lists, by its kinds and
// with no rule of this command's.
func storeBackups(f *facts, line place) []leftover {
	found, err := f.stores()
	if err != nil {
		return failed(line.at, err)
	}
	return found
}

// citations reads the lines of the commit the trunk's checkout holds that
// carry the name of a line's directory with either slash, under scope and
// outside the work records, which name documents without reading them.
func (f *facts) citations(line place, scope string) ([]string, error) {
	name := filepath.Base(f.directory(line))
	arguments := []string{"grep", "-n", "-z", "-I", "-i", "-F", "-e", name + "/", "-e", name + `\`, "HEAD", "--", scope}
	for _, record := range repoanalysis.WorkRecords {
		arguments = append(arguments, ":(exclude)"+record)
	}
	found, err := gitRead(f.checkout, arguments...)
	if errors.Is(err, noAnswer) {
		// No line matches.
		err = nil
	}
	return slices.Collect(strings.Lines(found)), err
}

// cited says which of git grep's lines names path, as its document and line
// number, or nothing. A line names a path when a run of path characters in it
// carries the path from base as whole elements, in any letter case and with
// either slash: the tail of a longer name is not it, and neither is a name it
// only opens. A full stop that ends the run ends a sentence. A path that
// holds any other character, a space or a sign, lies in no run, so no line
// is found to name it however a document writes it: unnamed keeps it.
func cited(lines []string, base, path string) string {
	inside, _ := filepath.Rel(base, path)
	inside = "/" + strings.ToLower(filepath.ToSlash(inside)) + "/"
	for _, line := range lines {
		document, rest, _ := strings.Cut(line, "\x00")
		number, text, _ := strings.Cut(rest, "\x00")
		text = strings.ToLower(strings.ReplaceAll(text, `\`, "/"))
		for run := range strings.FieldsFuncSeq(text, func(r rune) bool { return !strings.ContainsRune(pathRunes, r) }) {
			// The run is closed with runEnd: a slash, a character no path holds, and a slash. A path
			// holds no such character, so it matches only in the run and the first of those slashes:
			// the run names it wherever the run carries it. A path given with runEnd after it matches
			// only where the run ends: a run that ends with it names it, and a run that goes on
			// beneath it does not.
			if strings.Contains("/"+strings.TrimRight(run, ".")+runEnd, inside) {
				return document + ":" + number + " names it"
			}
		}
	}
	return ""
}

// unnamed decides an entry that no line names and nothing else holds: it is
// finished by its line's rule, unless no line could have named it. The test
// is the search itself, on a line of no document that is the entry's own path
// and no more: when even that is not found, the search's silence says nothing
// of the entry, and it is unknown.
func unnamed(line place, base, path string) (state, rule string) {
	if cited([]string{"\x00\x00" + path}, base, path) == "" {
		return stateUnknown, "its name holds a character the search of the committed lines ends a path at, so no line that names it could be found"
	}
	return stateFinished, line.finished
}

// storeFields keeps, of git grep's lines, those of a field whose key is
// store. A record's field opens a line with its key and a colon, and a line
// that opens with white space continues the field above it: a path on a
// continuation line is the store's as surely as one on the first. Each record
// is read whole, at the commit the lines came from; a line whose record or
// whose number cannot be read fails the listing, so it never reads as no field.
func (f *facts) storeFields(lines []string) ([]string, error) {
	var kept []string
	for _, line := range lines {
		document, rest, _ := strings.Cut(line, "\x00")
		number, _, _ := strings.Cut(rest, "\x00")
		text, err := gitRead(f.checkout, "cat-file", "blob", document)
		rows := strings.Split(text, "\n")
		at, numberErr := strconv.Atoi(number)
		if err = errors.Join(err, numberErr); err != nil || at < 1 || at > len(rows) {
			return nil, fmt.Errorf("git grep's line %q is no numbered line of its record: %v", line, err)
		}
		for at > 1 && strings.TrimLeft(rows[at-1], " \t") != rows[at-1] {
			at--
		}
		if strings.HasPrefix(strings.ToLower(rows[at-1]), "store:") {
			kept = append(kept, line)
		}
	}
	return kept, nil
}

// outputs lists a data directory's entries: the children of each experiment's
// directory on the line that deletes, the directory's own children on a line
// that reports. An entry is held by a committed record's line that names it,
// and by a store registration that is it, lies beneath it or holds it. Only
// the line that deletes finishes one, and then only a directory whose
// experiment a committed record names: a file there is a log, a marker or a
// lock whose owner nothing says, and an experiment with no record has nothing
// that keeps its result. On that line a record holds an entry only through a
// store field: a record names a run's output directory to say where the run
// wrote, which keeps nothing once the record is committed, and the rule of an
// entry so named says where. A run that is still going holds every entry of
// the experiment whose own directory it writes in: its lock lies there, above
// the entries, where the walk of an entry never sees it. The lock is probed
// only when its file is there, since taking it makes the file.
func outputs(f *facts, line place) []leftover {
	directory := f.directory(line)
	lines, err := f.citations(line, "docs")
	holding := lines
	if err == nil && line.action == actionDelete {
		holding, err = f.storeFields(lines)
	}
	if err != nil {
		return failed(directory, err)
	}
	located := f.locatedUnder(directory)
	decide := func(child os.DirEntry, path string) (string, string) {
		if at := cited(holding, f.data, path); at != "" {
			return stateHeld, at
		}
		for _, location := range located {
			if under(path, location) || under(location, path) {
				return stateHeld, "the store registers " + location
			}
		}
		switch {
		case f.unlocated != nil:
			return stateUnknown, f.unlocated.Error()
		case line.action != actionDelete:
			return stateUnknown, line.finished
		case !child.IsDir():
			return stateUnknown, "a file beside an experiment's outputs, and nothing says whose"
		case cited(lines, f.data, filepath.Dir(path)) == "":
			return stateUnknown, "no committed record names its experiment"
		}
		lock := filepath.Join(filepath.Dir(path), processlock.OutputLock)
		if _, err := os.Lstat(lock); !errors.Is(err, fs.ErrNotExist) && runHolds(lock) {
			return stateHeld, "a run holds its experiment: its lock " + lock + " is taken, or cannot be"
		}
		state, rule := unnamed(line, f.data, path)
		if at := cited(lines, f.data, path); at != "" {
			rule += "; " + at + " outside a store field, which keeps nothing"
		}
		return state, rule
	}
	if line.action != actionDelete {
		return each(directory, decide)
	}
	var found []leftover
	beside := each(directory, func(experiment os.DirEntry, path string) (string, string) {
		if !experiment.IsDir() {
			return decide(experiment, path)
		}
		found = append(found, each(path, decide)...)
		return "", ""
	})
	return append(beside, found...)
}

// executables lists a directory of the checkout whose entries a tracked file
// may name, anywhere in the tree.
func executables(f *facts, line place) []leftover {
	lines, err := f.citations(line, ".")
	if err != nil {
		return failed(f.directory(line), err)
	}
	return each(f.directory(line), func(_ os.DirEntry, path string) (string, string) {
		if at := cited(lines, f.checkout, path); at != "" {
			return stateHeld, at
		}
		return unnamed(line, f.checkout, path)
	})
}

// locatedUnder is the store's registrations that lie in directory, so that
// each entry there is compared with those alone: a comparison is by file
// identity and reads the disk. The store is read when the first line asks,
// so a listing of lines that never ask does not open it.
func (f *facts) locatedUnder(directory string) (inside []string) {
	if !f.asked {
		f.located, f.unlocated = f.registered()
		f.asked = true
	}
	for _, location := range f.located {
		if under(directory, location) {
			inside = append(inside, location)
		}
	}
	return inside
}

// landings reads, at the commit the trunk's checkout holds, the plan and the
// rows that landed: each row a commit on the first-parent chain landed a step
// of, with the newest such commit. A landing is a trailer of the commit's
// message as git reads trailers, naming a row and a step. A merge leaves the
// trunk's plan as it was, so a landing it brought in is none of the trunk's.
func (f *facts) landings() (committed plan.Plan, landed map[string]string, err error) {
	trunk := f.trees[0]
	if trunk.Branch == "" {
		return committed, nil, errors.New(noTrunk)
	}
	document, err := gitRead(f.checkout, "cat-file", "blob", trunk.Head+":"+plan.Path)
	if err == nil {
		committed, err = plan.Parse([]byte(document))
	}
	log, logErr := gitRead(f.checkout, "log", "--first-parent", "--format=%H%x00%(trailers:key="+plan.LandedRowTrailer+",valueonly,unfold,separator=%x00)", trunk.Head)
	landed = map[string]string{}
	for row := range strings.Lines(log) {
		steps := strings.Split(strings.TrimSpace(row), "\x00")
		for _, step := range steps[1:] {
			if id, _, found := strings.Cut(step, "/"); found && landed[id] == "" {
				landed[id] = steps[0]
			}
		}
	}
	return committed, landed, errors.Join(err, logErr)
}

// outOfPlace says where a file belongs when the scratch folder is not the
// place for its kind: an executable or a Python file, by the ending of its
// name in any letter case. Another file, and a directory, has no other place.
func outOfPlace(file os.DirEntry) string {
	switch ending := strings.ToLower(filepath.Ext(file.Name())); {
	case file.IsDir():
	case ending == ".exe":
		return "an executable belongs in bin/"
	case ending == repoanalysis.PythonSuffix:
		return "the one place for Python is " + repoanalysis.PythonPlace + ", which admits only what produces a golden reference: move it there, or write the tool in Go"
	}
	return ""
}

// scratchEntries lists the children of the checkout's scratch folder, each
// tied to the plan row that made it. A folder is a row's by its name. It is
// held while the working plan holds the row, while a committed line names it,
// while the store registers a location in it, and while the plan of the
// trunk's commit holds the row; a plan holds a row in any letter case, since
// two names that differ only in case are one folder on Windows. It is finished
// when neither plan holds the row and a commit on the trunk landed a step of a
// row of exactly its name, and unowned when none did. A plan, a trunk or
// registrations that cannot be read leave it unknown. A file is never
// finished: the scratch module's own two are kept, one a committed line or the
// store names is held, an executable or a Python file is misplaced, and any
// other is unowned. A folder's rule also names the misplaced files directly in
// it; a folder that cannot be read names none, and is not finished either,
// since the walk that measures it fails.
func scratchEntries(f *facts, line place) []leftover {
	directory := f.directory(line)
	lines, err := f.citations(line, ".")
	if err != nil {
		return failed(directory, err)
	}
	working, workingErr := plan.Load(filepath.Join(f.checkout, plan.Path))
	committed, landed, trunkErr := f.landings()
	located := f.locatedUnder(directory)
	return each(directory, func(child os.DirEntry, path string) (state, rule string) {
		name, holder := child.Name(), cited(lines, f.checkout, path)
		for _, location := range located {
			if same, _ := pathidentity.Same(path, location); holder == "" && (same || under(path, location)) {
				holder = "the store registers " + location
			}
		}
		holds := func(document plan.Plan) bool {
			return slices.ContainsFunc(document.Items, func(row plan.Item) bool { return strings.EqualFold(row.ID, name) })
		}
		switch belongs := outOfPlace(child); {
		case child.IsDir():
		case name == "go.mod" || name == "go.sum":
			return stateKept, "the scratch module's own file"
		case holder != "":
			return stateHeld, holder
		case belongs != "":
			return stateMisplaced, belongs
		default:
			return stateUnowned, "a file outside a row's folder"
		}
		switch {
		case workingErr != nil:
			state, rule = stateUnknown, workingErr.Error()
		case holds(working):
			state, rule = stateHeld, "the working plan holds its row"
		case holder != "":
			state, rule = stateHeld, holder
		case trunkErr != nil:
			state, rule = stateUnknown, trunkErr.Error()
		case holds(committed):
			state, rule = stateHeld, "the plan of the trunk's commit holds its row"
		case landed[name] == "":
			state, rule = stateUnowned, "no commit on the trunk landed a step of a row of this name"
		case f.unlocated != nil:
			state, rule = stateUnknown, f.unlocated.Error()
		default:
			state, rule = stateFinished, fmt.Sprintf("neither plan holds its row, and the commit %.*s landed a step of it", commitAbbreviation, landed[name])
		}
		inside, _ := os.ReadDir(path)
		for _, file := range inside {
			if belongs := outOfPlace(file); belongs != "" {
				rule += "; it holds " + file.Name() + ": " + belongs
			}
		}
		return state, rule
	})
}

// sessionFolders lists the harness's folder of each session. Where the
// folders are is computed from the checkout's path by a rule read off their
// names, so an absent folder is reported as the path tried, not as no session.
func sessionFolders(f *facts, line place) []leftover {
	if _, err := os.Stat(f.sessions); err != nil {
		return failed(f.sessions, err)
	}
	return each(f.sessions, func(os.DirEntry, string) (string, string) { return stateUnknown, line.finished })
}

// temporaryBuilds lists the folders whose name opens with the line's. The
// processes are read here, after every line before it has ended its own
// subprocesses: the store's listing is a go command.
func temporaryBuilds(f *facts, line place) []leftover {
	directory, prefix := filepath.Split(f.directory(line))
	running, err := f.processes()
	return each(directory, func(child os.DirEntry, path string) (string, string) {
		switch {
		case !strings.HasPrefix(child.Name(), prefix):
			return "", ""
		case under(path, f.executable):
			return stateHeld, "it holds this command's own executable"
		case err != nil:
			return stateUnknown, err.Error()
		case len(running) != 0:
			return stateHeld, "a go command or a test binary runs: " + running[0]
		}
		return stateFinished, line.finished
	})
}

// untrackedUnder lists the checkout's untracked paths under a line's
// directory, which the checkout holds itself.
func untrackedUnder(f *facts, line place) []leftover {
	var found []leftover
	for _, path := range f.untracked {
		if strings.HasPrefix(path, filepath.Base(f.directory(line))+"/") {
			found = append(found, leftover{path: filepath.Join(f.checkout, path), state: stateKept, rule: line.finished})
		}
	}
	return found
}

// samples lists the documents the checkout holds under a line's directory,
// tracked or not yet ignored, by the census a commit's checks hold every
// document to. One that something names is held. One that nothing reads is
// finished, whether a later sample replaced it or an export wrote it and no
// record names it yet: the listing cannot tell the two apart, and a commit's
// checks refuse either.
func samples(f *facts, line place) []leftover {
	census, err := repoanalysis.CheckoutDocuments(context.Background(), f.checkout)
	if err != nil {
		return failed(f.directory(line), err)
	}
	live := repoanalysis.LiveDocuments(census)
	_, beneath, _ := strings.Cut(line.at, ":")
	var found []leftover
	for _, document := range census {
		if !strings.HasPrefix(document.Path, beneath) {
			continue
		}
		entry := leftover{path: filepath.Join(f.checkout, document.Path), state: stateHeld, rule: "named by " + strings.Join(slices.Concat(document.Writers, document.Readers, document.ReferencedBy), ", ")}
		if !live[document.Path] {
			entry.state, entry.rule = stateFinished, line.finished
		}
		found = append(found, entry)
	}
	return found
}

// records lists the records of runs the checkout holds under a line's
// directory, at any depth, each kept. One whose text names a path or a test
// the tree no longer has is history: its rule says what it names, and the
// record stays as it was written. A record that is no text, a picture, names
// nothing.
func records(f *facts, line place) []leftover {
	documents, err := repoanalysis.TrackedDocuments(context.Background(), f.checkout)
	if err != nil {
		return failed(f.directory(line), err)
	}
	_, beneath, _ := strings.Cut(line.at, ":")
	texts := map[string]string{}
	var found []leftover
	for _, document := range documents {
		if !strings.HasPrefix(document, beneath) {
			continue
		}
		path := filepath.Join(f.checkout, document)
		if data, err := os.ReadFile(path); err == nil && utf8.Valid(data) {
			texts[path] = string(data)
		}
		found = append(found, leftover{path: path, state: stateKept, rule: line.finished})
	}
	absent := map[string][]string{}
	for _, dead := range repoanalysis.DeadReferences(f.checkout, texts) {
		// A reference holds no colon, so the last one ends the record's path.
		at := strings.LastIndex(dead, ": ")
		absent[dead[:at]] = append(absent[dead[:at]], dead[at+len(": "):])
	}
	for index, entry := range found {
		if names := absent[entry.path]; len(names) != 0 {
			found[index].rule = "history: it names what the tree no longer has: " + strings.Join(names, ", ")
		}
	}
	return found
}

// wholeDirectory lists a line's directory as one entry when it is there:
// kept on a line that never deletes, and unknown on one that reports.
func wholeDirectory(f *facts, line place) []leftover {
	directory, state := f.directory(line), stateUnknown
	if _, err := os.Lstat(directory); errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if line.action == actionNever {
		state = stateKept
	}
	return []leftover{{path: directory, state: state, rule: line.finished}}
}

// undeclared lists the directories nothing names: among the checkout's
// untracked ones, which git lists without a walk, and among the data root's
// own when the checkout does not hold it. A directory above a named one is
// read one level and its directories are tested the same way, since git
// lists a wholly untracked directory once.
func undeclared(f *facts, line place) (found []leftover) {
	var test func(path string, above bool)
	test = func(path string, above bool) {
		for _, directory := range f.named {
			if under(directory, path) {
				return
			}
			above = above || under(path, directory)
		}
		if !above {
			found = append(found, leftover{path: path, state: stateUndeclared, rule: line.finished})
			return
		}
		// Two statements: the read appends to found through test, and Go does not say
		// whether an append reads its slice before or after a call among its operands.
		unread := each(path, func(child os.DirEntry, inner string) (string, string) {
			if child.IsDir() {
				test(inner, false)
			}
			return "", ""
		})
		found = append(found, unread...)
	}
	for _, path := range f.untracked {
		if strings.HasSuffix(path, "/") {
			test(filepath.Join(f.checkout, path), false)
		}
	}
	if !under(f.checkout, f.data) {
		test(f.data, true)
	}
	return found
}

// readTool runs one tool that only reads and returns what it printed; a
// failure carries what it wrote to standard error.
func readTool(environment []string, arguments ...string) (string, error) {
	var printed, diagnostics strings.Builder
	err := runToolEnv(environment, nil, &printed, &diagnostics, arguments...)
	if err != nil {
		err = fmt.Errorf("%s: %w: %s", strings.Join(arguments, " "), err, strings.TrimSpace(diagnostics.String()))
	}
	return printed.String(), err
}

// gitRead runs one git command that only reads, in directory. A command
// whose answer is no, or nothing, fails with noAnswer.
func gitRead(directory string, arguments ...string) (string, error) {
	return readTool(gitauthority.ReaderEnvironment(), append([]string{"git", "-C", directory, "--no-replace-objects"}, arguments...)...)
}

// decodeStoreLeftovers reads the list overgodb precheck -leftovers printed:
// each store an earlier operation left, with its kind and whether it is the
// newest of that kind.
func decodeStoreLeftovers(printed []byte) (found []leftover, err error) {
	var listed []struct {
		Path, Kind string
		Keep       bool
	}
	if err := json.Unmarshal(printed, &listed); err != nil {
		return nil, fmt.Errorf("overgodb precheck -leftovers: %w", err)
	}
	for _, store := range listed {
		if store.Path == "" || store.Kind == "" {
			return nil, fmt.Errorf("overgodb precheck -leftovers listed an entry with no path or no kind: %s", printed)
		}
		found = append(found, leftover{path: filepath.Clean(store.Path), state: stateKept, rule: fmt.Sprintf("a %s is never deleted (the newest of its kind: %v)", store.Kind, store.Keep)})
	}
	return found, nil
}

// storeLocations reads the paths of the locations on this machine that the
// store at root registers, through a read-only open.
func storeLocations(root string, page int) ([]string, error) {
	store, err := overgodb.OpenReadOnly(root)
	if err != nil {
		return nil, err
	}
	defer store.Close()
	located, err := localLocations(store, page)
	var values []string
	for _, location := range located {
		values = append(values, location.Value)
	}
	return values, err
}

// localLocations reads the locations on this machine, a file's and a
// directory's, that an open store registers, each with its artifact, a page
// of the store's catalog at a time.
func localLocations(store *overgodb.Store, page int) (located []artifact.Location, err error) {
	query, ctx := overgodb.Query{Projection: overgodb.ProjectArtifacts, MaxResults: page}, context.Background()
	for {
		result, err := store.Query(ctx, query)
		if err != nil {
			return nil, err
		}
		for _, descriptor := range result.Artifacts {
			locations, err := store.Locations(ctx, descriptor.ID)
			if err != nil {
				return nil, err
			}
			for _, location := range locations {
				if location.Kind != artifact.LocationRemote {
					located = append(located, location)
				}
			}
		}
		if query.Cursor = result.Next; query.Cursor == nil {
			return located, nil
		}
	}
}

// goProcesses reads the rows tasklist prints, an image name and a process id
// first, and returns the go commands and test binaries among them but for the
// process parent: go run is this command's own parent. A list with no row is
// refused, since this command itself runs.
func goProcesses(listed string, parent int) (running []string, err error) {
	rows, err := csv.NewReader(strings.NewReader(listed)).ReadAll()
	if err == nil && len(rows) == 0 {
		err = errors.New("tasklist listed no process")
	}
	for _, row := range rows {
		if len(row) <= 1 {
			return nil, fmt.Errorf("tasklist listed a row with no process id: %q", row)
		}
		if name := strings.ToLower(row[0]); (name == "go.exe" || strings.HasSuffix(name, ".test.exe")) && row[1] != fmt.Sprint(parent) {
			running = append(running, row[0])
		}
	}
	return running, err
}

// leftoversCommand prints the listing, an entry a line of six fields (action,
// state, bytes, place, path, rule) and then the table, a line a place with
// what the listing found there, and writes the same text as a receipt in the
// process state of the trunk's checkout: the guard refuses to redirect this
// command's output into a file. It removes nothing.
func leftoversCommand(h hygiene, output io.Writer) error {
	found, err := listLeftovers(h)
	head, headErr := gitRead(h.checkout, "rev-parse", "HEAD")
	if err = errors.Join(err, headErr, processlock.EnsureStateDirectory(h.checkout)); err != nil {
		return err
	}
	var text strings.Builder
	entries, bytes := map[string]int{}, map[string]int64{}
	for _, entry := range found {
		size := "-"
		if entry.measured {
			size = fmt.Sprint(entry.bytes)
			bytes[entry.place] += entry.bytes
		}
		fmt.Fprintf(&text, "%s\t%s\t%s\t%s\t%s\t%s\n", entry.action, entry.state, size, entry.place, entry.path, strings.Join(strings.Fields(entry.rule), " "))
		entries[entry.place]++
		entries[entry.place+entry.state]++
	}
	for _, line := range leftoverPlaces {
		fmt.Fprintf(&text, "# %s at %s: %s; finished: %s; a finished entry: %s; %d entries, %d bytes measured, %d finished\n",
			line.name, line.at, line.holds, line.finished, line.action, entries[line.name], bytes[line.name], entries[line.name+stateFinished])
	}
	receipt := filepath.Join(h.checkout, processlock.StateDirectory, receiptPrefix+head[:min(len(head), commitAbbreviation)]+".txt")
	if err := os.WriteFile(receipt, []byte(text.String()), clioptions.PrivateFileMode); err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "%sloop: wrote this listing to %s\n", text.String(), receipt)
	return err
}

// sessionFolderName turns a checkout's path into the name of the folder the
// harness keeps that checkout's sessions in, as cmd/loophook memoryTexts
// turns it into the name of its memory folder. The folders seen also turn an
// underscore and a dot into a hyphen; a folder not found is reported as the
// path tried.
var sessionFolderName = strings.NewReplacer(":", "-", `\`, "-", "/", "-")

// said is one line of a removal: what was done with an entry or would be,
// its bytes when they were measured, its place, its path, and the rule or the
// failure, on one line.
func (entry leftover) said(verb, rule string) string {
	size := "-"
	if entry.measured {
		size = fmt.Sprint(entry.bytes)
	}
	return fmt.Sprintf("%s\t%s\t%s\t%s\t%s\n", verb, size, entry.place, entry.path, strings.Join(strings.Fields(rule), " "))
}

// removeEntry removes a finished entry that is a directory and no more: an
// experiment's output, a build folder, a row's scratch folder.
func removeEntry(h hygiene, path string) error { return h.removeTree(path) }

// removeCopy removes a helper copy's directory and then has git clear that
// one registration, only once the directory is gone: asked while it stands,
// git removes the directory itself and follows a junction in it, which is how
// 170 GB of weights were lost. It never prunes, which would also clear the
// registration of a copy whose directory is gone and whose state is unknown.
// When git refuses, a copy locked since it was listed, the failure says what
// is so by then: the directory is gone and the registration stays.
func removeCopy(h hygiene, path string) error {
	if err := h.removeTree(path); err != nil {
		return err
	}
	if _, err := os.Lstat(path); !errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("%s is still there after its removal (%v), so git keeps its registration", path, err)
	}
	if err := gitWrite(h.checkout, "worktree", "remove", path); err != nil {
		return fmt.Errorf("its directory is gone, and git still registers it: %w", err)
	}
	return nil
}

// removeBranch deletes a local branch as git does when it is merged, and
// never by force: a branch git finds unmerged stays.
func removeBranch(h hygiene, name string) error {
	return gitWrite(h.checkout, "branch", "-d", strings.TrimPrefix(name, "refs/heads/"))
}

// removing takes the lock of the one removal that runs at a time, without
// waiting: a second removal is refused, since each acts on a listing of its own.
func (h hygiene) removing() (*processlock.Lock, error) {
	if err := processlock.EnsureStateDirectory(h.checkout); err != nil {
		return nil, err
	}
	lock, err := processlock.Acquire(filepath.Join(h.checkout, processlock.StateDirectory, removalLock), clioptions.PrivateFileMode)
	if err != nil {
		return nil, fmt.Errorf("another removal holds the lock, or it cannot be taken, so nothing was removed: %w", err)
	}
	return lock, nil
}

// removeFinished removes each finished entry of a listing of one line, by
// that line's remover, and writes a line for each to output and to the
// receipt in the process state of the trunk's checkout: the word given for
// what was done, removed or withdrawn, or left with
// the failure, or part-removed when the failure came after the entry left its
// path, so that no line says an entry stands that does not: its directory is
// gone, or stands under the other name the failure gives. A failure goes on
// to the next entry; the failures are returned together.
func (h hygiene) removeFinished(remove func(h hygiene, path string) error, done string, found []leftover, output io.Writer) error {
	receipt, err := os.OpenFile(filepath.Join(h.checkout, processlock.StateDirectory, removalReceipt), os.O_APPEND|os.O_CREATE|os.O_WRONLY, clioptions.PrivateFileMode)
	if err != nil {
		return err
	}
	for _, entry := range found {
		if entry.state != stateFinished {
			continue
		}
		did := entry.said(done, entry.rule)
		if failure := remove(h, entry.path); failure != nil {
			verb := "left"
			if _, gone := os.Lstat(entry.path); filepath.IsAbs(entry.path) && errors.Is(gone, fs.ErrNotExist) {
				verb = "part-removed"
			}
			did, err = entry.said(verb, failure.Error()), errors.Join(err, fmt.Errorf("%s: %w", entry.path, failure))
		}
		_, writeErr := io.WriteString(io.MultiWriter(output, receipt), did)
		err = errors.Join(err, writeErr)
	}
	return errors.Join(err, receipt.Close())
}

// deleteLeftovers is the one command that removes leftovers. Named a place
// whose finished entries are deleted, it lists that place afresh under the
// removal lock and prints a line for each finished entry, a line with its
// state and rule for every other, and a token: the first digits of a digest
// over each finished entry's path, bytes and rule. It removes nothing until
// it is given that token back, and then only if the place still lists the
// same: the entries removed are the ones a listing showed. A place of the
// owner's data is refused from a worker's environment, as a push is.
func deleteLeftovers(h hygiene, name, token string, worker bool, output io.Writer) error {
	at := slices.IndexFunc(leftoverPlaces, func(line place) bool { return line.name == name && line.remove != nil })
	switch {
	case at < 0:
		var deleting []string
		for _, line := range leftoverPlaces {
			if line.remove != nil {
				deleting = append(deleting, line.name)
			}
		}
		return fmt.Errorf("-delete names a place whose finished entries are deleted, one of %s, and %q is none of them", strings.Join(deleting, ", "), name)
	case worker && slices.Contains(ownersData, name):
		return errors.New("the loop and its workers never remove the owner's data: the owner or the lead runs -delete " + name)
	}
	lock, err := h.removing()
	if err != nil {
		return err
	}
	defer lock.Close()
	found, err := listLeftovers(h, name)
	if err != nil {
		return err
	}
	if agreed, err := confirmed(found, token, "remove", "removed", "the place "+name, output); !agreed {
		return err
	}
	return h.removeFinished(leftoverPlaces[at].remove, "removed", found, output)
}

// confirmed is the half of a removal that removes nothing, which the deleting
// and the withdrawing command share. With no token it prints the listing, a
// line for each finished entry under the verb it would be taken by and a line
// with its state and rule for every other, and then the listing's token. With
// a token that is this listing's it prints only the lines of what is kept and
// reports that the caller may go on, to write a line for each entry it takes;
// a token that is not is refused, naming what holds the entries listed. The
// verb is given as it is asked and as done.
func confirmed(found []leftover, token, verb, done, holder string, output io.Writer) (bool, error) {
	var listing strings.Builder
	digest, finished := sha256.New(), 0
	for _, entry := range found {
		if entry.state != stateFinished {
			listing.WriteString(entry.said("kept", entry.state+": "+entry.rule))
			continue
		}
		finished++
		fmt.Fprintf(digest, "%s\x00%d\x00%s\x00", entry.path, entry.bytes, entry.rule)
		if token == "" {
			listing.WriteString(entry.said("would-"+verb, entry.rule))
		}
	}
	shown := hex.EncodeToString(digest.Sum(nil))[:tokenDigits]
	switch {
	case token == "":
		_, err := fmt.Fprintf(output, "%sloop: nothing was %s; to %s the %d entries marked would-%s, run this again with -confirm %s\n", listing.String(), done, verb, finished, verb, shown)
		return false, err
	case token != shown:
		return false, fmt.Errorf("-confirm %s is not the token of what %s holds now: list it again, and confirm what that listing shows", token, holder)
	}
	_, err := io.WriteString(output, listing.String())
	return err == nil, err
}

// withdrawRegistrations is the one command that withdraws from the store the
// locations of a run's checkpoints, so that the run's directory, once its
// record is committed, reads finished and the deleting command can take it.
// It is the deleting command's sibling: it lists under the removal lock,
// prints a token, changes nothing until it is given that token back, writes
// its lines to the same receipt, and is refused from a worker's environment,
// since a run's registrations are the owner's data. Only locations are
// withdrawn: every identity and every edge stays in the store, and no file
// is touched.
//
// The directory is a run's output: inside the experiments folder, and named
// itself by a committed record. For each folder in it that holds a checkpoint
// document, the locations withdrawn are those the store registers under that
// folder for the checkpoint, for its weights and their parts, for the
// momentum its resume document holds, and for each model the store says the
// checkpoint produced, with that model's components. Every other location
// under the directory is listed as kept and not touched: a model registered
// from a file there that no checkpoint of the run produced is a prototype,
// which is never withdrawn, whatever serves it, uses it or was made from it.
//
// A component other models also hold loses only its location in the
// checkpoint's folder. When that would leave it no location on this machine,
// the whole withdrawal is refused if one of those models is served, by an
// active alias, or is one no checkpoint of the run produced: whether a model
// is served is read only to refuse, never as grounds to withdraw.
func withdrawRegistrations(h hygiene, directory, token string, worker bool, output io.Writer) error {
	f := &facts{hygiene: h}
	line := leftoverPlaces[slices.IndexFunc(leftoverPlaces, func(line place) bool { return line.name == "experiment" })]
	directory, err := filepath.Abs(directory)
	lines, citeErr := f.citations(line, "docs")
	switch experiments := f.directory(line); {
	case worker:
		return errors.New("the loop and its workers never remove the owner's data: the owner or the lead runs -withdraw")
	case err != nil || citeErr != nil:
		return errors.Join(err, citeErr)
	case !under(experiments, directory) || under(directory, experiments):
		return fmt.Errorf("-withdraw names the output directory of a run, inside %s, and %s is not inside it", experiments, directory)
	// The record names the directory itself. A line that names only something beneath it does not
	// count: a record of one run names its experiment's folder that way, and that folder would
	// stand for every run in it, one no record names among them, whose weights the deleting
	// command takes once nothing registers them, because it asks only that the experiment be named.
	case cited(lines, h.data, directory+runEnd) == "":
		return fmt.Errorf("no committed record names %s, so nothing says the record of its run is kept: nothing is withdrawn", directory)
	}
	lock, err := h.removing()
	if err != nil {
		return err
	}
	defer lock.Close()
	// The store is opened for writing only to withdraw. An open store reads as it was when it was
	// opened, and the one commit is held to that head: a store another process has changed since
	// the listing was read refuses it, and is listed again.
	open := overgodb.OpenReadOnly
	if token != "" {
		open = overgodb.Open
	}
	store, err := open(h.store)
	if err != nil {
		return err
	}
	defer store.Close()
	// owned names, for each artifact a checkpoint of the run accounts for, the folders of those checkpoints.
	ctx, owned := context.Background(), map[artifact.ID][]string{}
	err = filepath.WalkDir(directory, func(path string, item fs.DirEntry, err error) error {
		if err != nil || item.Name() != trainingprogram.CheckpointFilename {
			return err
		}
		document, err := os.ReadFile(path)
		checkpoint, parseErr := trainingprogram.ParseCheckpoint(document)
		if err = errors.Join(err, parseErr); err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		members := []artifact.ID{checkpoint.ID(), checkpoint.Weights}
		sidecar, err := store.Parents(ctx, checkpoint.Resume)
		for _, edge := range sidecar {
			if edge.Relation == artifact.RelationContains {
				members = append(members, edge.Parent)
			}
		}
		produced, producedErr := store.Children(ctx, checkpoint.ID())
		for _, edge := range produced {
			if edge.Relation == artifact.RelationProducedBy && edge.Child.Kind() == artifact.KindModel {
				members = append(members, edge.Child)
			}
		}
		err = errors.Join(err, producedErr)
		// The range reads the members as they stood when it began: a component that a manifest
		// adds is accounted for, and is not itself read for a manifest.
		for _, member := range members {
			manifest, _, manifestErr := store.Manifest(ctx, member)
			err = errors.Join(err, manifestErr)
			for _, component := range manifest.Components {
				members = append(members, component.Artifact)
			}
		}
		for _, member := range members {
			owned[member] = append(owned[member], filepath.Dir(path))
		}
		return err
	})
	located, locatedErr := localLocations(store, registrationPage)
	if err = errors.Join(err, locatedErr); err != nil {
		return err
	}
	var found []leftover
	var withdrawn []artifact.LocationEvent
	// staying counts, for each artifact, the locations on this machine the withdrawal leaves it.
	staying := map[artifact.ID]int{}
	for _, location := range located {
		entry := leftover{place: line.name, path: location.Value, state: stateKept, rule: "not a run's checkpoint: a location of " + location.Artifact.String()}
		if slices.ContainsFunc(owned[location.Artifact], func(folder string) bool { return under(folder, location.Value) }) {
			entry.state, entry.rule = stateFinished, "a location of "+location.Artifact.String()+", which a checkpoint in the folder accounts for"
			withdrawn = append(withdrawn, artifact.LocationEvent{Location: location, Action: artifact.LocationRemove})
		} else if staying[location.Artifact]++; !under(directory, location.Value) {
			continue
		}
		found = append(found, entry)
	}
	// served holds each model an active alias names; an alias that is no activation names none,
	// and marks only the identity no model has.
	served := map[artifact.ID]bool{}
	err = store.VisitAliases(ctx, "", func(alias overgodb.AliasView) error {
		model, _, _ := modelrecipe.ParseActiveAlias(alias.Name)
		served[model] = true
		return nil
	})
	for _, event := range withdrawn {
		holders, holdersErr := store.Children(ctx, event.Artifact)
		err = errors.Join(err, holdersErr)
		for _, edge := range holders {
			if model := edge.Child; staying[event.Artifact] == 0 && edge.Relation == artifact.RelationContains && model.Kind() == artifact.KindModel && (served[model] || owned[model] == nil) {
				err = errors.Join(err, fmt.Errorf("the model %s is served, or is one no checkpoint of this run produced, and the withdrawal would leave its component %s no location on this machine: nothing is withdrawn", model, event.Artifact))
			}
		}
	}
	if err != nil {
		return err
	}
	if agreed, err := confirmed(found, token, "withdraw", "withdrawn", "the store under "+directory, output); !agreed || len(withdrawn) == 0 {
		return err
	}
	head, _ := store.Head()
	if _, err := store.Commit(ctx, artifact.Batch{Key: withdrawalKey + head.String(), ExpectedHead: &head, Locations: withdrawn}); err != nil {
		return err
	}
	// The one commit withdrew every location, so what is left for each is its line.
	return h.removeFinished(func(hygiene, string) error { return nil }, "withdrawn", found, output)
}

// leftoversFor lists the leftovers of the repository the current directory
// is a worktree of, against the real machine, or, named a place, removes that
// place's finished entries as deleteLeftovers does, or, named a run's
// directory, withdraws its checkpoints' locations as withdrawRegistrations
// does. The checkout is the trunk's, which git lists first, so a listing, a
// removal and their receipts are the same from any worktree; the store is
// the one named, or the one the data-root contract gives that checkout.
func leftoversFor(store, deleting, withdrawing, token string, worker bool, output io.Writer) error {
	trees, err := landing.Worktrees(context.Background(), loopWorktree)
	running, absErr := filepath.Abs(loopWorktree)
	if err = errors.Join(err, absErr); err != nil {
		return err
	}
	h := hygiene{checkout: trees[0].Root, running: running, temporary: os.TempDir(), removeTree: fsatomic.RemoveTree}
	roots, err := dataroot.Resolve(h.checkout)
	if err != nil {
		return err
	}
	if roots.Store, err = filepath.Abs(cmp.Or(strings.TrimSpace(store), roots.Store)); err != nil {
		return err
	}
	if h.executable, err = os.Executable(); err != nil {
		return err
	}
	cache, err := readTool(os.Environ(), "go", "env", "GOCACHE")
	if err != nil {
		return err
	}
	h.store, h.data, h.cache = roots.Store, filepath.Dir(roots.Store), strings.TrimSpace(cache)
	h.sessions = filepath.Join(os.TempDir(), "claude", sessionFolderName.Replace(h.checkout))
	h.kept = []string{roots.Store, roots.Models, roots.StoreModels, roots.Datasets, roots.Checkpoints, roots.AudioReference, roots.References(), roots.Libraries(), scorecard.Directory(roots.Store)}
	h.stores = func() ([]leftover, error) {
		printed, err := readTool(loopEnvironment(), "go", "-C", h.checkout, "run", "./cmd/overgodb", "precheck", "-leftovers", "-repo", roots.Store, "-checkout", h.checkout)
		if err != nil {
			return nil, err
		}
		return decodeStoreLeftovers([]byte(printed))
	}
	h.registered = func() ([]string, error) { return storeLocations(roots.Store, registrationPage) }
	h.processes = func() ([]string, error) {
		listed, err := readTool(os.Environ(), "tasklist", "/FO", "CSV", "/NH")
		if err != nil {
			return nil, err
		}
		return goProcesses(listed, os.Getppid())
	}
	switch {
	case deleting != "":
		return deleteLeftovers(h, deleting, token, worker, output)
	case withdrawing != "":
		return withdrawRegistrations(h, withdrawing, token, worker, output)
	}
	return leftoversCommand(h, output)
}
