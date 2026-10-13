package main

import (
	"bytes"
	"cmp"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/authoritylock"
	"overgo/internal/fsatomic"
	"overgo/internal/guard"
	"overgo/internal/hostoptimizer"
	"overgo/internal/landing"
	"overgo/internal/modelrecipe"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/processlock"
	"overgo/internal/recipe"
	"overgo/internal/repoanalysis"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

// TestLeftoversListsEveryDeclaredPlaceWithItsRule builds, under a temporary directory, a repository
// with one entry of each kind in each state: helper copies that are clean, changed, part deleted,
// emptied, holding ignored content, ahead of the trunk, held by a session and gone from disk; lane
// worktrees, branches and a stash; experiment outputs that a committed record names in its store
// field, that it names in another field, that a store registration names and that nothing names;
// checkpoints, scratch, session folders, temporary build
// folders, executables, a media sample that a read record names by its digest and one that nothing
// reads, a record of a run that names only what the tree has, one that names a command and a test
// it no longer has and one that is a picture, and directories no line of the table names. The
// listing gives exactly those verdicts, each with the rule that decided it: a sample is held by
// what names it and finished when nothing reads it, and a record is kept, as history when it
// names what the tree no longer has. A helper copy is finished only when a commit ran in it, which
// its process state says: a clean copy with none is of unknown state, and the emptied one is
// finished because a commit left the record of its run there. An experiment's output is held by a
// record's store field and by no other field, and every line that deletes has a remover while no
// other line has one. It measures bytes only where a line allows a delete
// or a report, reads a source that failed as unknown and never as nothing, on a line that deletes
// calls nothing finished that holds a link a recursive delete could follow, calls no entry finished
// for want of a committed line that names it when the search could not have found its name, prints
// one line an entry and writes the same text as a receipt named by the trunk's commit. And it
// removes nothing: the files, the worktrees, the branches and the stash are the same after it as
// before, but for the receipt.
func TestLeftoversListsEveryDeclaredPlaceWithItsRule(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	// Git names a worktree by its resolved path, so the fixture's root is resolved the same way.
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	at := func(slash string) string { return filepath.Join(root, filepath.FromSlash(slash)) }
	folder := func(slash string) string {
		t.Helper()
		if err := os.MkdirAll(at(slash), 0o755); err != nil {
			t.Fatal(err)
		}
		return at(slash)
	}
	write := func(slash, content string) {
		t.Helper()
		folder(filepath.ToSlash(filepath.Dir(filepath.FromSlash(slash))))
		if err := os.WriteFile(at(slash), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	git := func(directory string, arguments ...string) string {
		t.Helper()
		return gitFixture(t, directory, arguments...)
	}
	relative := func(path string) string {
		t.Helper()
		if !filepath.IsAbs(path) {
			return path
		}
		inside, err := filepath.Rel(root, path)
		if err != nil {
			t.Fatal(err)
		}
		return filepath.ToSlash(inside)
	}
	verdicts := func(found []leftover) []string {
		t.Helper()
		lines := make([]string, len(found))
		for index, entry := range found {
			lines[index] = strings.Join([]string{entry.place, relative(entry.path), entry.action, entry.state}, "|")
		}
		slices.Sort(lines)
		return lines
	}
	entryOf := func(t *testing.T, found []leftover, place, slash string) leftover {
		t.Helper()
		for _, entry := range found {
			if entry.place == place && relative(entry.path) == slash {
				return entry
			}
		}
		t.Fatalf("the listing holds no %s entry %s:\n%s", place, slash, strings.Join(verdicts(found), "\n"))
		return leftover{}
	}
	list := func(t *testing.T, h hygiene) []leftover {
		t.Helper()
		found, err := listLeftovers(h, true)
		if err != nil {
			t.Fatal(err)
		}
		return found
	}

	// The trunk's checkout: what it commits, then what accumulates in it.
	repo := folder("repo")
	git(repo, "init", "-q", "-b", "master")
	git(repo, "config", "core.autocrlf", "false")
	write("repo/.gitignore", "/bin/\n/tmp/\n/dist/\n/build/*\n!/build/held.dat\novergodb-store/\n/scripts/python_golden_references/\n")
	write("repo/.claude/tracked.fixture", "{}\n")
	write("repo/build/held.dat", "declared\n")
	write("repo/scripts/hook.txt", "run bin/named.exe\nrun bin\\slashed.exe\nrun bin/two words.exe\n")
	// An invented digest: a media sample is named by the digest of its bytes.
	sampled := strings.Repeat("ab", sha256.Size)
	// A record names a path by whole elements: a longer name before it, after it or around it is another path.
	// The last line, like the hook's, names an entry with a space in it, which the search reads as the end of a path.
	write("repo/docs/verification/run-record.txt", strings.Join([]string{
		"outputs: read once from Experiments/done.",
		"store: D:/data/experiments/kept/store",
		`reward: D:\data\experiments\kept\run\reward`,
		"model: checkpoints/cited, kept",
		"other: myexperiments/loose/out, experiments/orphan-old/out and experiments/coded.tar",
		"whole: (experiments/whole)",
		"sample: output:sha256:" + sampled,
		"final: experiments/done/final run",
		"",
	}, "\n"))
	// A record of a run whose command and test the tree no longer has: history, and kept as it was written.
	write("repo/docs/verification/old-probe.txt", "command: go run ./cmd/gone\nheld by: TestGone\n")
	// A record that is a picture, not text: bytes in it that spell a path name nothing.
	write("repo/docs/verification/chart.dat", "\xff\xd8 cmd/drawn \x00\xfe\n")
	// A record in a folder of its own is a record like any other.
	write("repo/docs/verification/suite/inputs.txt", "made by: go run ./cmd/maker\n")
	for _, record := range repoanalysis.WorkRecords {
		write("repo/"+record, "experiments/planned/out bin/loose.exe\n")
	}
	// Source reads the record, and the record carries one sample's digest; nothing carries the other's name.
	write("repo/internal/x/x.go", "// Package x names experiments/coded/out.\npackage x\n\nconst record = \"docs/verification/run-record.txt\"\n")
	write("repo/docs/media_samples/"+sampled+".dat", "a sample its record names\n")
	write("repo/docs/media_samples/sample.dat", "sample\n")
	git(repo, "add", "-A")
	git(repo, "commit", "-q", "-m", "base")
	trunk := git(repo, "rev-parse", "HEAD")
	write("repo/scripts/hook.txt", "work a stash holds\n")
	git(repo, "stash", "push", "-q", "-m", "kept work")
	git(repo, "branch", "merged-free", "master")
	if err := processlock.EnsureStateDirectory(repo); err != nil {
		t.Fatal(err)
	}
	copyAt := func(slash string, arguments ...string) string {
		t.Helper()
		git(repo, append(append([]string{"worktree", "add", "-q"}, arguments...), at(slash), "master")...)
		return at(slash)
	}
	copyAt("temporary/sessions/s1/scratchpad/clean", "--detach")
	copyAt("temporary/sessions/s1/scratchpad/dirty", "--detach")
	write("temporary/sessions/s1/scratchpad/dirty/note.txt", "uncommitted\n")
	partial := copyAt("temporary/sessions/s1/scratchpad/partial", "--detach")
	if err := os.Remove(filepath.Join(partial, "scripts", "hook.txt")); err != nil {
		t.Fatal(err)
	}
	emptied := copyAt("temporary/sessions/s1/scratchpad/emptied", "--detach")
	tracked := strings.Fields(git(emptied, "ls-files"))
	for _, file := range tracked {
		if err := os.Remove(filepath.Join(emptied, filepath.FromSlash(file))); err != nil {
			t.Fatal(err)
		}
	}
	// A commit ran in the emptied copy, and left the record of its run in its process state: the
	// copy with none is of unknown state.
	if err := processlock.EnsureStateDirectory(emptied); err != nil {
		t.Fatal(err)
	}
	write("temporary/sessions/s1/scratchpad/emptied/"+processlock.StateDirectory+"/landing_waits.json", "[]\n")
	// As many changed paths as tracked files, one of them a file added: not an emptied copy.
	replaced := copyAt("temporary/sessions/s1/scratchpad/replaced", "--detach")
	for _, file := range tracked[1:] {
		if err := os.Remove(filepath.Join(replaced, filepath.FromSlash(file))); err != nil {
			t.Fatal(err)
		}
	}
	write("temporary/sessions/s1/scratchpad/replaced/added.txt", "uncommitted\n")
	copyAt("temporary/sessions/s1/scratchpad/ignored", "--detach")
	write("temporary/sessions/s1/scratchpad/ignored/overgodb-store/blob", "ignored\n")
	ahead := copyAt("temporary/sessions/s2/scratchpad/ahead", "--detach")
	git(ahead, "commit", "-q", "--allow-empty", "-m", "ahead of the trunk")
	aheadCommit := git(ahead, "rev-parse", "HEAD")
	git(repo, "branch", "unmerged-free", aheadCommit)
	held := copyAt("temporary/sessions/s2/scratchpad/held", "--detach")
	if err := processlock.EnsureStateDirectory(held); err != nil {
		t.Fatal(err)
	}
	write("temporary/sessions/s2/scratchpad/held/"+processlock.StateDirectory+"/"+guard.WorktreeHolder, "s9\n")
	if err := os.RemoveAll(copyAt("temporary/sessions/s2/scratchpad/gone", "--detach")); err != nil {
		t.Fatal(err)
	}
	copyAt("repo/.claude/worktrees/agent-clean", "--detach")
	folder("repo/.claude/worktrees/agent-unlinked")
	folder("repo/.claude/other")
	git(copyAt("lanes/feature", "-b", "lane/feature"), "commit", "-q", "--allow-empty", "-m", "lane work")
	copyAt("lanes/merged", "-b", "merged-held")
	copyAt("repo/"+processlock.StateDirectory+"/cycle", "--detach")
	write("repo/bin/named.exe", "named")
	write("repo/bin/slashed.exe", "slashed")
	write("repo/bin/loose.exe", "7 bytes")
	write("repo/bin/two words.exe", "spaced")
	write("repo/tmp/probe.go", "package probe\n")
	write("repo/tmp/tool.exe", "tool")
	folder("repo/build/scrap")
	folder("repo/dist")
	folder("repo/scripts/python_golden_references")
	folder("repo/overgodb-store")
	write("repo/docs/local.json", "{}\n")
	write("repo/stray/x", "stray\n")

	// The data root: a real store that registers two locations, and what runs left beside it.
	storePath := folder("data/overgodb-store")
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	register := func(name string, kind artifact.LocationKind, value string) {
		t.Helper()
		id := testutil.ArtifactID(t, artifact.KindTensorSet, name)
		if _, err := store.Commit(ctx, artifact.Batch{
			Key:       "leftovers-fixture/" + name,
			Artifacts: []artifact.Descriptor{{ID: id, Size: uint64(len(name))}},
			Locations: []artifact.LocationEvent{{Location: artifact.Location{Artifact: id, Kind: kind, Value: value}, Action: artifact.LocationAdd}},
		}); err != nil {
			t.Fatal(err)
		}
	}
	write("data/experiments/done/weights/step.bin", "weights")
	write("data/experiments/done/run.log", "log\n")
	folder("data/experiments/done/final run")
	folder("data/experiments/kept/store")
	write("data/experiments/kept/store.lock", "")
	folder("data/experiments/kept/run")
	write("data/experiments/kept/model/weights.bin", "model")
	for _, experiment := range []string{"orphan", "planned", "coded", "loose", "whole"} {
		folder("data/experiments/" + experiment + "/out")
	}
	write("data/experiments/notes.txt", "beside the experiments\n")
	folder("data/checkpoints/proto")
	folder("data/checkpoints/cited")
	folder("data/checkpoints/loose")
	write("data/scratch/log.txt", "scratch\n")
	folder("data/build")
	folder("data/models")
	folder("data/mystery")
	folder("data/overgodb-store.superseded-1")
	// Spelled through another directory, so only a comparison by file identity places it under the entry.
	register("model", artifact.LocationFile, strings.Join([]string{at("data/experiments/done"), "..", "kept", "model", "weights.bin"}, string(filepath.Separator)))
	register("proto", artifact.LocationDirectory, at("data/checkpoints/proto"))
	// A registered directory holds what lies in it: the record names this experiment, and nothing names its output.
	register("whole", artifact.LocationDirectory, at("data/experiments/whole"))
	// A remote location is no file on this machine, whatever its value reads as.
	register("elsewhere", artifact.LocationRemote, at("data/experiments/done/weights"))
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	// A second repository whose main worktree is on no branch, with nothing tracked beside its agent
	// folder, and which is its own data root: the directories it tracks are no leftovers.
	second := folder("second")
	git(second, "init", "-q", "-b", "master")
	write("second/source/tracked.dat", "tracked\n")
	git(second, "add", "-A")
	git(second, "commit", "-q", "-m", "base")
	git(second, "checkout", "-q", "--detach")
	git(second, "worktree", "add", "-q", "--detach", at("second/.claude/worktrees/agent"), "master")
	git(second, "worktree", "add", "-q", "-b", "side", at("second-lane"), "master")
	folder("second/.claude/other")
	folder("second-temporary")

	// The temporary directory, the session folders in it and the build cache.
	write("temporary/go-build1/x", "object")
	executable := at("temporary/go-build2/b001/exe/loop.exe")
	write("temporary/go-build2/b001/exe/loop.exe", "this command")
	folder("temporary/overgo-gate-1")
	folder("temporary/other")
	folder("cache")
	write("outside/large.bin", "bytes behind a link")
	linked := fixtureLink(at("outside"), filepath.Join(folder("temporary/go-build4"), "b001")) &&
		fixtureLink(at("outside"), filepath.Join(folder("data/experiments/done/linked"), "out")) &&
		fixtureLink(at("outside"), at("data/scratch/out"))
	if !linked {
		t.Log("this machine makes neither a symbolic link nor a junction, so nothing about links is held here")
		for _, slash := range []string{"temporary/go-build4", "data/experiments/done/linked", "data/scratch/out"} {
			if err := os.RemoveAll(at(slash)); err != nil {
				t.Fatal(err)
			}
		}
	}

	printed := fmt.Sprintf(`[
		{"path": %q, "kind": "sealed-backup", "sequence": 7, "modified": "2026-10-01T00:00:00Z", "packed": true, "keep": true},
		{"path": %q, "kind": "superseded-store", "modified": "2026-10-02T00:00:00Z", "packed": false, "keep": true}
	]`, filepath.ToSlash(at("backups/sealed")), filepath.ToSlash(at("data/overgodb-store.superseded-1")))
	var storesReturned, processesReadEarly bool
	h := hygiene{
		checkout: repo, data: at("data"), sessions: at("temporary/sessions"), temporary: at("temporary"), cache: at("cache"),
		executable: executable,
		kept:       []string{storePath, at("data/models"), ""},
		stores: func() ([]leftover, error) {
			defer func() { storesReturned = true }()
			return decodeStoreLeftovers([]byte(printed))
		},
		registered: func() ([]string, error) { return storeLocations(storePath, 1) },
		processes: func() ([]string, error) {
			processesReadEarly = processesReadEarly || !storesReturned
			return nil, nil
		},
	}

	snapshot := func(t *testing.T) map[string]int64 {
		t.Helper()
		seen := map[string]int64{}
		if err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			seen[relative(path)] = -1
			if entry.Type().IsRegular() {
				info, err := entry.Info()
				if err != nil {
					return err
				}
				seen[relative(path)] = info.Size()
			}
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		for _, arguments := range [][]string{{"worktree", "list", "--porcelain"}, {"branch", "--list"}, {"stash", "list"}} {
			seen["git "+strings.Join(arguments, " ")+"\n"+gitFixture(t, repo, arguments...)] = 0
		}
		return seen
	}
	before := snapshot(t)
	registeredBefore, err := landing.Worktrees(ctx, repo)
	if err != nil || len(registeredBefore) != 14 || registeredBefore[0].Branch != "refs/heads/master" || registeredBefore[0].Head != trunk {
		t.Fatalf("the fixture registers %d worktrees, the first %+v (%v); want 14 with the trunk's checkout first, on master at %s", len(registeredBefore), registeredBefore[:min(len(registeredBefore), 1)], err, trunk)
	}

	found := list(t, h)
	t.Run("every entry has the verdict its state gives it", func(t *testing.T) {
		want := []string{
			"helper-copy|temporary/sessions/s1/scratchpad/clean|delete|unknown",
			"helper-copy|temporary/sessions/s1/scratchpad/dirty|delete|held",
			"helper-copy|temporary/sessions/s1/scratchpad/partial|delete|held",
			"helper-copy|temporary/sessions/s1/scratchpad/emptied|delete|finished",
			"helper-copy|temporary/sessions/s1/scratchpad/replaced|delete|held",
			"helper-copy|temporary/sessions/s1/scratchpad/ignored|delete|held",
			"helper-copy|temporary/sessions/s2/scratchpad/ahead|delete|held",
			"helper-copy|temporary/sessions/s2/scratchpad/held|delete|held",
			"helper-copy|temporary/sessions/s2/scratchpad/gone|delete|unknown",
			"helper-copy|repo/.claude/worktrees/agent-clean|delete|unknown",
			"helper-copy|repo/.claude/worktrees/agent-unlinked|delete|unknown",
			"worktree|lanes/feature|report|held",
			"worktree|lanes/merged|report|finished",
			"branch|refs/heads/lane/feature|delete|held",
			"branch|refs/heads/merged-free|delete|finished",
			"branch|refs/heads/merged-held|delete|held",
			"branch|refs/heads/unmerged-free|delete|held",
			"stash|stash@{0}|report|unknown",
			"store-backup|backups/sealed|never|kept",
			"store-backup|data/overgodb-store.superseded-1|never|kept",
			"experiment|data/experiments/done/weights|delete|finished",
			"experiment|data/experiments/done/run.log|delete|unknown",
			"experiment|data/experiments/done/final run|delete|unknown",
			"experiment|data/experiments/kept/store|delete|held",
			"experiment|data/experiments/kept/store.lock|delete|unknown",
			"experiment|data/experiments/kept/run|delete|finished",
			"experiment|data/experiments/kept/model|delete|held",
			"experiment|data/experiments/orphan/out|delete|unknown",
			"experiment|data/experiments/whole/out|delete|held",
			"experiment|data/experiments/notes.txt|delete|unknown",
			"experiment|data/experiments/planned/out|delete|unknown",
			"experiment|data/experiments/coded/out|delete|unknown",
			"experiment|data/experiments/loose/out|delete|unknown",
			"checkpoint|data/checkpoints/proto|report|held",
			"checkpoint|data/checkpoints/cited|report|held",
			"checkpoint|data/checkpoints/loose|report|unknown",
			"scratch|data/scratch/log.txt|report|unknown",
			"data-build|data/build|report|unknown",
			"session-scratch|temporary/sessions/s1|report|unknown",
			"session-scratch|temporary/sessions/s2|report|unknown",
			"temporary-build|temporary/go-build1|delete|finished",
			"temporary-build|temporary/go-build2|delete|held",
			"build-cache|cache|report|unknown",
			"bin|repo/bin/named.exe|report|held",
			"bin|repo/bin/slashed.exe|report|held",
			"bin|repo/bin/loose.exe|report|finished",
			"bin|repo/bin/two words.exe|report|unknown",
			"tmp|repo/tmp/probe.go|delete|unknown",
			"tmp|repo/tmp/tool.exe|delete|unknown",
			"build|repo/build|report|unknown",
			"release|repo/dist|report|unknown",
			"python-place|repo/scripts/python_golden_references|never|kept",
			"checkout-store|repo/overgodb-store|never|kept",
			"docs-untracked|repo/docs/local.json|never|kept",
			"media-samples|repo/docs/media_samples/" + sampled + ".dat|report|held",
			"media-samples|repo/docs/media_samples/sample.dat|report|finished",
			"verification|repo/docs/verification/run-record.txt|never|kept",
			"verification|repo/docs/verification/old-probe.txt|never|kept",
			"verification|repo/docs/verification/chart.dat|never|kept",
			"verification|repo/docs/verification/suite/inputs.txt|never|kept",
			"process-state|repo/.overgo-runtime|never|kept",
			"undeclared|repo/.claude/other|report|undeclared",
			"undeclared|repo/stray|report|undeclared",
			"undeclared|data/mystery|report|undeclared",
		}
		if linked {
			want = append(want, "temporary-build|temporary/go-build4|delete|unknown", "experiment|data/experiments/done/linked|delete|unknown", "scratch|data/scratch/out|report|unknown")
		}
		slices.Sort(want)
		if got := verdicts(found); !slices.Equal(got, want) {
			t.Fatalf("the listing's verdicts are\n%s\nwant\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
		}
		if processesReadEarly {
			t.Fatal("the running processes were read before the store's own listing returned, while its go command could still run")
		}
	})

	t.Run("every line of the table is whole and has an entry", func(t *testing.T) {
		names := map[string]bool{}
		for _, line := range leftoverPlaces {
			word, _, cut := strings.Cut(line.at, ":")
			if line.name == "" || names[line.name] || !cut || word == "" || line.holds == "" || line.finished == "" || line.list == nil ||
				!slices.Contains([]string{actionDelete, actionReport, actionNever}, line.action) {
				t.Errorf("line %+v is not whole: it needs a name of its own, a root word, what it holds, its finished state and an action", line)
			}
			if (line.remove != nil) != (line.action == actionDelete) {
				t.Errorf("line %s has the action %s and a remover: %v; a line that deletes has one, and no other line has", line.name, line.action, line.remove != nil)
			}
			names[line.name] = true
			if !slices.ContainsFunc(found, func(entry leftover) bool { return entry.place == line.name }) {
				t.Errorf("the fixture holds no entry of line %s, so nothing holds its verdicts", line.name)
			}
		}
	})

	t.Run("every entry carries the rule that decided it", func(t *testing.T) {
		for _, entry := range found {
			if strings.TrimSpace(entry.rule) == "" {
				t.Errorf("%s %s is %s by no rule", entry.place, relative(entry.path), entry.state)
			}
		}
		for _, named := range []struct{ place, slash, rule string }{
			{"experiment", "data/experiments/kept/store", "docs/verification/run-record.txt:2"},
			{"experiment", "data/experiments/kept/run", "docs/verification/run-record.txt:3"},
			{"experiment", "data/experiments/kept/model", "weights.bin"},
			{"checkpoint", "data/checkpoints/cited", "docs/verification/run-record.txt:4"},
			{"checkpoint", "data/checkpoints/proto", "proto"},
			{"experiment", "data/experiments/whole/out", filepath.ToSlash(at("data/experiments/whole"))},
			{"experiment", "data/experiments/orphan/out", "no committed record names its experiment"},
			{"experiment", "data/experiments/coded/out", "no committed record names its experiment"},
			{"experiment", "data/experiments/done/run.log", "a file"},
			{"experiment", "data/experiments/done/final run", "no line that names it could be found"},
			{"bin", "repo/bin/named.exe", "scripts/hook.txt:1"},
			{"bin", "repo/bin/slashed.exe", "scripts/hook.txt:2"},
			{"bin", "repo/bin/two words.exe", "no line that names it could be found"},
			{"experiment", "data/experiments/kept/run", "outside a store field"},
			{"helper-copy", "temporary/sessions/s1/scratchpad/clean", "no process state"},
			{"helper-copy", "repo/.claude/worktrees/agent-clean", "no process state"},
			{"helper-copy", "temporary/sessions/s2/scratchpad/ahead", aheadCommit[:12]},
			{"helper-copy", "temporary/sessions/s2/scratchpad/held", "s9"},
			{"helper-copy", "temporary/sessions/s1/scratchpad/ignored", "overgodb-store/"},
			{"helper-copy", "temporary/sessions/s1/scratchpad/emptied", fmt.Sprintf("every one of its %d tracked files is missing", len(tracked))},
			{"helper-copy", "temporary/sessions/s1/scratchpad/partial", fmt.Sprintf("1 of them deletions among its %d tracked files", len(tracked))},
			{"worktree", "lanes/merged", "on the trunk"},
			{"branch", "refs/heads/merged-held", filepath.ToSlash(at("lanes/merged"))},
			{"branch", "refs/heads/unmerged-free", "commits the trunk lacks"},
			{"helper-copy", "temporary/sessions/s1/scratchpad/dirty", "note.txt"},
			{"stash", "stash@{0}", "kept work"},
			{"store-backup", "backups/sealed", "sealed-backup"},
			{"tmp", "repo/tmp/tool.exe", "docs/plan.json"},
			{"temporary-build", "temporary/go-build2", "this command's own executable"},
			{"media-samples", "repo/docs/media_samples/" + sampled + ".dat", "named by docs/verification/run-record.txt"},
			{"media-samples", "repo/docs/media_samples/sample.dat", "nothing reads it"},
			{"verification", "repo/docs/verification/run-record.txt", "none: a record is kept"},
			{"verification", "repo/docs/verification/old-probe.txt", "history: it names what the tree no longer has: TestGone, cmd/gone"},
			{"verification", "repo/docs/verification/chart.dat", "none: a record is kept"},
			{"verification", "repo/docs/verification/suite/inputs.txt", "history: it names what the tree no longer has: cmd/maker"},
		} {
			if entry := entryOf(t, found, named.place, named.slash); !strings.Contains(filepath.ToSlash(entry.rule), named.rule) {
				t.Errorf("%s %s is %s by the rule %q, which does not name %q", named.place, named.slash, entry.state, entry.rule, named.rule)
			}
		}
	})

	t.Run("bytes are measured where a line deletes or reports and nowhere else", func(t *testing.T) {
		if loose := entryOf(t, found, "bin", "repo/bin/loose.exe"); !loose.measured || loose.bytes != 7 {
			t.Fatalf("bin/loose.exe measures %d bytes (measured %v), want its 7", loose.bytes, loose.measured)
		}
		if weights := entryOf(t, found, "experiment", "data/experiments/done/weights"); !weights.measured || weights.bytes != 7 {
			t.Fatalf("the finished experiment output measures %d bytes (measured %v), want the 7 of its one file", weights.bytes, weights.measured)
		}
		for _, entry := range found {
			if unmeasured := entry.action == actionNever || !filepath.IsAbs(entry.path); unmeasured && entry.measured {
				t.Errorf("%s %s is measured, and a line that never deletes, a branch and a stash are not", entry.place, relative(entry.path))
			}
		}
		if gone := entryOf(t, found, "helper-copy", "temporary/sessions/s2/scratchpad/gone"); gone.measured {
			t.Fatalf("the copy whose directory is gone measures %d bytes, and what cannot be walked is not measured", gone.bytes)
		}
		// A report's entry is measured without what lies behind a link, which the walk does not follow.
		if linked {
			if behind := entryOf(t, found, "scratch", "data/scratch/out"); !behind.measured || behind.bytes != 0 {
				t.Fatalf("the scratch entry that is a link measures %d bytes (measured %v), want none of what lies behind it", behind.bytes, behind.measured)
			}
		}
	})

	t.Run("a temporary build folder is held while a go command or a test binary runs", func(t *testing.T) {
		running := h
		running.processes = func() ([]string, error) { return []string{"loop.test.exe"}, nil }
		if entry := entryOf(t, list(t, running), "temporary-build", "temporary/go-build1"); entry.state != stateHeld || !strings.Contains(entry.rule, "loop.test.exe") {
			t.Fatalf("with a test binary running the folder is %s by %q, want held by the binary's name", entry.state, entry.rule)
		}
		unread := h
		unread.processes = func() ([]string, error) { return nil, errors.New("the process list is unreadable") }
		listed := list(t, unread)
		if entry := entryOf(t, listed, "temporary-build", "temporary/go-build1"); entry.state != stateUnknown || !strings.Contains(entry.rule, "the process list is unreadable") {
			t.Fatalf("with the process list unreadable the folder is %s by %q, want unknown by the failure", entry.state, entry.rule)
		}
		if entry := entryOf(t, listed, "temporary-build", "temporary/go-build2"); entry.state != stateHeld {
			t.Fatalf("the folder that holds this command's executable is %s when the process list is unreadable, want held", entry.state)
		}
		const sample = "\"System Idle Process\",\"0\",\"Services\",\"0\",\"8 K\"\r\n\"go.exe\",\"4100\",\"Console\",\"1\",\"21,004 K\"\r\n" +
			"\"GO.EXE\",\"4200\",\"Console\",\"1\",\"19,112 K\"\r\n\"loop.test.exe\",\"4300\",\"Console\",\"1\",\"9,000 K\"\r\n\"gopls.exe\",\"4400\",\"Console\",\"1\",\"9,000 K\"\r\n"
		if got, err := goProcesses(sample, 4100); err != nil || !slices.Equal(got, []string{"GO.EXE", "loop.test.exe"}) {
			t.Fatalf("the go commands and test binaries beside the parent 4100 are %q (%v), want the second go command and the test binary", got, err)
		}
		if got, err := goProcesses("\"go.exe\",\"4100\",\"Console\",\"1\",\"21,004 K\"\r\n", 4100); err != nil || len(got) != 0 {
			t.Fatalf("with only this command's parent running the list is %q (%v), want none", got, err)
		}
		for _, unreadable := range []string{"", "\"go.exe\"\r\n", "\"go.exe\",\"41\"00\"\r\n"} {
			if got, err := goProcesses(unreadable, 4100); err == nil {
				t.Fatalf("the process list %q read as %q, want it refused: an empty or malformed list says nothing about what runs", unreadable, got)
			}
		}
	})

	t.Run("a source that fails reads as unknown and never as nothing", func(t *testing.T) {
		failing := h
		failing.registered = func() ([]string, error) { return nil, errors.New("the store's registrations are unreadable") }
		failing.stores = func() ([]leftover, error) { return nil, errors.New("the store's own listing failed") }
		failing.sessions = at("temporary/absent")
		listed := list(t, failing)
		if entry := entryOf(t, listed, "experiment", "data/experiments/done/weights"); entry.state != stateUnknown || !strings.Contains(entry.rule, "the store's registrations are unreadable") {
			t.Fatalf("with the registrations unreadable the output is %s by %q, want unknown by the failure", entry.state, entry.rule)
		}
		if entry := entryOf(t, listed, "experiment", "data/experiments/kept/store"); entry.state != stateHeld {
			t.Fatalf("the output a committed record names is %s with the registrations unreadable, want held", entry.state)
		}
		var backups, sessions []leftover
		for _, entry := range listed {
			switch entry.place {
			case "store-backup":
				backups = append(backups, entry)
			case "session-scratch":
				sessions = append(sessions, entry)
			}
		}
		if len(backups) != 1 || backups[0].state != stateUnknown || !strings.Contains(backups[0].rule, "the store's own listing failed") {
			t.Fatalf("with the store's listing failed its line holds %+v, want one unknown entry carrying the failure", backups)
		}
		if len(sessions) != 1 || sessions[0].state != stateUnknown || sessions[0].path != at("temporary/absent") {
			t.Fatalf("with no session folder its line holds %+v, want one unknown entry naming the folder tried", sessions)
		}
		for _, malformed := range []string{`[{"path": "", "kind": "sealed-backup"}]`, `[{"path": "p"}]`, `{"path": "p"}`} {
			if decoded, err := decodeStoreLeftovers([]byte(malformed)); err == nil {
				t.Fatalf("the store listing %s decoded as %+v, want it refused", malformed, decoded)
			}
		}
		if decoded, err := decodeStoreLeftovers([]byte("null\n")); err != nil || len(decoded) != 0 {
			t.Fatalf("an empty store listing decoded as %+v (%v), want none", decoded, err)
		}
	})

	t.Run("with the main worktree on no branch nothing is on the trunk, and a checkout that is its data root is read once", func(t *testing.T) {
		detached := hygiene{
			checkout: second, data: second, sessions: at("second-sessions"), temporary: at("second-temporary"), cache: at("second-cache"),
			stores:     func() ([]leftover, error) { return nil, nil },
			registered: func() ([]string, error) { return nil, nil },
			processes:  func() ([]string, error) { return nil, nil },
		}
		want := []string{
			"branch|refs/heads/master|delete|unknown",
			"branch|refs/heads/side|delete|unknown",
			"helper-copy|second/.claude/worktrees/agent|delete|unknown",
			"session-scratch|second-sessions|report|unknown",
			"undeclared|second/.claude/other|report|undeclared",
			"worktree|second-lane|report|unknown",
		}
		if got := verdicts(list(t, detached)); !slices.Equal(got, want) {
			t.Fatalf("with the main worktree detached the verdicts are\n%s\nwant\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
		}
	})

	t.Run("the command prints the listing, writes it as a receipt and removes nothing", func(t *testing.T) {
		var output bytes.Buffer
		if err := leftoversCommand(h, false, &output); err != nil {
			t.Fatal(err)
		}
		receipt := filepath.Join(repo, processlock.StateDirectory, "leftovers-"+trunk[:12]+".txt")
		written, err := os.ReadFile(receipt)
		if err != nil {
			t.Fatal(err)
		}
		text, last := output.String(), ""
		if cut := strings.LastIndex(text, "loop: "); cut >= 0 {
			text, last = text[:cut], text[cut:]
		}
		if text != string(written) || !strings.Contains(last, receipt) || strings.Count(last, "\n") != 1 {
			t.Fatalf("the command printed\n%s\nand wrote\n%s\nwant the same listing in both and one last line naming %s", output.String(), written, receipt)
		}
		entries, places := 0, 0
		for line := range strings.Lines(text) {
			switch fields := strings.Split(strings.TrimSuffix(line, "\n"), "\t"); {
			case strings.HasPrefix(line, "# "):
				places++
			case len(fields) != 6 || slices.Contains(fields, ""):
				t.Errorf("the line %q is not an entry's six fields: action, state, bytes, place, path, rule", line)
			default:
				entries++
			}
		}
		if entries != len(found) || places != len(leftoverPlaces) {
			t.Fatalf("the command printed %d entries and %d places, want the listing's %d and the table's %d", entries, places, len(found), len(leftoverPlaces))
		}
		for _, line := range []string{
			"report\tfinished\t7\tbin\t" + at("repo/bin/loose.exe") + "\t",
			"never\tkept\t-\tprocess-state\t" + at("repo/.overgo-runtime") + "\t",
			"delete\theld\t-\tbranch\trefs/heads/merged-held\t",
			"# bin at checkout:bin/: ",
			"; a finished entry: report; 4 entries, 25 bytes measured, 1 finished\n",
			"# branch at git:refs/heads: ",
			"; a finished entry: delete; 4 entries, 0 bytes measured, 1 finished\n",
		} {
			if !strings.Contains(text, line) {
				t.Errorf("the printed listing lacks the line opening %q:\n%s", line, text)
			}
		}

		after := snapshot(t)
		delete(after, relative(receipt))
		if !maps.Equal(before, after) {
			var moved []string
			for path, size := range before {
				if now, kept := after[path]; !kept || now != size {
					moved = append(moved, fmt.Sprintf("%s was %d and is %d (still there: %v)", path, size, now, kept))
				}
			}
			for path := range after {
				if _, was := before[path]; !was {
					moved = append(moved, path+" is new")
				}
			}
			slices.Sort(moved)
			t.Fatalf("the listings changed the fixture, and a listing removes and writes nothing but its receipt:\n%s", strings.Join(moved, "\n"))
		}
		if registered, err := landing.Worktrees(ctx, repo); err != nil || !slices.Equal(registered, registeredBefore) {
			t.Fatalf("after the listings git registers %+v (%v), want the worktrees it registered before", registered, err)
		}
	})
}

// fixtureLink makes a link a recursive delete could follow, a symbolic link, or a junction where
// that needs a privilege, and reports whether this machine made one.
func fixtureLink(target, name string) bool {
	return os.Symlink(target, name) == nil || exec.Command("cmd", "/c", "mklink", "/J", name, target).Run() == nil
}

// TestTmpEntryBelongsToAnOpenRowOrIsFinished builds, under a temporary directory, a repository whose
// scratch folder holds one entry of each kind; the real scratch folder is never read. Nothing
// accumulates there. An entry is kept only as one of the scratch module's own two files, and held
// only while a plan holds a row of its name, the working plan or the plan of the trunk's commit, in
// any letter case, or while a committed line or a store registration names it. Every other entry
// is finished, whatever it is: a loose file, an executable, a Python file, a folder no row was ever
// named for, and the folder of a row that has left both plans, for which no landing is asked. What
// cannot be established is unknown and kept, since a failed read never reads as finished: every
// entry but the module's two files when the working plan cannot be read; every entry that neither
// the working plan nor a committed line holds when the main worktree is on no branch; every entry
// that nothing else holds when the store's registrations cannot be read; an entry whose name the
// search of the committed lines could not have found; and a finished entry that holds a link.
// Every entry's action is delete, an entry's bytes are measured, and no listing removes anything.
func TestTmpEntryBelongsToAnOpenRowOrIsFinished(t *testing.T) {
	t.Parallel()
	// Git names a worktree by its resolved path, so the fixture's root is resolved the same way.
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	at := func(slash string) string { return filepath.Join(root, filepath.FromSlash(slash)) }
	folder := func(slash string) string {
		t.Helper()
		if err := os.MkdirAll(at(slash), 0o755); err != nil {
			t.Fatal(err)
		}
		return at(slash)
	}
	write := func(slash, content string) {
		t.Helper()
		folder(filepath.ToSlash(filepath.Dir(filepath.FromSlash(slash))))
		if err := os.WriteFile(at(slash), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// savePlan writes the working plan of a checkout: the rows named, each with one open step whose
	// reasoning names a scratch folder, which holds nothing since the plan is a work record.
	savePlan := func(checkout string, rows ...string) {
		t.Helper()
		document := plan.Plan{Campaign: "hygiene", Doctrine: []plan.Paragraph{{Governs: "standing", Text: "the plan of a fixture"}}, Lane: "master-lead"}
		for _, row := range rows {
			document.Items = append(document.Items, plan.Item{ID: row, Title: "a row of the fixture", Status: plan.StatusOpen, Steps: []plan.Step{{
				ID: "do", Title: "its one step", Status: plan.StatusOpen, Verify: "go version", Rationale: "its probes lie under tmp/landed-row/x",
			}}})
		}
		if err := plan.Save(filepath.Join(checkout, filepath.FromSlash(plan.Path)), document); err != nil {
			t.Fatal(err)
		}
	}

	// The trunk's checkout: a record and a source file that name scratch entries, and the plan.
	repo := folder("repo")
	gitFixture(t, repo, "init", "-q", "-b", "master")
	gitFixture(t, repo, "config", "core.autocrlf", "false")
	write("repo/.gitignore", "/tmp/\n")
	write("repo/docs/verification/scratch-record.note", "overlay: tmp/cited-row/overlay.data\n")
	// The second name has no boundary on its left, so it names nothing in the scratch folder.
	write("repo/internal/x/x.go", "// Package x reads tmp/cited-file.go and mytmp/loose.go, and its probes build under tmp/go.mod.\npackage x\n")
	savePlan(repo, "open-row", "Case-Row", "tip-row")
	gitFixture(t, repo, "add", "-A")
	gitFixture(t, repo, "commit", "-q", "-m", "base")
	// The working plan after the last commit, uncommitted: one row landed whole and left it, one was filed.
	savePlan(repo, "open-row", "Case-Row", "filed-row")

	folders := []string{"open-row", "case-row", "tip-row", "filed-row", "cited-row", "registered-row", "outputs", "landed-row", "left-row", "prep", "folder.exe"}
	for _, name := range folders {
		folder("repo/tmp/" + name)
	}
	write("repo/tmp/open-row/probe.go", "package probe\n")
	write("repo/tmp/prep/old.py", "print('old')\n")
	write("repo/tmp/landed-row/probe.go", "package probe\n")
	files := []string{"loose.go", "cited-file.go", "weights.bin", "tool.exe", "script.py", "two words.txt", "notes.txt"}
	for _, name := range append([]string{"go.mod", "go.sum"}, files...) {
		write("repo/tmp/"+name, "notes")
	}
	write("outside/large.bin", "bytes behind a link")

	// A second repository whose main worktree is on no branch: nothing says what its trunk is.
	second := folder("second")
	gitFixture(t, second, "init", "-q", "-b", "master")
	gitFixture(t, second, "config", "core.autocrlf", "false")
	write("second/.gitignore", "/tmp/\n")
	folder("second/docs")
	savePlan(second, "open-row")
	gitFixture(t, second, "add", "-A")
	gitFixture(t, second, "commit", "-q", "-m", "base")
	gitFixture(t, second, "checkout", "-q", "--detach")
	folder("second/tmp/open-row")
	folder("second/tmp/landed-row")
	write("second/tmp/loose.go", "package loose\n")

	// The store registers a location in the folder of a row that has left the plans, one in a folder
	// that was never a row's, a file of the scratch folder itself, and one that is no scratch entry's.
	locations := []string{at("repo/tmp/registered-row/model.bin"), at("repo/tmp/outputs/run.bin"), at("repo/tmp/weights.bin"), at("data/elsewhere.bin")}
	h := hygiene{
		checkout: repo, data: folder("data"), sessions: folder("sessions"), temporary: folder("temporary"), cache: folder("cache"),
		executable: at("temporary/loop.exe"),
		stores:     func() ([]leftover, error) { return nil, nil },
		registered: func() ([]string, error) { return locations, nil },
		processes:  func() ([]string, error) { return nil, nil },
	}

	// scratch lists a checkout and returns the entries of its scratch folder by name, and their
	// states as sorted lines. Every entry is a child of the folder, and its action is delete.
	scratch := func(t *testing.T, h hygiene) (map[string]leftover, []string) {
		t.Helper()
		found, err := listLeftovers(h, true)
		if err != nil {
			t.Fatal(err)
		}
		entries, states := map[string]leftover{}, []string{}
		for _, entry := range found {
			if entry.place != "tmp" {
				continue
			}
			name := filepath.Base(entry.path)
			if entry.path != filepath.Join(h.checkout, "tmp", name) {
				t.Errorf("the scratch entry %s is no child of the scratch folder of %s", entry.path, h.checkout)
			}
			if entry.action != "delete" {
				t.Errorf("the scratch entry %s has the action %s, want delete", name, entry.action)
			}
			if strings.TrimSpace(entry.rule) == "" {
				t.Errorf("the scratch entry %s is %s by no rule", name, entry.state)
			}
			entries[name] = entry
			states = append(states, name+"|"+entry.state)
		}
		slices.Sort(states)
		return entries, states
	}
	finished := []string{"landed-row", "left-row", "prep", "folder.exe", "loose.go", "notes.txt", "tool.exe", "script.py"}
	base := map[string]string{
		"open-row": "held", "case-row": "held", "tip-row": "held", "filed-row": "held",
		"cited-row": "held", "registered-row": "held", "outputs": "held", "cited-file.go": "held", "weights.bin": "held",
		"two words.txt": "unknown", "go.mod": "kept", "go.sum": "kept",
	}
	for _, name := range finished {
		base[name] = "finished"
	}
	// wanted is the states of the fixture's entries as sorted lines, with those named given the state in place of the usual one.
	wanted := func(state string, names ...string) []string {
		lines := make([]string, 0, len(base))
		for name, usual := range base {
			if slices.Contains(names, name) {
				usual = state
			}
			lines = append(lines, name+"|"+usual)
		}
		slices.Sort(lines)
		return lines
	}
	// named fails unless an entry has the state given and its rule carries the text given, with either slash.
	named := func(t *testing.T, entries map[string]leftover, name, state, text string) {
		t.Helper()
		if entry := entries[name]; entry.state != state || !strings.Contains(filepath.ToSlash(entry.rule), filepath.ToSlash(text)) {
			t.Errorf("%s is %s by the rule %q, want %s by a rule that names %q", name, entry.state, entry.rule, state, text)
		}
	}
	snapshot := func(t *testing.T) map[string]int64 {
		t.Helper()
		seen := map[string]int64{}
		if err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			seen[path] = -1
			if entry.Type().IsRegular() {
				info, err := entry.Info()
				if err != nil {
					return err
				}
				seen[path] = info.Size()
			}
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		for _, repository := range []string{repo, second} {
			for _, arguments := range [][]string{{"log", "--all", "--format=%H"}, {"branch", "--list"}} {
				seen[repository+" git "+strings.Join(arguments, " ")+"\n"+gitFixture(t, repository, arguments...)] = 0
			}
		}
		return seen
	}
	before := snapshot(t)
	// unchanged fails when a listing removed, added or resized anything in the fixture, or moved a commit or a branch.
	unchanged := func(t *testing.T) {
		t.Helper()
		if after := snapshot(t); !maps.Equal(before, after) {
			var moved []string
			for path, size := range before {
				if now, kept := after[path]; !kept || now != size {
					moved = append(moved, fmt.Sprintf("%s was %d and is %d (still there: %v)", path, size, now, kept))
				}
			}
			for path := range after {
				if _, was := before[path]; !was {
					moved = append(moved, path+" is new")
				}
			}
			slices.Sort(moved)
			t.Fatalf("the listing changed the fixture, and a listing removes and writes nothing:\n%s", strings.Join(moved, "\n"))
		}
	}

	t.Run("every entry is finished unless a plan, a committed line or a registration holds it, by a rule that says which", func(t *testing.T) {
		entries, states := scratch(t, h)
		if want := wanted(""); !slices.Equal(states, want) {
			t.Fatalf("the scratch entries are\n%s\nwant\n%s", strings.Join(states, "\n"), strings.Join(want, "\n"))
		}
		for _, entry := range []struct{ name, state, text string }{
			{"open-row", "held", "the working plan holds its row"},
			{"case-row", "held", "the working plan holds its row"},
			{"filed-row", "held", "the working plan holds its row"},
			{"tip-row", "held", "the plan of the trunk's commit holds its row"},
			{"cited-row", "held", "docs/verification/scratch-record.note:1"},
			{"cited-file.go", "held", "internal/x/x.go:1"},
			{"registered-row", "held", locations[0]},
			{"outputs", "held", locations[1]},
			{"weights.bin", "held", locations[2]},
			{"go.mod", "kept", "the scratch module's own file"},
			{"two words.txt", "unknown", "no line that names it could be found"},
		} {
			named(t, entries, entry.name, entry.state, entry.text)
		}
		for _, name := range finished {
			named(t, entries, name, "finished", "no row of its name")
		}
		if notes := entries["notes.txt"]; !notes.measured || notes.bytes != 5 {
			t.Errorf("notes.txt measures %d bytes (measured %v), want its 5", notes.bytes, notes.measured)
		}
		unchanged(t)
	})

	t.Run("with the store's registrations unreadable nothing reads as finished, and nothing as registered", func(t *testing.T) {
		failing := h
		failing.registered = func() ([]string, error) { return nil, errors.New("the store's registrations are unreadable") }
		entries, states := scratch(t, failing)
		unread := append([]string{"registered-row", "outputs", "weights.bin", "two words.txt"}, finished...)
		if want := wanted("unknown", unread...); !slices.Equal(states, want) {
			t.Fatalf("with the registrations unreadable the scratch entries are\n%s\nwant\n%s", strings.Join(states, "\n"), strings.Join(want, "\n"))
		}
		for _, name := range unread {
			named(t, entries, name, "unknown", "the store's registrations are unreadable")
		}
		unchanged(t)
	})

	t.Run("with the main worktree on no branch a row's folder is held and every other entry is unknown", func(t *testing.T) {
		detached := h
		detached.checkout = second
		entries, states := scratch(t, detached)
		if want := []string{"landed-row|unknown", "loose.go|unknown", "open-row|held"}; !slices.Equal(states, want) {
			t.Fatalf("with the main worktree detached the scratch entries are %q, want %q", states, want)
		}
		named(t, entries, "landed-row", "unknown", "on no branch")
		named(t, entries, "loose.go", "unknown", "on no branch")
		unchanged(t)
	})

	t.Run("with the working plan unreadable every entry but the module's own two files is unknown", func(t *testing.T) {
		working := filepath.Join(repo, filepath.FromSlash(plan.Path))
		saved, err := os.ReadFile(working)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(working, []byte("no plan\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		entries, states := scratch(t, h)
		if err := os.WriteFile(working, saved, 0o644); err != nil {
			t.Fatal(err)
		}
		unread := append(slices.Clone(folders), files...)
		if want := wanted("unknown", unread...); !slices.Equal(states, want) {
			t.Fatalf("with the working plan unreadable the scratch entries are\n%s\nwant\n%s", strings.Join(states, "\n"), strings.Join(want, "\n"))
		}
		for _, name := range unread {
			named(t, entries, name, "unknown", working)
		}
		unchanged(t)
	})

	// Last, since it changes the fixture: a finished folder that holds a link is not finished.
	t.Run("a finished folder that holds a link a recursive delete could follow is unknown", func(t *testing.T) {
		if !fixtureLink(at("outside"), at("repo/tmp/left-row/out")) {
			t.Log("this machine makes neither a symbolic link nor a junction, so nothing about links is held here")
			return
		}
		entries, states := scratch(t, h)
		if want := wanted("unknown", "left-row"); !slices.Equal(states, want) {
			t.Fatalf("with a link in left-row the scratch entries are\n%s\nwant\n%s", strings.Join(states, "\n"), strings.Join(want, "\n"))
		}
		named(t, entries, "left-row", "unknown", "link")
	})
}

// sandbox is the temporary directory a test that removes builds its whole machine in: the trunk's
// checkout, the data root, the temporary directory. Every path such a test hands a remover lies
// beneath it, so nothing of the real machine is ever removed.
type sandbox struct {
	t    *testing.T
	root string
}

// newSandbox resolves the test's temporary directory as git does, which names a worktree by its
// resolved path.
func newSandbox(t *testing.T) sandbox {
	t.Helper()
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return sandbox{t, root}
}

func (s sandbox) at(slash string) string { return filepath.Join(s.root, filepath.FromSlash(slash)) }

func (s sandbox) folder(slash string) string {
	s.t.Helper()
	if err := os.MkdirAll(s.at(slash), 0o755); err != nil {
		s.t.Fatal(err)
	}
	return s.at(slash)
}

func (s sandbox) write(slash, content string) {
	s.t.Helper()
	s.folder(filepath.ToSlash(filepath.Dir(filepath.FromSlash(slash))))
	if err := os.WriteFile(s.at(slash), []byte(content), 0o644); err != nil {
		s.t.Fatal(err)
	}
}

// landedCopy makes a helper copy of the trunk's checkout as a session does, makes one commit in
// it, moves the trunk up to that commit, and leaves in the copy's process state what a commit run
// there writes: the ignore rule, the gate's lock and the records of its waits and its refusals.
func (s sandbox) landedCopy(repo, slash string) string {
	s.t.Helper()
	helper := s.at(slash)
	gitFixture(s.t, repo, "worktree", "add", "-q", "--detach", helper, "master")
	s.write(slash+"/work/"+filepath.Base(helper)+".txt", "the work of one helper copy\n")
	gitFixture(s.t, helper, "add", "-A")
	gitFixture(s.t, helper, "commit", "-q", "-m", "the commit of "+slash)
	gitFixture(s.t, repo, "merge", "-q", "--ff-only", gitFixture(s.t, helper, "rev-parse", "HEAD"))
	if err := processlock.EnsureStateDirectory(helper); err != nil {
		s.t.Fatal(err)
	}
	for _, written := range []string{"gate.lock", "landing_waits.json", "landing_refusals.json"} {
		s.write(slash+"/"+processlock.StateDirectory+"/"+written, "{}\n")
	}
	return helper
}

// seen walks the sandbox and returns every path beneath it, by its slash path from the root, with
// its size, a directory's as -1. The administrative directory of the repository is left out: git's
// own view of it is compared through git.
func (s sandbox) seen() map[string]int64 {
	s.t.Helper()
	seen := map[string]int64{}
	if err := filepath.WalkDir(s.root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		inside, err := filepath.Rel(s.root, path)
		if err != nil || entry.Name() == ".git" && entry.IsDir() {
			return cmp.Or(err, fs.SkipDir)
		}
		seen[filepath.ToSlash(inside)] = -1
		if entry.Type().IsRegular() {
			info, err := entry.Info()
			if err != nil {
				return err
			}
			seen[filepath.ToSlash(inside)] = info.Size()
		}
		return nil
	}); err != nil {
		s.t.Fatal(err)
	}
	return seen
}

// moved lists what differs between two walks of a sandbox, but for the paths that are one of
// those given or lie beneath one.
func moved(before, after map[string]int64, but ...string) []string {
	excepted := func(path string) bool {
		return slices.ContainsFunc(but, func(prefix string) bool { return path == prefix || strings.HasPrefix(path, prefix+"/") })
	}
	var differences []string
	for path, size := range before {
		if now, kept := after[path]; !excepted(path) && (!kept || now != size) {
			differences = append(differences, fmt.Sprintf("%s was %d and is %d (still there: %v)", path, size, now, kept))
		}
	}
	for path := range after {
		if _, was := before[path]; !was && !excepted(path) {
			differences = append(differences, path+" is new")
		}
	}
	slices.Sort(differences)
	return differences
}

// removalState is where the removing command keeps its lock and its receipt, as slash paths from
// a sandbox's root: the process state of the trunk's checkout, which no removal touches otherwise.
const removalState = "repo/" + processlock.StateDirectory

// removalRun names a place to the removing command over a machine and returns its lines, each split
// into its fields, the token its last line carries when it only listed, and its error.
func removalRun(h hygiene, place, token string) (lines [][]string, shown string, err error) {
	var output bytes.Buffer
	err = deleteLeftovers(h, place, token, &output)
	lines, shown = removalLines(output.String())
	return lines, shown, err
}

// removalLines splits what a removing or a withdrawing command printed into its lines, each split
// into its fields, and the token its last line carries when the command only listed.
func removalLines(printed string) (lines [][]string, shown string) {
	for line := range strings.Lines(printed) {
		if said, found := strings.CutPrefix(line, "loop: "); found {
			words := strings.Fields(said)
			shown = words[len(words)-1]
			continue
		}
		lines = append(lines, strings.Split(strings.TrimSuffix(line, "\n"), "\t"))
	}
	return lines, shown
}

// verbs is a run's lines as sorted "verb path" pairs, a path by its slash path from the root.
func (s sandbox) verbs(lines [][]string) []string {
	s.t.Helper()
	var pairs []string
	for _, fields := range lines {
		if len(fields) != 5 || slices.Contains(fields, "") {
			s.t.Fatalf("the line %q is not five fields: verb, bytes, place, path, rule", fields)
		}
		path := fields[3]
		if inside, err := filepath.Rel(s.root, path); filepath.IsAbs(path) && err == nil {
			path = filepath.ToSlash(inside)
		}
		pairs = append(pairs, fields[0]+" "+path)
	}
	slices.Sort(pairs)
	return pairs
}

// ruled fails unless the line of a path carries the verb and its rule the text.
func (s sandbox) ruled(lines [][]string, slash, verb, text string) {
	s.t.Helper()
	for _, fields := range lines {
		if fields[3] == s.at(slash) || fields[3] == slash {
			if fields[0] != verb || !strings.Contains(filepath.ToSlash(fields[4]), text) {
				s.t.Errorf("%s reads %s by %q, want %s by a rule that names %q", slash, fields[0], fields[4], verb, text)
			}
			return
		}
	}
	s.t.Errorf("no line names %s", slash)
}

// TestLeftoversDeletesOnlyWhatItsTableMarksFinished builds, under a temporary directory, a machine
// with entries of every place whose finished entries are deleted but the helper copies, which
// TestHelperCopyIsRemovedOnceItsCommitIsOnTheTrunk holds, and beside them what must never go: a
// store, a sealed backup, a model prototype the store registers, checkpoints, models, a stash,
// session folders, a run that is still going. The real machine is never read.
//
// Named with a place and no token, the command removes nothing: it prints a line for each finished
// entry, a line with its rule for every other entry of the place, and a token over the finished
// entries' paths, bytes and rules. Given that token it removes exactly those entries, a line each,
// and appends the lines to one receipt in the trunk's process state, whichever checkout it runs
// in. It refuses a token that is not the listing's, a token whose listing no longer holds (an
// entry's bytes changed), a place whose action is not delete, a place the table does not have, and a
// second removal while one holds the lock. Who runs it decides nothing. A refusal removes nothing.
//
// What it leaves, each with the rule that keeps it: an experiment's output a committed record
// names in a store field, also when the path stands on the field's continuation line, while a
// record's other mention holds nothing; one the store registers a location in; one that holds a
// store, a sealed backup or the lock of a run still going, and one whose experiment's own directory
// holds such a lock, while a lock a run left there untaken holds nothing; one that holds a link; a
// file; a branch a worktree has checked out or the trunk lacks, and the local copy of the branch a push publishes to; a build folder that holds this command's executable,
// and every build folder while another go command runs or the process list cannot be read; a
// scratch folder whose row the plan holds. A removal that is refused, by the remover, by a file held
// open in the folder or by a lock on a branch's ref, leaves its entry as it was, reads left in its
// line, removes the rest and fails the command.
func TestLeftoversDeletesOnlyWhatItsTableMarksFinished(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	s := newSandbox(t)
	repo := s.folder("repo")
	gitFixture(t, repo, "init", "-q", "-b", "master")
	gitFixture(t, repo, "config", "core.autocrlf", "false")
	s.write("repo/.gitignore", "/tmp/\n")
	s.write("repo/source/tracked.txt", "tracked\n")
	// The store field wraps: the path it names stands on its continuation line. The last field names
	// an output too, and is no store field.
	s.write("repo/docs/verification/sweep-note.txt", strings.Join([]string{
		"outputs: experiments/done, experiments/sealed, experiments/live, experiments/idle, experiments/linked, experiments/pilot, experiments/rerun and experiments/busy",
		"store: a copy of the store taken before the runs,",
		"  at D:/data/experiments/kept/store. The runs are kept out of the real store.",
		`runs: D:\data\experiments\kept\runs\order-1`,
		"",
	}, "\n"))
	document := plan.Plan{Campaign: "hygiene", Doctrine: []plan.Paragraph{{Governs: "standing", Text: "the plan of a fixture"}}, Lane: "master-lead"}
	document.Items = []plan.Item{{ID: "open-row", Title: "a row of the fixture", Status: plan.StatusOpen, Steps: []plan.Step{{
		ID: "do", Title: "its one step", Status: plan.StatusOpen, Verify: "go version", Rationale: "a row that is still open",
	}}}}
	if err := plan.Save(filepath.Join(repo, filepath.FromSlash(plan.Path)), document); err != nil {
		t.Fatal(err)
	}
	gitFixture(t, repo, "add", "-A")
	gitFixture(t, repo, "commit", "-q", "-m", "base")
	s.write("repo/source/tracked.txt", "work a stash holds\n")
	gitFixture(t, repo, "stash", "push", "-q", "-m", "kept work")
	gitFixture(t, repo, "branch", "merged-free", "master")
	gitFixture(t, repo, "branch", "main", "master")
	gitFixture(t, repo, "worktree", "add", "-q", "-b", "merged-held", s.at("lanes/merged"), "master")
	gitFixture(t, repo, "worktree", "add", "-q", "-b", "lane/feature", s.at("lanes/feature"), "master")
	gitFixture(t, s.at("lanes/feature"), "commit", "-q", "--allow-empty", "-m", "lane work")
	s.write("repo/tmp/landed-row/probe.go", "package probe\n")
	s.write("repo/tmp/open-row/probe.go", "package probe\n")
	s.write("repo/tmp/loose.go", "package loose\n")

	// The data root: a real store that registers a prototype's file inside one experiment's output.
	storePath := s.folder("data/overgodb-store")
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	prototype := testutil.ArtifactID(t, artifact.KindTensorSet, "prototype")
	s.write("data/experiments/pilot/run/reward/weights.bin", "a prototype")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "removal-fixture/prototype",
		Artifacts: []artifact.Descriptor{{ID: prototype, Size: 1}},
		Locations: []artifact.LocationEvent{{Location: artifact.Location{Artifact: prototype, Kind: artifact.LocationFile, Value: s.at("data/experiments/pilot/run/reward/weights.bin")}, Action: artifact.LocationAdd}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Backup(ctx, s.at("data/experiments/sealed/out")); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	for _, output := range []string{"done/weights", "kept/store", "kept/runs/order-1", "live/run", "idle/run", "rerun/control", "busy/weights", "orphan/out", "sealed/named/" + filepath.Base(storePath)} {
		s.write("data/experiments/"+output+"/step.bin", "weights")
	}
	s.write("data/experiments/done/run.log", "log\n")
	// A run that is still going holds the lock in its output directory, which is an entry or, for a
	// run that wrote into the experiment's own directory, the directory above its entries; a run
	// that ended left its lock file in either.
	for _, output := range []string{"live/run", "rerun"} {
		going, err := processlock.Acquire(s.at("data/experiments/"+output+"/"+processlock.OutputLock), 0o600)
		if err != nil {
			t.Fatal(err)
		}
		defer going.Close()
	}
	s.write("data/experiments/idle/run/"+processlock.OutputLock, "")
	s.write("data/experiments/done/"+processlock.OutputLock, "")
	s.write("outside/large.bin", "bytes behind a link")
	linked := fixtureLink(s.at("outside"), filepath.Join(s.folder("data/experiments/linked/weights"), "out"))
	if !linked {
		t.Log("this machine makes neither a symbolic link nor a junction, so nothing about links is held here")
		if err := os.RemoveAll(s.at("data/experiments/linked")); err != nil {
			t.Fatal(err)
		}
	}
	s.write("data/checkpoints/proto/model.bin", "a trained model")
	s.write("data/models/base/model.bin", "a model")
	s.write("backups/sealed/blob", "a backup")
	s.write("data/overgodb-store.superseded-1/blob", "a superseded store")
	s.write("temporary/go-build1/x", "object")
	s.write("temporary/go-build2/b001/exe/loop.exe", "this command")
	s.folder("cache")

	running := []string(nil)
	var listing error
	h := hygiene{
		checkout: repo, running: repo, data: s.at("data"), sessions: s.at("temporary/sessions"), temporary: s.at("temporary"), cache: s.at("cache"),
		executable: s.at("temporary/go-build2/b001/exe/loop.exe"),
		kept:       []string{storePath, s.at("data/models")},
		stores: func() ([]leftover, error) {
			return []leftover{{path: s.at("backups/sealed"), state: stateKept, rule: "a sealed-backup is never deleted"}, {path: s.at("data/overgodb-store.superseded-1"), state: stateKept, rule: "a superseded-store is never deleted"}}, nil
		},
		registered: func() ([]string, error) { return storeLocations(storePath, 1) },
		processes:  func() ([]string, error) { return running, listing },
		removeTree: fsatomic.RemoveTree,
	}

	run, verbs, ruled := removalRun, s.verbs, s.ruled
	gitState := func() string {
		t.Helper()
		var state strings.Builder
		for _, arguments := range [][]string{{"worktree", "list", "--porcelain"}, {"branch", "--list"}, {"stash", "list"}} {
			state.WriteString(gitFixture(t, repo, arguments...) + "\n")
		}
		return state.String()
	}
	locations, err := storeLocations(storePath, 1)
	if err != nil {
		t.Fatal(err)
	}
	first, registered := s.seen(), gitState()
	// untouched fails when anything in the sandbox or in git's view of it moved since the walk
	// given, but for the command's own lock and receipt and the paths excepted.
	untouched := func(t *testing.T, before map[string]int64, git string, but ...string) {
		t.Helper()
		if differences := moved(before, s.seen(), append(but, removalState)...); len(differences) != 0 {
			t.Fatalf("the command moved what it must leave:\n%s", strings.Join(differences, "\n"))
		}
		if now := gitState(); git != "" && now != git {
			t.Fatalf("git's worktrees, branches and stashes are\n%s\nwant them as they were:\n%s", now, git)
		}
	}
	receipt := s.at(removalState + "/leftovers-removed.txt")
	var receipted strings.Builder
	// receipts adds a run's removal lines to those the receipt must hold, in order.
	receipts := func(lines [][]string) {
		for _, fields := range lines {
			if fields[0] == "removed" || fields[0] == "left" {
				receipted.WriteString(strings.Join(fields, "\t") + "\n")
			}
		}
	}

	experiments := []string{
		"kept data/experiments/busy/weights", "kept data/experiments/done/run.log", "kept data/experiments/done/weights", "kept data/experiments/idle/run",
		"kept data/experiments/kept/runs", "kept data/experiments/kept/store", "kept data/experiments/live/run", "kept data/experiments/orphan/out",
		"kept data/experiments/pilot/run", "kept data/experiments/sealed/named", "kept data/experiments/sealed/out", "kept data/experiments/sealed/out.lock",
		"kept data/experiments/rerun/control", "kept data/experiments/rerun/" + processlock.OutputLock, "kept data/experiments/done/" + processlock.OutputLock,
	}
	if linked {
		experiments = append(experiments, "kept data/experiments/linked/weights")
	}
	finishedExperiments := []string{"data/experiments/busy/weights", "data/experiments/done/weights", "data/experiments/idle/run", "data/experiments/kept/runs"}
	// listed is a place's listing: every entry kept, but for the finished ones given.
	listed := func(all []string, finished ...string) []string {
		lines := slices.Clone(all)
		for index, line := range lines {
			if slices.Contains(finished, strings.TrimPrefix(line, "kept ")) {
				lines[index] = "would-remove " + strings.TrimPrefix(line, "kept ")
			}
		}
		slices.Sort(lines)
		return lines
	}

	var token string
	t.Run("a place named with no token is listed with a token, and nothing is removed", func(t *testing.T) {
		lines, shown, err := run(h, "experiment", "")
		if want := listed(experiments, finishedExperiments...); err != nil || !slices.Equal(verbs(lines), want) || len(shown) != 16 {
			t.Fatalf("the listing of the experiments is (%v, token %q)\n%s\nwant a token of 16 digits and\n%s", err, shown, strings.Join(verbs(lines), "\n"), strings.Join(want, "\n"))
		}
		for _, entry := range []struct{ slash, verb, text string }{
			{"data/experiments/done/weights", "would-remove", "experiment a committed record names"},
			{"data/experiments/kept/runs", "would-remove", "docs/verification/sweep-note.txt:4"},
			{"data/experiments/kept/store", "kept", "docs/verification/sweep-note.txt:3"},
			{"data/experiments/pilot/run", "kept", "the store registers"},
			{"data/experiments/sealed/out", "kept", "a store or a store backup"},
			{"data/experiments/sealed/named", "kept", "a store or a store backup"},
			{"data/experiments/live/run", "kept", "a run holds it"},
			{"data/experiments/rerun/control", "kept", "a run holds its experiment"},
			{"data/experiments/orphan/out", "kept", "no committed record names its experiment"},
			{"data/experiments/done/run.log", "kept", "a file"},
		} {
			ruled(lines, entry.slash, entry.verb, entry.text)
		}
		if linked {
			ruled(lines, "data/experiments/linked/weights", "kept", "link")
		}
		// Only what a removal would take is walked: a kept entry may be very large, and is not measured.
		for _, fields := range lines {
			if fields[3] == s.at("data/experiments/done/weights") && fields[1] != "7" {
				t.Errorf("the finished output is listed with %s bytes, want the 7 of its one file", fields[1])
			}
			if fields[0] == "kept" && fields[1] != "-" {
				t.Errorf("the kept entry %s is listed with %s bytes, want it unmeasured", fields[3], fields[1])
			}
		}
		again, repeated, err := run(h, "experiment", "")
		if err != nil || repeated != shown || !slices.Equal(verbs(again), verbs(lines)) {
			t.Fatalf("a second listing gave the token %q (%v), want the first one's %q: nothing changed between them", repeated, err, shown)
		}
		token = shown
		untouched(t, first, registered)
	})

	t.Run("a refusal removes nothing", func(t *testing.T) {
		for _, refused := range []struct {
			name, place, token, want string
		}{
			{"a token that is not the listing's", "experiment", "0123456789abcdef", "list it again"},
			{"a place that only reports", "checkpoint", "", "helper-copy, branch, experiment, temporary-build, tmp"},
			{"a place that never deletes", "store-backup", "", "helper-copy, branch, experiment, temporary-build, tmp"},
			{"the stashes", "stash", "", "helper-copy, branch, experiment, temporary-build, tmp"},
			{"the worktrees that are no helper copies", "worktree", "", "helper-copy, branch, experiment, temporary-build, tmp"},
			{"a place the table does not have", "nowhere", "", `"nowhere"`},
		} {
			lines, _, err := run(h, refused.place, refused.token)
			if err == nil || !strings.Contains(err.Error(), refused.want) || len(lines) != 0 {
				t.Errorf("%s: the command answered %v with the lines %q, want it refused, naming %q, with no line", refused.name, err, lines, refused.want)
			}
		}
		held, err := processlock.Acquire(s.at(removalState+"/leftovers-removal.lock"), 0o600)
		if err != nil {
			t.Fatal(err)
		}
		lines, _, err := run(h, "experiment", token)
		if closeErr := held.Close(); err == nil || closeErr != nil || len(lines) != 0 || !strings.Contains(err.Error(), "another removal") {
			t.Errorf("with the removal lock held the command answered %v with the lines %q (close: %v), want it refused as another removal's", err, lines, closeErr)
		}
		// The listing's token no longer holds once an entry it showed has other bytes.
		s.write("data/experiments/done/weights/step.bin", "weights, and more of them")
		lines, _, err = run(h, "experiment", token)
		if err == nil || !strings.Contains(err.Error(), "list it again") || len(lines) != 0 {
			t.Errorf("with a finished entry's bytes changed since the listing the command answered %v with the lines %q, want it refused with no line", err, lines)
		}
		s.write("data/experiments/done/weights/step.bin", "weights")
		if _, err := os.Stat(receipt); !errors.Is(err, fs.ErrNotExist) {
			t.Errorf("the refusals left a receipt (%v), and a refusal removes nothing", err)
		}
		untouched(t, first, registered)
	})

	t.Run("the experiments' token removes the finished outputs, and a removal that fails leaves its entry whole", func(t *testing.T) {
		// One entry's removal fails, and on Windows a second entry is in use: a file in it is open.
		failing := h
		failing.removeTree = func(path string) error {
			if path == s.at("data/experiments/kept/runs") {
				return errors.New("the disk refused")
			}
			return fsatomic.RemoveTree(path)
		}
		left := []string{"data/experiments/kept/runs"}
		if runtime.GOOS == "windows" {
			busy, err := os.Open(s.at("data/experiments/busy/weights/step.bin"))
			if err != nil {
				t.Fatal(err)
			}
			defer busy.Close()
			left = append(left, "data/experiments/busy/weights")
		}
		lines, _, err := run(failing, "experiment", token)
		receipts(lines)
		want := listed(experiments)
		for _, slash := range finishedExperiments {
			verb := "removed "
			if slices.Contains(left, slash) {
				verb = "left "
			}
			want = append(slices.DeleteFunc(want, func(line string) bool { return line == "kept "+slash }), verb+slash)
		}
		slices.Sort(want)
		if err == nil || !strings.Contains(err.Error(), "the disk refused") || !slices.Equal(verbs(lines), want) {
			t.Fatalf("the removal answered %v with\n%s\nwant it to fail by the entry it left and to say\n%s", err, strings.Join(verbs(lines), "\n"), strings.Join(want, "\n"))
		}
		ruled(lines, "data/experiments/kept/runs", "left", "the disk refused")
		gone := []string{"data/experiments/done/weights", "data/experiments/idle/run"}
		if runtime.GOOS != "windows" {
			gone = append(gone, "data/experiments/busy/weights")
		}
		untouched(t, first, registered, gone...)
		for _, slash := range gone {
			if _, err := os.Stat(s.at(slash)); !errors.Is(err, fs.ErrNotExist) {
				t.Errorf("%s is still there after its removal (%v)", slash, err)
			}
		}
	})

	t.Run("a later listing has a token of its own, and its removal is appended to the one receipt", func(t *testing.T) {
		if lines, _, err := run(h, "experiment", token); err == nil || len(lines) != 0 {
			t.Fatalf("the first listing's token was taken again (%v, lines %q) after the removal changed what the place holds", err, lines)
		}
		_, shown, err := run(h, "experiment", "")
		if err != nil || shown == token {
			t.Fatalf("the listing after the removal gave the token %q (%v), want one of its own", shown, err)
		}
		lines, _, err := run(h, "experiment", shown)
		receipts(lines)
		if err != nil {
			t.Fatalf("the removal of what the first one left: %v", err)
		}
		ruled(lines, "data/experiments/kept/runs", "removed", "docs/verification/sweep-note.txt:4")
		untouched(t, first, registered, finishedExperiments...)
		if written, err := os.ReadFile(receipt); err != nil || string(written) != receipted.String() {
			t.Fatalf("the receipt holds (%v)\n%s\nwant the removal lines of both runs, in order:\n%s", err, written, receipted.String())
		}
	})

	removed := slices.Clone(finishedExperiments)
	t.Run("a merged branch no worktree has checked out goes, and no other", func(t *testing.T) {
		before := gitState()
		lines, shown, err := run(h, "branch", "")
		all := []string{"kept refs/heads/lane/feature", "kept refs/heads/main", "kept refs/heads/merged-free", "kept refs/heads/merged-held"}
		if want := listed(all, "refs/heads/merged-free"); err != nil || !slices.Equal(verbs(lines), want) {
			t.Fatalf("the listing of the branches is (%v)\n%s\nwant\n%s", err, strings.Join(verbs(lines), "\n"), strings.Join(want, "\n"))
		}
		if now := gitState(); now != before {
			t.Fatalf("the listing of the branches changed git's view:\n%s\nwant\n%s", now, before)
		}
		// The local copy of the branch a push publishes to is as merged and as free as the one that
		// goes, and is held.
		ruled(lines, pushedBranch, "kept", "a push publishes")
		// A branch git cannot delete, its ref locked by another git command, is left, and its line
		// says left: a branch is no path whose absence could say it was removed in part.
		refLock := s.at("repo/.git/refs/heads/merged-free.lock")
		s.write("repo/.git/refs/heads/merged-free.lock", "")
		lines, _, err = run(h, "branch", shown)
		receipts(lines)
		if unlockErr := os.Remove(refLock); err == nil || unlockErr != nil {
			t.Fatalf("with the branch's ref locked the removal answered %v (unlocked: %v), want it to fail", err, unlockErr)
		}
		ruled(lines, "refs/heads/merged-free", "left", "git branch")
		lines, _, err = run(h, "branch", shown)
		receipts(lines)
		if err != nil {
			t.Fatal(err)
		}
		ruled(lines, "refs/heads/merged-free", "removed", "wholly in the trunk")
		if branches := strings.Fields(gitFixture(t, repo, "branch", "--format=%(refname:short)")); !slices.Equal(branches, []string{"lane/feature", "main", "master", "merged-held"}) {
			t.Fatalf("after the removal git holds the branches %q, want all but merged-free", branches)
		}
		untouched(t, first, "", removed...)
	})

	t.Run("a build folder goes only while no other go command runs, and never the one this command runs from", func(t *testing.T) {
		running = []string{"go.exe"}
		lines, shown, err := run(h, "temporary-build", "")
		if want := []string{"kept temporary/go-build1", "kept temporary/go-build2"}; err != nil || !slices.Equal(verbs(lines), want) {
			t.Fatalf("with a go command running the listing is (%v) %q, want every folder kept", err, verbs(lines))
		}
		if lines, _, err = run(h, "temporary-build", shown); err != nil || slices.ContainsFunc(lines, func(fields []string) bool { return fields[0] != "kept" }) {
			t.Fatalf("with a go command running the token of nothing removed something or failed: %v %q", err, lines)
		}
		running, listing = nil, errors.New("the process list is unreadable")
		if lines, _, err = run(h, "temporary-build", ""); err != nil || !slices.Equal(verbs(lines), []string{"kept temporary/go-build1", "kept temporary/go-build2"}) {
			t.Fatalf("with the process list unreadable the listing is (%v) %q, want every folder kept", err, verbs(lines))
		}
		ruled(lines, "temporary/go-build1", "kept", "the process list is unreadable")
		untouched(t, first, "", removed...)
		listing = nil
		_, shown, err = run(h, "temporary-build", "")
		if err != nil {
			t.Fatal(err)
		}
		// The token names the paths it was shown for: another folder of the same bytes by the same
		// rule is not what the listing showed.
		if err := os.Rename(s.at("temporary/go-build1"), s.at("temporary/go-build3")); err != nil {
			t.Fatal(err)
		}
		lines, _, err = run(h, "temporary-build", shown)
		if renameErr := os.Rename(s.at("temporary/go-build3"), s.at("temporary/go-build1")); err == nil || renameErr != nil || len(lines) != 0 {
			t.Fatalf("the token of one folder was taken for another of the same bytes (%v, lines %q; renamed back: %v)", err, lines, renameErr)
		}
		lines, _, err = run(h, "temporary-build", shown)
		receipts(lines)
		if want := []string{"kept temporary/go-build2", "removed temporary/go-build1"}; err != nil || !slices.Equal(verbs(lines), want) {
			t.Fatalf("with no other go command running the removal said (%v) %q, want %q", err, verbs(lines), want)
		}
		ruled(lines, "temporary/go-build2", "kept", "this command's own executable")
		removed = append(removed, "temporary/go-build1")
		untouched(t, first, "", removed...)
	})

	t.Run("every scratch entry that no plan holds goes, a loose file among them, and the folder of an open row stays", func(t *testing.T) {
		_, stale, err := run(h, "tmp", "")
		if err != nil {
			t.Fatal(err)
		}
		// The token names the entries it was shown for: once another entry lies in the folder, the
		// listing is another, and is listed again.
		s.write("repo/tmp/later.go", "package later\n")
		if lines, _, err := run(h, "tmp", stale); err == nil || len(lines) != 0 {
			t.Fatalf("the token of a listing was taken after another entry was written beside those it showed (%v, lines %q)", err, lines)
		}
		_, shown, err := run(h, "tmp", "")
		if err != nil || shown == stale {
			t.Fatalf("the listing with the later entry gave the token %q (%v), want one of its own", shown, err)
		}
		lines, _, err := run(h, "tmp", shown)
		receipts(lines)
		if want := []string{"kept repo/tmp/open-row", "removed repo/tmp/landed-row", "removed repo/tmp/later.go", "removed repo/tmp/loose.go"}; err != nil || !slices.Equal(verbs(lines), want) {
			t.Fatalf("the removal of the scratch entries said (%v) %q, want %q", err, verbs(lines), want)
		}
		ruled(lines, "repo/tmp/open-row", "kept", "the working plan holds its row")
		ruled(lines, "repo/tmp/loose.go", "removed", "no row of its name")
		removed = append(removed, "repo/tmp/landed-row", "repo/tmp/later.go", "repo/tmp/loose.go")
	})

	t.Run("a place goes with the listing flag and a token with a place", func(t *testing.T) {
		for _, flags := range []struct {
			leftovers    bool
			place, token string
			refused      bool
		}{
			{true, "", "", false}, {true, "experiment", "", false}, {true, "experiment", "0123456789abcdef", false}, {false, "", "", false},
			{false, "experiment", "", true}, {false, "experiment", "0123456789abcdef", true}, {true, "", "0123456789abcdef", true}, {false, "", "0123456789abcdef", true},
		} {
			if err := removalFlags(flags.leftovers, flags.place, "", flags.token); (err != nil) != flags.refused {
				t.Errorf("the flags %+v are answered %v, want refused: %v", flags, err, flags.refused)
			}
		}
	})

	t.Run("nothing else moved", func(t *testing.T) {
		untouched(t, first, "", removed...)
		for _, kept := range []string{
			"data/checkpoints/proto/model.bin", "data/models/base/model.bin", "data/overgodb-store", "backups/sealed/blob", "data/overgodb-store.superseded-1/blob",
			"data/experiments/pilot/run/reward/weights.bin", "data/experiments/kept/store/step.bin", "data/experiments/sealed/out", "data/experiments/live/run/step.bin",
			"data/experiments/rerun/control/step.bin", "outside/large.bin", "lanes/merged/source/tracked.txt", "temporary/go-build2/b001/exe/loop.exe", "repo/tmp/open-row/probe.go",
		} {
			if _, err := os.Stat(s.at(kept)); err != nil {
				t.Errorf("%s did not outlive the removals: %v", kept, err)
			}
		}
		if stash := gitFixture(t, repo, "stash", "list"); !strings.Contains(stash, "kept work") {
			t.Errorf("the stash list is %q after the removals, want the stash it held", stash)
		}
		if now, err := storeLocations(storePath, 1); err != nil || !slices.Equal(now, locations) {
			t.Errorf("the store registers %q (%v) after the removals, want what it registered before: %q", now, err, locations)
		}
		// Every path removed has its line, and so has the branch, which is no path.
		if written, err := os.ReadFile(receipt); err != nil || string(written) != receipted.String() || strings.Count(string(written), "removed\t") != len(removed)+1 {
			t.Errorf("the receipt holds (%v)\n%s\nwant the lines the removals printed, one for each of the %d paths removed and one for the branch:\n%s", err, written, len(removed), receipted.String())
		}
	})
}

// clearingWorld is a row sequence's world whose clearing is the real one over a test's machine. Its
// gate and git's confirmation answer as scripted; asked to clear, it records the call and clears
// whatever that machine lists as finished, as the world of a real commit does on the real one.
type clearingWorld struct {
	fakeRowWorld
	machine hygiene
}

func (w *clearingWorld) Cleared() string {
	w.calls = append(w.calls, clearedCall)
	return clearFinished(w.machine)
}

// TestCommitClearsWhatNothingHolds builds, under a temporary directory, a machine with what
// accumulates around a commit and, beside it, everything a commit must leave. The real machine is
// never read, and every removal happens beneath the test's own directory.
//
// A commit that git confirms clears every place whose finished entries are deleted, and no other.
// Those places are the helper copies, the branches, the experiments' outputs, the folders a go
// command builds in and the scratch folder. From the scratch folder goes everything that nothing
// holds, which is a loose file, an executable, a Python file, a folder no row was ever named for
// and the folder of a row that has left both plans. A build folder goes, and so does a helper copy
// whose commit is on the trunk and that holds nothing, a branch that is wholly in the trunk and
// checked out nowhere, and an experiment's output that a committed record names and the store
// registers nothing in. Each removal has its line in the receipt, and the commit's outcome says in
// one sentence how many went.
//
// What stays is the folder of a row the working plan holds, of one only the plan of the trunk's
// commit holds, and of one a plan names in another letter case. The scratch module's two files
// stay, and so does an entry a committed line names and one the store registers a location in,
// in the scratch folder or among an experiment's outputs. A helper copy git holds locked stays,
// one with uncommitted work, and the checkout the commit ran in. The build folder this command
// runs from stays, and every build folder while a go command runs. Every entry of a place that
// only reports or never deletes stays, which here is a stash, a model prototype, a model, a store
// and a backup.
//
// A landing the gate refuses, and one git does not confirm, clear nothing and take no lock. A
// clearing that cannot run, because another removal holds the lock, removes nothing. One whose
// remover of directories fails removes only the branch, which git removes. Each leaves the commit's
// outcome ready and says why in the outcome.
func TestCommitClearsWhatNothingHolds(t *testing.T) {
	t.Parallel()
	s := newSandbox(t)
	repo := s.folder("repo")
	gitFixture(t, repo, "init", "-q", "-b", "master")
	gitFixture(t, repo, "config", "core.autocrlf", "false")
	s.write("repo/.gitignore", "/tmp/\n")
	s.write("repo/source/tracked.txt", "tracked\n")
	s.write("repo/docs/verification/run-note.txt", "outputs: experiments/done\nprobes: tmp/cited-probe/out.data and tmp/cited.go\n")
	// savePlan writes the working plan of the trunk's checkout: the rows named, each open.
	savePlan := func(rows ...string) {
		t.Helper()
		document := plan.Plan{Campaign: "hygiene", Doctrine: []plan.Paragraph{{Governs: "standing", Text: "the plan of a fixture"}}, Lane: "master-lead"}
		for _, row := range rows {
			document.Items = append(document.Items, plan.Item{ID: row, Title: "a row of the fixture", Status: plan.StatusOpen, Steps: []plan.Step{{
				ID: "do", Title: "its one step", Status: plan.StatusOpen, Verify: "go version", Rationale: "a row that is still open",
			}}})
		}
		if err := plan.Save(filepath.Join(repo, filepath.FromSlash(plan.Path)), document); err != nil {
			t.Fatal(err)
		}
	}
	savePlan("open-row", "Case-Row", "tip-row")
	gitFixture(t, repo, "add", "-A")
	gitFixture(t, repo, "commit", "-q", "-m", "base")
	s.write("repo/source/tracked.txt", "work a stash holds\n")
	gitFixture(t, repo, "stash", "push", "-q", "-m", "kept work")
	gitFixture(t, repo, "branch", "merged-free", "master")
	// The local copy of the branch a push publishes to is as merged and as free, and a commit leaves it.
	gitFixture(t, repo, "branch", "main", "master")

	// The helper copies: one that is finished, and three a commit must leave.
	const copies = "temporary/sessions/s1/scratchpad/"
	landed := s.landedCopy(repo, copies+"landed")
	gitFixture(t, repo, "worktree", "lock", "--reason", "its session keeps it", s.landedCopy(repo, copies+"reserved"))
	s.landedCopy(repo, copies+"edited")
	s.write(copies+"edited/source/tracked.txt", "edited\n")
	committing := s.landedCopy(repo, copies+"committing")
	// The working plan after the last commit, uncommitted: one row landed whole and left it, one was filed.
	savePlan("open-row", "Case-Row", "filed-row")

	// The scratch folder: what nothing holds, and what a plan, a committed line or the store holds.
	held := []string{"open-row", "case-row", "tip-row", "filed-row", "cited-probe", "registered"}
	for _, name := range append([]string{"never-a-row", "landed-row"}, held...) {
		s.write("repo/tmp/"+name+"/probe.go", "package probe\n")
	}
	loose := []string{"loose.go", "tool.exe", "script.py"}
	for _, name := range append([]string{"go.mod", "go.sum", "cited.go"}, loose...) {
		s.write("repo/tmp/"+name, "a file of the fixture\n")
	}
	// The other places: an experiment's output that is finished and one the store registers a
	// location in, the build folders, and what is never anyone's to remove.
	for _, written := range []string{
		"data/experiments/done/weights/step.bin", "data/experiments/done/registered/step.bin", "data/checkpoints/proto/model.bin", "data/models/base/model.bin",
		"data/overgodb-store/blob", "backups/sealed/blob", "temporary/go-build1/x", "temporary/go-build2/b001/exe/loop.exe",
	} {
		s.write(written, "bytes of the fixture")
	}

	running := []string(nil)
	h := hygiene{
		checkout: repo, running: committing, data: s.at("data"), sessions: s.at("temporary/sessions"), temporary: s.at("temporary"), cache: s.folder("cache"),
		executable: s.at("temporary/go-build2/b001/exe/loop.exe"),
		kept:       []string{s.at("data/overgodb-store"), s.at("data/models")},
		stores:     func() ([]leftover, error) { return nil, nil },
		registered: func() ([]string, error) {
			return []string{s.at("repo/tmp/registered/probe.go"), s.at("data/experiments/done/registered/step.bin")}, nil
		},
		processes:  func() ([]string, error) { return running, nil },
		removeTree: fsatomic.RemoveTree,
	}
	gitState := func() string {
		t.Helper()
		var state strings.Builder
		for _, arguments := range [][]string{{"worktree", "list", "--porcelain"}, {"branch", "--list"}, {"stash", "list"}} {
			state.WriteString(gitFixture(t, repo, arguments...) + "\n")
		}
		return state.String()
	}
	before, registered := s.seen(), gitState()
	// branch is the one finished branch, and unbranched is what git holds once it is gone.
	const branch = "refs/heads/merged-free"
	unbranched := strings.Replace(registered, "  "+path.Base(branch)+"\n", "", 1)
	if unbranched == registered {
		t.Fatalf("git lists no branch %s to remove:\n%s", path.Base(branch), registered)
	}
	// commit runs the row sequence over a machine, with a gate that answers as given and git
	// confirming unless an error is given, and returns its outcome and the calls it made.
	commit := func(t *testing.T, machine hygiene, gate string, unconfirmed error) (landOutcome, []string) {
		t.Helper()
		world := &clearingWorld{fakeRowWorld: fakeRowWorld{gate: gate, landedErr: unconfirmed}, machine: machine}
		outcome, err := landRow(world, "row/do", "message.txt")
		if (err != nil) != (unconfirmed != nil) {
			t.Fatalf("the row sequence answered %v, want an error only when git does not confirm: %v", err, unconfirmed)
		}
		return outcome, world.calls
	}
	// left fails when anything in the sandbox moved but the paths given and the removal's own lock
	// and receipt, or git's worktrees, branches and stashes are not those given.
	left := func(t *testing.T, git string, gone ...string) {
		t.Helper()
		// A copy: the paths given may be the head of a longer list, which an append in place would overwrite.
		if differences := moved(before, s.seen(), append(slices.Clone(gone), removalState)...); len(differences) != 0 {
			t.Fatalf("the commit moved what it must leave:\n%s", strings.Join(differences, "\n"))
		}
		if now := gitState(); now != git {
			t.Fatalf("git's worktrees, branches and stashes are\n%s\nwant\n%s", now, git)
		}
	}
	receipt := s.at(removalState + "/" + removalReceipt)
	finished := []string{"repo/tmp/never-a-row", "repo/tmp/landed-row"}
	for _, name := range loose {
		finished = append(finished, "repo/tmp/"+name)
	}
	// finished are the finished entries that are directories and files, the build folder last. The
	// branch is one more, which git removes.
	finished = append(finished, copies+"landed", "data/experiments/done/weights", "temporary/go-build1")
	// receipted fails unless the receipt holds exactly a line for each finished directory and file
	// under each verb given, and the one line of the branch's removal, as sorted "verb path" pairs.
	receipted := func(t *testing.T, verbs ...string) {
		t.Helper()
		written, err := os.ReadFile(receipt)
		if err != nil {
			t.Fatal(err)
		}
		want := []string{"removed " + branch}
		for _, verb := range verbs {
			for _, slash := range finished {
				want = append(want, verb+" "+slash)
			}
		}
		slices.Sort(want)
		lines, _ := removalLines(string(written))
		if got := s.verbs(lines); !slices.Equal(got, want) {
			t.Errorf("the receipt holds the lines\n%s\nwant one for each finished entry under each of %q:\n%s", strings.Join(got, "\n"), verbs, strings.Join(want, "\n"))
		}
	}
	// said is what an outcome says of a clearing that removed so many of so many finished entries.
	said := func(removed, finished int) string {
		return fmt.Sprintf("cleared %d of %d finished entries", removed, finished)
	}

	t.Run("a commit clears every place whose finished entries are deleted", func(t *testing.T) {
		if got, want := deletingPlaces(), []string{"helper-copy", "branch", "experiment", "temporary-build", "tmp"}; !slices.Equal(got, want) {
			t.Fatalf("the table deletes the finished entries of the places %q, want exactly %q", got, want)
		}
	})

	t.Run("a landing that is refused, or that git does not confirm, clears nothing", func(t *testing.T) {
		for _, refused := range []struct {
			gate        string
			unconfirmed error
			calls       []string
		}{
			{"land: test: failed", nil, []string{phaseGate}},
			{"", errors.New("HEAD is not the landing"), []string{phaseGate, phaseLanded}},
		} {
			outcome, calls := commit(t, h, refused.gate, refused.unconfirmed)
			if outcome.Outcome != outcomeRefused || outcome.Cleared != "" || !slices.Equal(calls, refused.calls) {
				t.Errorf("the landing ended %+v after the calls %v, want it refused after %v with nothing cleared", outcome, calls, refused.calls)
			}
			// Not the removal's lock and not its receipt either: a landing that failed comes near no removal.
			if differences := moved(before, s.seen()); len(differences) != 0 {
				t.Fatalf("a landing that ended %s moved:\n%s", outcome.Outcome, strings.Join(differences, "\n"))
			}
		}
		left(t, registered)
	})

	t.Run("a clearing that cannot run, or whose remover of directories fails, leaves the commit ready and says why", func(t *testing.T) {
		if err := processlock.EnsureStateDirectory(repo); err != nil {
			t.Fatal(err)
		}
		other, err := processlock.Acquire(s.at(removalState+"/"+removalLock), 0o600)
		if err != nil {
			t.Fatal(err)
		}
		outcome, calls := commit(t, h, "", nil)
		if closeErr := other.Close(); closeErr != nil || outcome.Outcome != outcomeReady || outcome.Commit == "" || !strings.Contains(outcome.Cleared, "another removal") || !slices.Equal(calls, []string{phaseGate, phaseLanded, clearedCall}) {
			t.Fatalf("with the removal lock held the commit ended %+v after %v (close: %v), want it ready, saying another removal holds the lock", outcome, calls, closeErr)
		}
		if _, err := os.Stat(receipt); !errors.Is(err, fs.ErrNotExist) {
			t.Errorf("a clearing that could not take the lock left a receipt (%v)", err)
		}
		left(t, registered)
		failing := h
		failing.removeTree = func(string) error { return errors.New("the disk refused") }
		outcome, _ = commit(t, failing, "", nil)
		// The branch is no directory: git removes it, and every directory and file is left.
		if all := len(finished) + 1; outcome.Outcome != outcomeReady || outcome.Commit == "" || !strings.Contains(outcome.Cleared, "the disk refused") || !strings.Contains(outcome.Cleared, said(1, all)) {
			t.Fatalf("with a remover of directories that fails the commit ended %+v, want it ready, saying that the branch alone of the %d finished entries went and why", outcome, all)
		}
		receipted(t, "left")
		left(t, unbranched)
	})

	t.Run("a confirmed commit clears what nothing holds, and a build folder only once no go command runs", func(t *testing.T) {
		running = []string{"go.exe"}
		outcome, calls := commit(t, h, "", nil)
		gone := finished[:len(finished)-1]
		if outcome.Outcome != outcomeReady || outcome.Commit == "" || !strings.Contains(outcome.Cleared, said(len(gone), len(gone))) || !strings.Contains(outcome.Cleared, receipt) || !slices.Equal(calls, []string{phaseGate, phaseLanded, clearedCall}) {
			t.Fatalf("the commit ended %+v after %v, want it ready, saying that each of %d finished entries went and naming the receipt %s", outcome, calls, len(gone), receipt)
		}
		if strings.Contains(outcome.Cleared, "\n") {
			t.Errorf("the outcome says %q of the clearing, want one line", outcome.Cleared)
		}
		// Git registers every worktree it did but the one helper copy that went, whose lines end with an empty one.
		at := strings.Index(unbranched, "worktree "+filepath.ToSlash(landed)+"\n")
		if at < 0 {
			t.Fatalf("git never registered the finished helper copy %s:\n%s", landed, unbranched)
		}
		rest := unbranched[at:]
		git := unbranched[:at] + rest[strings.Index(rest, "\n\n")+len("\n\n"):]
		left(t, git, gone...)
		for _, slash := range gone {
			if _, err := os.Lstat(s.at(slash)); !errors.Is(err, fs.ErrNotExist) {
				t.Errorf("%s is still there after the commit (%v)", slash, err)
			}
		}
		running = nil
		if outcome, _ = commit(t, h, "", nil); outcome.Outcome != outcomeReady || !strings.Contains(outcome.Cleared, said(1, 1)) {
			t.Fatalf("with no go command running the next commit ended %+v, want it ready, saying that the one build folder went", outcome)
		}
		left(t, git, finished...)
		// Every removal has its line, after the lines of the clearing whose remover failed.
		receipted(t, "left", "removed")
		// A commit after it finds nothing to clear, and moves nothing.
		if outcome, _ = commit(t, h, "", nil); !strings.Contains(outcome.Cleared, said(0, 0)) {
			t.Errorf("a commit with nothing to clear says %q, want it to say that none went", outcome.Cleared)
		}
		left(t, git, finished...)
		receipted(t, "left", "removed")
	})

	t.Run("nothing finished is left, and what a rule keeps is still there", func(t *testing.T) {
		found, err := listLeftovers(h, true, deletingPlaces()...)
		if err != nil {
			t.Fatal(err)
		}
		if at := slices.IndexFunc(found, func(listed leftover) bool { return listed.state == stateFinished }); at >= 0 {
			t.Errorf("after the commits %s is still listed as finished by the place %s", found[at].path, found[at].place)
		}
		if !slices.ContainsFunc(found, func(listed leftover) bool {
			return listed.path == s.at("data/experiments/done/registered") && listed.state == stateHeld
		}) {
			t.Errorf("the experiment's output the store registers a location in is not listed as held: %+v", found)
		}
		if _, err := os.Stat(s.at("data/experiments/done/registered/step.bin")); err != nil {
			t.Errorf("the experiment's output the store registers a location in did not outlive the commits: %v", err)
		}
		for _, name := range append([]string{"go.mod", "go.sum", "cited.go"}, held...) {
			if _, err := os.Stat(s.at("repo/tmp/" + name)); err != nil {
				t.Errorf("the scratch entry %s did not outlive the commits: %v", name, err)
			}
		}
	})
}

// TestHelperCopyIsRemovedOnceItsCommitIsOnTheTrunk builds, under a temporary directory, a
// repository with helper copies in every state a session leaves one in. The real machine is never
// read.
//
// This test holds the deleting command. A confirmed commit clears the finished copies too, by the
// same listing and the same remover, which TestCommitClearsWhatNothingHolds holds: a build checkout
// that is to be used again is kept by git's lock on it, as the copy here that git holds locked is.
//
// Named the helper copies with no token, the command lists the finished ones as would-remove and
// removes nothing: a copy a commit ran in whose commit is on the trunk, and that holds nothing
// uncommitted, nothing ignored and nothing in its process state but what a commit writes, or that
// is emptied of every tracked file. Every other copy is kept with the rule that keeps it: one with
// an edited file, one with an ignored file, one a session's holder file names, one that holds a
// session's state, one whose commit the trunk lacks, one a commit is running in, one git holds
// locked against removal, and the checkout the command runs in; and, of unknown state, one with no
// process state, one whose process state holds no record of a commit's run and one whose directory
// is gone. A lane's worktree and the cycle checkout are no helper copies and are not listed, and no
// fact of another place is read, the store's least of all.
//
// A token that is not the listing's, the listing's own token once a copy it showed has changed,
// and a removal while another holds the lock remove nothing. A remover that fails, one that leaves
// the directory in place and, on Windows, a file held open in a copy leave every copy whole and
// registered, each on a line that reads left, and fail the command. Given its token, the command
// removes exactly the copies it listed, clears each one's registration and appends a line for each
// to the receipt in the trunk's process state, from whichever checkout it runs in; every other
// copy stays whole and registered. A copy locked after it was listed and before git is asked loses
// its directory and keeps its registration: its line reads part-removed, never removed and never
// left, and says its directory is gone and git still registers it.
func TestHelperCopyIsRemovedOnceItsCommitIsOnTheTrunk(t *testing.T) {
	t.Parallel()
	s := newSandbox(t)
	repo := s.folder("repo")
	gitFixture(t, repo, "init", "-q", "-b", "master")
	gitFixture(t, repo, "config", "core.autocrlf", "false")
	s.write("repo/.gitignore", "/tmp/\n")
	s.write("repo/source/tracked.txt", "tracked\n")
	gitFixture(t, repo, "add", "-A")
	gitFixture(t, repo, "commit", "-q", "-m", "base")
	const copies = "temporary/sessions/s1/scratchpad/"
	state := "/" + processlock.StateDirectory + "/"
	landed := s.landedCopy(repo, copies+"landed")
	emptied := s.landedCopy(repo, copies+"emptied")
	for file := range strings.FieldsSeq(gitFixture(t, emptied, "ls-files")) {
		if err := os.Remove(filepath.Join(emptied, filepath.FromSlash(file))); err != nil {
			t.Fatal(err)
		}
	}
	raced := s.landedCopy(repo, copies+"raced")
	s.landedCopy(repo, copies+"edited")
	s.write(copies+"edited/source/tracked.txt", "edited\n")
	s.landedCopy(repo, copies+"heldby")
	s.write(copies+"heldby"+state+guard.WorktreeHolder, "s9\n")
	s.landedCopy(repo, copies+"stated")
	s.write(copies+"stated"+state+guard.SessionState("s3"), "{}\n")
	s.landedCopy(repo, copies+"ignored")
	s.write(copies+"ignored/tmp/probe.txt", "ignored\n")
	committing, err := authoritylock.Acquire(s.landedCopy(repo, copies+"locked"))
	if err != nil {
		t.Fatal(err)
	}
	defer committing.Close()
	gitFixture(t, repo, "worktree", "lock", "--reason", "its session keeps it", s.landedCopy(repo, copies+"reserved"))
	gitFixture(t, repo, "worktree", "add", "-q", "--detach", s.at(copies+"fresh"), "master")
	// Process state that no commit made: what a command that only takes the gate's lock leaves.
	gitFixture(t, repo, "worktree", "add", "-q", "--detach", s.at(copies+"prepared"), "master")
	if err := processlock.EnsureStateDirectory(s.at(copies + "prepared")); err != nil {
		t.Fatal(err)
	}
	s.write(copies+"prepared"+state+"gate.lock", "")
	gitFixture(t, repo, "worktree", "add", "-q", "--detach", s.at(copies+"gone"), "master")
	if err := os.RemoveAll(s.at(copies + "gone")); err != nil {
		t.Fatal(err)
	}
	ahead := s.landedCopy(repo, copies+"ahead")
	gitFixture(t, ahead, "commit", "-q", "--allow-empty", "-m", "ahead of the trunk")
	gitFixture(t, repo, "worktree", "add", "-q", "-b", "merged-held", s.at("lanes/merged"), "master")
	if err := processlock.EnsureStateDirectory(repo); err != nil {
		t.Fatal(err)
	}
	gitFixture(t, repo, "worktree", "add", "-q", "--detach", s.at("repo"+state+"cycle"), "master")

	unread := func() ([]string, error) {
		t.Error("the removal of helper copies read a fact of another place: the store's registrations or the running processes")
		return nil, nil
	}
	h := hygiene{
		checkout: repo, running: repo, temporary: s.at("temporary"), removeTree: fsatomic.RemoveTree, registered: unread, processes: unread,
		stores: func() ([]leftover, error) {
			t.Error("the removal of helper copies asked the store for its backups")
			return nil, nil
		},
	}
	names := []string{"ahead", "edited", "emptied", "fresh", "gone", "heldby", "ignored", "landed", "locked", "prepared", "raced", "reserved", "stated"}
	// said is what a run says of the copies: each kept, but for those given another verb.
	said := func(verbs map[string]string) []string {
		var lines []string
		for _, name := range names {
			lines = append(lines, cmp.Or(verbs[name], "kept")+" "+copies+name)
		}
		slices.Sort(lines)
		return lines
	}
	finished := map[string]string{"emptied": "would-remove", "landed": "would-remove", "raced": "would-remove"}
	registrations := func() string {
		t.Helper()
		return gitFixture(t, repo, "worktree", "list", "--porcelain")
	}
	// roots are the worktrees git registers, each by its path as git writes it.
	roots := func(registrations string) (paths []string) {
		for line := range strings.Lines(registrations) {
			if path, found := strings.CutPrefix(strings.TrimSpace(line), "worktree "); found {
				paths = append(paths, path)
			}
		}
		return paths
	}
	before, registered := s.seen(), registrations()
	// whole fails when anything in the sandbox moved but the command's own lock and receipt, or git
	// registers anything other than it did.
	whole := func(t *testing.T) {
		t.Helper()
		if differences := moved(before, s.seen(), removalState); len(differences) != 0 {
			t.Fatalf("a run that removes nothing moved:\n%s", strings.Join(differences, "\n"))
		}
		if now := registrations(); now != registered {
			t.Fatalf("a run that removes nothing left git registering\n%s\nwant\n%s", now, registered)
		}
	}
	receipt := s.at(removalState + "/leftovers-removed.txt")
	var receipted strings.Builder
	// receipts adds a run's removal lines to those the receipt must hold, in order.
	receipts := func(lines [][]string) {
		for _, fields := range lines {
			if fields[0] != "kept" {
				receipted.WriteString(strings.Join(fields, "\t") + "\n")
			}
		}
	}

	var token string
	t.Run("named with no token the finished copies are listed with a token, and nothing is removed", func(t *testing.T) {
		lines, shown, err := removalRun(h, "helper-copy", "")
		if want := said(finished); err != nil || !slices.Equal(s.verbs(lines), want) || len(shown) != tokenDigits {
			t.Fatalf("the listing of the helper copies is (%v, token %q)\n%s\nwant a token of %d digits and\n%s", err, shown, strings.Join(s.verbs(lines), "\n"), tokenDigits, strings.Join(want, "\n"))
		}
		for _, entry := range []struct{ name, verb, text string }{
			{"landed", "would-remove", gitFixture(t, landed, "rev-parse", "HEAD")[:commitAbbreviation]},
			{"emptied", "would-remove", "emptied"},
			{"edited", "kept", "1 uncommitted or ignored paths, the first source/tracked.txt"},
			{"ignored", "kept", "the first tmp/"},
			{"heldby", "kept", "the session s9 holds it"},
			{"stated", "kept", "its process state holds " + guard.SessionState("s3")},
			{"ahead", "kept", "is not on the trunk"},
			{"locked", "kept", "a commit is running in it"},
			{"reserved", "kept", "git holds it locked against removal: its session keeps it"},
			{"fresh", "kept", "unknown: it has no process state"},
			{"prepared", "kept", "unknown: it has no process state that holds the record of a commit's run"},
			{"gone", "kept", "unknown: "},
		} {
			s.ruled(lines, copies+entry.name, entry.verb, entry.text)
		}
		for _, fields := range lines {
			if measured := fields[1] != "-" && fields[1] != "0"; measured != (fields[0] == "would-remove") {
				t.Errorf("the line %q carries the bytes %s, want a copy that would go measured and no other", fields, fields[1])
			}
		}
		again, repeated, err := removalRun(h, "helper-copy", "")
		if err != nil || repeated != shown || !slices.Equal(s.verbs(again), s.verbs(lines)) {
			t.Fatalf("a second listing gave the token %q (%v), want the listing and the token %q of the first: nothing changed between them", repeated, err, shown)
		}
		// Run in a finished copy, the command keeps that copy: its listing is another, with a token of its own.
		inside := h
		inside.running = landed
		lines, own, err := removalRun(inside, "helper-copy", "")
		if want := said(map[string]string{"emptied": "would-remove", "raced": "would-remove"}); err != nil || own == shown || !slices.Equal(s.verbs(lines), want) {
			t.Fatalf("run in a finished copy the listing is (%v, token %q beside %q)\n%s\nwant\n%s", err, own, shown, strings.Join(s.verbs(lines), "\n"), strings.Join(want, "\n"))
		}
		s.ruled(lines, copies+"landed", "kept", "it is the checkout this command runs in")
		token = shown
		whole(t)
	})

	t.Run("a token that is not the listing's, or is no longer, removes nothing", func(t *testing.T) {
		// refused fails unless the command, given a token, answers with an error that names want and prints no line.
		refused := func(name, given, want string) {
			t.Helper()
			lines, _, err := removalRun(h, "helper-copy", given)
			if err == nil || !strings.Contains(err.Error(), want) || len(lines) != 0 {
				t.Errorf("%s: the command answered %v with the lines %q, want it refused, naming %q, with no line", name, err, lines, want)
			}
		}
		refused("a token that is not the listing's", "0123456789abcdef", "list it again")
		held, err := processlock.Acquire(s.at(removalState+"/leftovers-removal.lock"), 0o600)
		if err != nil {
			t.Fatal(err)
		}
		refused("the listing's token while another removal holds the lock", token, "another removal")
		if err := held.Close(); err != nil {
			t.Fatal(err)
		}
		// A copy the listing showed as finished is written in before the token comes back.
		later := copies + "landed/work/later.txt"
		s.write(later, "written after the listing\n")
		refused("the listing's token once a copy it showed has changed", token, "list it again")
		if written, err := os.ReadFile(s.at(later)); err != nil || string(written) != "written after the listing\n" {
			t.Fatalf("the file written after the listing reads %q (%v), want it as it was written", written, err)
		}
		if err := os.Remove(s.at(later)); err != nil {
			t.Fatal(err)
		}
		if _, again, err := removalRun(h, "helper-copy", ""); err != nil || again != token {
			t.Fatalf("with the copy as it was listed the token is %q (%v), want the first listing's %q", again, err, token)
		}
		if _, err := os.Stat(receipt); !errors.Is(err, fs.ErrNotExist) {
			t.Errorf("the refusals left a receipt (%v), and a refusal removes nothing", err)
		}
		whole(t)
	})

	t.Run("a removal that fails leaves its copy whole and registered", func(t *testing.T) {
		left := said(map[string]string{"emptied": "left", "landed": "left", "raced": "left"})
		// failed fails unless the token's run, over a machine whose remover fails, reads every finished
		// copy as left by a rule that names want, fails, and moves nothing.
		failed := func(remover func(path string) error, want string) [][]string {
			t.Helper()
			failing := h
			failing.removeTree = remover
			lines, _, err := removalRun(failing, "helper-copy", token)
			receipts(lines)
			if err == nil || !strings.Contains(err.Error(), want) || !slices.Equal(s.verbs(lines), left) {
				t.Fatalf("the removal answered %v with\n%s\nwant it to fail, naming %q, and to say\n%s", err, strings.Join(s.verbs(lines), "\n"), want, strings.Join(left, "\n"))
			}
			s.ruled(lines, copies+"emptied", "left", want)
			whole(t)
			return lines
		}
		refuse := func(string) error { return errors.New("the disk refused") }
		failed(refuse, "the disk refused")
		// A remover that reports success and leaves the directory: git is not asked to clear the
		// registration, since git would then remove the directory itself, following what it holds.
		failed(func(string) error { return nil }, "is still there")
		if runtime.GOOS == "windows" {
			// A file held open in one copy: Windows refuses the remover that follows no link.
			busy, err := os.Open(filepath.Join(landed, "source", "tracked.txt"))
			if err != nil {
				t.Fatal(err)
			}
			defer busy.Close()
			lines := failed(func(path string) error {
				if path == landed {
					return fsatomic.RemoveTree(path)
				}
				return refuse(path)
			}, "the disk refused")
			for _, fields := range lines {
				if fields[3] == landed && strings.Contains(fields[4], "the disk refused") {
					t.Errorf("the copy with a file held open reads %q, want it left by Windows's own refusal", fields)
				}
			}
		}
	})

	t.Run("given its token the command removes exactly the copies it listed, from whichever checkout it runs in", func(t *testing.T) {
		commit := gitFixture(t, landed, "rev-parse", "HEAD")
		// A session locks one copy after the listing found it finished and before git is asked to
		// clear its registration: the directory is gone by then, and git refuses.
		elsewhere := h
		elsewhere.running = s.at(copies + "fresh")
		elsewhere.removeTree = func(path string) error {
			if path == raced {
				gitFixture(t, repo, "worktree", "lock", path)
			}
			return fsatomic.RemoveTree(path)
		}
		lines, _, err := removalRun(elsewhere, "helper-copy", token)
		receipts(lines)
		gone := "its directory is gone, and git still registers it"
		if want := said(map[string]string{"emptied": "removed", "landed": "removed", "raced": "part-removed"}); err == nil || !strings.Contains(err.Error(), raced+": "+gone) || !slices.Equal(s.verbs(lines), want) {
			t.Fatalf("the removal answered %v with\n%s\nwant it to fail only by the copy git still registers and to say\n%s", err, strings.Join(s.verbs(lines), "\n"), strings.Join(want, "\n"))
		}
		s.ruled(lines, copies+"landed", "removed", commit[:commitAbbreviation])
		s.ruled(lines, copies+"emptied", "removed", "emptied")
		s.ruled(lines, copies+"raced", "part-removed", gone)
		s.ruled(lines, copies+"fresh", "kept", "it is the checkout this command runs in")
		for _, fields := range lines {
			if fields[0] != "kept" && (fields[1] == "-" || fields[1] == "0" || fields[2] != "helper-copy") {
				t.Errorf("the line %q of a copy that went carries the bytes %s and the place %s, want its measured bytes and helper-copy", fields, fields[1], fields[2])
			}
		}
		taken := []string{copies + "landed", copies + "emptied", copies + "raced"}
		for _, slash := range taken {
			if _, err := os.Lstat(s.at(slash)); !errors.Is(err, fs.ErrNotExist) {
				t.Errorf("%s is still on disk after its removal (%v)", slash, err)
			}
		}
		// The two removed are no longer registered; the one locked in between is, and so is every other.
		want := slices.DeleteFunc(roots(registered), func(path string) bool {
			return path == filepath.ToSlash(landed) || path == filepath.ToSlash(emptied)
		})
		if now := roots(registrations()); !slices.Equal(now, want) || !slices.Contains(now, filepath.ToSlash(raced)) {
			t.Fatalf("after the removal git registers\n%s\nwant\n%s", strings.Join(now, "\n"), strings.Join(want, "\n"))
		}
		if differences := moved(before, s.seen(), append(taken, removalState)...); len(differences) != 0 {
			t.Fatalf("the removal moved more than the copies it listed:\n%s", strings.Join(differences, "\n"))
		}
		if written, err := os.ReadFile(receipt); err != nil || string(written) != receipted.String() || strings.Count(string(written), "\nremoved\t") != len(taken)-1 {
			t.Fatalf("run from a helper copy, the receipt in the trunk's process state holds (%v)\n%s\nwant every line the removals printed, in order, two of them removed:\n%s", err, written, receipted.String())
		}
		// The token is of a listing the place no longer gives; the next listing has nothing to remove.
		if lines, _, err := removalRun(h, "helper-copy", token); err == nil || len(lines) != 0 {
			t.Fatalf("the token was taken again (%v, lines %q) after the removal changed what the place holds", err, lines)
		}
		lines, shown, err := removalRun(h, "helper-copy", "")
		if want := said(nil); err != nil || shown == token || !slices.Equal(s.verbs(lines), slices.DeleteFunc(want, func(line string) bool {
			return line == "kept "+copies+"landed" || line == "kept "+copies+"emptied"
		})) {
			t.Fatalf("the listing after the removal is (%v, token %q)\n%s\nwant every copy that is left kept, under a token of its own", err, shown, strings.Join(s.verbs(lines), "\n"))
		}
		if differences := moved(before, s.seen(), append(taken, removalState)...); len(differences) != 0 {
			t.Fatalf("the runs after the removal moved:\n%s", strings.Join(differences, "\n"))
		}
	})
}

// recorded is one checkpoint a fixture's run recorded in its store: its folder, as a slash path
// from the sandbox's root, its identity and its resume document's, the model registered as
// produced by it, and that model's two shards: the one only it has, and the one whose bytes the
// fixture chose.
type recorded struct {
	slash                                  string
	checkpoint, resume, model, own, shared artifact.ID
}

// files are the paths the store registers for a recorded checkpoint, as slash paths from the
// sandbox's root: the checkpoint's folder twice, once for the checkpoint and once for its model,
// then its weights, its momentum and the model's two shards.
func (r recorded) files() []string {
	return []string{r.slash, r.slash, r.slash + "/" + trainingprogram.CheckpointWeights, r.slash + "/momentum.bin", r.slash + "/model-00001-of-00002.gguf", r.slash + "/model-00002-of-00002.gguf"}
}

// written is a writer that hands each write to a function, so that a test can act at the moment a
// command prints.
type written func(line []byte) (int, error)

func (w written) Write(line []byte) (int, error) { return w(line) }

// commit commits one batch of a fixture to its store.
func (s sandbox) commit(store *overgodb.Store, batch artifact.Batch) {
	s.t.Helper()
	if _, err := store.Commit(s.t.Context(), batch); err != nil {
		s.t.Fatalf("the batch %s: %v", batch.Key, err)
	}
}

// location registers a file or a directory of the sandbox for an artifact, spelled as a run spells
// the locations it records.
func (s sandbox) location(id artifact.ID, kind artifact.LocationKind, slash string) artifact.LocationEvent {
	s.t.Helper()
	location, err := artifact.CanonicalLocalLocation(id, kind, s.at(slash))
	if err != nil {
		s.t.Fatal(err)
	}
	return artifact.LocationEvent{Location: location, Action: artifact.LocationAdd}
}

// registerModel registers a model whose components are the files given, a shard each: every file
// with its location and the model with the directory of its first file, as the intake of a model
// registers one. A model a checkpoint produced names that checkpoint, as a run's trained model does.
func (s sandbox) registerModel(store *overgodb.Store, name string, producedBy artifact.ID, files ...string) artifact.ID {
	s.t.Helper()
	batch := artifact.Batch{Key: "withdrawal-fixture/model/" + name}
	var components []artifact.Component
	for index, slash := range files {
		data, err := os.ReadFile(s.at(slash))
		if err != nil {
			s.t.Fatal(err)
		}
		shard := testutil.ArtifactBytesID(s.t, artifact.KindTensorSet, data)
		components = append(components, artifact.Component{Role: artifact.ComponentWeightsShard, Ordinal: uint32(index), Name: fmt.Sprintf("weights-%d", index), Artifact: shard})
		batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: shard, Size: uint64(len(data))})
		batch.Locations = append(batch.Locations, s.location(shard, artifact.LocationFile, slash))
	}
	manifest, err := artifact.NewManifest(artifact.KindModel, components)
	if err != nil {
		s.t.Fatal(err)
	}
	batch.Manifests = []artifact.Manifest{manifest}
	batch.Locations = append(batch.Locations, s.location(manifest.ID, artifact.LocationDirectory, path.Dir(files[0])))
	if producedBy.Valid() {
		batch.Lineage = []artifact.Lineage{{Child: manifest.ID, Parent: producedBy, Relation: artifact.RelationProducedBy}}
	}
	s.commit(store, batch)
	return manifest.ID
}

// activate gives a model an active alias, as an activation does: the alias of the model and a
// task, bound to the recipe that serves it.
func (s sandbox) activate(store *overgodb.Store, model artifact.ID) {
	s.t.Helper()
	served := testutil.ArtifactID(s.t, artifact.KindRecipe, "the recipe that serves "+model.String())
	s.commit(store, artifact.Batch{
		Key: "withdrawal-fixture/active/" + model.String(), Artifacts: []artifact.Descriptor{{ID: served}},
		Aliases: []artifact.AliasBinding{{Name: modelrecipe.ActiveAlias(model, recipe.TaskInference), Target: served}},
	})
}

// recordCheckpoint publishes a one-step checkpoint in a folder of the sandbox and records it in the
// store as a run records a checkpoint that holds a trained model: the checkpoint's document, its
// weights, its momentum and their locations, by the checkpoint's own batch, and beside them a model
// of two shards registered as produced by the checkpoint. The second shard's bytes are those given,
// so that a model of another run, or of no run, can hold the same shard. The step is taken along
// the gradient given: two checkpoints of one gradient hold the same weights and the same momentum,
// byte for byte, and two of different gradients share neither.
func (s sandbox) recordCheckpoint(store *overgodb.Store, slash, shared string, gradient float32) recorded {
	s.t.Helper()
	id := func(kind artifact.Kind, name string) artifact.ID {
		return testutil.ArtifactID(s.t, kind, slash+" "+name)
	}
	const parameters = 4
	groups, err := hostoptimizer.CompilePlan(parameters, []hostoptimizer.GroupSpec{{Name: "weight", End: parameters, Rows: 2, Cols: 2}})
	if err != nil {
		s.t.Fatal(err)
	}
	config, err := trainingprogram.BuiltinOptimizerPolicy().Config(parameters)
	if err != nil {
		s.t.Fatal(err)
	}
	weights := make([]float32, parameters)
	muon, err := hostoptimizer.New(weights, []float32{gradient, -gradient, gradient / 2, gradient / 4}, groups, config)
	if err != nil {
		s.t.Fatal(err)
	}
	muon.Step()
	state := muon.Snapshot()
	base, rng := id(artifact.KindModel, "base"), id(artifact.KindProfile, "rng")
	checkpoint, err := trainingprogram.PublishCheckpoint(s.at(slash), trainingprogram.CheckpointSpec{
		RunPlan: id(artifact.KindRecipe, "run plan"), Program: id(artifact.KindRecipe, "program"), Model: base,
		Dataset: id(artifact.KindDataset, "dataset"), Split: id(artifact.KindDatasetShard, "split"),
		Stream:    trainingprogram.DatasetState{Identity: id(artifact.KindProfile, "stream"), Position: 1},
		Optimizer: state, ParameterCount: state.Parameters(),
		RNG:        []trainingprogram.RNGState{{Name: "augmentation", Algorithm: rng}, {Name: "data", Algorithm: rng}},
		Processors: []artifact.ID{id(artifact.KindProfile, "processor")},
		Lineage:    []trainingprogram.LineageParent{{Artifact: base, Relation: artifact.RelationTrainedFrom}},
	}, func(stage string) error {
		return optimizer.SavePlanWeights(filepath.Join(stage, trainingprogram.CheckpointWeights), groups, weights)
	})
	if err != nil {
		s.t.Fatal(err)
	}
	batch, err := checkpoint.Batch("withdrawal-fixture/checkpoint/"+slash, s.at(slash))
	if err != nil {
		s.t.Fatal(err)
	}
	// The store refuses an edge to an artifact it does not know, and the base model is a stand-in.
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: base})
	s.commit(store, batch)
	trained := "the weights the run at " + slash + " trained"
	s.write(slash+"/model-00001-of-00002.gguf", trained)
	s.write(slash+"/model-00002-of-00002.gguf", shared)
	model := s.registerModel(store, slash, checkpoint.ID(), slash+"/model-00001-of-00002.gguf", slash+"/model-00002-of-00002.gguf")
	return recorded{slash, checkpoint.ID(), checkpoint.Resume, model, testutil.ArtifactID(s.t, artifact.KindTensorSet, trained), testutil.ArtifactID(s.t, artifact.KindTensorSet, shared)}
}

// recordSplitWeights records, in a folder of the sandbox, a checkpoint whose weights are no one
// file but a manifest of one shard, which is how a checkpoint over a GGUF base names its weights,
// and which produced no model: the store locates the shard in the folder, and nothing but the
// manifest of the checkpoint's weights says the shard is the checkpoint's. The document is that of
// a checkpoint the fixture already recorded, with the manifest in place of its weights, so the two
// share a resume document; no GGUF file is written, and the store is given no model, which the
// batch of a real checkpoint over a GGUF base also publishes. It returns the paths the store
// registers for the checkpoint, as slash paths from the sandbox's root: its folder and its shard.
func (s sandbox) recordSplitWeights(store *overgodb.Store, from recorded, slash string) []string {
	s.t.Helper()
	const shardName, content = "/weights-00001-of-00001.gguf", "the one shard of weights that are a manifest"
	s.write(slash+shardName, content)
	shard := testutil.ArtifactID(s.t, artifact.KindTensorSet, content)
	weights, err := artifact.NewManifest(artifact.KindTensorSet, []artifact.Component{{Role: artifact.ComponentWeightsShard, Name: "weights-0", Artifact: shard}})
	if err != nil {
		s.t.Fatal(err)
	}
	document, err := os.ReadFile(s.at(from.slash + "/" + trainingprogram.CheckpointFilename))
	if err != nil {
		s.t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(document, &fields); err != nil {
		s.t.Fatal(err)
	}
	if fields["weights"], err = json.Marshal(weights.ID); err != nil {
		s.t.Fatal(err)
	}
	if document, err = json.Marshal(fields); err != nil {
		s.t.Fatal(err)
	}
	checkpoint, err := trainingprogram.ParseCheckpoint(document)
	if err != nil || checkpoint.Weights != weights.ID {
		s.t.Fatalf("the document with a manifest for its weights reads as a checkpoint of the weights %s (%v), want %s", checkpoint.Weights, err, weights.ID)
	}
	s.write(slash+"/"+trainingprogram.CheckpointFilename, string(document))
	s.commit(store, artifact.Batch{
		Key:       "withdrawal-fixture/split-weights/" + slash,
		Artifacts: []artifact.Descriptor{{ID: checkpoint.ID()}, {ID: shard, Size: uint64(len(content))}},
		Manifests: []artifact.Manifest{weights},
		Locations: []artifact.LocationEvent{
			s.location(checkpoint.ID(), artifact.LocationDirectory, slash), s.location(shard, artifact.LocationFile, slash+shardName),
		},
	})
	return []string{slash, slash + shardName}
}

// TestWithdrawalTakesOnlyARunsCheckpointRegistrations builds, under a temporary directory, a data
// root whose store registers the checkpoints of several experiment runs, each with the model it
// produced, and beside them what a withdrawal must never take: a model prototype registered from a
// file in a run's directory, a second there whose file holds the same bytes as a shard of the
// checkpoint's model, two registered from files in a checkpoint's own folder, a file there
// that is no model, and models of no run that hold a shard a checkpoint's model also holds. The
// real machine and the real store are never read.
//
// Named a run's directory and no token, the command changes nothing, and asks the store for no
// write access: it prints a line for every location the store registers under a folder that holds
// a checkpoint document, when the location is of that checkpoint, of its weights or its momentum,
// or of a model the checkpoint produced or one of that model's shards; a line that reads kept for
// every other location under the directory, which is not a run's checkpoint; and a token. Given
// that token it withdraws exactly the locations it listed, in the store alone: no file moves, every
// identity and every edge stays, the same weights and momentum recorded by a checkpoint elsewhere
// keep their locations there, and each withdrawn location has its line in the receipt the removals
// share. A checkpoint accounts for a location only under its own folder: the same bytes located
// elsewhere in the run, as a prototype's file, keep that location. A prototype keeps every location, also in a checkpoint's own folder, whatever else the
// store says of it: that the checkpoint's resume document depends on it, or that it was derived
// from the checkpoint. So does a file the checkpoint is said to have produced that is no model.
// What is kept still holds the run's directory against deletion; a run with nothing kept reads
// finished once its registrations are withdrawn, and the deleting command then removes it.
//
// A shard another model also holds loses only its location in the checkpoint's folder. When that
// is the only location the shard has, the whole withdrawal is refused and names the model: one
// that is served, whether or not a checkpoint of the run produced it, and one that no checkpoint
// of the run produced, served or not. Also refused, with nothing withdrawn: the experiments folder
// itself, a directory outside it, a run no committed record names, a run with a checkpoint
// document that cannot be read as one, a token that is not the listing's,
// and a withdrawal while a removal holds the lock.
//
// A committed record must name the directory itself. The pilot's experiment holds a second run
// that no record names, and the record of the pilot's run names the experiment's folder only
// through that run beneath it: the second run is refused, and so is the experiment's folder, which
// would otherwise list the second run's registrations beside the first's.
//
// A checkpoint's weights may be a manifest of shards and no one file. One checkpoint of the run
// with no prototype has such weights and produced no model: the shard the store locates in its
// folder is withdrawn with the checkpoint's own location, by the manifest of its weights alone.
func TestWithdrawalTakesOnlyARunsCheckpointRegistrations(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	s := newSandbox(t)
	repo := s.folder("repo")
	gitFixture(t, repo, "init", "-q", "-b", "master")
	gitFixture(t, repo, "config", "core.autocrlf", "false")
	s.write("repo/docs/verification/pilot-record.txt", "out: D:/data/experiments/pilot/run\nruns: experiments/clean/run, experiments/strand/run, experiments/adopted/run, experiments/lone/run and experiments/torn/run\n")
	gitFixture(t, repo, "add", "-A")
	gitFixture(t, repo, "commit", "-q", "-m", "base")

	const (
		pilotRun, cleanRun, strandRun   = "data/experiments/pilot/run", "data/experiments/clean/run", "data/experiments/strand/run"
		adoptedRun, loneRun, unrecorded = "data/experiments/adopted/run", "data/experiments/lone/run", "data/experiments/unrecorded/run"
		tornRun, siblingRun             = "data/experiments/torn/run", "data/experiments/pilot/other"
		pilotShard                      = "the shard the pilot's model shares with a served one"
	)
	storePath := s.folder("data/overgodb-store")
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	// The pilot's run: one checkpoint with its model, and prototypes the owner registered by hand,
	// first one from a folder of its own and two from files beside the checkpoint's own files. A
	// served model of no run holds the shard the checkpoint's model holds, at a path of its own.
	pilot := s.recordCheckpoint(store, pilotRun+"/reward/groups-001", pilotShard, 1)
	s.write(pilotRun+"/proto/model.gguf", "a prototype the owner made by hand")
	s.write(pilot.slash+"/handmade.gguf", "a second prototype, beside a checkpoint's own files")
	s.write(pilot.slash+"/derived.gguf", "a third prototype, made by hand from the checkpoint")
	s.write(pilot.slash+"/notes.txt", "what the owner wrote down about the checkpoint")
	apart := s.registerModel(store, "prototype-apart", artifact.ID{}, pilotRun+"/proto/model.gguf")
	beside := s.registerModel(store, "prototype-beside", artifact.ID{}, pilot.slash+"/handmade.gguf")
	derived := s.registerModel(store, "prototype-derived", artifact.ID{}, pilot.slash+"/derived.gguf")
	// A fourth prototype, in the run and outside every checkpoint's folder, whose one file holds the
	// same bytes as the shard only the checkpoint's model has: the store locates that shard twice,
	// and a checkpoint accounts only for the location in its own folder.
	s.write(pilotRun+"/proto/copy.gguf", "the weights the run at "+pilot.slash+" trained")
	copied := s.registerModel(store, "prototype-copied", artifact.ID{}, pilotRun+"/proto/copy.gguf")
	// What else the store says of the two prototypes in the checkpoint's folder does not make them
	// the run's: the checkpoint's resume document is said to depend on one, and the other is said
	// to be derived from the checkpoint, which is not to be produced by it. The notes are said to
	// be produced by the checkpoint, and are no model.
	notes := testutil.ArtifactID(t, artifact.KindFile, "what the owner wrote down about the checkpoint")
	s.commit(store, artifact.Batch{
		Key: "withdrawal-fixture/by-hand", Artifacts: []artifact.Descriptor{{ID: notes, Size: 1}},
		Locations: []artifact.LocationEvent{s.location(notes, artifact.LocationFile, pilot.slash+"/notes.txt")},
		Lineage: []artifact.Lineage{
			{Child: pilot.resume, Parent: beside, Relation: artifact.RelationDependsOn},
			{Child: derived, Parent: pilot.checkpoint, Relation: artifact.RelationDerivedFrom},
			{Child: notes, Parent: pilot.checkpoint, Relation: artifact.RelationProducedBy},
		},
	})
	kept := []string{
		pilotRun + "/proto", pilotRun + "/proto/model.gguf", pilot.slash, pilot.slash + "/handmade.gguf",
		pilot.slash, pilot.slash + "/derived.gguf", pilot.slash + "/notes.txt",
		pilotRun + "/proto", pilotRun + "/proto/copy.gguf",
	}
	s.write("data/models/served/own.gguf", "the weights of a model that is served")
	s.write("data/models/served/meta.gguf", pilotShard)
	s.activate(store, s.registerModel(store, "served", artifact.ID{}, "data/models/served/own.gguf", "data/models/served/meta.gguf"))
	// A run with no prototype: two arms of one gradient, whose checkpoints hold the same weights and
	// the same momentum, which no other checkpoint holds, and whose models hold one shard between them.
	clean := []recorded{
		s.recordCheckpoint(store, cleanRun+"/control/groups-001", "the shard the two arms share", 2),
		s.recordCheckpoint(store, cleanRun+"/reward/groups-001", "the shard the two arms share", 2),
	}
	// A third checkpoint of that run, whose weights are a manifest of a shard in its folder and which
	// produced no model: only the manifest of its weights says the shard is a checkpoint's.
	split := s.recordSplitWeights(store, clean[0], cleanRun+"/split/groups-001")
	// Three runs whose withdrawal would leave a model with a shard no location names: a served model
	// of no run, the run's own model once it is served, and a prototype of no run that nothing serves.
	strand := s.recordCheckpoint(store, strandRun+"/reward/groups-001", "the shard only the checkpoint of strand has on disk", 3)
	s.write("data/models/leaning/own.gguf", "the weights of a served model that leans on a run's shard")
	leaning := s.registerModel(store, "leaning", artifact.ID{}, "data/models/leaning/own.gguf", strand.slash+"/model-00002-of-00002.gguf")
	s.activate(store, leaning)
	adopted := s.recordCheckpoint(store, adoptedRun+"/reward/groups-001", "the shard of the arm that was adopted", 4)
	s.activate(store, adopted.model)
	lone := s.recordCheckpoint(store, loneRun+"/reward/groups-001", "the shard only the checkpoint of lone has on disk", 5)
	s.write("data/models/borrowing/own.gguf", "the weights of a prototype that borrows a run's shard")
	borrowing := s.registerModel(store, "borrowing", artifact.ID{}, "data/models/borrowing/own.gguf", lone.slash+"/model-00002-of-00002.gguf")
	// A run no committed record names and a recorded checkpoint outside the experiments, both of the
	// pilot's gradient: the same weights and the same momentum as the pilot's checkpoint, elsewhere.
	s.recordCheckpoint(store, unrecorded+"/groups-001", "the shard of a run with no record", 1)
	s.recordCheckpoint(store, "data/checkpoints/kept/groups-001", "the shard of a trained model that is kept", 1)
	// A second run of the pilot's experiment, which no committed record names: the record names the
	// pilot's run, and so names the experiment's folder only through that run beneath it.
	s.recordCheckpoint(store, siblingRun+"/groups-001", "the shard of a sibling run with no record", 6)
	// A run whose checkpoint document cannot be read as one.
	s.write(tornRun+"/groups-001/"+trainingprogram.CheckpointFilename, "{}")
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	h := hygiene{
		checkout: repo, running: repo, data: s.at("data"), store: storePath, temporary: s.at("temporary"),
		kept:       []string{storePath, s.at("data/models")},
		registered: func() ([]string, error) { return storeLocations(storePath, 1) },
		removeTree: fsatomic.RemoveTree,
	}
	withdraw := func(slash, token string) ([][]string, string, error) {
		var output bytes.Buffer
		err := withdrawRegistrations(h, s.at(slash), token, &output)
		lines, shown := removalLines(output.String())
		return lines, shown, err
	}
	// read opens the fixture's store for reading alone and hands it to a check.
	read := func(t *testing.T, check func(store *overgodb.Store)) {
		t.Helper()
		store, err := overgodb.OpenReadOnly(storePath)
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		check(store)
	}
	// slashes turns locations into their sorted slash paths from the sandbox's root.
	slashes := func(t *testing.T, values []string) []string {
		t.Helper()
		inside := make([]string, len(values))
		for index, value := range values {
			relative, err := filepath.Rel(s.root, value)
			if err != nil {
				t.Fatal(err)
			}
			inside[index] = filepath.ToSlash(relative)
		}
		slices.Sort(inside)
		return inside
	}
	// registered is every location the store registers, a path once for each artifact it locates,
	// and the store's head, which every commit moves.
	registered := func(t *testing.T) (paths []string, head artifact.CommitID) {
		t.Helper()
		values, err := storeLocations(storePath, 1)
		if err != nil {
			t.Fatal(err)
		}
		read(t, func(store *overgodb.Store) { head, _ = store.Head() })
		return slashes(t, values), head
	}
	// located is where the store locates one artifact.
	located := func(t *testing.T, id artifact.ID) (paths []string) {
		t.Helper()
		read(t, func(store *overgodb.Store) {
			locations, err := store.Locations(ctx, id)
			if err != nil {
				t.Fatal(err)
			}
			for _, location := range locations {
				paths = append(paths, location.Value)
			}
		})
		return slashes(t, paths)
	}
	// said is a command's expected lines: every path of each list under the list's verb, sorted.
	said := func(listed map[string][]string) (lines []string) {
		for verb, paths := range listed {
			for _, path := range paths {
				lines = append(lines, verb+" "+path)
			}
		}
		slices.Sort(lines)
		return lines
	}
	disk := s.seen()
	// unmoved fails when a file of the sandbox moved since the fixture was built, but for the store's
	// own files, the command's lock and receipt, and the paths excepted.
	unmoved := func(t *testing.T, but ...string) {
		t.Helper()
		if differences := moved(disk, s.seen(), append(but, removalState, "data/overgodb-store")...); len(differences) != 0 {
			t.Fatalf("the command moved files, and a withdrawal moves none:\n%s", strings.Join(differences, "\n"))
		}
	}
	receipt := s.at(removalState + "/leftovers-removed.txt")
	first, firstHead := registered(t)

	var token string
	t.Run("a run named with no token is listed with a token, and nothing is withdrawn", func(t *testing.T) {
		lines, shown, err := withdraw(pilotRun, "")
		want := said(map[string][]string{"would-withdraw": pilot.files(), "kept": kept})
		if err != nil || !slices.Equal(s.verbs(lines), want) || len(shown) != 16 {
			t.Fatalf("the listing of the pilot's run is (%v, token %q)\n%s\nwant a token of 16 digits and\n%s", err, shown, strings.Join(s.verbs(lines), "\n"), strings.Join(want, "\n"))
		}
		for _, fields := range lines {
			if (fields[0] == "kept") != strings.Contains(fields[4], "not a run's checkpoint") {
				t.Errorf("the line %q does not say, by its rule, whether the location is a run's checkpoint's", fields)
			}
		}
		// A listing asks the store for no write access: it lists while a writer holds the store's
		// write lock, the file beside the store that one writer takes at a time.
		writing, err := processlock.Acquire(storePath+".lock", 0o600)
		if err != nil {
			t.Fatal(err)
		}
		again, repeated, err := withdraw(pilotRun, "")
		if closeErr := writing.Close(); err != nil || closeErr != nil || repeated != shown || !slices.Equal(s.verbs(again), want) {
			t.Fatalf("a second listing, made while a writer held the store, gave the token %q (%v, close: %v), want the first one's %q: nothing changed between them", repeated, err, closeErr, shown)
		}
		// The listing's last line says what was done, which is nothing, and what the token would do.
		var printed bytes.Buffer
		if err := withdrawRegistrations(h, s.at(pilotRun), "", &printed); err != nil || !strings.HasSuffix(printed.String(), "\nloop: nothing was withdrawn; to withdraw the 6 entries marked would-withdraw, run this again with -confirm "+shown+"\n") {
			t.Errorf("the listing ends (%v)\n%s\nwant its last line to say nothing was withdrawn and to give the token of its 6 entries", err, printed.String())
		}
		token = shown
	})

	t.Run("a refusal withdraws nothing", func(t *testing.T) {
		for _, refused := range []struct {
			name, slash, token, want string
		}{
			{"the experiments folder itself", "data/experiments", "", "inside"},
			{"the data root, which holds the experiments folder", "data", "", "inside"},
			{"a directory outside the experiments folder", "data/checkpoints/kept", "", "inside"},
			{"a run no committed record names", unrecorded, "", "no committed record"},
			{"a run no committed record names, beside one of its experiment that a record names", siblingRun, "", "no committed record"},
			{"an experiment's folder, which a record names only through a run in it", path.Dir(pilotRun), "", "no committed record"},
			{"a token that is not the listing's", pilotRun, "0123456789abcdef", "holds now: list it again"},
			{"a token that is not the listing's, by what it is said not to be the token of", pilotRun, "0123456789abcdef", "the store under " + s.at(pilotRun)},
			{"a run that has the only copy of a shard of a served model of no run", strandRun, "", leaning.String()},
			{"a run whose own model is served", adoptedRun, "", adopted.model.String()},
			{"a run that has the only copy of a shard of a prototype nothing serves", loneRun, "", borrowing.String()},
			{"a run whose checkpoint document cannot be read as one", tornRun, "", trainingprogram.CheckpointFilename},
		} {
			lines, _, err := withdraw(refused.slash, refused.token)
			if err == nil || !strings.Contains(err.Error(), refused.want) || len(lines) != 0 {
				t.Errorf("%s: the command answered %v with the lines %q, want it refused, naming %q, with no line", refused.name, err, lines, refused.want)
			}
		}
		// Committed records that cannot be searched are a failure of their own, never read as no
		// record: a checkout that is no repository refuses with what git said.
		blind := h
		blind.checkout = s.at("data/models")
		var printed bytes.Buffer
		if err := withdrawRegistrations(blind, s.at(pilotRun), "", &printed); err == nil || !strings.Contains(err.Error(), "git") || strings.Contains(err.Error(), "no committed record") || printed.Len() != 0 {
			t.Errorf("with the committed records unreadable the command answered %v and printed %q, want it refused with git's own failure and no line", err, printed.String())
		}
		held, err := processlock.Acquire(s.at(removalState+"/leftovers-removal.lock"), 0o600)
		if err != nil {
			t.Fatal(err)
		}
		lines, _, err := withdraw(pilotRun, token)
		if closeErr := held.Close(); err == nil || closeErr != nil || len(lines) != 0 || !strings.Contains(err.Error(), "another removal") {
			t.Errorf("with the removal lock held the command answered %v with the lines %q (close: %v), want it refused as another removal's", err, lines, closeErr)
		}
		if now, head := registered(t); !slices.Equal(now, first) || head != firstHead {
			t.Fatalf("the listings and the refusals changed the store: it registers\n%s\nat the head %s, want what it registered before them at %s:\n%s", strings.Join(now, "\n"), head, firstHead, strings.Join(first, "\n"))
		}
		// The withdrawal is held to the store its listing read. Another process commits between that
		// reading and the withdrawal's own commit, here at the moment the command prints what it
		// keeps: the withdrawal is refused and nothing is withdrawn. What that process registered has
		// no location, so the listing is unchanged and its token still stands.
		raced := false
		racing := written(func(line []byte) (int, error) {
			if raced {
				return len(line), nil
			}
			raced = true
			other, err := overgodb.Open(storePath)
			if err != nil {
				return 0, err
			}
			meanwhile := testutil.ArtifactID(t, artifact.KindFile, "what another process registered meanwhile")
			_, err = other.Commit(ctx, artifact.Batch{Key: "withdrawal-fixture/meanwhile", Artifacts: []artifact.Descriptor{{ID: meanwhile, Size: 1}}})
			return len(line), errors.Join(err, other.Close())
		})
		if err := withdrawRegistrations(h, s.at(pilotRun), token, racing); !raced || !errors.Is(err, artifact.ErrCommitPrecondition) {
			t.Errorf("with the store changed by another process since the listing was read (it was: %v), the withdrawal answered %v, want it refused as a store that has moved", raced, err)
		}
		if now, head := registered(t); !slices.Equal(now, first) || head == firstHead {
			t.Fatalf("after the refused withdrawal the store registers\n%s\nat the head %s, want what it registered before at a head the other process moved from %s", strings.Join(now, "\n"), head, firstHead)
		}
		if _, err := os.Stat(receipt); !errors.Is(err, fs.ErrNotExist) {
			t.Errorf("the listings and the refusals left a receipt (%v), and they withdraw nothing", err)
		}
		unmoved(t)
	})

	t.Run("the token withdraws the locations of the run's checkpoint, and no other", func(t *testing.T) {
		lines, _, err := withdraw(pilotRun, token)
		want := said(map[string][]string{"withdrawn": pilot.files(), "kept": kept})
		if err != nil || !slices.Equal(s.verbs(lines), want) {
			t.Fatalf("the withdrawal said (%v)\n%s\nwant\n%s", err, strings.Join(s.verbs(lines), "\n"), strings.Join(want, "\n"))
		}
		var receipted strings.Builder
		for _, fields := range lines {
			if fields[0] == "withdrawn" {
				receipted.WriteString(strings.Join(fields, "\t") + "\n")
			}
		}
		if written, err := os.ReadFile(receipt); err != nil || string(written) != receipted.String() {
			t.Fatalf("the receipt holds (%v)\n%s\nwant the lines of the locations withdrawn, in order:\n%s", err, written, receipted.String())
		}
		// Every other location the store registered is still registered: those of the other runs'
		// checkpoints, whose weights are the same bytes as the pilot's, among them.
		left := slices.Clone(first)
		for _, path := range pilot.files() {
			at := slices.Index(left, path)
			if at < 0 {
				t.Fatalf("the fixture's store never registered %s", path)
			}
			left = slices.Delete(left, at, at+1)
		}
		if now, _ := registered(t); !slices.Equal(now, left) {
			t.Fatalf("after the withdrawal the store registers\n%s\nwant all it registered but the checkpoint's own:\n%s", strings.Join(now, "\n"), strings.Join(left, "\n"))
		}
		prototype := func(content string) artifact.ID { return testutil.ArtifactID(t, artifact.KindTensorSet, content) }
		for _, held := range []struct {
			what string
			id   artifact.ID
			want []string
		}{
			{"the checkpoint", pilot.checkpoint, nil}, {"the checkpoint's model", pilot.model, nil},
			{"the model's own shard, which a prototype's file elsewhere in the run also is", pilot.own, []string{pilotRun + "/proto/copy.gguf"}},
			{"the prototype whose file is that copy", copied, []string{pilotRun + "/proto"}},
			{"the shard a served model also holds", pilot.shared, []string{"data/models/served/meta.gguf"}},
			{"the prototype in a folder of its own", apart, []string{pilotRun + "/proto"}},
			{"that prototype's file", prototype("a prototype the owner made by hand"), []string{pilotRun + "/proto/model.gguf"}},
			{"the prototype beside the checkpoint's files", beside, []string{pilot.slash}},
			{"that prototype's file", prototype("a second prototype, beside a checkpoint's own files"), []string{pilot.slash + "/handmade.gguf"}},
			{"the prototype made from the checkpoint", derived, []string{pilot.slash}},
			{"that prototype's file", prototype("a third prototype, made by hand from the checkpoint"), []string{pilot.slash + "/derived.gguf"}},
			{"the notes, which are no model", notes, []string{pilot.slash + "/notes.txt"}},
		} {
			if now := located(t, held.id); !slices.Equal(now, held.want) {
				t.Errorf("after the withdrawal %s is located at %q, want %q", held.what, now, held.want)
			}
		}
		// Only locations are withdrawn: the checkpoint, its model and the edge between them stay.
		read(t, func(store *overgodb.Store) {
			_, known, err := store.Artifact(ctx, pilot.checkpoint)
			_, modelled, manifestErr := store.Manifest(ctx, pilot.model)
			produced, lineageErr := store.Children(ctx, pilot.checkpoint)
			if err = errors.Join(err, manifestErr, lineageErr); err != nil || !known || !modelled ||
				!slices.Contains(produced, artifact.Lineage{Child: pilot.model, Parent: pilot.checkpoint, Relation: artifact.RelationProducedBy}) {
				t.Errorf("after the withdrawal the store holds the checkpoint: %v, its model: %v, and the edges %v (%v), want both identities and the edge from the model to the checkpoint", known, modelled, produced, err)
			}
		})
		unmoved(t)
		// The prototypes still hold the run's directory against the deleting command.
		listed, _, err := removalRun(h, "experiment", "")
		if err != nil {
			t.Fatal(err)
		}
		s.ruled(listed, pilotRun, "kept", "the store registers")
		// The token is of a listing the run no longer gives; the next one has nothing to withdraw,
		// and its token withdraws nothing.
		if lines, _, err := withdraw(pilotRun, token); err == nil || len(lines) != 0 {
			t.Fatalf("the token was taken again (%v, lines %q) after the withdrawal changed what the store registers", err, lines)
		}
		after, afterHead := registered(t)
		lines, shown, err := withdraw(pilotRun, "")
		if want := said(map[string][]string{"kept": kept}); err != nil || shown == token || !slices.Equal(s.verbs(lines), want) {
			t.Fatalf("the listing after the withdrawal is (%v, token %q)\n%s\nwant only the locations that are no checkpoint's, kept, under a token of its own", err, shown, strings.Join(s.verbs(lines), "\n"))
		}
		if _, _, err := withdraw(pilotRun, shown); err != nil {
			t.Fatalf("the token of a listing with nothing to withdraw: %v", err)
		}
		if now, head := registered(t); !slices.Equal(now, after) || head != afterHead {
			t.Fatalf("the token of a listing with nothing to withdraw changed the store: it registers\n%s", strings.Join(now, "\n"))
		}
		if written, err := os.ReadFile(receipt); err != nil || string(written) != receipted.String() {
			t.Fatalf("the receipt holds (%v)\n%s\nafter runs that withdrew nothing, want it as the withdrawal left it:\n%s", err, written, receipted.String())
		}
	})

	t.Run("a run with no prototype reads finished once its registrations are withdrawn, and is then deleted", func(t *testing.T) {
		listed, _, err := removalRun(h, "experiment", "")
		if err != nil {
			t.Fatal(err)
		}
		s.ruled(listed, cleanRun, "kept", "the store registers")
		_, shown, err := withdraw(cleanRun, "")
		if err != nil {
			t.Fatal(err)
		}
		lines, _, err := withdraw(cleanRun, shown)
		if want := said(map[string][]string{"withdrawn": slices.Concat(clean[0].files(), clean[1].files(), split)}); err != nil || !slices.Equal(s.verbs(lines), want) {
			t.Fatalf("the withdrawal of a run of two arms, and of a checkpoint whose weights are a manifest, said (%v)\n%s\nwant\n%s", err, strings.Join(s.verbs(lines), "\n"), strings.Join(want, "\n"))
		}
		unmoved(t)
		listed, shown, err = removalRun(h, "experiment", "")
		if err != nil {
			t.Fatal(err)
		}
		s.ruled(listed, cleanRun, "would-remove", "experiment a committed record names")
		s.ruled(listed, pilotRun, "kept", "the store registers")
		s.ruled(listed, strandRun, "kept", "the store registers")
		// The deleting command's own sentences are as they were: the two commands share the code that
		// prints a listing and holds a token to it, each under its own verb.
		var printed bytes.Buffer
		if err := deleteLeftovers(h, "experiment", "", &printed); err != nil || !strings.HasSuffix(printed.String(), "\nloop: nothing was removed; to remove the 2 entries marked would-remove, run this again with -confirm "+shown+"\n") {
			t.Errorf("the deleting command's listing ends (%v)\n%s\nwant its last line to say nothing was removed and to give the token of its 2 entries", err, printed.String())
		}
		if err := deleteLeftovers(h, "experiment", "0123456789abcdef", &printed); err == nil || !strings.Contains(err.Error(), "is not the token of what the place experiment holds now") {
			t.Errorf("the deleting command answered a token that is not its listing's with %v, want it refused as not the token of what the place holds", err)
		}
		listed, _, err = removalRun(h, "experiment", shown)
		if err != nil {
			t.Fatal(err)
		}
		s.ruled(listed, cleanRun, "removed", "experiment a committed record names")
		if _, err := os.Stat(s.at(cleanRun)); !errors.Is(err, fs.ErrNotExist) {
			t.Errorf("the run whose registrations were withdrawn is still there after its removal (%v)", err)
		}
		// The run whose checkpoint document is torn went with it: no registration ever held it.
		unmoved(t, cleanRun, tornRun)
		// The files are gone and the identities stay: the store still knows both checkpoints.
		read(t, func(store *overgodb.Store) {
			for _, arm := range clean {
				if _, known, err := store.Artifact(ctx, arm.checkpoint); err != nil || !known {
					t.Errorf("after its run was deleted the store no longer holds the checkpoint of %s (%v)", arm.slash, err)
				}
			}
		})
	})

	t.Run("a directory goes with the listing flag, alone or with a token, and never with a place", func(t *testing.T) {
		for _, flags := range []struct {
			leftovers               bool
			place, directory, token string
			refused                 bool
		}{
			{true, "", "D:/data/experiments/pilot/run", "", false}, {true, "", "D:/data/experiments/pilot/run", "0123456789abcdef", false},
			{false, "", "D:/data/experiments/pilot/run", "", true}, {false, "", "D:/data/experiments/pilot/run", "0123456789abcdef", true},
			{true, "experiment", "D:/data/experiments/pilot/run", "", true}, {true, "experiment", "D:/data/experiments/pilot/run", "0123456789abcdef", true},
		} {
			if err := removalFlags(flags.leftovers, flags.place, flags.directory, flags.token); (err != nil) != flags.refused {
				t.Errorf("the flags %+v are answered %v, want refused: %v", flags, err, flags.refused)
			}
		}
	})
}
