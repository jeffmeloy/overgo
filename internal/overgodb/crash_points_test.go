//go:build crashpoints

package overgodb

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"runtime/debug"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"unicode"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
)

// The crash harness, run by TestEveryDurableOperationIsACrashPoint and TestEveryDurableOperationCanFail over the
// overlay that routes the store's durable calls through the crash recorder.

// The workload's shape: its commits, the commits that hold a blob or an output no alias reaches, the commit a recovered
// store takes next, the pack's limit in blobs, and the workers that check states at once.
const (
	crashFirstRun       = 6
	crashSecondRun      = 9
	crashNextCommit     = 1000
	crashPackBlobs      = 4
	crashWorkers        = 8
	crashBlobBytes      = 2 * inlineContentLimit
	crashSector         = 512
	crashNoiseDigits    = 5
	crashNoiseHexDigits = 12
	crashCloseEveryTail = 1
)

type crashCommit struct {
	key      string
	batch    artifact.Batch
	id       artifact.CommitID
	sequence uint64
}

// crashLedger is what the workload was told: each acknowledged commit, and the head at each sequence.
type crashLedger struct {
	mu      sync.Mutex
	commits []crashCommit
	heads   map[uint64]artifact.CommitID
}

func newCrashLedger() *crashLedger { return &crashLedger{heads: map[uint64]artifact.CommitID{}} }

func (l *crashLedger) head(id artifact.CommitID, sequence uint64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.heads[sequence] = id
}

func crashContent(t testing.TB, kind artifact.Kind, data []byte) artifact.Content {
	id, err := artifact.IdentifyBytes(kind, data)
	testcheck.NoError(t, err)
	return artifact.Content{Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(data)), MediaType: "application/octet-stream"}, Data: data}
}

// crashBatch is commit index's batch, the same every run. Most hold small inline content under an alias; some hold a
// blob, and output kinds no alias reaches are what the release may take.
func crashBatch(t testing.TB, index int) artifact.Batch {
	alias := fmt.Sprintf("crash/%d", index)
	inline := crashContent(t, artifact.KindModel, []byte(fmt.Sprintf("record %d", index)))
	batch := artifact.Batch{
		Key:      fmt.Sprintf("crash/commit/%d", index),
		Contents: []artifact.Content{inline},
		Aliases:  []artifact.AliasBinding{{Name: alias, Target: inline.Descriptor.ID}},
	}
	switch {
	case slices.Contains([]int{3, 10, 12}, index):
		blob := crashContent(t, artifact.KindTensorSet, bytes.Repeat([]byte{byte(index), 1, 2, 3}, crashBlobBytes/4))
		batch.Contents = append(batch.Contents, blob)
		batch.Aliases = append(batch.Aliases, artifact.AliasBinding{Name: alias + "/blob", Target: blob.Descriptor.ID})
	case slices.Contains([]int{5, 14}, index):
		batch.Contents = append(batch.Contents, crashContent(t, artifact.KindOutput, bytes.Repeat([]byte{5, 9}, crashBlobBytes/2)))
	case index == 7:
		batch.Contents = append(batch.Contents, crashContent(t, artifact.KindOutput, []byte("an inline output nothing reaches")))
	}
	return batch
}

// crashDriver runs the workload against a store under root, as a process would, reopening after a fault.
type crashDriver struct {
	t        testing.TB
	ctx      context.Context
	root     string
	recorder *crashRecorder
	ledger   *crashLedger
	store    *Store
	// swallowed names public operations that returned success although an injected failure fell inside them.
	swallowed []string
}

func (d *crashDriver) guard(stage string, do func() error) error {
	d.recorder.setStage(stage)
	d.recorder.mu.Lock()
	before := d.recorder.injected
	d.recorder.mu.Unlock()
	err := do()
	d.recorder.mu.Lock()
	after := d.recorder.injected
	d.recorder.mu.Unlock()
	if err == nil && after != before {
		d.swallowed = append(d.swallowed, stage)
	}
	return err
}

func (d *crashDriver) open(stage string) {
	if d.store != nil {
		return
	}
	_ = d.guard(stage, func() error {
		store, err := Create(d.root)
		if err == nil {
			d.store = store
		}
		return err
	})
}

func (d *crashDriver) refault(err error) {
	if errors.Is(err, ErrStoreFaulted) && d.store != nil {
		_ = d.store.Close()
		d.store = nil
	}
}

func (d *crashDriver) commit(index int) {
	d.open("reopen-after-failure")
	if d.store == nil {
		return
	}
	batch := crashBatch(d.t, index)
	err := d.guard(fmt.Sprintf("commit-%d", index), func() error {
		id, err := d.store.Commit(d.ctx, batch)
		if err != nil {
			return err
		}
		head, sequence := d.store.Head()
		d.ledger.mu.Lock()
		d.ledger.commits = append(d.ledger.commits, crashCommit{key: batch.Key, batch: batch, id: id, sequence: sequence})
		d.ledger.heads[sequence] = head
		d.ledger.mu.Unlock()
		d.recorder.ack()
		return nil
	})
	d.refault(err)
}

func (d *crashDriver) snapshot(stage string) {
	d.open("reopen-after-failure")
	if d.store == nil {
		return
	}
	err := d.guard(stage, func() error {
		_, err := d.store.Snapshot(d.ctx)
		return err
	})
	d.refault(err)
}

func (d *crashDriver) close(stage string, tail uint64) {
	if d.store == nil {
		return
	}
	d.store.tail = tail
	_ = d.guard(stage, func() error { return d.store.Close() })
	d.store = nil
}

func (d *crashDriver) release(stage string) {
	d.open("reopen-after-failure")
	if d.store == nil {
		return
	}
	err := d.guard(stage, func() error {
		_, err := Release(d.ctx, d.store, nil, RetentionPolicy{}, crashReleasable)
		return err
	})
	if d.store != nil {
		head, sequence := d.store.Head()
		d.ledger.head(head, sequence)
	}
	d.refault(err)
}

func (d *crashDriver) pack(stage string) {
	d.open("reopen-after-failure")
	if d.store == nil {
		return
	}
	err := d.guard(stage, func() error {
		_, err := packBlobsUnder(d.ctx, d.store, crashPackBlobs*crashBlobBytes)
		return err
	})
	d.refault(err)
}

// run is the workload: commits, a seal, a close that takes maintenance, reopens, a release, a pack, and two snapshots.
func (d *crashDriver) run() {
	d.open("create")
	for index := 1; index <= crashFirstRun; index++ {
		d.commit(index)
	}
	d.snapshot("snapshot-seal")
	for index := crashFirstRun + 1; index <= crashSecondRun; index++ {
		d.commit(index)
	}
	d.close("close-with-maintenance", crashCloseEveryTail)
	d.open("reopen-1")
	d.commit(10)
	d.commit(11)
	d.close("close-plain", ^uint64(0))
	d.open("reopen-2")
	d.release("release")
	d.pack("pack")
	d.commit(14)
	d.commit(12)
	d.snapshot("snapshot-2")
	d.snapshot("snapshot-rewrite")
	d.commit(13)
	d.close("close-last", ^uint64(0))
}

// crashFile is a file of the model: its bytes, and those a flush made durable.
type crashFile struct{ data, durable []byte }

// crashTree is the model of the store's tree that the recorded operations build.
type crashTree struct {
	files map[string]*crashFile
	dirs  map[string]bool
}

func newCrashTree() *crashTree {
	return &crashTree{files: map[string]*crashFile{}, dirs: map[string]bool{}}
}

func (t *crashTree) mkdirs(path string) {
	for path != "." && path != "" && !t.dirs[path] {
		t.dirs[path] = true
		path = filepath.ToSlash(filepath.Dir(path))
	}
}

func (t *crashTree) apply(op crashOp, durableTruncate bool) {
	file := t.files[op.path]
	switch op.kind {
	case crashMkdir:
		t.mkdirs(op.path)
	case crashCreate:
		t.files[op.path] = &crashFile{}
	case crashWriteOp:
		if file == nil {
			return
		}
		if end := op.offset + int64(len(op.data)); end > int64(len(file.data)) {
			file.data = append(file.data, make([]byte, end-int64(len(file.data)))...)
		}
		copy(file.data[op.offset:], op.data)
	case crashTrunc:
		if file == nil {
			return
		}
		file.data = file.data[:op.size:op.size]
		if durableTruncate {
			file.durable = file.durable[:min(int64(len(file.durable)), op.size)]
		}
	case crashFsync:
		if file != nil {
			file.durable = slices.Clone(file.data)
		}
	case crashRenameOp, crashReplace:
		if file != nil {
			t.files[op.to] = file
			delete(t.files, op.path)
		}
	case crashRemoveOp:
		delete(t.files, op.path)
		delete(t.dirs, op.path)
	}
}

// crashState is one tree a crash can leave, as bytes.
type crashState struct {
	mode, stage, what string
	position, acked   int
	files             map[string][]byte
	dirs              []string
}

func (s crashState) key() string {
	hash := sha256.New()
	fmt.Fprintln(hash, s.acked)
	for _, path := range slices.Sorted(maps.Keys(s.files)) {
		fmt.Fprintln(hash, path, len(s.files[path]))
		hash.Write(s.files[path])
	}
	for _, path := range slices.Sorted(slices.Values(s.dirs)) {
		fmt.Fprintln(hash, path)
	}
	return string(hash.Sum(nil))
}

func (t *crashTree) volatile() (map[string][]byte, []string) {
	files := make(map[string][]byte, len(t.files))
	for path, file := range t.files {
		files[path] = slices.Clone(file.data)
	}
	return files, slices.Collect(maps.Keys(t.dirs))
}

// flushed is the tree a power loss leaves: each file as of its last flush, and with zeroTail its unflushed extension
// present but unwritten.
func (t *crashTree) flushed(zeroTail bool) (map[string][]byte, []string) {
	files := make(map[string][]byte, len(t.files))
	for path, file := range t.files {
		content := slices.Clone(file.durable)
		if zeroTail && len(file.data) > len(content) {
			content = append(content, make([]byte, len(file.data)-len(content))...)
		}
		files[path] = content
	}
	return files, slices.Collect(maps.Keys(t.dirs))
}

func crashWriteTree(root string, files map[string][]byte, dirs []string) error {
	if err := os.MkdirAll(root, storeDirectoryMode); err != nil {
		return err
	}
	for _, directory := range dirs {
		if err := os.MkdirAll(filepath.Join(root, filepath.FromSlash(directory)), storeDirectoryMode); err != nil {
			return err
		}
	}
	for path, data := range files {
		target := filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(target), storeDirectoryMode); err != nil {
			return err
		}
		if err := os.WriteFile(target, data, storeFileMode); err != nil {
			return err
		}
	}
	return nil
}

func crashReadTree(root string) (map[string][]byte, []string, error) {
	files, dirs := map[string][]byte{}, []string{}
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		relative, err := filepath.Rel(root, path)
		if err != nil || relative == "." {
			return err
		}
		if entry.IsDir() {
			dirs = append(dirs, filepath.ToSlash(relative))
			return nil
		}
		data, err := os.ReadFile(path)
		files[filepath.ToSlash(relative)] = data
		return err
	})
	return files, dirs, err
}

func crashAckAfter(ops []crashOp, position, final int) int {
	if position+1 < len(ops) {
		return ops[position+1].acked
	}
	return final
}

// crashKillStates are what a killed process leaves: every operation that returned is there, the operating system keeps
// them all, and the one under way may have written part of its buffer.
func crashKillStates(ops []crashOp, final int) []crashState {
	var states []crashState
	tree := newCrashTree()
	for position, op := range ops {
		if op.kind == crashWriteOp && len(op.data) > 1 {
			for _, part := range []int{1, len(op.data) / 2, len(op.data) - 1} {
				torn := op
				torn.data = op.data[:part]
				probe := crashCloneTree(tree)
				probe.apply(torn, true)
				files, dirs := probe.volatile()
				states = append(states, crashState{mode: "kill-torn", stage: op.stage, what: fmt.Sprintf("%s %s %d of %d bytes", op.kind, crashShort(op.path), part, len(op.data)), position: position, acked: op.acked, files: files, dirs: dirs})
			}
		}
		tree.apply(op, true)
		files, dirs := tree.volatile()
		states = append(states, crashState{mode: "kill", stage: op.stage, what: fmt.Sprintf("after %s %s", op.kind, crashShort(op.path)), position: position, acked: crashAckAfter(ops, position, final), files: files, dirs: dirs})
	}
	return states
}

func crashCloneTree(t *crashTree) *crashTree {
	clone := newCrashTree()
	maps.Copy(clone.dirs, t.dirs)
	for path, file := range t.files {
		clone.files[path] = &crashFile{data: slices.Clone(file.data), durable: slices.Clone(file.durable)}
	}
	return clone
}

func crashMetadata(kind crashOpKind) bool {
	return slices.Contains([]crashOpKind{crashMkdir, crashCreate, crashTrunc, crashRenameOp, crashReplace, crashRemoveOp}, kind)
}

// crashNTFSStates are what a power loss leaves on a journaled volume: metadata reaches the log in order and any flush
// carries all of it, so a prefix of the metadata since the last flush survives; a file's data survives only as of its own
// flush, and its unflushed extension may be there unwritten, as zeros, or written up to any sector.
func crashNTFSStates(ops []crashOp, final int) []crashState {
	var states []crashState
	flushed := -1
	for position, op := range ops {
		if op.kind == crashFsync || op.kind == crashReplace {
			flushed = position
		}
		limits := []int{flushed}
		for index := flushed + 1; index <= position; index++ {
			if crashMetadata(ops[index].kind) {
				limits = append(limits, index)
			}
		}
		for _, limit := range limits {
			tree := newCrashTree()
			for index, replayed := range ops[:position+1] {
				if crashMetadata(replayed.kind) && index > limit {
					continue
				}
				tree.apply(replayed, true)
			}
			for _, zeroTail := range []bool{false, true} {
				files, dirs := tree.flushed(zeroTail)
				states = append(states, crashState{mode: fmt.Sprintf("ntfs-power zeroTail=%t", zeroTail), stage: op.stage, what: fmt.Sprintf("%s %s, metadata through %d of %d", op.kind, crashShort(op.path), limit, position), position: position, acked: crashAckAfter(ops, position, final), files: files, dirs: dirs})
			}
			for path, file := range tree.files {
				for boundary := (len(file.durable)/crashSector + 1) * crashSector; boundary < len(file.data); boundary += crashSector {
					files, dirs := tree.flushed(true)
					files[path] = append(slices.Clone(file.data[:boundary]), make([]byte, len(file.data)-boundary)...)
					states = append(states, crashState{mode: "ntfs-power sector", stage: op.stage, what: fmt.Sprintf("%s %s through byte %d of %d", op.kind, crashShort(path), boundary, len(file.data)), position: position, acked: crashAckAfter(ops, position, final), files: files, dirs: dirs})
				}
			}
		}
	}
	return states
}

// crashWinRuleStates hold the store to the rule fsatomic writes for Windows: SyncDirectory does nothing, so a rename
// reaches the disk only through Replace or a flush of the file through its new name. Everything else is as generous as
// it can be, so a finding here is a rename and nothing else.
func crashWinRuleStates(ops []crashOp, final int) []crashState {
	var states []crashState
	for position, op := range ops {
		live := newCrashTree()
		tree := newCrashTree()
		flushedLater := func(index int, path string) bool {
			return slices.ContainsFunc(ops[index+1:position+1], func(later crashOp) bool { return later.kind == crashFsync && later.path == path })
		}
		for index, replayed := range ops[:position+1] {
			switch replayed.kind {
			case crashMkdir:
				tree.mkdirs(replayed.path)
				live.mkdirs(replayed.path)
			case crashCreate:
				live.apply(replayed, false)
				tree.files[replayed.path] = live.files[replayed.path]
			case crashRenameOp:
				file := live.files[replayed.path]
				live.apply(replayed, false)
				if flushedLater(index, replayed.to) {
					tree.files[replayed.to] = file
					delete(tree.files, replayed.path)
				}
			case crashReplace:
				file := live.files[replayed.path]
				live.apply(replayed, false)
				tree.files[replayed.to] = file
				delete(tree.files, replayed.path)
			case crashRemoveOp:
				live.apply(replayed, false)
				delete(tree.files, replayed.path)
				delete(tree.dirs, replayed.path)
			default:
				live.apply(replayed, false)
			}
		}
		files, dirs := tree.flushed(false)
		states = append(states, crashState{mode: "winrule", stage: op.stage, what: fmt.Sprintf("after %s %s", op.kind, crashShort(op.path)), position: position, acked: crashAckAfter(ops, position, final), files: files, dirs: dirs})
	}
	return states
}

// crashShort is a path or a message with the words that differ from run to run, temporary directories, digests and
// counters, each written #, so findings group by where they fell. A word runs between separators: anything but a
// letter, a digit, '.', '-' or '_'.
func crashShort(text string) string {
	var short, word strings.Builder
	digits, hex := 0, 0
	flush := func() {
		if digits >= crashNoiseDigits || hex >= crashNoiseHexDigits {
			short.WriteByte('#')
		} else {
			short.WriteString(word.String())
		}
		word.Reset()
		digits, hex = 0, 0
	}
	for _, r := range filepath.ToSlash(text) {
		if !unicode.IsLetter(r) && !unicode.IsDigit(r) && !strings.ContainsRune(".-_", r) {
			flush()
			short.WriteRune(r)
			continue
		}
		word.WriteRune(r)
		if unicode.IsDigit(r) {
			digits++
		}
		if unicode.Is(unicode.ASCII_Hex_Digit, r) {
			hex++
		}
	}
	flush()
	return short.String()
}

// crashFinding is a promise a state broke.
type crashFinding struct {
	invariant, detail string
	state             crashState
}

func crashHash(files map[string][]byte) string {
	hash := sha256.New()
	for _, path := range slices.Sorted(maps.Keys(files)) {
		fmt.Fprintln(hash, path, len(files[path]))
		hash.Write(files[path])
	}
	return string(hash.Sum(nil))
}

// crashCheck reopens one state and holds it to what the store promises: a read-only open neither refuses nor writes, the
// journal alone replays to the writable open's head and state, history is the workload's, no acknowledged commit,
// content or alias is lost, live blobs verify, an interrupted release finishes, the store goes on, and a reader opened
// before the recovery follows it.
func crashCheck(t testing.TB, ctx context.Context, directory string, state crashState, ledger *crashLedger) (findings []crashFinding) {
	report := func(invariant, format string, arguments ...any) {
		findings = append(findings, crashFinding{invariant: invariant, detail: fmt.Sprintf(format, arguments...), state: state})
	}
	defer func() {
		if failure := recover(); failure != nil {
			report("panic", "%v\n%s", failure, debug.Stack())
		}
	}()
	primary := filepath.Join(directory, "store")
	replay := filepath.Join(directory, "replay")
	defer os.RemoveAll(directory)
	testcheck.NoError(t, crashWriteTree(primary, state.files, state.dirs))
	withoutCheckpoints := map[string][]byte{}
	for path, data := range state.files {
		if !strings.HasPrefix(path, checkpointDirectory+"/") {
			withoutCheckpoints[path] = data
		}
	}
	replayDirs := slices.DeleteFunc(slices.Clone(state.dirs), func(path string) bool {
		return path == checkpointDirectory || strings.HasPrefix(path, checkpointDirectory+"/")
	})
	testcheck.NoError(t, crashWriteTree(replay, withoutCheckpoints, replayDirs))
	log, hasLog := state.files[storeFilename]
	// A store that acknowledged nothing and has no whole header never began: a writer starts it, and a reader may refuse.
	if state.acked == 0 && hasLog && (len(log) < storeHeaderBytes || len(bytes.TrimRight(log, "\x00")) == 0) {
		hasLog = false
	}

	// The reader stays open through the writable open's recovery, which it must follow.
	var reader *Store
	var readHead artifact.CommitID
	var readSequence uint64
	if hasLog {
		opened, err := OpenReadOnly(primary)
		if err != nil {
			report("read-only open refuses", "%v", err)
		} else {
			reader = opened
			defer reader.Close()
			readHead, readSequence = reader.Head()
		}
		after, _, err := crashReadTree(primary)
		testcheck.NoError(t, err)
		if crashHash(after) != crashHash(state.files) {
			report("read-only open wrote", "tree changed")
		}
	}

	var replayed *Store
	if hasLog {
		full, err := OpenReadOnly(replay)
		if err != nil {
			report("full replay refuses", "%v", err)
		} else {
			replayed = full
			defer full.Close()
		}
	}

	store, err := Create(primary)
	if err != nil {
		report("writable open refuses", "%v", err)
		return findings
	}
	closed := false
	defer func() {
		if !closed {
			_ = store.Close()
		}
	}()
	head, sequence := store.Head()
	if reader != nil && (head != readHead || sequence != readSequence) {
		report("read-only and writable opens disagree", "read-only %d, writable %d", readSequence, sequence)
	}
	if replayed != nil {
		replayHead, replaySequence := replayed.Head()
		if head != replayHead || sequence != replaySequence {
			report("open stops short of the journal", "open at %d, journal at %d", sequence, replaySequence)
		}
		ours, theirs := projections(&store.state), projections(&replayed.state)
		for index := range ours {
			mine, errMine := ours[index].view.checkpoint()
			yours, errYours := theirs[index].view.checkpoint()
			if errMine != nil || errYours != nil || !bytes.Equal(mine, yours) {
				report("state differs from a full replay", "projection %s", ours[index].name)
			}
		}
	}
	if sequence != 0 {
		if truth, known := ledger.heads[sequence]; known && truth != head {
			report("history forked", "sequence %d is %s, the workload's was %s", sequence, head, truth)
		}
	}
	if got := len(store.state.commits.ordered); uint64(got) != sequence {
		report("commit facet disagrees with the sequence", "%d commits at sequence %d", got, sequence)
	}

	for _, committed := range ledger.commits[:min(state.acked, len(ledger.commits))] {
		held, found := store.state.commit(committed.key)
		if !found || held.id != committed.id {
			report("acknowledged commit lost", "%s at sequence %d; the store is at %d", committed.key, committed.sequence, sequence)
			continue
		}
		for _, content := range committed.batch.Contents {
			locator, recorded := store.state.contents.locator(content.Descriptor.ID)
			if !recorded {
				report("acknowledged content lost", "%s %s", committed.key, content.Descriptor.ID)
				continue
			}
			if locator.released != 0 {
				if content.Descriptor.ID.Kind() != artifact.KindOutput {
					report("reachable content released", "%s", content.Descriptor.ID)
				}
				continue
			}
			_, source, present, err := store.OpenContent(ctx, content.Descriptor.ID)
			if err != nil || !present {
				report("acknowledged content unreadable", "%s: %v", committed.key, err)
				continue
			}
			read, err := io.ReadAll(source)
			if closer, ok := source.(io.Closer); ok {
				_ = closer.Close()
			}
			if err != nil || !bytes.Equal(read, content.Data) {
				report("acknowledged content differs", "%s: %v", committed.key, err)
			}
		}
		for _, alias := range committed.batch.Aliases {
			if target, bound, err := store.ResolveAlias(ctx, alias.Name); err != nil || !bound || target != alias.Target {
				report("acknowledged alias lost", "%s: %v", alias.Name, err)
			}
		}
	}
	if _, err := store.VerifyBlobs(ctx); err != nil {
		report("a live blob does not verify", "%v", err)
	}
	if _, err := Release(ctx, store, nil, RetentionPolicy{}, crashReleasable); err != nil {
		report("a release after recovery refuses", "%v", err)
	}
	if _, err := store.VerifyBlobs(ctx); err != nil {
		report("a live blob does not verify after a release", "%v", err)
	}
	head, sequence = store.Head()

	// The store goes on: a repeat of an acknowledged batch replays, the next commit takes the next number, and what
	// follows replays whole.
	if state.acked != 0 && state.acked <= len(ledger.commits) {
		last := ledger.commits[state.acked-1]
		if id, err := store.Commit(ctx, last.batch); err != nil || id != last.id {
			report("a repeat of an acknowledged batch did not replay", "%v", err)
		}
	}
	if _, err := store.Commit(ctx, crashBatch(t, crashNextCommit)); err != nil {
		report("the next commit refuses", "%v", err)
		return findings
	}
	nextHead, nextSequence := store.Head()
	if nextSequence != sequence+1 {
		report("the next commit took the wrong number", "%d after %d", nextSequence, sequence)
	}
	if reader != nil {
		if err := reader.Refresh(ctx); err != nil {
			report("a reader from before the recovery refuses to refresh", "%v", err)
		} else if readerHead, readerSequence := reader.Head(); readerHead != nextHead || readerSequence != nextSequence {
			report("a reader from before the recovery stops short", "%d, want %d", readerSequence, nextSequence)
		}
	}
	if _, err := store.Snapshot(ctx); err != nil {
		report("the snapshot after recovery refuses", "%v", err)
	}
	closed = true
	if err := store.Close(); err != nil {
		report("the close after recovery refuses", "%v", err)
	}
	if replayed != nil {
		_ = replayed.Close()
	}
	testcheck.NoError(t, os.RemoveAll(filepath.Join(primary, checkpointDirectory)))
	final, err := OpenReadOnly(primary)
	if err != nil {
		report("a full replay after recovery refuses", "%v", err)
		return findings
	}
	defer final.Close()
	if finalHead, finalSequence := final.Head(); finalHead != nextHead || finalSequence != nextSequence {
		report("a full replay after recovery differs", "%d, want %d", finalSequence, nextSequence)
	}
	return findings
}

func crashReleasable(descriptor artifact.Descriptor) bool {
	return descriptor.ID.Kind() == artifact.KindOutput
}

func crashRecordWorkload(t testing.TB, ctx context.Context, root string) ([]crashOp, *crashLedger, int) {
	recorder := newCrashRecorder(root)
	detach := crashAttach(recorder)
	defer detach()
	ledger := newCrashLedger()
	driver := &crashDriver{t: t, ctx: ctx, root: root, recorder: recorder, ledger: ledger}
	driver.run()
	return recorder.ops, ledger, recorder.acked
}

// crashSummary groups findings by model, promise and where they fell, each with the first one's place.
func crashSummary(findings []crashFinding) []string {
	type group struct {
		count int
		first crashFinding
	}
	groups := map[string]*group{}
	for _, finding := range findings {
		key := fmt.Sprintf("%s | %s | %s", finding.state.mode, finding.invariant, crashShort(strings.SplitN(finding.detail, "\n", 2)[0]))
		if groups[key] == nil {
			groups[key] = &group{first: finding}
		}
		groups[key].count++
	}
	var lines []string
	for _, key := range slices.Sorted(maps.Keys(groups)) {
		g := groups[key]
		lines = append(lines, fmt.Sprintf("%5d  %s  [first: position %d, stage %s, %s, acked %d]", g.count, key, g.first.state.position, g.first.state.stage, g.first.state.what, g.first.state.acked))
	}
	return lines
}

func crashTemp(t *testing.T) string {
	root, err := filepath.EvalSymlinks(t.TempDir())
	testcheck.NoError(t, err)
	return root
}

// crashCheckAll checks states on crashWorkers at once, each under a directory of its own beneath base.
func crashCheckAll(t *testing.T, ctx context.Context, base string, states []crashState, ledger *crashLedger) []crashFinding {
	work := make(chan int)
	var mu sync.Mutex
	var findings []crashFinding
	var workers sync.WaitGroup
	for range crashWorkers {
		workers.Go(func() {
			for index := range work {
				found := crashCheck(t, ctx, filepath.Join(base, strconv.Itoa(index)), states[index], ledger)
				mu.Lock()
				findings = append(findings, found...)
				mu.Unlock()
			}
		})
	}
	for index := range states {
		work <- index
	}
	close(work)
	workers.Wait()
	return findings
}

// TestCrashStatesKeepTheStoresPromises records the workload's durable operations, holds the recorder to rebuilding the
// tree the workload left, and checks every distinct state a kill, a power loss on a journaled volume and the Windows
// rename rule leave.
func TestCrashStatesKeepTheStoresPromises(t *testing.T) {
	// Serial: its workers share the process's recorder registry, and it runs alone under the crashpoints tag.
	ctx := t.Context()
	base := crashTemp(t)
	root := filepath.Join(base, "workload", "store")
	ops, ledger, acked := crashRecordWorkload(t, ctx, root)

	model := newCrashTree()
	counts := map[crashOpKind]int{}
	for _, op := range ops {
		model.apply(op, true)
		counts[op.kind]++
	}
	wanted, _ := model.volatile()
	real, _, err := crashReadTree(root)
	testcheck.NoError(t, err)
	if crashHash(wanted) != crashHash(real) {
		var differ []string
		for path := range real {
			if !bytes.Equal(real[path], wanted[path]) {
				differ = append(differ, path)
			}
		}
		for path := range wanted {
			if _, found := real[path]; !found {
				differ = append(differ, "absent "+path)
			}
		}
		t.Fatalf("the recorded operations do not rebuild the tree, so a durable call escaped the recorder: %v", differ)
	}
	t.Logf("workload: %d durable operations %v, %d commits acknowledged", len(ops), counts, acked)
	var all []crashFinding
	for _, crashModel := range []struct {
		name   string
		states []crashState
	}{
		{"kill", crashKillStates(ops, acked)},
		{"ntfs", crashNTFSStates(ops, acked)},
		{"winrule", crashWinRuleStates(ops, acked)},
	} {
		seen := map[string]bool{}
		unique := slices.DeleteFunc(crashModel.states, func(state crashState) bool {
			key := state.key()
			if seen[key] {
				return true
			}
			seen[key] = true
			return false
		})
		findings := crashCheckAll(t, ctx, filepath.Join(base, crashModel.name), unique, ledger)
		t.Logf("model %s: %d distinct crash states, %d findings", crashModel.name, len(unique), len(findings))
		for _, line := range crashSummary(findings) {
			t.Log(line)
		}
		all = append(all, findings...)
	}
	if len(all) != 0 {
		t.Errorf("%d crash states broke a promise", len(all))
	}
}

// TestRefusedOperationsKeepTheStoresPromises refuses each durable operation of the workload in turn, once and from there
// on, lets the workload go on, and checks the store it leaves.
func TestRefusedOperationsKeepTheStoresPromises(t *testing.T) {
	// Serial: its workers share the process's recorder registry, and it runs alone under the crashpoints tag.
	ctx := t.Context()
	base := crashTemp(t)
	countRoot := filepath.Join(base, "count", "store")
	recorder := newCrashRecorder(countRoot)
	detach := crashAttach(recorder)
	driver := &crashDriver{t: t, ctx: ctx, root: countRoot, recorder: recorder, ledger: newCrashLedger()}
	driver.run()
	detach()
	total := recorder.seen
	t.Logf("fallible operations: %d", total)
	var all []crashFinding
	for _, sticky := range []bool{false, true} {
		mode := map[bool]string{false: "fail-once", true: "fail-from-here-on"}[sticky]
		work := make(chan int)
		var mu sync.Mutex
		var findings []crashFinding
		swallowed := map[string]int{}
		var workers sync.WaitGroup
		for range crashWorkers {
			workers.Go(func() {
				for index := range work {
					found, quiet := crashFailRun(t, ctx, filepath.Join(base, mode, strconv.Itoa(index)), mode, index, sticky)
					mu.Lock()
					findings = append(findings, found...)
					for _, stage := range quiet {
						swallowed[stage]++
					}
					mu.Unlock()
				}
			})
		}
		for index := range total {
			work <- index
		}
		close(work)
		workers.Wait()
		t.Logf("mode %s: %d runs, %d findings", mode, total, len(findings))
		for _, line := range crashSummary(findings) {
			t.Log(line)
		}
		for _, stage := range slices.Sorted(maps.Keys(swallowed)) {
			t.Logf("  returned success through an injected failure: %s x%d", stage, swallowed[stage])
		}
		all = append(all, findings...)
	}
	if len(all) != 0 {
		t.Errorf("%d refused operations broke a promise", len(all))
	}
}

func crashFailRun(t testing.TB, ctx context.Context, directory, mode string, failAt int, sticky bool) (findings []crashFinding, swallowed []string) {
	root := filepath.Join(directory, "live", "store")
	testcheck.NoError(t, os.MkdirAll(filepath.Dir(root), storeDirectoryMode))
	recorder := newCrashRecorder(root)
	recorder.failAt, recorder.sticky = failAt, sticky
	ledger := newCrashLedger()
	driver := &crashDriver{t: t, ctx: ctx, root: root, recorder: recorder, ledger: ledger}
	detach := crashAttach(recorder)
	panicked := func() (failure any) {
		defer func() { failure = recover() }()
		driver.run()
		return nil
	}()
	detach()
	if driver.store != nil {
		_ = driver.store.Close()
	}
	state := crashState{mode: mode, what: fmt.Sprintf("operation %d refused", failAt), position: failAt, acked: len(ledger.commits)}
	if panicked != nil {
		findings = append(findings, crashFinding{invariant: "panic in the workload", detail: fmt.Sprint(panicked), state: state})
	}
	// A store whose directory was never made is an empty tree: refused from its first operation, it began nothing.
	files, dirs, err := crashReadTree(root)
	if errors.Is(err, fs.ErrNotExist) {
		files, dirs, err = map[string][]byte{}, nil, nil
	}
	if err != nil {
		return append(findings, crashFinding{invariant: "no tree", detail: err.Error(), state: state}), driver.swallowed
	}
	state.files, state.dirs = files, dirs
	if len(ledger.commits) != 0 && len(recorder.ops) != 0 {
		state.stage = recorder.ops[min(failAt, len(recorder.ops)-1)].stage
	}
	found := crashCheck(t, ctx, filepath.Join(directory, "check"), state, ledger)
	_ = os.RemoveAll(directory)
	return append(findings, found...), driver.swallowed
}
