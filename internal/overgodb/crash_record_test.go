//go:build crashpoints

package overgodb

import (
	"errors"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"

	"overgo/internal/fsatomic"
)

// The crash recorder sees every durable operation the store makes: TestEveryDurableOperationIsACrashPoint builds the
// package with its production files rewritten so that each durable call reaches it (crashOverlay), and only under the
// crashpoints tag, so an ordinary build has neither the recorder nor the rewritten calls. A call on a path no attached
// recorder's root holds passes straight through.

var errCrashInjected = errors.New("injected durable-operation failure")

type crashOpKind string

const (
	crashMkdir    crashOpKind = "mkdir"
	crashCreate   crashOpKind = "create"
	crashWriteOp  crashOpKind = "write"
	crashTrunc    crashOpKind = "truncate"
	crashFsync    crashOpKind = "fsync"
	crashDirSync  crashOpKind = "dirsync"
	crashRenameOp crashOpKind = "rename"
	crashReplace  crashOpKind = "replace"
	crashRemoveOp crashOpKind = "remove"
)

// crashOp is one durable operation as it returned: what it did and to what, the workload's stage, and how many commits
// were acknowledged before it.
type crashOp struct {
	kind     crashOpKind
	path, to string
	offset   int64
	data     []byte
	size     int64
	stage    string
	acked    int
}

// crashRecorder records the operations under root, and refuses the one at failAt, and with sticky every one after it,
// or else the next of target's kind on target's path.
type crashRecorder struct {
	mu       sync.Mutex
	root     string
	ops      []crashOp
	stage    string
	acked    int
	failAt   int
	sticky   bool
	target   crashOp
	seen     int
	injected int
}

func newCrashRecorder(root string) *crashRecorder {
	return &crashRecorder{root: filepath.Clean(root), failAt: -1}
}

var (
	crashTraced    atomic.Int64
	crashRecorders sync.Map
)

func crashAttach(recorder *crashRecorder) func() {
	crashRecorders.Store(recorder.root, recorder)
	crashTraced.Add(1)
	return func() {
		crashRecorders.Delete(recorder.root)
		crashTraced.Add(-1)
	}
}

// crashFind names the recorder whose root holds path, and path under it.
func crashFind(path string) (*crashRecorder, string) {
	if crashTraced.Load() == 0 {
		return nil, ""
	}
	var found *crashRecorder
	var relative string
	crashRecorders.Range(func(key, value any) bool {
		rel, err := filepath.Rel(key.(string), path)
		if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			return true
		}
		found, relative = value.(*crashRecorder), filepath.ToSlash(rel)
		return false
	})
	return found, relative
}

func (r *crashRecorder) setStage(stage string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.stage = stage
}

func (r *crashRecorder) ack() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.acked++
}

func (r *crashRecorder) record(op crashOp) {
	op.stage, op.acked = r.stage, r.acked
	r.ops = append(r.ops, op)
}

// refuse says whether this operation, of kind on path, is the one to fail; the caller holds the lock.
func (r *crashRecorder) refuse(kind crashOpKind, path string) bool {
	position := r.seen
	r.seen++
	switch {
	case r.target.kind != "":
		if kind != r.target.kind || path != r.target.path {
			return false
		}
		r.target = crashOp{}
	case r.failAt < 0 || position < r.failAt || position > r.failAt && !r.sticky:
		return false
	}
	r.injected++
	return true
}

func crashRun(op crashOp, do func() error) error {
	recorder, relative := crashFind(op.path)
	if recorder == nil {
		return do()
	}
	op.path = relative
	if op.to != "" {
		_, op.to = crashFind(op.to)
	}
	recorder.mu.Lock()
	defer recorder.mu.Unlock()
	if recorder.refuse(op.kind, op.path) {
		return errCrashInjected
	}
	if err := do(); err != nil {
		return err
	}
	recorder.record(op)
	return nil
}

// crashOSType stands in for the os and fsatomic calls the rewritten files make (crashos).
type crashOSType struct{}

var crashos crashOSType

func (crashOSType) Rename(from, to string) error {
	return crashRun(crashOp{kind: crashRenameOp, path: from, to: to}, func() error { return os.Rename(from, to) })
}

func (crashOSType) Replace(from, to string) error {
	return crashRun(crashOp{kind: crashReplace, path: from, to: to}, func() error { return fsatomic.Replace(from, to) })
}

func (crashOSType) Remove(path string) error {
	return crashRun(crashOp{kind: crashRemoveOp, path: path}, func() error { return os.Remove(path) })
}

func (crashOSType) FSRemove(path string) error {
	return crashRun(crashOp{kind: crashRemoveOp, path: path}, func() error { return fsatomic.Remove(path) })
}

// RemoveAll takes the tree apart one entry at a time, so a crash can fall between any two of the removals os.RemoveAll
// would make.
func (crashOSType) RemoveAll(path string) error {
	if recorder, _ := crashFind(path); recorder == nil {
		return os.RemoveAll(path)
	}
	var entries []string
	_ = filepath.WalkDir(path, func(entry string, _ fs.DirEntry, err error) error {
		if err == nil {
			entries = append(entries, entry)
		}
		return nil
	})
	slices.Reverse(entries)
	for _, entry := range entries {
		if err := crashos.Remove(entry); err != nil {
			return err
		}
	}
	return nil
}

func (crashOSType) MkdirAll(path string, mode fs.FileMode) error {
	return crashRun(crashOp{kind: crashMkdir, path: path}, func() error { return os.MkdirAll(path, mode) })
}

func (crashOSType) Mkdir(path string, mode fs.FileMode) error {
	return crashRun(crashOp{kind: crashMkdir, path: path}, func() error { return os.Mkdir(path, mode) })
}

func (crashOSType) Link(from, to string) error { return os.Link(from, to) }

func (crashOSType) CreateTemp(directory, pattern string) (*os.File, error) {
	recorder, relative := crashFind(directory)
	if recorder == nil {
		return os.CreateTemp(directory, pattern)
	}
	recorder.mu.Lock()
	defer recorder.mu.Unlock()
	if recorder.refuse(crashCreate, relative) {
		return nil, errCrashInjected
	}
	file, err := os.CreateTemp(directory, pattern)
	if err != nil {
		return nil, err
	}
	_, relative = crashFind(file.Name())
	recorder.record(crashOp{kind: crashCreate, path: relative})
	return file, nil
}

func (crashOSType) OpenFile(path string, flag int, mode fs.FileMode) (*os.File, error) {
	_, statErr := os.Stat(path)
	if flag&os.O_CREATE == 0 || !errors.Is(statErr, os.ErrNotExist) {
		return os.OpenFile(path, flag, mode)
	}
	var file *os.File
	err := crashRun(crashOp{kind: crashCreate, path: path}, func() (err error) {
		file, err = os.OpenFile(path, flag, mode)
		return err
	})
	return file, err
}

func (crashOSType) SyncDirectory(path string) error {
	if recorder, _ := crashFind(path); recorder == nil {
		return fsatomic.SyncDirectory(path)
	}
	return crashRun(crashOp{kind: crashDirSync, path: path}, func() error { return nil })
}

func (crashOSType) SyncFile(path string) error {
	if recorder, _ := crashFind(path); recorder == nil {
		return fsatomic.SyncFile(path)
	}
	return crashRun(crashOp{kind: crashFsync, path: path}, func() error { return nil })
}

func crashSync(target interface{ Sync() error }) error {
	file, ok := target.(*os.File)
	if !ok {
		return target.Sync()
	}
	if recorder, _ := crashFind(file.Name()); recorder == nil {
		return file.Sync()
	}
	// A real flush adds nothing to a simulated crash and costs milliseconds.
	return crashRun(crashOp{kind: crashFsync, path: file.Name()}, func() error { return nil })
}

func crashTruncate(file *os.File, size int64) error {
	return crashRun(crashOp{kind: crashTrunc, path: file.Name(), size: size}, func() error { return file.Truncate(size) })
}

func crashWrite(writer io.Writer, data []byte) (int, error) {
	file, ok := writer.(*os.File)
	if !ok {
		return writer.Write(data)
	}
	recorder, relative := crashFind(file.Name())
	if recorder == nil {
		return file.Write(data)
	}
	recorder.mu.Lock()
	defer recorder.mu.Unlock()
	offset, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return 0, err
	}
	refused := recorder.refuse(crashWriteOp, relative)
	if refused {
		// A full volume takes part of a buffer, then refuses the rest.
		data = data[:len(data)/2]
	}
	written, err := file.Write(data)
	if written > 0 {
		recorder.record(crashOp{kind: crashWriteOp, path: relative, offset: offset, data: slices.Clone(data[:written])})
	}
	if refused {
		return written, errCrashInjected
	}
	return written, err
}

// crashTee records what a stream writes through a tee.
type crashTee struct{ destination io.Writer }

func (c crashTee) Write(data []byte) (int, error) { return crashWrite(c.destination, data) }

// crashCopy copies a stream into a file the recorder sees as one write.
func crashCopy(destination io.Writer, source io.Reader) (int64, error) {
	if file, ok := destination.(*os.File); !ok || crashTraced.Load() == 0 || file == nil {
		return io.Copy(destination, source)
	}
	data, err := io.ReadAll(source)
	if err != nil {
		return 0, err
	}
	if err := writeAll(destination, data); err != nil {
		return 0, err
	}
	return int64(len(data)), nil
}
