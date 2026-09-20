package gate

import (
	"os"
	"path/filepath"
	"strconv"

	"overgo/internal/processcontrol"
)

// A gate removes its temporaries in deferred calls, which a killed gate never
// reaches: candidate worktrees of the whole tree stayed registered and on
// disk. Every temporary therefore lives under one root named for the process
// that owns it, so the next gate can tell whose are orphaned.

// gateTemporaryRoot is where the gate process pid keeps its temporaries.
func gateTemporaryRoot(pid int) string {
	return filepath.Join(os.TempDir(), "overgo-gate", strconv.Itoa(pid))
}

// gateTemporary creates a directory under this process's root.
func gateTemporary(pattern string) (string, error) {
	root := gateTemporaryRoot(os.Getpid())
	if err := os.MkdirAll(root, gatePrivateDirectoryMode); err != nil {
		return "", err
	}
	return os.MkdirTemp(root, pattern)
}

// sweepOrphanedTemporaries removes the roots of gate processes that are gone
// and prunes the worktree registrations their candidates held. A root whose
// name is no process id is not the gate's and is left alone.
func sweepOrphanedTemporaries(repo string) error {
	parent := filepath.Dir(gateTemporaryRoot(os.Getpid()))
	entries, _ := os.ReadDir(parent)
	for _, entry := range entries {
		pid, err := strconv.Atoi(entry.Name())
		if err != nil || processcontrol.ProcessAlive(pid) {
			continue
		}
		if err := os.RemoveAll(filepath.Join(parent, entry.Name())); err != nil {
			return err
		}
	}
	_, err := gitWriterCommand(repo, "worktree", "prune")
	return err
}
