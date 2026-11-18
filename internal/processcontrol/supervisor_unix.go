//go:build !windows

package processcontrol

import (
	"context"
	"errors"
	"os/exec"
	"syscall"
)

func claimResource(string) error {
	return errors.New("processcontrol: physical resource containment requires Windows job objects")
}

func shareResource(string) (func() error, error) {
	return nil, errors.New("processcontrol: shared physical resource admission requires Windows")
}

func holdResource(string) (func() error, error) {
	return nil, errors.New("processcontrol: physical resource holds require Windows")
}

func resourceTransaction(string, func() error) error {
	return errors.New("processcontrol: physical resource transactions require Windows")
}

func resourceTransactionHeld(string) (bool, error) {
	return false, errors.New("processcontrol: physical resource transactions require Windows")
}

// Unix containment uses process groups: the command leads its own
// group, an interrupt delivers SIGTERM to the whole group, and
// termination delivers SIGKILL to the group.
type processTree struct {
	pgid  int
	bound jobBound
}

// detach starts command in a session of its own, which its starter's terminal and process group do not reach.
func detach(command *exec.Cmd) error {
	command.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	return command.Start()
}

func configureSysProc(command *exec.Cmd) {
	command.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

func newProcessTree(command *exec.Cmd, bound jobBound) (processTree, error) {
	pgid, err := syscall.Getpgid(command.Process.Pid)
	if err != nil {
		return processTree{}, err
	}
	return processTree{pgid: pgid, bound: bound}, nil
}

func (t processTree) terminate() error {
	if t.pgid == 0 {
		return errors.New("processcontrol: no process group")
	}
	return syscall.Kill(-t.pgid, syscall.SIGKILL)
}

func (t processTree) close() error { return nil }

// peakProcessBytes is zero: the job counter that holds a tree's peak is Windows'.
func (t processTree) peakProcessBytes() uint64 { return 0 }

// peakJobBytes is zero: the job counter that holds a tree's commit is Windows'.
func (t processTree) peakJobBytes() uint64 { return 0 }

// ProcessAlive reports whether the process still runs: signal 0 performs
// the existence and permission checks without delivering anything.
func ProcessAlive(pid int) bool {
	return pid > 0 && syscall.Kill(pid, 0) == nil
}

// ProcessStart reads no start where the system's process times are Windows'.
func ProcessStart(int) (int64, bool) { return 0, false }

func (t processTree) wait() (bool, error) { return false, nil }

// releaseWaiter would observe a resource's holders; contention itself is Windows-only.
type releaseWaiter struct{}

func openReleaseWaiter(string) (*releaseWaiter, error) { return &releaseWaiter{}, nil }

func (*releaseWaiter) arm() error { return nil }

func (*releaseWaiter) wait(context.Context) error {
	return errors.New("processcontrol: waiting on a physical resource holder requires Windows")
}

func (*releaseWaiter) close() error { return nil }
