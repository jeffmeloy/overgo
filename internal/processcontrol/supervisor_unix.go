//go:build !windows

package processcontrol

import (
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

func resourceTransaction(string, func() error) error {
	return errors.New("processcontrol: physical resource transactions require Windows")
}

// Unix containment uses process groups: the command leads its own
// group, an interrupt delivers SIGTERM to the whole group, and
// termination delivers SIGKILL to the group.
type processTree struct {
	pgid int
}

func configureSysProc(command *exec.Cmd) {
	command.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

func newProcessTree(command *exec.Cmd) (processTree, error) {
	pgid, err := syscall.Getpgid(command.Process.Pid)
	if err != nil {
		return processTree{}, err
	}
	return processTree{pgid: pgid}, nil
}

func (t processTree) interrupt(*exec.Cmd) error {
	if t.pgid == 0 {
		return errors.New("processcontrol: no process group")
	}
	return syscall.Kill(-t.pgid, syscall.SIGTERM)
}

func (t processTree) terminate() error {
	if t.pgid == 0 {
		return errors.New("processcontrol: no process group")
	}
	return syscall.Kill(-t.pgid, syscall.SIGKILL)
}

func (t processTree) close() error { return nil }

// ProcessAlive reports whether the process still runs: signal 0 performs
// the existence and permission checks without delivering anything.
func ProcessAlive(pid int) bool {
	return pid > 0 && syscall.Kill(pid, 0) == nil
}

// DetachedSysProcAttr starts a child in its own session, so it outlives the
// process that spawned it.
func DetachedSysProcAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{Setsid: true}
}

func (t processTree) wait() error { return nil }
