package processcontrol

import (
	"errors"
	"io"
	"os/exec"
)

// StartDetached starts a command that outlives its parent: its own process
// group or session, no inherited console, output to the caller's writers,
// and the process handle released once the pid is known. The caller keeps
// the pid in a locator; ProcessAlive answers for it afterwards.
func StartDetached(command Command, output io.Writer) (int, error) {
	if command.Path == "" {
		return 0, errors.New("processcontrol: detached command requires a path")
	}
	child := exec.Command(command.Path, command.Args...)
	child.Dir = command.Dir
	child.Env = command.Env
	child.Stdout, child.Stderr = output, output
	child.SysProcAttr = DetachedSysProcAttr()
	if err := child.Start(); err != nil {
		return 0, err
	}
	pid := child.Process.Pid
	return pid, child.Process.Release()
}
