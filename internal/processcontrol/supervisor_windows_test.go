//go:build windows

package processcontrol

import (
	"bytes"
	"os/exec"
	"strings"
	"syscall"
	"testing"
)

func TestWindowsContainmentPrecedesExecution(t *testing.T) {
	run := exec.Command("cmd", "/c", "echo contained")
	configureSysProc(run)
	// CREATE_SUSPENDED is the external Win32 creation contract, not a timing
	// assertion: the command cannot exit or spawn before assignment to its job.
	if run.SysProcAttr == nil || run.SysProcAttr.CreationFlags&0x00000004 == 0 {
		t.Fatal("process can execute before job assignment")
	}
	var output bytes.Buffer
	run.Stdout = &output
	if err := run.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = run.Process.Kill()
		_ = run.Wait()
	})
	handle, err := syscall.OpenProcess(syscall.SYNCHRONIZE, false, uint32(run.Process.Pid))
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.CloseHandle(handle)
	state, err := syscall.WaitForSingleObject(handle, 0)
	if err != nil || state != syscall.WAIT_TIMEOUT {
		t.Fatalf("suspended child already terminated: %d, %v", state, err)
	}
	tree, err := newProcessTree(run)
	if err != nil {
		t.Fatal(err)
	}
	defer tree.close()
	if err := run.Wait(); err != nil || strings.TrimSpace(output.String()) != "contained" {
		t.Fatalf("contained execution: %q, %v", output.String(), err)
	}
}

func TestWindowsConcurrentShortCommands(t *testing.T) {
	for _, script := range []string{"echo first", "echo second", "exit 0", "exit 7"} {
		t.Run(script, func(t *testing.T) {
			t.Parallel()
			receipt, err := Run(t.Context(), shellCommand(t, script))
			want := 0
			if script == "exit 7" {
				want = 7
			}
			if err != nil || receipt.ExitCode != want || receipt.TreeTerminated {
				t.Fatalf("short command: %+v, %v", receipt, err)
			}
		})
	}
}
