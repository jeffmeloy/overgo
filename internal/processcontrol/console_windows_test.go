//go:build windows

package processcontrol

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"unsafe"

	"overgo/internal/testcheck"
)

// consoleProbeEnvironment marks a process of this test binary as TestAProcessStartsAfterItsConsoleHostEnds's child.
const consoleProbeEnvironment = "OVERGO_TEST_CONSOLE_PROBE"

// TestConsoleProbeProcess is a process of TestAProcessStartsAfterItsConsoleHostEnds, and passes at once in any other
// run: it says it is ready, waits for its parent's word, starts a console program through Run, and says how it ended.
func TestConsoleProbeProcess(t *testing.T) {
	if os.Getenv(consoleProbeEnvironment) == "" {
		return
	}
	fmt.Println("ready")
	_, _ = bufio.NewReader(os.Stdin).ReadString('\n')
	receipt, err := Run(context.Background(), Command{Path: os.Args[0], Args: []string{"-test.run=^$"}})
	if err != nil {
		fmt.Println("start", err)
		return
	}
	fmt.Println("exit", receipt.ExitCode)
}

// TestAProcessStartsAfterItsConsoleHostEnds holds every process Run starts to a console of its own (landing-console):
// a process whose own console's host has ended still starts a console program through Run, which exits 0. A landing
// whose session ended, its console's host gone with it, started git, which exited 0xC0000142 (STATUS_DLL_INIT_FAILED),
// and its commit attempt was refused unjudged.
func TestAProcessStartsAfterItsConsoleHostEnds(t *testing.T) {
	child := exec.Command(os.Args[0], "-test.run=^TestConsoleProbeProcess$", "-test.v")
	child.Env = append(os.Environ(), consoleProbeEnvironment+"=1")
	child.SysProcAttr = &syscall.SysProcAttr{CreationFlags: createNoWindow}
	input := testcheck.Must(child.StdinPipe())(t)
	output := testcheck.Must(child.StdoutPipe())(t)
	testcheck.NoError(t, child.Start())
	said := bufio.NewScanner(output)
	for said.Scan() && said.Text() != "ready" {
	}
	host := consoleHostOf(t, uint32(child.Process.Pid))
	handle := testcheck.Must(syscall.OpenProcess(syscall.PROCESS_TERMINATE|syscall.SYNCHRONIZE, false, host))(t)
	testcheck.NoError(t, syscall.TerminateProcess(handle, 1))
	_, _ = syscall.WaitForSingleObject(handle, syscall.INFINITE)
	testcheck.NoError(t, syscall.CloseHandle(handle))
	testcheck.Must(fmt.Fprintln(input, "go"))(t)
	ended := ""
	for said.Scan() {
		if text := said.Text(); strings.HasPrefix(text, "exit ") || strings.HasPrefix(text, "start ") {
			ended = text
		}
	}
	testcheck.NoError(t, child.Wait())
	if ended != "exit 0" {
		t.Fatalf("a process whose console's host ended started a console program through Run, which ended %q; want exit 0", ended)
	}
}

// consoleHostOf is the console host (conhost.exe) of the process pid, the conhost whose parent it is.
func consoleHostOf(t *testing.T, pid uint32) uint32 {
	t.Helper()
	snapshot := testcheck.Must(syscall.CreateToolhelp32Snapshot(syscall.TH32CS_SNAPPROCESS, 0))(t)
	defer syscall.CloseHandle(snapshot)
	var entry syscall.ProcessEntry32
	entry.Size = uint32(unsafe.Sizeof(entry))
	for err := syscall.Process32First(snapshot, &entry); err == nil; err = syscall.Process32Next(snapshot, &entry) {
		if entry.ParentProcessID == pid && syscall.UTF16ToString(entry.ExeFile[:]) == "conhost.exe" {
			return entry.ProcessID
		}
	}
	t.Fatalf("process %s has no console host", strconv.Itoa(int(pid)))
	return 0
}
