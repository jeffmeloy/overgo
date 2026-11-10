//go:build windows

package processcontrol

import (
	"bytes"
	"os/exec"
	"slices"
	"strings"
	"syscall"
	"testing"
	"time"

	"overgo/internal/testcheck"
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
	testcheck.NoError(t, run.Start())
	t.Cleanup(func() {
		_ = run.Process.Kill()
		_ = run.Wait()
	})
	handle := testcheck.Must(syscall.OpenProcess(syscall.SYNCHRONIZE, false, uint32(run.Process.Pid)))(t)
	defer syscall.CloseHandle(handle)
	state, err := syscall.WaitForSingleObject(handle, 0)
	if err != nil || state != syscall.WAIT_TIMEOUT {
		t.Fatalf("suspended child already terminated: %d, %v", state, err)
	}
	tree := testcheck.Must(newProcessTree(run))(t)
	defer tree.close()
	if err := run.Wait(); err != nil || strings.TrimSpace(output.String()) != "contained" {
		t.Fatalf("contained execution: %q, %v", output.String(), err)
	}
}

// TestASupervisedStartCostsWhatAPlainOneDoes holds a contained start to about the cost of the child it starts: run in
// turn with a plain exec of the same command, so the two share the machine's load, a supervised start takes less than
// twice a plain one at the median. A start that walked every thread on the machine to resume its child took three
// times as long here (48.8 ms against 15 ms, BenchmarkSupervisedStart).
func TestASupervisedStartCostsWhatAPlainOneDoes(t *testing.T) {
	command := Command{Path: "cmd", Args: []string{"/c", "exit 0"}}
	var plain, supervised []time.Duration
	for range 15 {
		began := time.Now()
		testcheck.NoError(t, exec.Command(command.Path, command.Args...).Run())
		plain = append(plain, time.Since(began))
		began = time.Now()
		testcheck.Must(Run(t.Context(), command))(t)
		supervised = append(supervised, time.Since(began))
	}
	slices.Sort(plain)
	slices.Sort(supervised)
	if median, bar := supervised[len(supervised)/2], 2*plain[len(plain)/2]; median >= bar {
		t.Fatalf("a supervised start took %v at the median, a plain one %v: want under %v", median, plain[len(plain)/2], bar)
	}
}

// BenchmarkSupervisedStart measures a contained short command, one at a time and side by side as a landing's
// scenarios start theirs: a start's cost is its child's creation, job and resume. Before the resume took one call, a
// walk of every thread on the machine cost 28 ms of each start, and eight side by side took 40 ms a start.
func BenchmarkSupervisedStart(b *testing.B) {
	command := Command{Path: "cmd", Args: []string{"/c", "exit 0"}}
	b.Run("alone", func(b *testing.B) {
		for b.Loop() {
			if _, err := Run(b.Context(), command); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("side-by-side", func(b *testing.B) {
		b.SetParallelism(2)
		b.RunParallel(func(next *testing.PB) {
			for next.Next() {
				if _, err := Run(b.Context(), command); err != nil {
					b.Error(err)
				}
			}
		})
	})
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
