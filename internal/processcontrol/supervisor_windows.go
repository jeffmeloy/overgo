//go:build windows

package processcontrol

import (
	"errors"
	"fmt"
	"os/exec"
	"syscall"
	"unsafe"
)

// Windows containment uses a job object with kill-on-close: every
// descendant the command spawns joins the job, TerminateJobObject
// takes the whole tree down at once, and closing the handle -- even
// through supervisor death -- kills anything still running. Reached
// through loaded DLLs per repository doctrine; no cgo.
var (
	kernel32               = syscall.NewLazyDLL("kernel32.dll")
	procCreateJobObjectW   = kernel32.NewProc("CreateJobObjectW")
	procSetInformationJob  = kernel32.NewProc("SetInformationJobObject")
	procAssignProcessToJob = kernel32.NewProc("AssignProcessToJobObject")
	procTerminateJobObject = kernel32.NewProc("TerminateJobObject")
	procOpenProcess        = kernel32.NewProc("OpenProcess")
	procCloseHandle        = kernel32.NewProc("CloseHandle")
)

const (
	// jobObjectExtendedLimitClass is JobObjectExtendedLimitInformation,
	// the information class selector SetInformationJobObject takes for
	// extended limits including kill-on-close.
	jobObjectExtendedLimitClass = 9
	// windowsFalse is the Win32 BOOL FALSE argument value.
	windowsFalse = 0
	// jobObjectLimitKillOnJobClose is JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE:
	// closing the last job handle terminates every process in the job.
	jobObjectLimitKillOnJobClose = 0x2000
	// processAccessForJob combines PROCESS_SET_QUOTA (0x0100) and
	// PROCESS_TERMINATE (0x0001), the rights AssignProcessToJobObject
	// requires on the target process handle.
	processAccessForJob = 0x0101
	// terminatedTreeExitCode is the exit status TerminateJobObject
	// stamps on every process it kills.
	terminatedTreeExitCode = 1
)

type jobBasicLimits struct {
	PerProcessUserTimeLimit int64
	PerJobUserTimeLimit     int64
	LimitFlags              uint32
	MinimumWorkingSetSize   uintptr
	MaximumWorkingSetSize   uintptr
	ActiveProcessLimit      uint32
	Affinity                uintptr
	PriorityClass           uint32
	SchedulingClass         uint32
}

type ioCounters struct {
	ReadOperationCount  uint64
	WriteOperationCount uint64
	OtherOperationCount uint64
	ReadTransferCount   uint64
	WriteTransferCount  uint64
	OtherTransferCount  uint64
}

type jobExtendedLimits struct {
	BasicLimitInformation jobBasicLimits
	IoInfo                ioCounters
	ProcessMemoryLimit    uintptr
	JobMemoryLimit        uintptr
	PeakProcessMemoryUsed uintptr
	PeakJobMemoryUsed     uintptr
}

type processTree struct {
	job uintptr
}

func configureSysProc(*exec.Cmd) {}

func newProcessTree(command *exec.Cmd) (processTree, error) {
	job, _, callErr := procCreateJobObjectW.Call(0, 0)
	if job == 0 {
		return processTree{}, fmt.Errorf("create job object: %w", callErr)
	}
	limits := jobExtendedLimits{}
	limits.BasicLimitInformation.LimitFlags = jobObjectLimitKillOnJobClose
	if ok, _, callErr := procSetInformationJob.Call(
		job, uintptr(jobObjectExtendedLimitClass),
		uintptr(unsafe.Pointer(&limits)), unsafe.Sizeof(limits),
	); ok == 0 {
		_, _, _ = procCloseHandle.Call(job)
		return processTree{}, fmt.Errorf("configure job object: %w", callErr)
	}
	process, _, callErr := procOpenProcess.Call(processAccessForJob, windowsFalse, uintptr(command.Process.Pid))
	if process == 0 {
		_, _, _ = procCloseHandle.Call(job)
		return processTree{}, fmt.Errorf("open process %d: %w", command.Process.Pid, callErr)
	}
	assigned, _, callErr := procAssignProcessToJob.Call(job, process)
	_, _, _ = procCloseHandle.Call(process)
	if assigned == 0 {
		_, _, _ = procCloseHandle.Call(job)
		return processTree{}, fmt.Errorf("assign process %d to job: %w", command.Process.Pid, callErr)
	}
	return processTree{job: job}, nil
}

// interrupt has no cooperative tree signal on Windows without a shared
// console; the request is recorded and termination is the effective
// stop. Callers needing cooperative shutdown speak an application
// protocol instead.
func (t processTree) interrupt(*exec.Cmd) error { return nil }

func (t processTree) terminate() error {
	if t.job == 0 {
		return errors.New("processcontrol: no job handle")
	}
	if ok, _, callErr := procTerminateJobObject.Call(t.job, terminatedTreeExitCode); ok == 0 {
		return fmt.Errorf("processcontrol: terminate job: %w", callErr)
	}
	return nil
}

func (t processTree) close() error {
	if t.job == 0 {
		return nil
	}
	_, _, _ = procCloseHandle.Call(t.job)
	return nil
}
