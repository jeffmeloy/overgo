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
	kernel32                 = syscall.NewLazyDLL("kernel32.dll")
	procCreateJobObjectW     = kernel32.NewProc("CreateJobObjectW")
	procSetInformationJob    = kernel32.NewProc("SetInformationJobObject")
	procAssignProcessToJob   = kernel32.NewProc("AssignProcessToJobObject")
	procTerminateJobObject   = kernel32.NewProc("TerminateJobObject")
	procOpenProcess          = kernel32.NewProc("OpenProcess")
	procCloseHandle          = kernel32.NewProc("CloseHandle")
	procCreateCompletionPort = kernel32.NewProc("CreateIoCompletionPort")
	procGetCompletionStatus  = kernel32.NewProc("GetQueuedCompletionStatus")
	procGetExitCodeProcess   = kernel32.NewProc("GetExitCodeProcess")
	// NtResumeProcess resumes the child created suspended in one call. Go closes CreateProcess's thread handle, and
	// the documented way back to it, a Toolhelp walk of every thread on the machine, cost 28 ms a start and ran one
	// start at a time (eight side by side took 40 ms each), since the child is always the walk's last entry.
	procNtResumeProcess = syscall.NewLazyDLL("ntdll.dll").NewProc("NtResumeProcess")
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
	// processAccessForJob combines PROCESS_SET_QUOTA, PROCESS_TERMINATE,
	// PROCESS_DUP_HANDLE and PROCESS_SUSPEND_RESUME: job assignment,
	// resource-handle retention, and the child's resume once it is contained.
	processAccessForJob = 0x0941
	// terminatedTreeExitCode is the exit status TerminateJobObject
	// stamps on every process it kills.
	terminatedTreeExitCode = 1
	// createSuspended is CREATE_SUSPENDED. No child instructions execute before
	// the child is resumed following successful job assignment.
	createSuspended = 0x00000004
	// JobObjectAssociateCompletionPortInformation and JOB_OBJECT_MSG_ACTIVE_PROCESS_ZERO.
	jobCompletionPortClass = 7
	jobActiveProcessZero   = 4
	// processQueryLimitedInformation is PROCESS_QUERY_LIMITED_INFORMATION,
	// the least OpenProcess right that still admits GetExitCodeProcess.
	processQueryLimitedInformation = 0x1000
	// stillActive is STILL_ACTIVE, the exit code GetExitCodeProcess reports
	// for a process that has not exited.
	stillActive = 259
)

// ProcessAlive reports whether the process still runs.
func ProcessAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	handle, _, _ := procOpenProcess.Call(processQueryLimitedInformation, windowsFalse, uintptr(pid))
	if handle == 0 {
		return false
	}
	defer procCloseHandle.Call(handle)
	var code uint32
	ok, _, _ := procGetExitCodeProcess.Call(handle, uintptr(unsafe.Pointer(&code)))
	return ok != 0 && code == stillActive
}

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
	job        uintptr
	completion uintptr
	process    uintptr
}

func configureSysProc(command *exec.Cmd) {
	if command.SysProcAttr == nil {
		command.SysProcAttr = &syscall.SysProcAttr{}
	}
	command.SysProcAttr.CreationFlags |= createSuspended
}

func newProcessTree(command *exec.Cmd) (processTree, error) {
	if command.Process == nil || command.SysProcAttr == nil || command.SysProcAttr.CreationFlags&createSuspended == 0 {
		return processTree{}, errors.New("job assignment requires a child created suspended")
	}
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
	// A standalone port uses the Windows concurrency default; wait is its sole consumer.
	completion, _, callErr := procCreateCompletionPort.Call(uintptr(syscall.InvalidHandle), 0, 0, 0)
	if completion == 0 {
		_, _, _ = procCloseHandle.Call(job)
		return processTree{}, fmt.Errorf("create job completion port: %w", callErr)
	}
	ready := false
	defer func() {
		if !ready {
			_, _, _ = procCloseHandle.Call(completion)
		}
	}()
	association := struct{ Key, Port uintptr }{job, completion}
	if ok, _, callErr := procSetInformationJob.Call(job, jobCompletionPortClass,
		uintptr(unsafe.Pointer(&association)), unsafe.Sizeof(association)); ok == 0 {
		_, _, _ = procCloseHandle.Call(job)
		return processTree{}, fmt.Errorf("associate job completion port: %w", callErr)
	}
	process, _, callErr := procOpenProcess.Call(processAccessForJob|syscall.SYNCHRONIZE, windowsFalse, uintptr(command.Process.Pid))
	if process == 0 {
		_, _, _ = procCloseHandle.Call(job)
		return processTree{}, fmt.Errorf("open process %d: %w", command.Process.Pid, callErr)
	}
	defer func() {
		if !ready {
			_, _, _ = procCloseHandle.Call(process)
		}
	}()
	assigned, _, callErr := procAssignProcessToJob.Call(job, process)
	if assigned != 0 {
		if err := retainChildResources(process); err != nil {
			_, _, _ = procCloseHandle.Call(job)
			return processTree{}, err
		}
	}
	if assigned == 0 {
		_, _, _ = procCloseHandle.Call(job)
		return processTree{}, fmt.Errorf("assign process %d to job: %w", command.Process.Pid, callErr)
	}
	// The child runs from here, inside its job.
	if status, _, _ := procNtResumeProcess.Call(process); status != 0 {
		_, _, _ = procCloseHandle.Call(job)
		return processTree{}, fmt.Errorf("resume contained child %d: NTSTATUS %#x", command.Process.Pid, status)
	}
	ready = true
	return processTree{job: job, completion: completion, process: process}, nil
}

func (t processTree) terminate() error {
	if t.job == 0 {
		return errors.New("processcontrol: no job handle")
	}
	if ok, _, callErr := procTerminateJobObject.Call(t.job, terminatedTreeExitCode); ok == 0 {
		return fmt.Errorf("processcontrol: terminate job: %w", callErr)
	}
	return nil
}

// peakProcessBytes is the most memory any one process of the job has committed; zero when the job cannot be read.
func (t processTree) peakProcessBytes() uint64 {
	limits := jobExtendedLimits{}
	if t.job == 0 {
		return 0
	}
	if ok, _, _ := procQueryInformationJob.Call(t.job, uintptr(jobObjectExtendedLimitClass), uintptr(unsafe.Pointer(&limits)), unsafe.Sizeof(limits), 0); ok == 0 {
		return 0
	}
	return uint64(limits.PeakProcessMemoryUsed)
}

func (t processTree) close() error {
	if t.job == 0 {
		return nil
	}
	_, _, _ = procCloseHandle.Call(t.job)
	if t.process != 0 {
		_, _, _ = procCloseHandle.Call(t.process)
	}
	if t.completion != 0 {
		_, _, _ = procCloseHandle.Call(t.completion)
	}
	return nil
}

// Parent exit ends its owned tree. Terminate leftover descendants, then wait for
// the empty-job notification before draining pipes or releasing resources.
func (t processTree) wait() error {
	state, err := syscall.WaitForSingleObject(syscall.Handle(t.process), syscall.INFINITE)
	if err != nil || state != syscall.WAIT_OBJECT_0 {
		return errors.Join(fmt.Errorf("parent wait returned state %d", state), err, t.terminate())
	}
	if err := t.terminate(); err != nil {
		return err
	}
	for {
		var message uint32
		var key, overlapped uintptr
		if ok, _, callErr := procGetCompletionStatus.Call(t.completion,
			uintptr(unsafe.Pointer(&message)), uintptr(unsafe.Pointer(&key)),
			uintptr(unsafe.Pointer(&overlapped)), syscall.INFINITE); ok == 0 {
			return fmt.Errorf("wait for job completion: %w", callErr)
		}
		if key == t.job && message == jobActiveProcessZero {
			return nil
		}
	}
}
