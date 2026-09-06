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
	procThread32First      = kernel32.NewProc("Thread32First")
	procThread32Next       = kernel32.NewProc("Thread32Next")
	procOpenThread         = kernel32.NewProc("OpenThread")
	procResumeThread       = kernel32.NewProc("ResumeThread")
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
	// createSuspended is CREATE_SUSPENDED. No child instructions execute before
	// the primary thread is resumed following successful job assignment.
	createSuspended = 0x00000004
	// threadSuspendResume is THREAD_SUSPEND_RESUME, the minimum ResumeThread right.
	threadSuspendResume = 0x0002
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
	if err := resumeCreatedThread(uint32(command.Process.Pid)); err != nil {
		_, _, _ = procCloseHandle.Call(job)
		return processTree{}, err
	}
	return processTree{job: job}, nil
}

// threadEntry has the documented THREADENTRY32 ABI from tlhelp32.h.
type threadEntry struct {
	Size, Usage, ThreadID, ProcessID uint32
	BasePriority, DeltaPriority      int32
	Flags                            uint32
}

// resumeCreatedThread resumes only the unstarted child's primary thread. Go
// closes CreateProcess's thread handle, so obtain it through documented Toolhelp
// APIs. An unexpected thread set fails closed rather than resuming other work.
func resumeCreatedThread(pid uint32) error {
	snapshot, err := syscall.CreateToolhelp32Snapshot(syscall.TH32CS_SNAPTHREAD, 0)
	if err != nil {
		return fmt.Errorf("snapshot suspended child threads: %w", err)
	}
	defer syscall.CloseHandle(snapshot)
	var entry threadEntry
	var threadID uint32
	entry.Size = uint32(unsafe.Sizeof(entry))
	ok, _, callErr := procThread32First.Call(uintptr(snapshot), uintptr(unsafe.Pointer(&entry)))
	for ok != 0 {
		if entry.ProcessID == pid {
			if threadID != 0 {
				return errors.New("suspended child has more than one initial thread")
			}
			threadID = entry.ThreadID
		}
		entry.Size = uint32(unsafe.Sizeof(entry))
		ok, _, callErr = procThread32Next.Call(uintptr(snapshot), uintptr(unsafe.Pointer(&entry)))
	}
	if !errors.Is(callErr, syscall.ERROR_NO_MORE_FILES) {
		return fmt.Errorf("enumerate suspended child threads: %w", callErr)
	}
	if threadID == 0 {
		return errors.New("suspended child primary thread is absent")
	}
	thread, _, callErr := procOpenThread.Call(threadSuspendResume, windowsFalse, uintptr(threadID))
	if thread == 0 {
		return fmt.Errorf("open suspended child thread: %w", callErr)
	}
	defer procCloseHandle.Call(thread)
	previous, _, callErr := procResumeThread.Call(thread)
	if uint32(previous) == ^uint32(0) {
		return fmt.Errorf("resume contained child: %w", callErr)
	}
	if previous != 1 {
		return fmt.Errorf("contained child has unexpected suspend count %d", previous)
	}
	return nil
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
