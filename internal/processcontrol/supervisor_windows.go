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
	procThread32First        = kernel32.NewProc("Thread32First")
	procThread32Next         = kernel32.NewProc("Thread32Next")
	procOpenThread           = kernel32.NewProc("OpenThread")
	procResumeThread         = kernel32.NewProc("ResumeThread")
	procCreateCompletionPort = kernel32.NewProc("CreateIoCompletionPort")
	procGetCompletionStatus  = kernel32.NewProc("GetQueuedCompletionStatus")
	procGetExitCodeProcess   = kernel32.NewProc("GetExitCodeProcess")
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
	// processAccessForJob combines PROCESS_SET_QUOTA, PROCESS_TERMINATE and
	// PROCESS_DUP_HANDLE: job assignment plus resource-handle retention.
	processAccessForJob = 0x0141
	// terminatedTreeExitCode is the exit status TerminateJobObject
	// stamps on every process it kills.
	terminatedTreeExitCode = 1
	// createSuspended is CREATE_SUSPENDED. No child instructions execute before
	// the primary thread is resumed following successful job assignment.
	createSuspended = 0x00000004
	// threadSuspendResume is THREAD_SUSPEND_RESUME, the minimum ResumeThread right.
	threadSuspendResume = 0x0002
	// JobObjectAssociateCompletionPortInformation and JOB_OBJECT_MSG_ACTIVE_PROCESS_ZERO.
	jobCompletionPortClass = 7
	jobActiveProcessZero   = 4
	// detachedProcess is DETACHED_PROCESS: the child gets no console, so it
	// outlives the console session of the process that started it.
	detachedProcess = 0x00000008
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

// DetachedSysProcAttr starts a child in its own process group without a
// console, so it outlives the process that spawned it.
func DetachedSysProcAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP | detachedProcess}
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
	if err := resumeCreatedThread(uint32(command.Process.Pid)); err != nil {
		_, _, _ = procCloseHandle.Call(job)
		return processTree{}, err
	}
	ready = true
	return processTree{job: job, completion: completion, process: process}, nil
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
