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
	// jobObjectLimitBreakawayOK is JOB_OBJECT_LIMIT_BREAKAWAY_OK: a process started asking to leave the job
	// (createBreakawayFromJob) leaves it, as StartDetached asks, and every other stays in it.
	jobObjectLimitBreakawayOK = 0x800
	// createBreakawayFromJob, createNewProcessGroup and createNoWindow are CREATE_BREAKAWAY_FROM_JOB,
	// CREATE_NEW_PROCESS_GROUP and CREATE_NO_WINDOW, the creation flags of a detached start (detach).
	createBreakawayFromJob = 0x01000000
	createNewProcessGroup  = 0x00000200
	createNoWindow         = 0x08000000
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
	// jobObjectLimitJobMemory is JOB_OBJECT_LIMIT_JOB_MEMORY: the job's processes together commit no more than
	// JobMemoryLimit; jobMemoryLimitReached is JOB_OBJECT_MSG_JOB_MEMORY_LIMIT, posted when a commit would pass it.
	jobObjectLimitJobMemory = 0x200
	jobMemoryLimitReached   = 10
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

// ProcessStart is when the process of pid started, in the system's count of 100-nanosecond intervals since 1601
// (GetProcessTimes' creation time), and whether it was read: with its id it names one process, where the id alone names
// any the system has given it to since (process-identity/start-time).
func ProcessStart(pid int) (int64, bool) {
	if pid <= 0 {
		return 0, false
	}
	handle, _, _ := procOpenProcess.Call(processQueryLimitedInformation, windowsFalse, uintptr(pid))
	if handle == 0 {
		return 0, false
	}
	defer procCloseHandle.Call(handle)
	var creation, exit, kernel, user syscall.Filetime
	if err := syscall.GetProcessTimes(syscall.Handle(handle), &creation, &exit, &kernel, &user); err != nil {
		return 0, false
	}
	return int64(creation.HighDateTime)<<32 | int64(creation.LowDateTime), true
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
	bound      jobBound
}

// detach starts command out of the job this process runs in, whose close would end it, in a process group of its own,
// which its starter's console's interrupt does not reach, with a console of its own that shows no window: a process
// with none (DETACHED_PROCESS) gives each console program it starts a window of its own, as go run's compiler.
func detach(command *exec.Cmd) error {
	command.SysProcAttr = &syscall.SysProcAttr{CreationFlags: createBreakawayFromJob | createNewProcessGroup | createNoWindow}
	err := command.Start()
	if errors.Is(err, syscall.ERROR_ACCESS_DENIED) {
		return fmt.Errorf("processcontrol: the job this process runs in admits no process leaving it: %w", err)
	}
	return err
}

// configureSysProc creates a supervised child suspended, put in its tree's job before it runs, and with a console of its
// own that shows no window (createNoWindow), so no process it starts depends on its starter's console: a landing whose
// session had ended, the session's console host gone with it, started git, which could not initialize (0xC0000142), and
// its commit attempt was refused unjudged (landing-console). Its output still comes through the pipes the supervisor
// drains, and the tree still ends with its job.
func configureSysProc(command *exec.Cmd) {
	if command.SysProcAttr == nil {
		command.SysProcAttr = &syscall.SysProcAttr{}
	}
	command.SysProcAttr.CreationFlags |= createSuspended | createNoWindow
}

func newProcessTree(command *exec.Cmd, bound jobBound) (processTree, error) {
	if command.Process == nil || command.SysProcAttr == nil || command.SysProcAttr.CreationFlags&createSuspended == 0 {
		return processTree{}, errors.New("job assignment requires a child created suspended")
	}
	job, _, callErr := procCreateJobObjectW.Call(0, 0)
	if job == 0 {
		return processTree{}, fmt.Errorf("create job object: %w", callErr)
	}
	limits := jobExtendedLimits{}
	// A process a supervised one starts detached (StartDetached) leaves the tree: a closing landing's lanes runner,
	// started by a landing the gate runs in a tree, outlives it (lanes-on-close).
	limits.BasicLimitInformation.LimitFlags = jobObjectLimitKillOnJobClose | jobObjectLimitBreakawayOK
	if bound.limit > 0 {
		limits.BasicLimitInformation.LimitFlags |= jobObjectLimitJobMemory
		limits.JobMemoryLimit = uintptr(bound.limit)
	}
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
	return processTree{job: job, completion: completion, process: process, bound: bound}, nil
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
	return uint64(t.extendedLimits().PeakProcessMemoryUsed)
}

// extendedLimits reads the job's extended limits and counters; zero when the job cannot be read.
func (t processTree) extendedLimits() jobExtendedLimits {
	limits := jobExtendedLimits{}
	if t.job == 0 {
		return limits
	}
	if ok, _, _ := procQueryInformationJob.Call(t.job, uintptr(jobObjectExtendedLimitClass), uintptr(unsafe.Pointer(&limits)), unsafe.Sizeof(limits), 0); ok == 0 {
		return jobExtendedLimits{}
	}
	return limits
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

// wait waits for the tree to end, saying whether a commit of it would have passed its bound (jobBound). Parent exit
// ends its owned tree: leftover descendants are terminated, then the empty-job notification is awaited before pipes
// are drained or resources released. The job's notifications are read while the tree runs, and a commit that would
// pass the bound ends the tree there: a Go child whose commit failed at its bound never ended, its own timeout's
// timer stalled with it (check-memory/do).
func (t processTree) wait() (limited bool, err error) {
	type drained struct {
		limited bool
		err     error
	}
	notified := make(chan drained, 1)
	go func() {
		var limited bool
		for {
			var message uint32
			var key, overlapped uintptr
			if ok, _, callErr := procGetCompletionStatus.Call(t.completion,
				uintptr(unsafe.Pointer(&message)), uintptr(unsafe.Pointer(&key)),
				uintptr(unsafe.Pointer(&overlapped)), syscall.INFINITE); ok == 0 {
				notified <- drained{limited, fmt.Errorf("wait for job completion: %w", callErr)}
				return
			}
			switch {
			case key == t.job && message == jobMemoryLimitReached:
				limited = true
				_ = t.terminate()
			case key == t.job && message == jobActiveProcessZero:
				notified <- drained{limited, nil}
				return
			}
		}
	}()
	state, err := syscall.WaitForSingleObject(syscall.Handle(t.process), syscall.INFINITE)
	if err != nil || state != syscall.WAIT_OBJECT_0 {
		return false, errors.Join(fmt.Errorf("parent wait returned state %d", state), err, t.terminate())
	}
	if err := t.terminate(); err != nil {
		return false, err
	}
	ended := <-notified
	return ended.limited, ended.err
}
