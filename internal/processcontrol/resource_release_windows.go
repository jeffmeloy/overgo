//go:build windows

package processcontrol

import (
	"context"
	"errors"
	"fmt"
	"syscall"
	"unsafe"
)

var (
	procCreateEventW           = kernel32.NewProc("CreateEventW")
	procSetEvent               = kernel32.NewProc("SetEvent")
	procResetEvent             = kernel32.NewProc("ResetEvent")
	procWaitForMultipleObjects = kernel32.NewProc("WaitForMultipleObjects")
	procQueryInformationJob    = kernel32.NewProc("QueryInformationJobObject")
	procNtQueryInformationFile = syscall.NewLazyDLL("ntdll.dll").NewProc("NtQueryInformationFile")
)

const (
	// Win32 and NT constants: the job's process-id list class, the file's
	// process-ids-using-file class, the NT statuses a short buffer earns,
	// WAIT_FAILED, MAXIMUM_WAIT_OBJECTS, FILE_READ_ATTRIBUTES and the
	// manual-reset flag of CreateEventW.
	jobObjectBasicProcessIDList        = 3
	fileProcessIDsUsingFileInformation = 47
	statusInfoLengthMismatch           = 0xC0000004
	statusBufferOverflow               = 0x80000005
	waitFailed                         = 0xFFFFFFFF
	maximumWaitObjects                 = 64
	fileReadAttributes                 = 0x0080
	manualResetEvent                   = 1
	// holderListStart is the first slot of a process-id list after its count word.
	holderListStart = 1
)

// releaseEventName names the manual-reset event every release of the
// resource sets; waiters reset it before each claim.
func releaseEventName(name string) (*uint16, error) {
	return syscall.UTF16PtrFromString("Global\\" + resourceObjectName(name) + ".Released")
}

func openReleaseEvent(name string) (syscall.Handle, error) {
	encoded, err := releaseEventName(name)
	if err != nil {
		return 0, err
	}
	handle, _, callErr := procCreateEventW.Call(0, manualResetEvent, 0, uintptr(unsafe.Pointer(encoded)))
	if handle == 0 {
		return 0, fmt.Errorf("processcontrol: open release event: %w", callErr)
	}
	return syscall.Handle(handle), nil
}

// signalResourceRelease wakes every waiter on the resource.
func signalResourceRelease(name string) {
	handle, err := openReleaseEvent(name)
	if err != nil {
		return
	}
	_, _, _ = procSetEvent.Call(uintptr(handle))
	_ = syscall.CloseHandle(handle)
}

// releaseWaiter observes a resource's holders: their exit through their
// process handles, their release through the resource's release event.
type releaseWaiter struct {
	name  string
	event syscall.Handle
	armed bool
	// unattributed records a refusal no listed holder explains.
	unattributed bool
}

func openReleaseWaiter(name string) (*releaseWaiter, error) {
	event, err := openReleaseEvent(name)
	if err != nil {
		return nil, err
	}
	return &releaseWaiter{name: name, event: event}, nil
}

// arm clears the release event ahead of a claim.
func (w *releaseWaiter) arm() error {
	if ok, _, err := procResetEvent.Call(uintptr(w.event)); ok == 0 {
		return fmt.Errorf("processcontrol: arm release event: %w", err)
	}
	w.armed = true
	return nil
}

// wait returns when a holder exits or releases, or when ctx ends; the
// caller's next claim reads the outcome.
func (w *releaseWaiter) wait(ctx context.Context) error {
	cancel, _, callErr := procCreateEventW.Call(0, manualResetEvent, 0, 0)
	if cancel == 0 {
		return fmt.Errorf("processcontrol: create cancellation event: %w", callErr)
	}
	defer syscall.CloseHandle(syscall.Handle(cancel))
	handles := []uintptr{uintptr(w.event), cancel}
	holders, err := resourceHolders(w.name)
	if err != nil {
		return err
	}
	// A holder gone between the claim and this listing left a stale verdict;
	// the next claim reads the release. Only a second unattributed refusal in
	// a row waits on the event alone, for a holder the listing cannot see.
	if len(holders) == 0 && !w.unattributed {
		w.unattributed = true
		return nil
	}
	w.unattributed = len(holders) == 0
	for _, pid := range holders {
		if len(handles) == maximumWaitObjects {
			break
		}
		process, err := syscall.OpenProcess(syscall.SYNCHRONIZE, false, pid)
		if err != nil {
			continue
		}
		defer syscall.CloseHandle(process)
		handles = append(handles, uintptr(process))
	}
	stop := make(chan struct{})
	finished := make(chan struct{})
	go func() {
		defer close(finished)
		select {
		case <-ctx.Done():
			_, _, _ = procSetEvent.Call(cancel)
		case <-stop:
		}
	}()
	result, _, callErr := procWaitForMultipleObjects.Call(uintptr(len(handles)), uintptr(unsafe.Pointer(&handles[0])), windowsFalse, resourceInfiniteWait)
	close(stop)
	<-finished
	w.armed = false
	if result == waitFailed {
		return fmt.Errorf("processcontrol: wait for resource release: %w", callErr)
	}
	if cause := context.Cause(ctx); cause != nil {
		return cause
	}
	return nil
}

// close releases the event; a wait armed and abandoned hands its wake-up
// on, so a sibling waiter never sleeps through a release this waiter took.
func (w *releaseWaiter) close() error {
	if w.armed {
		_, _, _ = procSetEvent.Call(uintptr(w.event))
	}
	return syscall.CloseHandle(w.event)
}

// resourceHolders lists the other processes holding the resource: those
// with its admission file open and the members of its exclusive job.
func resourceHolders(name string) ([]uint32, error) {
	self := uint32(syscall.Getpid())
	var holders []uint32
	fileUsers, err := admissionFileUsers(name)
	if err != nil {
		return nil, err
	}
	jobMembers, err := exclusiveJobMembers(name)
	if err != nil {
		return nil, err
	}
	for _, pid := range append(fileUsers, jobMembers...) {
		if pid != self && pid != 0 {
			holders = append(holders, pid)
		}
	}
	return holders, nil
}

// admissionFileUsers asks the kernel which processes hold the admission
// file open; attribute access joins no sharing check, so the query sees an
// exclusive holder too.
func admissionFileUsers(name string) ([]uint32, error) {
	path, err := resourceAdmissionPath(name)
	if err != nil {
		return nil, err
	}
	encoded, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return nil, err
	}
	handle, err := syscall.CreateFile(encoded, fileReadAttributes,
		syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE, nil, syscall.OPEN_EXISTING, syscall.FILE_ATTRIBUTE_NORMAL, 0)
	if errors.Is(err, syscall.ERROR_FILE_NOT_FOUND) || errors.Is(err, syscall.ERROR_PATH_NOT_FOUND) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("processcontrol: inspect resource admission %q: %w", name, err)
	}
	defer syscall.CloseHandle(handle)
	buffer := make([]uintptr, maximumWaitObjects)
	for {
		var status [2]uintptr
		result, _, _ := procNtQueryInformationFile.Call(uintptr(handle), uintptr(unsafe.Pointer(&status[0])),
			uintptr(unsafe.Pointer(&buffer[0])), uintptr(len(buffer))*unsafe.Sizeof(buffer[0]), fileProcessIDsUsingFileInformation)
		switch uint32(result) {
		case 0:
			return processIDs(buffer, uint32(buffer[0])), nil
		case statusInfoLengthMismatch, statusBufferOverflow:
			buffer = make([]uintptr, len(buffer)*2)
		default:
			return nil, fmt.Errorf("processcontrol: list holders of %q: NTSTATUS %#x", name, uint32(result))
		}
	}
}

// exclusiveJobMembers lists the processes in the resource's exclusive job,
// none when no exclusive claim exists.
func exclusiveJobMembers(name string) ([]uint32, error) {
	jobName, err := syscall.UTF16PtrFromString("Global\\" + resourceObjectName(name))
	if err != nil {
		return nil, err
	}
	job, _, callErr := procOpenResourceJob.Call(resourceJobQuery, 0, uintptr(unsafe.Pointer(jobName)))
	if job == 0 {
		if errors.Is(callErr, syscall.ERROR_FILE_NOT_FOUND) {
			return nil, nil
		}
		return nil, fmt.Errorf("processcontrol: inspect exclusive resource %q: %w", name, callErr)
	}
	defer syscall.CloseHandle(syscall.Handle(job))
	buffer := make([]uintptr, maximumWaitObjects)
	for {
		var returned uint32
		ok, _, callErr := procQueryInformationJob.Call(job, jobObjectBasicProcessIDList, uintptr(unsafe.Pointer(&buffer[0])),
			uintptr(len(buffer))*unsafe.Sizeof(buffer[0]), uintptr(unsafe.Pointer(&returned)))
		if ok != 0 {
			// The count word holds the assigned count low and the listed count high.
			return processIDs(buffer, uint32(buffer[0]>>32)), nil
		}
		if !errors.Is(callErr, syscall.ERROR_MORE_DATA) {
			return nil, fmt.Errorf("processcontrol: list members of %q: %w", name, callErr)
		}
		buffer = make([]uintptr, len(buffer)*2)
	}
}

// processIDs reads count pids from a process-id list buffer.
func processIDs(buffer []uintptr, count uint32) []uint32 {
	pids := make([]uint32, 0, count)
	for index := holderListStart; index < len(buffer) && uint32(index-holderListStart) < count; index++ {
		pids = append(pids, uint32(buffer[index]))
	}
	return pids
}
