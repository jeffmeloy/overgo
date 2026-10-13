package landing

import (
	"fmt"
	"syscall"
	"unsafe"
)

// freeSpaceCall is the kernel's read of a volume's free space.
var freeSpaceCall = syscall.NewLazyDLL("kernel32.dll").NewProc("GetDiskFreeSpaceExW")

// VolumeFree reads how many bytes this process may still write on the volume
// that holds directory, which is less than the volume's free bytes under a quota.
func VolumeFree(directory string) (uint64, error) {
	name, err := syscall.UTF16PtrFromString(directory)
	if err != nil {
		return 0, err
	}
	var free uint64
	// The call's later results, the volume's size and its free bytes for every user, are not asked for.
	if read, _, failure := freeSpaceCall.Call(uintptr(unsafe.Pointer(name)), uintptr(unsafe.Pointer(&free)), 0, 0); read == 0 {
		return 0, fmt.Errorf("free space of %s: %w", directory, failure)
	}
	return free, nil
}
