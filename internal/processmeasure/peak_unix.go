//go:build unix

package processmeasure

import (
	"runtime"
	"syscall"
)

// kibibyte is the bytes in one unit of getrusage's maximum resident set, where the kernel counts it in kibibytes.
const kibibyte = 1 << 10

// SelfPeakWorkingSet reports the calling process's own peak resident set, the RSS high-water, as getrusage gives it:
// in kibibytes on Linux and the BSDs, in bytes on Darwin.
func SelfPeakWorkingSet() (uint64, error) {
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		return 0, err
	}
	if runtime.GOOS == "darwin" {
		return uint64(usage.Maxrss), nil
	}
	return uint64(usage.Maxrss) * kibibyte, nil
}
