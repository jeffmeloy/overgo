//go:build !windows

package testprocess

import (
	"errors"
	"os"
)

// peakHandle would retain a child's memory counters across its exit.
type peakHandle struct{}

func retainPeak(*os.Process) (peakHandle, error) {
	return peakHandle{}, errors.New("process peak measurement is unavailable on this platform")
}

func (peakHandle) read() (uint64, error) {
	return 0, errors.New("process peak measurement is unavailable on this platform")
}

func (peakHandle) close() error { return nil }
