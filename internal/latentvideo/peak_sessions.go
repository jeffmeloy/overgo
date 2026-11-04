package latentvideo

import (
	"errors"

	"overgo/internal/cuda/driver"
)

// memoryMeter is a device session's allocation ledger.
type memoryMeter interface {
	MemoryStats() (driver.MemoryStats, error)
}

// peakAcrossSessions is a runtime's device high-water mark: its sessions are resident together, so each session's
// peak counts with what the others hold now, and the largest such sum is the mark. A session whose ledger cannot be
// read is left out, and the reads' errors return only when no session could be read.
func peakAcrossSessions(sessions ...memoryMeter) (uint64, error) {
	var stats []driver.MemoryStats
	var errs []error
	for _, session := range sessions {
		read, err := session.MemoryStats()
		if err != nil {
			errs = append(errs, err)
			continue
		}
		stats = append(stats, read)
	}
	var held, peak uint64
	for _, read := range stats {
		held += read.CurrentBytes
	}
	for _, read := range stats {
		peak = max(peak, held-read.CurrentBytes+read.PeakBytes)
	}
	if len(stats) == 0 && len(errs) > 0 {
		return 0, errors.Join(errs...)
	}
	return peak, nil
}
