package latentvideo

import (
	"errors"
	"testing"

	"overgo/internal/cuda/driver"
)

// meter is a session ledger that reads as given.
type meter struct {
	stats driver.MemoryStats
	err   error
}

func (m meter) MemoryStats() (driver.MemoryStats, error) { return m.stats, m.err }

// TestPeakAcrossSessionsCountsWhatTheOthersHold reads a media runtime's sessions as resident together: each session's
// peak counts with what the others hold now, and the largest sum is the runtime's mark, above any one session's peak.
// A session whose ledger cannot be read is left out, and only no reading at all is an error.
func TestPeakAcrossSessionsCountsWhatTheOthersHold(t *testing.T) {
	t.Parallel()
	denoiser := meter{stats: driver.MemoryStats{CurrentBytes: 600, PeakBytes: 900}}
	decoder := meter{stats: driver.MemoryStats{CurrentBytes: 300, PeakBytes: 700}}
	codec := meter{stats: driver.MemoryStats{CurrentBytes: 100, PeakBytes: 100}}
	broken := meter{err: errors.New("ledger unreadable")}
	for _, check := range []struct {
		sessions []memoryMeter
		peak     uint64
	}{
		// The decoder's peak of 700 beside the denoiser's 600 and the codec's 100 is 1,400, above the denoiser's 900 with
		// 400 beside it, and above either peak alone.
		{[]memoryMeter{denoiser, decoder, codec}, 1400},
		{[]memoryMeter{denoiser, decoder}, 1300},
		{[]memoryMeter{denoiser, broken}, 900},
	} {
		peak, err := peakAcrossSessions(check.sessions...)
		if err != nil || peak != check.peak {
			t.Errorf("peak of %d sessions = %d, %v, want %d", len(check.sessions), peak, err, check.peak)
		}
	}
	if _, err := peakAcrossSessions(broken); err == nil {
		t.Error("a runtime no session of which could be read reported a peak")
	}
}
