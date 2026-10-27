package executor

import (
	"testing"
	"time"
)

// TestHostSlotLimitTradesACopyForTheCPU: an expert is the CPU's while its call's fixed time and a slot's for each slot
// cost less than its copy over the link, never past the slots the staging reserves, and the reserve where nothing was
// measured. Flash-Next's 2.33 MB expert on this PC's costs (13.4 GB/s, 27 us a call, 10.2 us a slot) is the CPU's up to
// 14 slots; the constant it replaces held 32.
func TestHostSlotLimitTradesACopyForTheCPU(t *testing.T) {
	t.Parallel()
	const expertBytes = 2_330_000
	measured := MachineCosts{LinkBytesPerSecond: 13.4e9, CPUFixed: 27 * time.Microsecond, CPUSlot: 10200 * time.Nanosecond}
	for _, row := range []struct {
		name  string
		costs MachineCosts
		bytes uint64
		want  uint32
	}{
		{"this PC", measured, expertBytes, 14},
		{"a copy cheaper than one call", measured, 64 << 10, 0},
		{"a slow link, past the reserve", MachineCosts{LinkBytesPerSecond: 1e9, CPUFixed: measured.CPUFixed, CPUSlot: measured.CPUSlot}, expertBytes, hostExpertSlotsMost},
		{"nothing measured", MachineCosts{}, expertBytes, hostExpertSlotsUnmeasured},
	} {
		if got := row.costs.hostSlotLimit(row.bytes); got != row.want {
			t.Errorf("%s: an expert of %d bytes is the CPU's up to %d slots, want %d", row.name, row.bytes, got, row.want)
		}
	}
}
