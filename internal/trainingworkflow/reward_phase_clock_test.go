package trainingworkflow

import (
	"testing"

	"overgo/internal/processmeasure"
)

// TestRewardPhaseClockReadsAPhaseShorterThanATick holds the clock a reward
// session times its phases by to reading a span shorter than the system's
// tick. On a small model every phase is that short, and a clock that steps
// by the tick read a whole session's sampling as no time at all whenever no
// phase happened to cross a tick, which the session's own test then refused.
// The span here is as short as the high-resolution counter can tell.
func TestRewardPhaseClockReadsAPhaseShorterThanATick(t *testing.T) {
	t.Parallel()
	began := phaseClock()
	first, err := processmeasure.Counter()
	for now := first; err == nil && now <= first; now, err = processmeasure.Counter() {
	}
	if err != nil {
		t.Fatal(err)
	}
	if elapsed := phaseClock() - began; elapsed <= 0 {
		t.Fatalf("a span the counter saw pass read as %v on the phase clock", elapsed)
	}
}
