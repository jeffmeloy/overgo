package landing

import (
	"slices"
	"testing"
)

// TestNoCycleLaneRunsUnderGoTestsClock holds every cycle lane whose command is a go test to no clock of go test's: a lane's
// tests wait their card turns behind a render, and go test's default ten minutes ended the audio lane in the middle of
// TestASRResidentReference on 2026-11-14, its command alone among the lanes naming no -timeout (no-clocks).
func TestNoCycleLaneRunsUnderGoTestsClock(t *testing.T) {
	t.Parallel()
	for _, lane := range cycleLanes {
		command := lane.command("")
		if len(command) == 0 || command[0] != "test" {
			continue
		}
		if !slices.Contains(command, "-timeout=0") {
			t.Errorf("the cycle lane %s runs go test %v under go test's default clock; want -timeout=0", lane.name, command[1:])
		}
	}
}
