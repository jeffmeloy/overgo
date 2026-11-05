package main

import (
	"io"
	"strings"
	"testing"
)

// TestTheRewardPilotIsRetired holds training-objective to the reward pilot's retirement: the pilot's one run neither
// gained nor lost (docs/verification/instruction-pilot-run-qwen25-05b.txt), so its path is gone, and -pilot, which
// fixed its acceptance, is a flag the command does not have, as are the flags that went with it.
func TestTheRewardPilotIsRetired(t *testing.T) {
	t.Parallel()
	for _, retired := range []string{"pilot", "pilot-gain", "pilot-seed", "pilot-window", "pilot-tokens", "pilot-draws", "pilot-passes", "pilot-memory"} {
		if err := run([]string{"-" + retired, "1"}, io.Discard); err == nil || !strings.Contains(err.Error(), "flag provided but not defined: -"+retired) {
			t.Errorf("training-objective -%s = %v; want it refused as a flag the command does not have", retired, err)
		}
	}
}
