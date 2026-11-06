package main

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"overgo/internal/processcontrol"
	"overgo/internal/testcheck"
)

// TestTheRewardPilotIsRetired holds train to the reward pilot's retirement: the pilot's one run neither gained nor
// lost (docs/verification/instruction-pilot-run-qwen25-05b.txt), so its path is gone, and -pilot, with the flag that
// kept its checkpoints, is refused as a flag train does not have before anything else is read.
func TestTheRewardPilotIsRetired(t *testing.T) {
	t.Parallel()
	for _, retired := range []string{"pilot", "keep-checkpoints"} {
		var refusal bytes.Buffer
		receipt := testcheck.Must(processcontrol.Run(t.Context(), processcontrol.Command{
			Path: os.Args[0], Args: []string{"-" + retired, "evidence"}, Stderr: &refusal,
			Env: append(os.Environ(), commandChildEnvironment+"="+childStandIn),
		}))(t)
		if receipt.ExitCode == 0 || !strings.Contains(refusal.String(), "flag provided but not defined: -"+retired) {
			t.Errorf("train -%s ended with status %d and %q; want it refused as a flag train does not have", retired, receipt.ExitCode, refusal.String())
		}
	}
}
