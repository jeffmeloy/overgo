package main

import (
	"strings"
	"testing"

	"overgo/internal/loop"
)

// The driver's world lands a worker's passed step itself (TestTheDriverLandsAWorkersPassedStep).
var _ loop.LandingWorld = (*execWorld)(nil)

// TestTheDriverLandsAWorkersPassedStep holds the worker's prompt to what its tools can do (internal/loop holds the
// driver's side): cmd/loop -tools gives a worker nine tools, none of which lands, so the prompt asks it to work its step
// until the check passes and says the driver lands it, and names no landing command for the worker to run.
func TestTheDriverLandsAWorkersPassedStep(t *testing.T) {
	t.Parallel()
	if strings.Contains(defaultPrompt, "-land") || !strings.Contains(defaultPrompt, "check passes") || !strings.Contains(defaultPrompt, "the driver lands it") {
		t.Fatalf("the worker's prompt reads %q; want it to work until the check passes and leave the landing to the driver", defaultPrompt)
	}
}
