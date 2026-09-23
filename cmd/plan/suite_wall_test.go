package main

import (
	"testing"

	"overgo/internal/testutil"
)

// TestCommandSuiteRunsInParallel holds every top-level test of the plan
// command to running in parallel unless it changes process-wide state. The
// suite measured 56 s in a gate's test phase (review candidate on 5a9b22ea)
// with 55 of its 60 tests serial; each command test drives run with its own
// arguments, so nothing but process-wide state needs the serial phase.
func TestCommandSuiteRunsInParallel(t *testing.T) {
	testutil.RequireParallel(t)
}
