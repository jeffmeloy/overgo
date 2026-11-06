package main

import (
	"slices"
	"testing"
)

// TestADeviceRunReplaysWhatNothingChanged holds the device lane to go test's cache: a correctness run carries no
// -count=1, so go test replays a pass whose binary, files and environment are unchanged, while -fresh runs every
// selected test again and a measurement always runs; and the lane prints go test's line for each package that passed,
// replayed or run, and nothing else of the stream, so the landing counts them.
//
// Serial: it sets the lane's -fresh flag, which every step it builds reads.
func TestADeviceRunReplaysWhatNothingChanged(t *testing.T) {
	if step := deviceTestStep("./internal/cuda/kernel"); slices.Contains(step, freshRun) {
		t.Fatalf("a correctness run is %v, want go test's cache to decide it", step)
	}
	if _, measurement := splitTestStep(deviceTestStep("./pkg"), []string{"TestMeasured"}); !slices.Contains(measurement, freshRun) {
		t.Fatalf("a measurement run is %v, want it run whatever the cache holds", measurement)
	}
	*freshFlag = true
	defer func() { *freshFlag = false }()
	if step := deviceTestStep("./internal/cuda/kernel"); !slices.Contains(step, freshRun) {
		t.Fatalf("a run under -fresh is %v, want every test run again", step)
	}
	_, measurement := splitTestStep(deviceTestStep("./pkg"), []string{"TestMeasured"})
	if fresh := slices.DeleteFunc(slices.Clone(measurement), func(argument string) bool { return argument != freshRun }); len(fresh) != 1 {
		t.Fatalf("a measurement run under -fresh is %v, want its flag once", measurement)
	}
	const stream = `{"Action":"start","Package":"overgo/a"}
{"Action":"output","Package":"overgo/a","Output":"ok  \tovergo/a\t(cached)\n"}
{"Action":"pass","Package":"overgo/a"}
{"Action":"output","Package":"overgo/b","Test":"TestB","Output":"--- PASS: TestB (0.01s)\n"}
{"Action":"output","Package":"overgo/b","Output":"ok  \tovergo/b\t1.250s\n"}
{"Action":"output","Package":"overgo/c","Output":"FAIL\tovergo/c\t0.100s\n"}
not json
`
	if got, want := passedPackageLines(stream), "ok  \tovergo/a\t(cached)\nok  \tovergo/b\t1.250s\n"; got != want {
		t.Fatalf("the lane prints %q of the stream, want the passed packages' lines %q", got, want)
	}
}
