package main

import (
	"slices"
	"testing"
)

// TestADeviceRunRunsEverySelectedTestAfresh holds the device lane to running each test it selects, never replaying a
// cached pass, since go test keys no file outside the module, the driver and the data root's models among them: the
// correctness run and the measurement run each carry -count=1 once. The lane prints go test's line for each package
// that passed and nothing else of the stream, so the landing counts them.
func TestADeviceRunRunsEverySelectedTestAfresh(t *testing.T) {
	t.Parallel()
	once := func(command []string) bool {
		return len(slices.DeleteFunc(slices.Clone(command), func(argument string) bool { return argument != freshRun })) == 1
	}
	step := deviceTestStep("./pkg")
	if !once(step) {
		t.Fatalf("a correctness run is %v, want it run afresh once", step)
	}
	if _, measurement := splitTestStep(step, []string{"TestMeasured"}); !once(measurement) {
		t.Fatalf("a measurement run is %v, want it run afresh once", measurement)
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
