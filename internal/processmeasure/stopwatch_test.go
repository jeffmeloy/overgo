package processmeasure

import (
	"errors"
	"math"
	"testing"
)

// Native counter brackets test the clock binding without a latency deadline.
func TestStopwatchReadsCounter(t *testing.T) {
	before, err := Counter()
	if err != nil {
		t.Fatal(err)
	}
	stopwatch := NewStopwatch()
	after, err := Counter()
	if err != nil {
		t.Fatal(err)
	}
	if stopwatch.err != nil || stopwatch.started < before || stopwatch.started > after {
		t.Fatalf("start %v outside counter bounds [%v, %v]: %v", stopwatch.started, before, after, stopwatch.err)
	}
	elapsed, err := stopwatch.Elapsed()
	if err != nil {
		t.Fatal(err)
	}
	end, err := Counter()
	if err != nil {
		t.Fatal(err)
	}
	if elapsed < uint64(after-stopwatch.started) || elapsed > uint64(end-stopwatch.started) {
		t.Fatalf("elapsed %d outside native subtraction bounds [%d, %d]", elapsed, after-stopwatch.started, end-stopwatch.started)
	}
}

func TestStopwatchRetainsFailure(t *testing.T) {
	failure := errors.New("counter unavailable")
	failed := Stopwatch{set: true, err: failure}
	if _, err := failed.Elapsed(); !errors.Is(err, failure) {
		t.Fatalf("start failure lost: %v", err)
	}
	if _, err := (Stopwatch{set: true, started: math.MaxInt64}).Elapsed(); err == nil {
		t.Fatal("backwards counter accepted")
	}
	var walls Walls
	walls.Elapsed(failed)
	walls.Elapsed(NewStopwatch())
	walls.Elapsed(Stopwatch{})
	if !errors.Is(walls.Err, failure) {
		t.Fatalf("first failure lost: %v", walls.Err)
	}
}

func TestStopwatchElapsedIsMonotone(t *testing.T) {
	stopwatch := NewStopwatch()
	first, err := stopwatch.Elapsed()
	if err != nil {
		t.Fatal(err)
	}
	second, err := stopwatch.Elapsed()
	if err != nil {
		t.Fatal(err)
	}
	if second < first {
		t.Fatalf("elapsed moved backwards: %d then %d", first, second)
	}
}

func TestStopwatchRefusesAnUnstartedRead(t *testing.T) {
	if _, err := (Stopwatch{}).Elapsed(); err == nil {
		t.Fatal("unstarted stopwatch reported an elapsed wall")
	}
	var walls Walls
	if walls.Elapsed(Stopwatch{}) != 0 || walls.Err == nil {
		t.Fatal("walls did not keep the failed read")
	}
	var clean Walls
	clean.Elapsed(NewStopwatch())
	if clean.Err != nil {
		t.Fatal(clean.Err)
	}
}
