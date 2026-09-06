package loop

import (
	"math"
	"testing"
	"time"
)

// TestRequeueBackoffAndIdleIntervals pins: the requeue delay doubles per
// attempt from the base, jitter adds at most the declared fraction, the
// maximum clamps it, an enormous attempt count cannot overflow, a zero
// declaration delays nothing, invalid declarations are refused; the idle
// interval doubles on empty ticks, clamps at its maximum and resets when a
// tick finds work.
func TestRequeueBackoffAndIdleIntervals(t *testing.T) {
	backoff := Backoff{Base: time.Second, Max: time.Minute, Jitter: 0.5}
	if err := backoff.Validate(); err != nil {
		t.Fatal(err)
	}
	for attempt, want := range map[int]time.Duration{0: 0, 1: time.Second, 2: 2 * time.Second, 3: 4 * time.Second, 6: 32 * time.Second, 7: time.Minute, 8: time.Minute} {
		if got := backoff.Delay(attempt, 0); got != want {
			t.Fatalf("attempt %d delay = %s, want %s", attempt, got, want)
		}
	}
	if got := backoff.Delay(2, 1); got != 3*time.Second {
		t.Fatalf("full jitter on attempt 2 = %s, want 3s", got)
	}
	if got := backoff.Delay(2, 0.5); got != 2500*time.Millisecond {
		t.Fatalf("half jitter on attempt 2 = %s, want 2.5s", got)
	}
	for _, attempt := range []int{40, 63, 64, 1000, math.MaxInt} {
		if got := backoff.Delay(attempt, 1); got != time.Minute {
			t.Fatalf("attempt %d delay = %s, want the maximum without overflow", attempt, got)
		}
	}
	if got := (Backoff{}).Delay(5, 1); got != 0 {
		t.Fatalf("zero declaration delayed %s", got)
	}
	for _, invalid := range []Backoff{{Base: -time.Second, Max: time.Minute}, {Base: time.Minute, Max: time.Second}, {Base: time.Second, Max: time.Minute, Jitter: 1.5}} {
		if err := invalid.Validate(); err == nil {
			t.Fatalf("invalid backoff %+v validated", invalid)
		}
	}

	idle, err := NewIdleInterval(time.Second, 8*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	var observed []time.Duration
	for _, found := range []bool{false, false, false, false, false, true, false} {
		observed = append(observed, idle.Next(found))
	}
	want := []time.Duration{2 * time.Second, 4 * time.Second, 8 * time.Second, 8 * time.Second, 8 * time.Second, time.Second, 2 * time.Second}
	for index := range want {
		if observed[index] != want[index] {
			t.Fatalf("idle intervals = %v, want %v", observed, want)
		}
	}
	if _, err := NewIdleInterval(0, time.Second); err == nil {
		t.Fatal("a zero idle base was accepted")
	}
}
