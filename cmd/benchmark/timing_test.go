package main

import (
	"errors"
	"testing"
	"time"
)

func TestGenerationTimingCounter(t *testing.T) {
	for _, test := range []struct {
		name       string
		readings   []uint64
		tokens     int
		ttft, wall time.Duration
	}{
		{name: "single or batch first output", readings: []uint64{7, 19}, tokens: 3, ttft: 7, wall: 19},
		{name: "zero first output stays first", readings: []uint64{0, 19}, tokens: 3, wall: 19},
		{name: "no output", readings: []uint64{19}, ttft: 19, wall: 19},
		{name: "zero interval is not floored", readings: []uint64{0, 0}, tokens: 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			reads := 0
			timing := generationTiming{elapsed: func() (uint64, error) {
				if reads >= len(test.readings) {
					t.Fatal("read counter after the first output before finish")
				}
				value := test.readings[reads]
				reads++
				return value, nil
			}}
			for range test.tokens {
				if err := timing.token(); err != nil {
					t.Fatal(err)
				}
			}
			wall, ttft, decode, err := timing.finish()
			if err != nil || wall != test.wall || ttft != test.ttft || decode != test.wall-test.ttft || reads != len(test.readings) {
				t.Fatalf("wall=%v ttft=%v decode=%v reads=%d err=%v", wall, ttft, decode, reads, err)
			}
		})
	}
}

func TestGenerationTimingMeasurementFailure(t *testing.T) {
	failure := errors.New("counter read failed")
	timing := generationTiming{elapsed: func() (uint64, error) { return 0, failure }}
	if err := timing.token(); !errors.Is(err, failure) || timing.hasToken {
		t.Fatalf("failed first token: err=%v present=%v", err, timing.hasToken)
	}
	if _, _, _, err := timing.finish(); !errors.Is(err, failure) {
		t.Fatalf("failed finish: %v", err)
	}
	timing.firstToken, timing.hasToken = 7, true
	if _, _, _, err := timing.finish(); !errors.Is(err, failure) {
		t.Fatalf("failed finish after output: %v", err)
	}
	timing.elapsed = func() (uint64, error) { return timing.firstToken - 1, nil }
	if _, _, _, err := timing.finish(); err == nil {
		t.Fatal("accepted backwards counter between first output and finish")
	}
}
