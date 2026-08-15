package main

import (
	"testing"

	"overgo/internal/runrecord"
)

func TestNonOverlappingConfirmation(t *testing.T) {
	latest := runrecord.Advisory{WindowStart: 20, WindowEnd: 22, LatestSequence: 23}
	for _, test := range []struct {
		name  string
		prior runrecord.Advisory
		want  bool
	}{
		{name: "disjoint", prior: runrecord.Advisory{WindowStart: 16, WindowEnd: 18, LatestSequence: 19}, want: true},
		{name: "shared boundary", prior: runrecord.Advisory{WindowStart: 17, WindowEnd: 19, LatestSequence: 20}},
		{name: "overlapping", prior: runrecord.Advisory{WindowStart: 18, WindowEnd: 20, LatestSequence: 21}},
		{name: "same latest", prior: latest},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := nonOverlappingConfirmation(test.prior, latest); got != test.want {
				t.Fatalf("nonOverlappingConfirmation() = %v, want %v", got, test.want)
			}
		})
	}
}
