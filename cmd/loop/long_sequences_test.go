package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/landing"
	"overgo/internal/testcheck"
)

// TestChecksRunTheLongSequences holds a full run's checks to the long generated sequences a commit's checks leave
// out: the lanes phase runs the cycle's lanes (measurePhases, validate -run -lanes), and one of them runs the store's
// long lifecycles under the cycle's tags, which compile the test that runs them and which no commit's checks build.
func TestChecksRunTheLongSequences(t *testing.T) {
	t.Parallel()
	at := slices.IndexFunc(landing.CycleLanes(), func(lane landing.CycleLane) bool {
		return slices.Contains(lane.Arguments, "^TestLongStoreLifecycles$") && slices.Contains(lane.Arguments, "./internal/overgodb")
	})
	if at < 0 {
		t.Fatal("no lane of the cycle runs the store's long lifecycles")
	}
	arguments := landing.CycleLanes()[at].Arguments
	if tags := slices.Index(arguments, "-tags"); tags < 0 || tags+1 >= len(arguments) || arguments[tags+1] != landing.CycleTags {
		t.Fatalf("the lane runs %v, want it under the cycle's tags %s", arguments, landing.CycleTags)
	}
	source, err := os.ReadFile(filepath.Join("..", "..", "internal", "overgodb", "long_sequences_test.go"))
	testcheck.NoError(t, err)
	if !strings.Contains(string(source), "//go:build integration") || !strings.Contains(string(source), "func TestLongStoreLifecycles(") {
		t.Fatal("the store's long lifecycles are not a test the cycle's tags alone compile")
	}
}
