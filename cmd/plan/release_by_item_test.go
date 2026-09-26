package main

import (
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/worklease"
)

// TestReleaseClaimByItem holds -release-claim to accepting the item a worker
// holds instead of the claim id copied out of plan -next -json: the item's
// task alias names the claim, another worker is still refused, and an item
// nothing claims names no claim.
func TestReleaseClaimByItem(t *testing.T) {
	t.Setenv(plan.AutomationRoleEnvironment, worklease.UnassignedRole)
	t.Setenv(plan.AutomationWorkerEnvironment, "")
	root := initializePlanTestRepository(t, mutationPlan(t, "claimed task"))
	t.Chdir(root)
	captureStdout(t, func() {
		if err := run(cli{prompt: true, worker: "session-one", retireLegacyLeases: noLegacyLeaseRetirement}, nil); err != nil {
			t.Fatal(err)
		}
	})
	var released strings.Builder
	if err := releaseDispatchClaim(root, "absent", "session-one", "handoff", &released); err == nil || !strings.Contains(err.Error(), "names 0 claims") {
		t.Fatalf("an unclaimed item released: %v", err)
	}
	if err := releaseDispatchClaim(root, "row", "session-two", "handoff", &released); err == nil {
		t.Fatal("another worker released the claim by its item")
	}
	if err := releaseDispatchClaim(root, "row/do", "session-one", "handoff", &released); err != nil || !strings.Contains(released.String(), "state=released") {
		t.Fatalf("release by item = %v: %s", err, released.String())
	}
}
