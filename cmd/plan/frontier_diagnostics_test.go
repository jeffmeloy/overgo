package main

import (
	"strings"
	"testing"
	"time"

	"overgo/internal/plan"
	"overgo/internal/worklease"
)

// TestFrontierLeaseDiagnostics drives the exact classifier printReadyFrontier
// uses, so the report keeps the ready frontier authoritative and separate from
// advisory lease health: a lease claiming a row outside the frontier and an
// expired lease are advisory, two leases on one ready row are a blocking live
// conflict, and a valid isolated lease raises nothing.
func TestFrontierLeaseDiagnostics(t *testing.T) {
	t.Parallel()
	frontier := []plan.Ref{{Item: "alpha", Step: "do"}, {Item: "beta", Step: "do"}}
	past := time.Now().Add(-time.Hour).UTC().Format(time.RFC3339Nano)
	leases := []worklease.Lease{
		{Worktree: "C:/w/stale", Task: "retired/row", Worker: "stale", Role: "master-lead"},
		{Worktree: "C:/w/alpha", Task: "alpha/do", Worker: "alice", Role: "master-lead"},
		{Worktree: "C:/w/alpha2", Task: "alpha/do", Worker: "bob", Role: "developer"},
		{Worktree: "C:/w/expired", Task: "beta/do", Worker: "carol", Role: "reviewer", ExpiresAt: past},
		{Worktree: "C:/w/beta", Task: "beta/do", Worker: "dave", Role: "auditor"},
	}

	diagnostics := frontierLeaseDiagnostics(frontier, leases, time.Now())
	byWorktree := make(map[string]leaseDiagnostic, len(diagnostics))
	for _, diagnostic := range diagnostics {
		if _, seen := byWorktree[diagnostic.worktree]; seen {
			t.Fatalf("duplicate diagnostic for %s", diagnostic.worktree)
		}
		byWorktree[diagnostic.worktree] = diagnostic
	}

	assert := func(worktree string, wantBlocking bool, wantReason string) {
		diagnostic, ok := byWorktree[worktree]
		if !ok {
			t.Fatalf("no diagnostic for %s; got %v", worktree, diagnostics)
		}
		if diagnostic.blocking != wantBlocking || !strings.Contains(diagnostic.reason, wantReason) {
			t.Fatalf("%s diagnostic = {blocking:%v reason:%q}, want blocking=%v reason~=%q", worktree, diagnostic.blocking, diagnostic.reason, wantBlocking, wantReason)
		}
	}
	assert("C:/w/stale", false, "outside the ready frontier")
	assert("C:/w/alpha2", true, "shares ready row")
	assert("C:/w/expired", false, "expired")
	for _, valid := range []string{"C:/w/alpha", "C:/w/beta"} {
		if _, raised := byWorktree[valid]; raised {
			t.Fatalf("valid isolated lease %s raised a diagnostic: %v", valid, byWorktree[valid])
		}
	}
}
