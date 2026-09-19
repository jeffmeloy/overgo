package main

import (
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/worklease"
)

// TestPreparedMergePortability drives the production source-store resolver and
// the -merge-source-store CLI wiring, not a reimplementation: a prepared merge
// is portable across registered worktrees because an explicit selector names
// the source store directly, multiple checkouts at one commit are reported as
// ambiguous rather than silently resolved (their store authorities can differ),
// a unique candidate or branch still resolves on its own, and the selector is
// rejected outside -prepare-merge.
func TestPreparedMergePortability(t *testing.T) {
	const snapshot = "0123456789abcdef0123456789abcdef01234567"
	twoAtOneCommit := []byte(
		"worktree C:/first\x00HEAD " + snapshot + "\x00branch refs/heads/topic\x00\x00" +
			"worktree C:/second\x00HEAD " + snapshot + "\x00branch refs/heads/other\x00\x00",
	)
	store := func(worktree string) string {
		return filepath.Join(filepath.FromSlash(worktree), "overgodb-store")
	}
	always := func(string) bool { return true }

	t.Run("selector picks one registered worktree", func(t *testing.T) {
		want := store("C:/second")
		got, err := sourceOvergoDBFromPorcelain(twoAtOneCommit, snapshot, "", "C:/second/overgodb-store", always)
		if err != nil || got != want {
			t.Fatalf("selector resolution = %q, %v; want %q", got, err, want)
		}
	})

	t.Run("multiple checkouts without a selector are reported", func(t *testing.T) {
		_, err := sourceOvergoDBFromPorcelain(twoAtOneCommit, snapshot, "", "", always)
		if err == nil || !strings.Contains(err.Error(), "-merge-source-store") {
			t.Fatalf("ambiguity error = %v; want guidance to -merge-source-store", err)
		}
	})

	t.Run("selector must name a candidate", func(t *testing.T) {
		_, err := sourceOvergoDBFromPorcelain(twoAtOneCommit, snapshot, "", "C:/absent/overgodb-store", always)
		if err == nil || !strings.Contains(err.Error(), "not a registered OvergoDB worktree") {
			t.Fatalf("unmatched selector error = %v", err)
		}
	})

	t.Run("a unique candidate resolves without a selector", func(t *testing.T) {
		one := []byte("worktree C:/only\x00HEAD " + snapshot + "\x00detached\x00\x00")
		got, err := sourceOvergoDBFromPorcelain(one, snapshot, "", "", always)
		if err != nil || got != store("C:/only") {
			t.Fatalf("unique resolution = %q, %v", got, err)
		}
	})

	t.Run("selector overrides an otherwise unique branch match", func(t *testing.T) {
		got, err := sourceOvergoDBFromPorcelain(twoAtOneCommit, snapshot, "refs/heads/topic", "C:/second/overgodb-store", always)
		if err != nil || got != store("C:/second") {
			t.Fatalf("selector did not override branch preference = %q, %v", got, err)
		}
	})

	t.Run("selector is rejected outside prepare-merge", func(t *testing.T) {
		t.Setenv(plan.AutomationRoleEnvironment, worklease.UnassignedRole)
		t.Setenv(plan.AutomationWorkerEnvironment, "")
		root := initializePlanTestRepository(t, mutationPlan(t, "Portable merge"))
		t.Chdir(root)
		err := run(cli{mergeSourceStore: store("C:/first"), retireLegacyLeases: noLegacyLeaseRetirement}, nil)
		if err == nil || !strings.Contains(err.Error(), "requires -prepare-merge") {
			t.Fatalf("selector accepted without -prepare-merge: %v", err)
		}
	})
}
