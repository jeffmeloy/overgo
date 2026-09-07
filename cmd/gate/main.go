// Command gate owns flags and composition only: every phase of the one
// gate transaction path lives in internal/gate.
package main

import (
	"flag"

	"overgo/internal/clioptions"
	"overgo/internal/gate"
	"overgo/internal/runrecord"
)

func main() {
	clioptions.MainNamed("gate", run)
}

func run() error {
	messageFile := flag.String("message-file", "", "commit message file (required; never -m: the shell eats backticks)")
	pathsCSV := flag.String("paths", "", "comma-separated repo-relative paths this commit ships (required unless -merge)")
	storePath := flag.String("store", gate.StorePath, "canonical OvergoDB store directory")
	merge := flag.Bool("merge", false, "finalize an in-progress merge: derive the shipped paths from the staged merge set and let the commit record both parents (stage it first with git merge --no-ff --no-commit <branch>)")
	planProjection := flag.String("plan-projection", "", "with -merge only: explicit target-plan projection (first-parent-target); empty keeps semantic union")
	mergeSourceStore := flag.String("merge-source-store", "", "with a first-parent-target merge: canonical OvergoDB store of the registered source worktree")
	planRef := flag.String("plan", "", "item/step this commit serves; MUST equal the current open step, including for -merge. Off-plan commits are refused.")
	reconcile := flag.Bool("reconcile", false, "finalize the deterministic OvergoDB batch in tmp/gate_debt.json")
	recordFailure := flag.Bool("record-failure", false, "recover an unbatchable post-commit record as a typed failed finalization")
	recoverPreparation := flag.String("preparation", "", "exact stale preparation ID -record-failure closes when recovery debt is ambiguous")
	recoverInterrupted := flag.Bool("recover-interrupted", false, "roll back an interrupted gate commit and cancel its prepared store lifecycle")
	admitReview := flag.String("admit-review", "", "read-only: admit a OvergoDB review-verdict ID against the current HEAD")
	watchdog := flag.Bool("watchdog", false, "print typed JSON liveness from tmp/gate_lifecycle.json")
	inspectPlan := flag.Bool("inspect-plan", false, "non-committing: construct and print the exact manifest-bound verification plan without executing checks; may write temporary Git object/index state")
	preflight := flag.Bool("preflight", false, "diagnose working-tree validation failures without admission, store repair or acceptance credit")
	staleAfter := flag.Duration("stale-after", runrecord.DefaultHeartbeatStaleAfter, "heartbeat age classified stale by -watchdog")
	flag.Parse()
	return gate.Run(gate.Options{
		MessageFile: *messageFile, PathsCSV: *pathsCSV, StorePath: *storePath,
		Merge: *merge, PlanProjection: *planProjection, MergeSourceStore: *mergeSourceStore,
		PlanRef: *planRef, Reconcile: *reconcile, RecordFailure: *recordFailure,
		RecoverPreparation: *recoverPreparation, RecoverInterrupted: *recoverInterrupted,
		AdmitReview: *admitReview, Watchdog: *watchdog, InspectPlan: *inspectPlan, Preflight: *preflight,
		StaleAfter: *staleAfter,
	})
}
