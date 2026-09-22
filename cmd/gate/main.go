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
	pathsCSV := flag.String("paths", "", "comma-separated repo-relative paths this commit ships; empty derives the dirty tree, sorted, refusing an untracked top-level entry (-merge ships the staged set)")
	storePath := flag.String("store", gate.StorePath, "canonical OvergoDB store directory")
	merge := flag.Bool("merge", false, "finalize an in-progress merge: derive the shipped paths from the staged merge set and let the commit record both parents (stage it first with git merge --no-ff --no-commit <branch>)")
	planProjection := flag.String("plan-projection", "", "with -merge only: explicit target-plan projection (first-parent-target); empty keeps semantic union")
	mergeSourceStore := flag.String("merge-source-store", "", "with a first-parent-target merge: canonical OvergoDB store of the registered source worktree")
	checkpoint := flag.String("checkpoint", "", "with -plan and -paths: publish this checkpoint of the step's verification batch with the affected owner tests and stop before the cumulative suites and the commit")
	planRef := flag.String("plan", "", "item/step this commit serves; MUST equal the current open step, including for -merge. Off-plan commits are refused.")
	reconcile := flag.Bool("reconcile", false, "finalize the deterministic OvergoDB batch the gate left owing in its runtime directory")
	recordFailure := flag.Bool("record-failure", false, "recover an unbatchable post-commit record as a typed failed finalization")
	recoverPreparation := flag.String("preparation", "", "exact stale preparation ID -record-failure closes when recovery debt is ambiguous")
	recoverInterrupted := flag.Bool("recover-interrupted", false, "roll back an interrupted gate commit and cancel its prepared store lifecycle")
	admitReview := flag.String("admit-review", "", "read-only: admit a OvergoDB review-verdict ID against the current HEAD")
	watchdog := flag.Bool("watchdog", false, "print typed JSON liveness from the lifecycle record in the gate runtime directory")
	packageReceipt := flag.String("package-receipt", "", "read-only: print the gate receipt for PACKAGE[@complete|short] at the working tree, keyed to its exact inputs")
	lanes := flag.Bool("lanes", false, "run the deferred lanes of the last landed commit and record their outcome; a successful gate starts this itself")
	inspectPlan := flag.Bool("inspect-plan", false, "non-committing: construct and print the exact manifest-bound verification plan without executing checks; may write temporary Git object/index state")
	preflight := flag.Bool("preflight", false, "apply the derived-file repairs to the working tree, then diagnose its changes without admission, store repair or acceptance credit; -paths cmd,internal checks all Go sources")
	staleAfter := flag.Duration("stale-after", runrecord.DefaultHeartbeatStaleAfter, "heartbeat age classified stale by -watchdog")
	flag.Parse()
	return gate.Run(gate.Options{
		MessageFile: *messageFile, PathsCSV: *pathsCSV, StorePath: *storePath,
		Merge: *merge, PlanProjection: *planProjection, MergeSourceStore: *mergeSourceStore,
		PlanRef: *planRef, Checkpoint: *checkpoint, Reconcile: *reconcile, RecordFailure: *recordFailure,
		RecoverPreparation: *recoverPreparation, RecoverInterrupted: *recoverInterrupted,
		AdmitReview: *admitReview, Watchdog: *watchdog, PackageReceipt: *packageReceipt, InspectPlan: *inspectPlan, Preflight: *preflight,
		Lanes: *lanes, StaleAfter: *staleAfter,
	})
}
