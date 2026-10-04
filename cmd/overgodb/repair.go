package main

import (
	"context"
	"flag"
	"fmt"

	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
)

// repair is overgodb repair: it replays the store through the canonical
// reader, the one authority on the journal's format. A torn tail, the
// incomplete final frame a dying writer leaves, is truncated when -apply opens
// the store for writing; every other damage -- a complete frame whose
// checksum, version, identity or chain does not hold, or a transaction the
// store cannot apply -- is refused with the reader's own error, since only a
// sealed backup restores committed history.
func repair(flags *flag.FlagSet, args []string) error {
	repo := flags.String("repo", "overgodb-store", "OvergoDB store directory")
	apply := flags.Bool("apply", false, "open the store for writing, which truncates a torn journal tail; default replays it read-only")
	lineage := flags.Bool("lineage", false, "commit the lineage edges committed lifecycle events declare but the store lacks (additive typed-document repair)")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *lineage {
		return reconcileLineage(*repo)
	}
	open := overgodb.OpenReadOnly
	if *apply {
		open = overgodb.Open
	}
	store, err := open(*repo)
	if err != nil {
		return fmt.Errorf("overgodb repair:%s does not replay; restore it from a sealed backup: %w", *repo, err)
	}
	// A verified checkpoint anchors the replay, as it does every open: the
	// active journal past it replays frame by frame.
	head, sequence := store.Head()
	fmt.Printf("journal replays: head=%s sequence=%d from_checkpoint=%t\n", head, sequence, store.SnapshotReplay().Loaded)
	return store.Close()
}

// reconcileLineage repairs the typed-document layer: lifecycle events
// published without their declared lineage edges leave evidence invisible
// to lineage-closure consumers such as compaction's retained set.
func reconcileLineage(repo string) error {
	store, err := overgodb.Open(repo)
	if err != nil {
		return err
	}
	defer store.Close()
	report, err := modelrecipe.ReconcileLifecycleLineage(context.Background(), store)
	if err != nil {
		return err
	}
	fmt.Printf("lineage reconciled: events=%d missing_edges_committed=%d skipped_events=%d\n",
		report.Events, report.MissingEdges, report.SkippedEvents)
	return nil
}
