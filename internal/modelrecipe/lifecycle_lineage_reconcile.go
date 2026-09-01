package modelrecipe

// Lifecycle events published before the re-verification path carried
// event.Lineage() reference their evidence only inside the document body, so
// lineage-closure consumers -- compaction's retained set above all -- cannot
// see the gate/run evidence an active chain depends on. Reconciliation
// derives each event's declared lineage, compares it against the committed
// edges, and commits exactly the missing ones: an additive repair of
// recorded facts, never a rewrite.

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
)

// LifecycleLineageReport counts one reconciliation.
type LifecycleLineageReport struct {
	Events        int
	MissingEdges  int
	SkippedEvents int
}

// ReconcileLifecycleLineage commits the lineage edges every committed
// lifecycle event declares but the store does not hold. Events whose
// referenced artifacts are absent from the store are skipped and counted:
// an edge to an unknown artifact would be refused at commit, and absence is
// a separate finding, not this repair's job.
func ReconcileLifecycleLineage(ctx context.Context, store *overgodb.Store) (LifecycleLineageReport, error) {
	if ctx == nil || store == nil {
		return LifecycleLineageReport{}, errors.New("model recipe: lineage reconciliation requires a store and context")
	}
	report := LifecycleLineageReport{}
	var missing []artifact.Lineage
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{recipe.LifecycleDocumentContract()},
		Order:     overgodb.DocumentOldestFirst,
	}, recipe.ParseLifecycleEvent, func(_ overgodb.DocumentView, event recipe.LifecycleEvent) error {
		report.Events++
		committed, err := store.Parents(ctx, event.ID)
		if err != nil {
			return err
		}
		present := make(map[artifact.ID]bool, len(committed))
		for _, edge := range committed {
			present[edge.Parent] = true
		}
		skipped := false
		for _, edge := range event.Lineage() {
			if present[edge.Parent] {
				continue
			}
			if _, known, err := store.Artifact(ctx, edge.Parent); err != nil {
				return err
			} else if !known {
				skipped = true
				continue
			}
			missing = append(missing, edge)
		}
		if skipped {
			report.SkippedEvents++
		}
		return nil
	})
	if err != nil {
		return report, err
	}
	report.MissingEdges = len(missing)
	if len(missing) == 0 {
		return report, nil
	}
	batch, err := artifact.NewDocumentBatch("recipe/lifecycle-lineage-reconcile", nil, missing, nil)
	if err != nil {
		return report, err
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return report, fmt.Errorf("model recipe: lineage reconciliation commit: %w", err)
	}
	return report, nil
}
