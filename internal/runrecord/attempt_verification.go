package runrecord

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

func gateFinalizationsForPreparation(
	ctx context.Context,
	store *overgodb.Store,
	preparationID artifact.ID,
) ([]gateLifecycle, error) {
	if ctx == nil || store == nil {
		return nil, errors.New("run record: gate finalizations require the store")
	}
	preparation, err := requireGateLifecycle(ctx, store, preparationID)
	if err != nil {
		return nil, err
	}
	if preparation.State != gatePrepared {
		return nil, errors.New("run record: gate finalizations require a preparation")
	}
	edges, err := store.Children(ctx, preparationID)
	if err != nil {
		return nil, err
	}
	var finalizations []gateLifecycle
	for _, edge := range edges {
		if edge.Parent != preparationID || edge.Relation != artifact.RelationDerivedFrom {
			continue
		}
		lifecycle, typed, candidateErr := readGateLifecycleCandidate(ctx, store, edge.Child)
		if candidateErr != nil {
			return nil, candidateErr
		}
		if !typed {
			continue
		}
		if lifecycle.State != gateFinalized || lifecycle.Preparation == nil || *lifecycle.Preparation != preparationID ||
			lifecycle.TreeKey != preparation.TreeKey || lifecycle.Environment != preparation.Environment ||
			lifecycle.Started != preparation.Started {
			return nil, errors.New("run record: finalization lineage contradicts its preparation")
		}
		finalizations = append(finalizations, lifecycle)
	}
	return canonicalLifecycles(finalizations), nil
}

// GateFinalizationForPreparation returns the sole canonical finalization for a
// preparation. Absence is explicit; ambiguous append-only authority is always
// an error so callers cannot accidentally select one by iteration order.
func GateFinalizationForPreparation(
	ctx context.Context,
	store *overgodb.Store,
	preparationID artifact.ID,
) (gateLifecycle, bool, error) {
	finalizations, err := gateFinalizationsForPreparation(ctx, store, preparationID)
	if err != nil {
		return gateLifecycle{}, false, err
	}
	var finalization gateLifecycle
	found := false
	for _, candidate := range finalizations {
		if found {
			return gateLifecycle{}, false, fmt.Errorf(
				"run record: preparation %s has %d canonical finalizations",
				preparationID, len(finalizations),
			)
		}
		finalization, found = candidate, true
	}
	return finalization, found, nil
}

func readGateLifecycleCandidate(
	ctx context.Context,
	store *overgodb.Store,
	id artifact.ID,
) (gateLifecycle, bool, error) {
	content, found, err := artifact.ReadContent(ctx, store, id)
	if err != nil || !found {
		return gateLifecycle{}, false, err
	}
	if content.Descriptor.ID.Kind() != artifact.KindEvidence || content.Descriptor.MediaType != gateLifecycleMediaType ||
		content.Descriptor.Schema != GateLifecycleSchema {
		return gateLifecycle{}, false, nil
	}
	lifecycle, err := parseGateLifecycle(content.Data)
	if err != nil {
		return gateLifecycle{}, false, err
	}
	if lifecycle.ID != id {
		return gateLifecycle{}, false, errors.New("run record: gate lifecycle candidate identity mismatch")
	}
	return lifecycle, true, nil
}

func canonicalLifecycles(values []gateLifecycle) []gateLifecycle {
	slices.SortFunc(values, func(left, right gateLifecycle) int {
		return artifact.CompareID(left.ID, right.ID)
	})
	result := values[:0]
	for _, value := range values {
		if len(result) == 0 || result[len(result)-1].ID != value.ID {
			result = append(result, value)
		}
	}
	return result
}
