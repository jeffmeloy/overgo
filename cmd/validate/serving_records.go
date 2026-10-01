package main

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// An inference model's serving cell holds what the scorecard's serving cells
// read: a measured attempt of its active recipe, recipe measure's run of the
// recipe on the request its verification ran, which times the load beside
// the prefill and decode. A page's request runs on a model already loaded and
// times no load. The attempt binds no commit: the cell runs again when a
// commit moves the inference surface, as every inference cell does, and
// otherwise holds while one measured attempt is recorded.

// servingValidation is the validation whose record is a measured serving
// attempt.
const servingValidation = "serving"

// errUnmeasured is why a serving cell holds no record.
var errUnmeasured = errors.New("no attempt of this recipe timed its load yet")

// measuredRecipes reads the recipes the store holds a measured serving
// attempt of: one that succeeded and timed its load.
func measuredRecipes(ctx context.Context, store *overgodb.Store) (map[artifact.ID]bool, error) {
	measured := map[artifact.ID]bool{}
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvidence, MediaType: runrecord.ServingObservationMediaType, Schema: runrecord.ServingObservationSchema,
		}},
		// Membership is order-independent; any one order reads them all.
		Order: overgodb.DocumentOldestFirst,
	}, runrecord.ParseServingObservation, func(_ overgodb.DocumentView, observation runrecord.ServingObservation) error {
		if observation.Outcome == runrecord.OutcomeSucceeded &&
			slices.ContainsFunc(observation.Phases, func(phase runrecord.PhaseMetric) bool { return phase.Phase == runrecord.PhaseLoad }) {
			measured[observation.Recipe] = true
		}
		return nil
	})
	return measured, err
}
