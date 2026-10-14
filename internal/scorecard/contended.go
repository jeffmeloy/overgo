package scorecard

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// contendedMediaType and contendedSchema identify the record of a serving
// attempt that was measured beside other work.
const (
	contendedMediaType = "application/vnd.overgo.contended-reading+json"
	contendedSchema    = "overgo/contended-reading/v1"
)

// contendedReading says that a serving attempt was measured while the work it
// names ran on the same machine. The scorecard reads nothing from an attempt
// such a record depends on, so the cell keeps the times its earlier attempts
// gave it.
type contendedReading struct {
	Version   uint16      `json:"version"`
	Attempt   artifact.ID `json:"attempt"`
	Competing []string    `json:"competing"`
	ID        artifact.ID `json:"-"`
}

var contendedCodec = artifact.JSONDocumentCodec(
	"contended reading", artifact.KindEvidence, contendedMediaType, contendedSchema,
	func(value *contendedReading) error {
		if value == nil || value.Version != artifact.InitialDocumentVersion || value.Attempt.Kind() != artifact.KindEvidence ||
			len(value.Competing) == 0 || slices.Contains(value.Competing, "") {
			return errors.New("scorecard: invalid contended reading")
		}
		return nil
	},
	func(value contendedReading) artifact.ID { return value.ID },
	func(value *contendedReading, id artifact.ID) { value.ID = id },
)

// errEarlier ends the walk of the serving attempts at the first one the store
// held before the measurement began.
var errEarlier = errors.New("an attempt the store held before the measurement")

// SetAsideContended records, for each serving attempt that succeeded and that
// the store first held after the sequence given, that it was measured beside
// the work named, and returns how many it recorded. A measurement that saw
// other work on its machine gives the store's sequence from before it began,
// so the attempts it wrote give the scorecard no reading. A failed attempt is
// not recorded, and counts as the failure it is.
func SetAsideContended(ctx context.Context, store *overgodb.Store, after uint64, competing []string) (int, error) {
	if err := store.Refresh(ctx); err != nil {
		return 0, err
	}
	var batches []artifact.Batch
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvidence, MediaType: runrecord.ServingObservationMediaType, Schema: runrecord.ServingObservationSchema,
		}},
		Order: overgodb.DocumentNewestFirst,
	}, runrecord.ParseServingObservation, func(view overgodb.DocumentView, observation runrecord.ServingObservation) error {
		attempt := view.Content.Descriptor.ID
		introduced, found, err := store.ArtifactIntroduction(ctx, attempt)
		switch {
		case err != nil:
			return err
		case !found || introduced.Sequence <= after:
			return errEarlier
		case observation.Outcome != runrecord.OutcomeSucceeded:
			return nil
		}
		reading, err := contendedCodec.New(contendedReading{Version: artifact.InitialDocumentVersion, Attempt: attempt, Competing: slices.Clone(competing)})
		if err != nil {
			return err
		}
		batch, err := contendedCodec.Batch("scorecard/contended/"+reading.ID.String(), reading,
			[]artifact.Lineage{{Child: reading.ID, Parent: attempt, Relation: artifact.RelationDependsOn}}, nil)
		batches = append(batches, batch)
		return err
	})
	if err != nil && !errors.Is(err, errEarlier) {
		return 0, err
	}
	// The walk has ended, so the store is written only now.
	for recorded, batch := range batches {
		if _, err := artifact.Publish(ctx, store, batch); err != nil {
			return recorded, err
		}
	}
	return len(batches), nil
}
