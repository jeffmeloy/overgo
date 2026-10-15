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

// SetAsideContended excludes timing only from the named successful attempts.
// Failed attempts remain counted. All records are read and validated before
// exclusions are published; replay uses the same immutable record identities.
func SetAsideContended(ctx context.Context, store *overgodb.Store, attempts []artifact.ID, competing []string) (int, error) {
	if err := store.Refresh(ctx); err != nil {
		return 0, err
	}
	var batches []artifact.Batch
	for _, attempt := range attempts {
		observation, err := runrecord.RequireServingObservation(ctx, store, attempt)
		if err != nil {
			return 0, err
		}
		if observation.Outcome != runrecord.OutcomeSucceeded {
			continue
		}
		reading, err := contendedCodec.New(contendedReading{Version: artifact.InitialDocumentVersion, Attempt: attempt, Competing: slices.Clone(competing)})
		if err != nil {
			return 0, err
		}
		batch, err := contendedCodec.Batch("scorecard/contended/"+reading.ID.String(), reading, []artifact.Lineage{{Child: reading.ID, Parent: attempt, Relation: artifact.RelationDependsOn}}, nil)
		if err != nil {
			return 0, err
		}
		batches = append(batches, batch)
	}
	for recorded, batch := range batches {
		if _, err := artifact.Publish(ctx, store, batch); err != nil {
			return recorded, err
		}
	}
	return len(batches), nil
}
