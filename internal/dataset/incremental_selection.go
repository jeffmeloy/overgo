package dataset

import (
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
)

// IncrementalSelection splits one bounded selection against a previously
// admitted source set: arcs the consumer already holds are referenced by
// exact identity without duplicating bytes, new arcs carry their complete
// citation, and every admitted source appears exactly once — the
// optimization can reference, never omit.
type IncrementalSelection struct {
	PriorSourceSet artifact.ID                  `json:"prior_source_set"`
	SourceSet      artifact.ID                  `json:"source_set"`
	Reused         []artifact.ID                `json:"reused,omitempty"`
	Fresh          []InteractionSelectionSource `json:"fresh,omitempty"`
	TransferTokens uint64                       `json:"transfer_tokens"`
	TransferBytes  uint64                       `json:"transfer_bytes"`
}

// SelectIncremental derives the deterministic cursor delta between a held
// selection and the current admitted selection. Sources cited by both are
// reused by identity with their provenance preserved through the current
// boundary's own citations; only sources the holder lacks transfer bytes.
func SelectIncremental(prior, current InteractionSelection) (IncrementalSelection, error) {
	if prior.SourceSet.Kind() != artifact.KindEvidence || current.SourceSet.Kind() != artifact.KindEvidence ||
		len(current.Sources) == 0 {
		return IncrementalSelection{}, errors.New("dataset: incremental selection requires identified boundaries")
	}
	held := make(map[artifact.ID]struct{}, len(prior.Sources))
	for _, source := range prior.Sources {
		held[source.Source] = struct{}{}
	}
	delta := IncrementalSelection{PriorSourceSet: prior.SourceSet, SourceSet: current.SourceSet}
	for _, source := range current.Sources {
		if _, reusable := held[source.Source]; reusable {
			delta.Reused = append(delta.Reused, source.Source)
			continue
		}
		delta.Fresh = append(delta.Fresh, source)
		delta.TransferTokens += source.Tokens
		delta.TransferBytes += source.Bytes
	}
	return delta, nil
}

// Validate proves a delta against its exact boundaries: recomputing the
// decomposition must reproduce it bit for bit, so a delta that omitted,
// duplicated, or reordered an admitted stimulus can never verify.
func (delta IncrementalSelection) Validate(prior, current InteractionSelection) error {
	rebuilt, err := SelectIncremental(prior, current)
	if err != nil {
		return err
	}
	if delta.PriorSourceSet != rebuilt.PriorSourceSet || delta.SourceSet != rebuilt.SourceSet ||
		!slices.Equal(delta.Reused, rebuilt.Reused) || !slices.Equal(delta.Fresh, rebuilt.Fresh) ||
		delta.TransferTokens != rebuilt.TransferTokens || delta.TransferBytes != rebuilt.TransferBytes {
		return errors.New("dataset: incremental selection differs from its admitted boundaries")
	}
	if len(delta.Reused)+len(delta.Fresh) != len(current.Sources) {
		return fmt.Errorf(
			"dataset: incremental selection covers %d of %d admitted sources",
			len(delta.Reused)+len(delta.Fresh), len(current.Sources),
		)
	}
	return nil
}
