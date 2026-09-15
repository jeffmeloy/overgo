package plan

import (
	"context"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/worklease"
)

// RetireLegacyLeases removes the alias binding of every work-lease document
// that predates the current lease contract, through per-alias compare-and-set
// on the document each alias currently targets. The reviewed expectation must
// equal the number of legacy documents found; retired documents remain in the
// store's history. It returns the retired alias names.
func RetireLegacyLeases(ctx context.Context, store *overgodb.Store, expected int) ([]string, error) {
	if store == nil {
		return nil, fmt.Errorf("plan: nil lease store")
	}
	var retirements []artifact.AliasBinding
	var retired []string
	_, err := store.VisitDocuments(ctx, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvidence, MediaType: worklease.MediaType, Schema: worklease.Schema,
		}}, AliasPrefixes: []string{worklease.AliasRoot}, Order: overgodb.DocumentOldestFirst,
	}, func(view overgodb.DocumentView) error {
		if _, parseErr := worklease.Parse(view.Content.Data); parseErr == nil {
			return nil
		}
		target := view.Content.Descriptor.ID
		for _, alias := range view.Aliases {
			retirements = append(retirements, artifact.AliasBinding{
				Name: alias, Target: target, Previous: artifact.IDPointer(target), Remove: true,
			})
			retired = append(retired, alias)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if len(retired) != expected {
		return nil, fmt.Errorf(
			"plan: reviewed retirement expected %d legacy lease(s), found %d", expected, len(retired),
		)
	}
	if len(retirements) == 0 {
		return nil, nil
	}
	batch := artifact.Batch{Key: "automation/lease-retirement", Aliases: retirements}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		return nil, err
	}
	return retired, nil
}
