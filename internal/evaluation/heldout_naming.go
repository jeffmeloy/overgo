package evaluation

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// NameEarlierParts admits what was charged before a part named what it
// holds. A part published then cannot be found from another registration of
// its documents, so material that had answered confirmations read as fresh
// when it was imported again. Every part a grant of confirmations stands on
// that was published without names is given the names it would be published
// with today (containContents): its exposure then follows its documents'
// content to wherever they are registered (exposure), and the part answers
// confirmations again, counted like any other.
//
// It is run where held-out material is admitted, before a pool or an
// acceptance is built, and never for a confirmation: it lists the store's
// grants of confirmations, which no confirmation does. A part named before is
// passed by, so a store whose parts are all named publishes nothing. It
// returns how many parts it named.
//
// A part that answered confirmations and whose documents have no content
// recorded cannot have what it answered follow them. That is refused, and
// with it the admission that asked: nothing further is admitted to such a
// store until the part is decided.
func NameEarlierParts(ctx context.Context, store *overgodb.Store) (int, error) {
	return nameEarlierParts(ctx, store, func() (map[string]string, artifact.ID) {
		contents := map[string]string{}
		prompts, cases, err := StoreInstructionPrompts(ctx, store)
		if err != nil {
			// No catalog of instruction cases names an instruction part's documents.
			return contents, artifact.ID{}
		}
		for _, asked := range prompts.Cases() {
			contents[asked.Record] = asked.Content
		}
		return contents, cases
	})
}

// nameEarlierParts is NameEarlierParts with the store's instruction cases
// given: each case's content by its record, and the dataset the cases are
// records of. A pool's part is named by the contents its dataset recorded for
// its files, an instruction acceptance's by its cases', as each is when it is
// published today.
func nameEarlierParts(ctx context.Context, store *overgodb.Store, instruction func() (map[string]string, artifact.ID)) (int, error) {
	var naming []artifact.Batch
	seen := map[artifact.ID]bool{}
	_, err := store.VisitDocuments(ctx, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{Kind: artifact.KindEvidence, MediaType: runrecord.BudgetMediaType, Schema: runrecord.BudgetSchema}},
		Order:     overgodb.DocumentOldestFirst,
	}, func(view overgodb.DocumentView) error {
		budget, charges, err := runrecord.BudgetCharges(ctx, store, view.Content.Descriptor.ID)
		if err != nil || budget.Unit != poolUnit || seen[budget.Split] {
			return err
		}
		if published, err := partContents(ctx, store, budget.Split); err != nil || len(published) != 0 {
			return err
		}
		membership, found, err := dataset.LoadMembership(ctx, store, budget.Split)
		if err != nil || !found {
			return errors.Join(fmt.Errorf("evaluation: the part %s a grant of confirmations stands on is not committed", budget.Split), err)
		}
		var contents map[string]string
		if authority, _, err := store.Artifact(ctx, budget.Authority); err != nil {
			return err
		} else if authority.MediaType != instructionAcceptanceMedia {
			if contents, err = recordContents(ctx, store, membership.Source); err != nil {
				return err
			}
		} else if cases, of := instruction(); of == membership.Source {
			contents = cases
		}
		batch := artifact.Batch{Key: poolKey + "named/" + budget.Split.String()}
		if err := containContents(&batch, []dataset.Membership{membership}, contents); err != nil {
			return err
		}
		switch {
		case len(batch.Lineage) != 0:
			naming, seen[budget.Split] = append(naming, batch), true
		case len(charges) != 0:
			return fmt.Errorf("evaluation: part %s answered %d confirmations before parts named what they hold, and no content is recorded for its documents, so what they answered cannot follow them to where they are registered again; nothing further is admitted to this store until that part is decided", budget.Split, len(charges))
		}
		return nil
	})
	if err != nil {
		return 0, err
	}
	for _, batch := range naming {
		if _, err := artifact.Publish(ctx, store, batch); err != nil {
			return 0, err
		}
	}
	return len(naming), nil
}
