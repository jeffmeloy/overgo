package evaluation

import (
	"context"
	"errors"
	"fmt"
	"slices"

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
// given no more names, so a store whose parts are all named publishes
// nothing. It returns how many parts it named.
//
// A part is named whole or not at all: one that holds a document with no
// content recorded is given no names, for that document would read as fresh
// where it is registered again. If it answered confirmations the naming is
// refused, a pool's part also when named before, and with it the admission
// that asked. An acceptance's named part is followed by the names it has.
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
// published today. A pool's part that has names and has answered is read again.
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
		// A named part that has answered nothing is passed by unread. One that has answered is read again: the names it has may be those of some of its documents only.
		published, err := partContents(ctx, store, budget.Split)
		if err != nil || len(published) != 0 && len(charges) == 0 {
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
		} else if len(published) != 0 {
			// An acceptance's named part is followed by the names it has: the cases the store lists now may be named otherwise than the ones it was built from.
			return nil
		} else if cases, of := instruction(); of == membership.Source {
			contents = cases
		}
		// Whole is asked of each record and not of how many names there are: records of one content share a name.
		whole := !slices.ContainsFunc(membership.Records, func(record dataset.Record) bool { return contents[record.ID] == "" })
		if whole && len(published) != 0 {
			// Named, and a content is recorded for every record: whoever named it named them all in one batch (containContents), so it is followed by the names it has.
			return nil
		}
		batch := artifact.Batch{Key: poolKey + "named/" + budget.Split.String()}
		if whole {
			if err := containContents(&batch, []dataset.Membership{membership}, contents); err != nil {
				return err
			}
		}
		switch {
		case len(batch.Lineage) != 0:
			naming, seen[budget.Split] = append(naming, batch), true
		case len(charges) != 0:
			return fmt.Errorf("evaluation: part %s answered %d confirmations and not every document it holds is named by a recorded content, so what they answered cannot follow them to where they are registered again; nothing further is admitted to this store until that part is decided", budget.Split, len(charges))
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
