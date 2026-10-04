package dataset

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
)

// PublicationBatch defines atomic dataset document publication.
func PublicationBatch(
	key string,
	documents []Document,
	aliases []artifact.AliasBinding,
) (artifact.Batch, error) {
	contents := make([]artifact.Content, 0, len(documents))
	lineage := make([]artifact.Lineage, 0)
	published := make(map[artifact.ID]struct{}, len(documents))
	for _, document := range documents {
		if _, duplicate := published[document.ID]; duplicate {
			return artifact.Batch{}, fmt.Errorf("dataset: duplicate publication %s", document.ID)
		}
		content, err := document.Content()
		if err != nil {
			return artifact.Batch{}, err
		}
		contents = append(contents, content)
		lineage = append(lineage, document.Lineage()...)
		published[document.ID] = struct{}{}
	}
	for _, alias := range aliases {
		if _, ok := published[alias.Target]; !ok {
			return artifact.Batch{}, fmt.Errorf("dataset: alias %q targets unpublished document", alias.Name)
		}
	}
	return artifact.NewDocumentBatch(key, contents, lineage, aliases)
}

// Load resolves one inline dataset document.
func Load(ctx context.Context, store artifact.Reader, id artifact.ID) (Document, bool, error) {
	return documentCodec.Read(ctx, store, id)
}

// Resolve returns the dataset document bound to an alias.
func Resolve(ctx context.Context, store artifact.Reader, alias string) (Document, bool, error) {
	if store == nil {
		return Document{}, false, errors.New("dataset: nil repository")
	}
	id, ok, err := artifact.ResolveAlias(ctx, store, alias)
	if err != nil || !ok {
		return Document{}, ok, err
	}
	document, found, err := Load(ctx, store, id)
	if err != nil {
		return Document{}, false, fmt.Errorf("dataset: resolve %q: %w", alias, err)
	}
	return document, found, nil
}

// LoadMembership resolves one inline split-membership document.
func LoadMembership(ctx context.Context, store artifact.Reader, id artifact.ID) (Membership, bool, error) {
	return membershipCodec.Read(ctx, store, id)
}

// SplitPartner loads the membership a published split pairs with
// membership: the other partition of each two-partition split whose view
// selects it. A membership no such split holds, or one that splits pair
// with different partners, is refused rather than guessed.
func SplitPartner(ctx context.Context, store artifact.Reader, membership artifact.ID) (Membership, error) {
	views, err := childDocuments(ctx, store, membership, artifact.RelationDependsOn)
	if err != nil {
		return Membership{}, err
	}
	var partners []artifact.ID
	for _, view := range views {
		if view.Type != TypeView || view.Selector == nil || *view.Selector != membership {
			continue
		}
		splits, err := childDocuments(ctx, store, view.ID, artifact.RelationContains)
		if err != nil {
			return Membership{}, err
		}
		for _, split := range splits {
			if split.Type != TypeSplit || len(split.Partitions) != splitPair {
				continue
			}
			for _, partition := range split.Partitions {
				if partition.View == view.ID {
					continue
				}
				other, found, err := Load(ctx, store, partition.View)
				if err != nil || !found || other.Selector == nil {
					return Membership{}, errors.Join(fmt.Errorf("dataset: split %s partition %q names no committed view", split.ID, partition.Name), err)
				}
				if !slices.Contains(partners, *other.Selector) {
					partners = append(partners, *other.Selector)
				}
			}
		}
	}
	if len(partners) != 1 {
		return Membership{}, fmt.Errorf("dataset: %d published two-partition splits pair membership %s with a partner, want one", len(partners), membership)
	}
	partner, found, err := LoadMembership(ctx, store, partners[0])
	if err == nil && !found {
		err = fmt.Errorf("dataset: split partner %s of membership %s is not committed", partners[0], membership)
	}
	return partner, err
}

// splitPair is how many partitions a split pairing a training membership
// with its held-out one holds.
const splitPair = 2

// childDocuments loads the dataset documents id's children that relate to it
// by relation.
func childDocuments(ctx context.Context, store artifact.Reader, id artifact.ID, relation artifact.Relation) ([]Document, error) {
	edges, err := store.Children(ctx, id)
	if err != nil {
		return nil, err
	}
	var documents []Document
	for _, edge := range edges {
		if edge.Relation != relation || edge.Child.Kind() != artifact.KindDataset {
			continue
		}
		descriptor, found, err := store.Artifact(ctx, edge.Child)
		if err != nil {
			return nil, err
		}
		if !found || descriptor.MediaType != MediaType || descriptor.Schema != Schema {
			continue
		}
		document, found, err := Load(ctx, store, edge.Child)
		if err != nil {
			return nil, err
		}
		if found {
			documents = append(documents, document)
		}
	}
	return documents, nil
}
