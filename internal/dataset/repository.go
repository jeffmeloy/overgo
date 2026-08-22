package dataset

import (
	"context"
	"errors"
	"fmt"

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
