package dataset

import (
	"context"
	"errors"
	"fmt"

	"llamacpp2go/internal/artifact"
)

// PublicationBatch: atomic dataset document publication.
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

// Publish: commits one dataset document.
func Publish(
	ctx context.Context,
	store artifact.Repository,
	key string,
	document Document,
	alias *artifact.AliasBinding,
) (artifact.CommitID, error) {
	if store == nil {
		return artifact.CommitID{}, errors.New("dataset: nil repository")
	}
	var aliases []artifact.AliasBinding
	if alias != nil {
		aliases = append(aliases, *alias)
	}
	batch, err := PublicationBatch(key, []Document{document}, aliases)
	if err != nil {
		return artifact.CommitID{}, err
	}
	return store.Commit(ctx, batch)
}

// Load: resolves one inline dataset document.
func Load(ctx context.Context, store artifact.Reader, id artifact.ID) (Document, bool, error) {
	content, ok, err := artifact.ReadDocument(ctx, store, id, documentContract)
	if err != nil || !ok {
		return Document{}, ok, err
	}
	document, err := Parse(content.Data)
	if err != nil {
		return Document{}, false, err
	}
	if document.ID != id {
		return Document{}, false, errors.New("dataset: stored identity mismatch")
	}
	return document, true, nil
}

// Resolve: resolves an alias to one dataset document.
func Resolve(ctx context.Context, store artifact.Reader, alias string) (Document, bool, error) {
	if store == nil {
		return Document{}, false, errors.New("dataset: nil repository")
	}
	id, ok, err := store.ResolveAlias(ctx, alias)
	if err != nil || !ok {
		return Document{}, ok, err
	}
	document, found, err := Load(ctx, store, id)
	if err != nil {
		return Document{}, false, fmt.Errorf("dataset: resolve %q: %w", alias, err)
	}
	return document, found, nil
}
