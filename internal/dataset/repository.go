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
	batch := artifact.Batch{Key: key, Aliases: cloneAliases(aliases)}
	published := make(map[artifact.ID]struct{}, len(documents))
	for _, document := range documents {
		if _, duplicate := published[document.ID]; duplicate {
			return artifact.Batch{}, fmt.Errorf("dataset: duplicate publication %s", document.ID)
		}
		content, err := document.Content()
		if err != nil {
			return artifact.Batch{}, err
		}
		batch.Contents = append(batch.Contents, content)
		batch.Lineage = append(batch.Lineage, document.Lineage()...)
		published[document.ID] = struct{}{}
	}
	for _, alias := range aliases {
		if _, ok := published[alias.Target]; !ok {
			return artifact.Batch{}, fmt.Errorf("dataset: alias %q targets unpublished document", alias.Name)
		}
	}
	if err := batch.Validate(); err != nil {
		return artifact.Batch{}, err
	}
	return batch, nil
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
	if store == nil {
		return Document{}, false, errors.New("dataset: nil repository")
	}
	if id.Kind() != artifact.KindDataset {
		return Document{}, false, errors.New("dataset: invalid document identity")
	}
	content, ok, err := store.Content(ctx, id)
	if err != nil || !ok {
		return Document{}, ok, err
	}
	if err := documentContract.ValidateContent(content, id); err != nil {
		return Document{}, false, errors.New("dataset: incompatible content contract")
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

func cloneAliases(aliases []artifact.AliasBinding) []artifact.AliasBinding {
	result := make([]artifact.AliasBinding, len(aliases))
	for index, alias := range aliases {
		result[index] = alias
		if alias.Previous != nil {
			previous := *alias.Previous
			result[index].Previous = &previous
		}
	}
	return result
}
