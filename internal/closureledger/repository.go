package closureledger

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
)

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
		return Document{}, false, errors.New("closure ledger: stored identity mismatch")
	}
	return document, true, nil
}

func Resolve(ctx context.Context, store artifact.Reader, alias string) (Document, bool, error) {
	id, ok, err := artifact.ResolveAlias(ctx, store, alias)
	if err != nil || !ok {
		return Document{}, ok, err
	}
	document, found, err := Load(ctx, store, id)
	if err != nil {
		return Document{}, false, fmt.Errorf("closure ledger: resolve %q: %w", alias, err)
	}
	return document, found, nil
}
