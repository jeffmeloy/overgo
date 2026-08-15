package closureledger

import (
	"context"
	"fmt"

	"overgo/internal/artifact"
)

func Load(ctx context.Context, store artifact.Reader, id artifact.ID) (Document, bool, error) {
	return documentCodec.Read(ctx, store, id)
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
