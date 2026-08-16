package closureledger

import (
	"context"

	"overgo/internal/artifact"
)

func Load(ctx context.Context, store artifact.Reader, id artifact.ID) (Document, bool, error) {
	return documentCodec.Read(ctx, store, id)
}
