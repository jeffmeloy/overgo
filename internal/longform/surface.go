package longform

import (
	"context"

	"overgo/internal/inferencesurface"
)

// Surface digests the inference code surface of the tree at root, the key a
// long-form record is held to. What the surface is belongs to
// inferencesurface, which imports none of it: this package measures with the
// device stack, and a tool that only asks whether a change moves the surface
// must not have to link that stack to find out.
func Surface(ctx context.Context, root string) (string, error) {
	return inferencesurface.Digest(ctx, root)
}
