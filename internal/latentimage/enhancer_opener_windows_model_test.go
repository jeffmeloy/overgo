//go:build modeltest && windows

package latentimage

import (
	"context"

	"overgo/internal/artifact"
	"overgo/internal/enhancerchat"
)

// enhancerOpener opens a MOSS model's enhancer chat through the store's profiles, as the image capability does.
func enhancerOpener(store artifact.Reader) EnhancerOpener {
	return func(ctx context.Context, directory string) (func(context.Context, string, string, int) (string, error), func() error, error) {
		return enhancerchat.Open(ctx, store, directory, EnhancerSampling(), PythonSpace)
	}
}
