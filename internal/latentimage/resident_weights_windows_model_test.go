//go:build modeltest && windows

package latentimage

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
)

// TestASecondMOSSRequestCopiesNoWeights holds a MOSS session's transformer weights on the card from one request to the
// next (moss-resident): on the MOSS Ultra workflow's files, a session's first sampling copies the transformer's weights
// to the card, a second request's copies under a tenth of that, and a request naming an adapter copies the adapter again,
// so a file replaced under its name is read as it is, and the BF16 form of the int8 projections it patches
// (rotaryGraph.projection, moss-int8), which the next request naming it keeps, copying under a quarter. Each request had
// copied the transformer's 9.5 GB again, about 33 s at 2368 x 1344 (moss-speed/do).
func TestASecondMOSSRequestCopiesNoWeights(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	root := ultraModelRoot(t)
	profile := testcheck.Must(ResolveProfile(root))(t)
	request := Request{Prompt: "a lighthouse at dusk", Width: 1024, Height: 576, Seed: 7, Steps: 2}
	generator, loaded := testcheck.Must(LoadGenerator(ctx, root, profile, request))(t), true
	defer closeReporting(t, context.WithoutCancel(ctx), generator)
	type sampled struct {
		copied uint64
		wall   time.Duration
	}
	sample := func(asked Request) sampled {
		t.Helper()
		if !loaded {
			testcheck.NoError(t, generator.Reset(ctx, asked))
		}
		loaded = false
		session := testcheck.Must(generator.prepare(ctx, asked))(t)
		before := testcheck.Must(generator.rotary.runtime.Stats(ctx))(t).Execution
		start := time.Now()
		testcheck.Must(generator.integrate(ctx, session))(t)
		wall := time.Since(start)
		after := testcheck.Must(generator.rotary.runtime.Stats(ctx))(t).Execution
		testcheck.Must(generator.decode(ctx, session))(t)
		return sampled{copied: after.HostToDeviceBytes - before.HostToDeviceBytes, wall: wall}
	}
	first := sample(request)
	second := sample(request)
	adapted := request
	adapted.Adapter = "moss_image_sft"
	withAdapter, again := sample(adapted), sample(adapted)
	adapterFile := testcheck.Must(os.Stat(filepath.Join(root, "adapters", "moss_image_sft.safetensors")))(t)
	t.Logf("a session's first sampling copied %.2f GiB in %v, its second %.3f GiB in %v, one naming an adapter %.3f GiB in %v, and the next %.3f GiB in %v",
		float64(first.copied)/(1<<30), first.wall.Round(time.Millisecond), float64(second.copied)/(1<<30), second.wall.Round(time.Millisecond),
		float64(withAdapter.copied)/(1<<30), withAdapter.wall.Round(time.Millisecond), float64(again.copied)/(1<<30), again.wall.Round(time.Millisecond))
	if second.copied*10 >= first.copied {
		t.Errorf("a second request copied %d bytes to the card, the first %d; want under a tenth", second.copied, first.copied)
	}
	adapterBytes := uint64(adapterFile.Size())
	if withAdapter.copied < second.copied+adapterBytes/4 || again.copied < adapterBytes/4 || again.copied*4 >= withAdapter.copied {
		t.Errorf("requests naming an adapter file of %d bytes copied %d and then %d bytes; want the adapter read again each time and the projections it patches kept",
			adapterBytes, withAdapter.copied, again.copied)
	}
}
