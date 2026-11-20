//go:build modeltest && windows

package latentimage

import (
	"context"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
)

// TestASecondMOSSRequestCopiesNoWeights holds a MOSS session's transformer weights on the card from one request to the
// next (moss-resident): on the MOSS Ultra workflow's files, a session's first sampling copies the transformer's weights
// to the card, a second request's copies under a tenth of that, and a request naming an adapter merges it into the int8
// projections it patches (mergedProjection, moss-int8-adapters), which the next request naming the same file keeps,
// copying under a quarter, the merge reading the weights it merges into on the card and copying its pairs alone; a
// file replaced under its name is merged again (TestAKeptMergeFollowsItsAdaptersContent).
// Each request had copied the transformer's 9.5 GB again, about 33 s at 2368 x 1344 (moss-speed/do).
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
	if withAdapter.copied < second.copied+adapterBytes/4 || again.copied*4 >= withAdapter.copied {
		t.Errorf("requests naming an adapter file of %d bytes copied %d and then %d bytes; want its merge into the projections it patches made once and kept",
			adapterBytes, withAdapter.copied, again.copied)
	}
	// The merge reads the weights it merges into where the session's earlier program holds them, and copies its
	// adapter's pairs alone (moss-merge-time): it had read 4.75 GB of int8 rows from the file and copied them again.
	if withAdapter.copied > 2*adapterBytes {
		t.Errorf("a request merging an adapter file of %d bytes after a request without it copied %d bytes; want its pairs alone, the weights it merges into read on the card",
			adapterBytes, withAdapter.copied)
	}
}

// TestAKeptMergeFollowsItsAdaptersContent holds the merges a MOSS session keeps on the card to the adapter files as they
// stand (moss-merge-time/identity): over a copy of the MOSS Ultra workflow's model whose moss_image_sft is its own file,
// a request naming the adapter merges it into the int8 projections it patches, the next request naming it copies under
// a quarter of that, the merge kept, and a request after the file is replaced under its name, one bit of its last value
// changed, merges it again. The merge had been kept under the adapter's path, and the replaced file was answered with
// the earlier file's merge (SQA-406's fault in the merge's place).
func TestAKeptMergeFollowsItsAdaptersContent(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	root := mirroredModel(t, ultraModelRoot(t), filepath.Join("adapters", "moss_image_sft.safetensors"))
	profile := testcheck.Must(ResolveProfile(root))(t)
	request := Request{Prompt: "a lighthouse at dusk", Width: 1024, Height: 576, Seed: 7, Steps: 2, Adapter: "moss_image_sft"}
	generator, loaded := testcheck.Must(LoadGenerator(ctx, root, profile, request))(t), true
	defer closeReporting(t, context.WithoutCancel(ctx), generator)
	copied := func() uint64 {
		t.Helper()
		if !loaded {
			testcheck.NoError(t, generator.Reset(ctx, request))
		}
		loaded = false
		session := testcheck.Must(generator.prepare(ctx, request))(t)
		before := testcheck.Must(generator.rotary.runtime.Stats(ctx))(t).Execution
		testcheck.Must(generator.integrate(ctx, session))(t)
		after := testcheck.Must(generator.rotary.runtime.Stats(ctx))(t).Execution
		testcheck.Must(generator.decode(ctx, session))(t)
		return after.HostToDeviceBytes - before.HostToDeviceBytes
	}
	merged, kept := copied(), copied()
	adapter := filepath.Join(root, "adapters", "moss_image_sft.safetensors")
	file := testcheck.Must(os.OpenFile(adapter, os.O_RDWR, 0))(t)
	size := testcheck.Must(file.Seek(0, io.SeekEnd))(t)
	last := make([]byte, 1)
	testcheck.Must(file.ReadAt(last, size-4))(t)
	last[0] ^= 1
	testcheck.Must(file.WriteAt(last, size-4))(t)
	testcheck.NoError(t, file.Close())
	replaced := copied()
	t.Logf("requests naming the adapter copied %d bytes, then %d, then %d after its file was replaced", merged, kept, replaced)
	if kept*4 >= merged || replaced*4 < merged*3 {
		t.Errorf("requests naming the adapter copied %d, %d and, its file replaced under its name, %d bytes; want its merge kept, then made again from the replaced file",
			merged, kept, replaced)
	}
}

// mirroredModel is a model directory that is root's own files, linked, except the file at changeable, which is a copy
// a test may change without touching root.
func mirroredModel(t *testing.T, root, changeable string) string {
	t.Helper()
	mirror := t.TempDir()
	testcheck.NoError(t, filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		relative := testcheck.Must(filepath.Rel(root, path))(t)
		target := filepath.Join(mirror, relative)
		switch {
		case entry.IsDir():
			return os.MkdirAll(target, 0o700)
		case relative == changeable:
			data, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			return os.WriteFile(target, data, 0o600)
		default:
			return os.Link(path, target)
		}
	}))
	return mirror
}
