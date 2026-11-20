//go:build modeltest && windows

package latentimage

import (
	"context"
	"encoding/base64"
	"math"
	"testing"
	"time"

	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
)

// TestADepthMapCostsASecondOnce holds a depth control from a picture (moss-depth-card) to what its cure removes, by
// count: the depth is estimated on the card, and Depth Anything and the picture encoder are built once a session. Two
// controls of the recorder's picture at the depth sections' canvas, through one session's picture sections, give the
// same latent bit for bit; the first leaves the depth model on the card, the only estimator the sections hold; the
// second builds neither again, puts less on the encoder's card than the weights it holds there, and leaves the process
// holding no more device memory than the first did, where building either again, or the depth's program, would put its
// weights there. Each control's time is logged: the row's information, read with the card's turn held alone, not its
// verdict.
func TestADepthMapCostsASecondOnce(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	root := ultraModelRoot(t)
	ultraDepth(t, root)
	const width, height = 1024, 576
	control := DepthControl{Picture: base64.StdEncoding.EncodeToString(ultraPictureBytes(t))}
	sections := newPictureSections(root, width, height)
	t.Cleanup(func() { testcheck.NoError(t, sections.close(context.WithoutCancel(ctx))) })
	controlled := func(which string) ([]float32, uint64, uint64) {
		t.Helper()
		began := time.Now()
		latent := testcheck.Must(sections.controlLatent(ctx, control))(t)
		t.Logf("the %s control took %v", which, time.Since(began).Round(time.Millisecond))
		if sections.encoder == nil || sections.depth == nil {
			t.Fatalf("the %s control from a picture left its sections without a picture encoder or the depth model on the card", which)
		}
		return latent, testcheck.Must(sections.encoder.session.Stats(ctx))(t).Execution.HostToDeviceBytes, driver.ProcessCurrentBytes()
	}

	first, encoderUploads, held := controlled("first")
	encoder, depth := sections.encoder, sections.depth
	second, encoderUploadsAfter, heldAfter := controlled("second")

	if sections.encoder != encoder || sections.depth != depth {
		t.Error("the second control built its picture encoder or its depth model again; want the first's held")
	}
	if len(second) != len(first) {
		t.Fatalf("the second control holds %d values; the first %d", len(second), len(first))
	}
	for index, value := range first {
		if math.Float32bits(second[index]) != math.Float32bits(value) {
			t.Fatalf("the second control's value %d is %v; want the first's %v bit for bit", index, second[index], value)
		}
	}
	stats := testcheck.Must(sections.encoder.session.Stats(ctx))(t)
	uploads := encoderUploadsAfter - encoderUploads
	t.Logf("the second control put %d bytes on the encoder's card, of %d the first had, beside its %d bytes of weights; the process held %d device bytes after the first, %d after the second",
		uploads, encoderUploads, stats.StaticBytes, held, heldAfter)
	if uploads >= stats.StaticBytes {
		t.Errorf("the second control put %d bytes on the encoder's card; want fewer than the %d bytes of weights it holds there", uploads, stats.StaticBytes)
	}
	if heldAfter > held {
		t.Errorf("the process held %d device bytes after the second control, %d more than after the first; want nothing more held", heldAfter, heldAfter-held)
	}
}
