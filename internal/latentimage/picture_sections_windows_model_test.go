//go:build modeltest && windows

package latentimage

import (
	"bytes"
	"context"
	"encoding/base64"
	"image"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/graphruntime"
	"overgo/internal/tensor"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestThePictureSectionsHoldWhatTheirSourcesSay holds a session's picture sections (picture-sections) to what their
// sources say, by count:
//   - a full-denoise request whose picture's crop leaves nothing is refused, as a partial one is (SQA-432)
//   - the device memory a depth control from a picture leaves held is all in the generator's session, the process's
//     held bytes growing by what that session's ledger counts, so its peak counts them (SQA-434)
//   - the picture encoder keeps no host copy of the weights it put on the card (SQA-435)
//   - the depth model is kept for its checkpoint's bytes and read again for other bytes under the same name, a
//     checkpoint that cannot be read refusing the map, and the same bytes read again giving the same latent (SQA-406)
func TestThePictureSectionsHoldWhatTheirSourcesSay(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	const width, height = 1024, 576

	var tall bytes.Buffer
	testcheck.NoError(t, png.Encode(&tall, image.NewRGBA(image.Rect(0, 0, 1, 2))))
	full := &Generator{request: Request{Image: base64.StdEncoding.EncodeToString(tall.Bytes()), Width: width, Height: height},
		modelDir: t.TempDir(), schedule: FlowSchedule{Sigmas: []float64{1, 0}}}
	if _, err := full.startLatent(ctx, make([]float32, 4)); err == nil || !strings.Contains(err.Error(), "crop") {
		t.Errorf("a full-denoise request whose 1 x 2 picture's crop leaves nothing at %d x %d started (%v); want it refused for its crop", width, height, err)
	}

	root := ultraModelRoot(t)
	ultraDepth(t, root)
	session := testcheck.Must(graphruntime.NewResidentSession(tensor.FirstOffset))(t)
	t.Cleanup(func() { testcheck.NoError(t, session.Close(context.Background())) })
	generator := &Generator{request: Request{Width: width, Height: height}, modelDir: root, rotary: &rotaryGeneration{runtime: session}}
	sections := generator.sections()
	t.Cleanup(func() { testcheck.NoError(t, sections.close(context.Background())) })
	held := func() uint64 {
		t.Helper()
		return testcheck.Must(session.Stats(ctx))(t).Device.CurrentBytes
	}
	control := DepthControl{Picture: base64.StdEncoding.EncodeToString(ultraPictureBytes(t))}
	controlled := func(which string) []float32 {
		t.Helper()
		began := time.Now()
		latent := testcheck.Must(sections.controlLatent(ctx, control))(t)
		t.Logf("the %s control took %v", which, time.Since(began).Round(time.Millisecond))
		return latent
	}
	same := func(which string, got, want []float32) {
		t.Helper()
		for index, value := range want {
			if index >= len(got) || math.Float32bits(got[index]) != math.Float32bits(value) {
				t.Fatalf("the %s control's latent differs from the first's at %d; want it bit for bit", which, index)
			}
		}
	}

	processBefore, sessionBefore := driver.ProcessCurrentBytes(), held()
	first := controlled("first")
	process, inSession := driver.ProcessCurrentBytes()-processBefore, held()-sessionBefore
	peak := testcheck.Must(generator.PeakDeviceBytes())(t)
	t.Logf("a control from a picture left the process holding %d device bytes more and the generator's session %d; the session's peak %d", process, inSession, peak)
	if process != inSession || peak < inSession {
		t.Errorf("a control from a picture left %d device bytes held, %d of them in the generator's session, whose peak reads %d; want all of them there", process, inSession, peak)
	}
	if feeds, operations := len(sections.encoder.graph.feeds), len(sections.encoder.encoder.Operations); feeds != 0 || operations != 0 {
		t.Errorf("the picture encoder keeps %d weight feeds and %d operations on the host after their upload; want none", feeds, operations)
	}

	depth := sections.depth
	same("second", controlled("second"), first)
	if sections.depth != depth {
		t.Error("the second control read the depth model again from the same bytes; want the first's kept")
	}
	// The root's checkpoint is a hard link into the model store: it is removed and replaced, never written through.
	checkpoint := filepath.Join(root, depthDirectory, depthCheckpoint)
	testcheck.NoError(t, os.Remove(checkpoint))
	testcheck.NoError(t, os.WriteFile(checkpoint, nil, 0o600))
	if _, err := sections.controlLatent(ctx, control); err == nil {
		t.Error("a control mapped its picture from an empty checkpoint under the depth model's name; want the checkpoint refused")
	}
	if sections.depth != nil {
		t.Error("the sections kept a depth model read from other bytes than their checkpoint holds")
	}
	testcheck.NoError(t, os.Remove(checkpoint))
	testcheck.NoError(t, os.Link(filepath.Join(testutil.ModelArtifactDir(t, "depth-anything-v2-small"), depthCheckpoint), checkpoint))
	same("restored", controlled("restored"), first)
	if sections.depth == nil || sections.depth == depth {
		t.Error("the checkpoint restored was not read again; want a depth model read from its bytes")
	}
}
