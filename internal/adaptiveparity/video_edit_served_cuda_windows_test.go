//go:build windows

package adaptiveparity

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"image"
	"image/draw"
	"image/gif"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/latentvideo"
	"overgo/internal/media"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/modeltest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// The reference case of the edit model: the clip its own repository tests
// with, as the earlier implementation recorded the frames it fed, the
// instruction and seed of that run, and the clip that run drew.
const (
	referenceEditInstruction = "Change the red currants to deep black grapes."
	referenceEditSeed        = 42
	referenceEditFrames      = 81
	referenceEditHeight      = 480
	referenceEditWidth       = 832
	referenceEditFPS         = 16
	// referenceEditBlock is the side of the squares two clips are compared
	// over. A served clip is a GIF, dithered to a fixed palette, so single
	// pixels differ by the dither; the mean of a square does not.
	referenceEditBlock = 8
)

// TestServedEditMatchesTheReferenceEdit holds the served edit path, the one
// a request reaches, to the edit model's reference case. The runtime under
// it is already held to the earlier implementation's latent and clip; this
// is the path around the runtime: a source clip committed as a GIF, a
// request naming a prompt, a seed and that clip, the text context from the
// base model's text encoder through the edit checkpoint's projection, and
// the edited clip as it is served. Each thing that path does is held to
// what the earlier run recorded: the source frames it resolves, the text
// context it resolves, and the clip it serves, which must agree with the
// clip the earlier implementation drew for the same frames, instruction and
// seed, square by square, and hold its dark and bright levels.
func TestServedEditMatchesTheReferenceEdit(t *testing.T) {
	cudatest.Require(t)
	edit := modeltest.Directory(t, "LiveEdit")
	wan, checkpoint := latentvideo.LiveEditArtifacts(edit)
	evidence := filepath.Join(filepath.Dir(filepath.Dir(edit)), ".claude", "worktrees", "image_gen", ".media_artifacts", "video_artifacts")
	sourcePixels := filepath.Join(evidence, "liveedit_native_go_currants_seed42.mp4.tensors", "source_pixels.f32le")
	referenceClip := filepath.Join(evidence, "liveedit_native_no_pruning_current_seed42.mp4")
	for _, required := range []string{checkpoint, latentvideo.WanTextConditioningSpec(wan, 1).EncoderCheckpoint, sourcePixels, referenceClip} {
		if _, err := os.Stat(required); err != nil {
			t.Skipf("UNAVAILABLE: %v", err)
		}
	}
	ffmpeg, err := media.ResolveFFmpeg(os.Getenv("OVERGO_FFMPEG"))
	if err != nil {
		testskip.NotApplicable(t, "no FFmpeg to read the reference clip with")
	}
	// The whole reference clip keeps most of the device resident, so the edit
	// runs in a process of its own and leaves the device as it found it.
	if cudatest.MeasurementProcess(t, 0) {
		return
	}
	profile, err := latentvideo.ResolveProfile(wan)
	if err != nil {
		t.Fatal(err)
	}
	profileContent, err := profile.Content()
	if err != nil {
		t.Fatal(err)
	}
	modelID := testutil.ArtifactID(t, artifact.KindModel, "liveedit-reference-edit")
	definition, err := modelrecipe.ReferenceVideoEditDefinition(modelID, profile.ID)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	// The model's artifact stands in the store at the size of its directory.
	var extent uint64
	if err := filepath.WalkDir(edit, func(_ string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil || entry.IsDir() {
			return walkErr
		}
		info, err := entry.Info()
		extent += uint64(info.Size())
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key: "test/artifact/" + modelID.String(), Artifacts: []artifact.Descriptor{{ID: modelID, Size: extent}},
	}); err != nil {
		t.Fatal(err)
	}

	// The source clip, from the frames the earlier run fed: channel, then frame, then row, 0 to 1.
	plane := referenceEditHeight * referenceEditWidth
	recorded, err := os.Open(sourcePixels)
	if err != nil {
		t.Fatal(err)
	}
	defer recorded.Close()
	raw := make([]byte, media.RGBChannels*referenceEditFrames*plane*binary.Size(float32(0)))
	if _, err := io.ReadFull(recorded, raw); err != nil {
		t.Fatal(err)
	}
	encoder, err := latentvideo.NewClipEncoder(referenceEditFPS, latentvideo.UnitPixels)
	if err != nil {
		t.Fatal(err)
	}
	for frame := range referenceEditFrames {
		values := make([]float32, media.RGBChannels*plane)
		for channel := range media.RGBChannels {
			from := (channel*referenceEditFrames + frame) * plane
			for index := range plane {
				values[channel*plane+index] = math.Float32frombits(binary.LittleEndian.Uint32(raw[(from+index)*binary.Size(float32(0)):]))
			}
		}
		if err := encoder.Add(frame, values, referenceEditHeight, referenceEditWidth); err != nil {
			t.Fatal(err)
		}
	}
	sourceVideo, err := encoder.Finish()
	if err != nil {
		t.Fatal(err)
	}
	clip, err := latentvideo.GIFContent(sourceVideo)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "test/video-edit-reference", Contents: []artifact.Content{profileContent, clip},
	}); err != nil {
		t.Fatal(err)
	}
	form := latentvideo.ReferenceEditRequest{Prompt: referenceEditInstruction, Seed: referenceEditSeed, SourceArtifact: clip.Descriptor.ID}
	request, err := json.Marshal(form)
	if err != nil {
		t.Fatal(err)
	}

	// What the request resolves to, held to what the earlier run recorded: the
	// source frames it fed and the text context it conditioned on. A source
	// read from -1 to 1 misses the first by the whole range, and a context
	// projected by the base model's projection misses the second by a quarter.
	resolved, err := latentvideo.ResolveClipRequest(t.Context(), store, edit, form)
	if err != nil {
		t.Fatal(err)
	}
	var sourceError float64
	for index, value := range resolved.Source.Pixels {
		want := math.Float32frombits(binary.LittleEndian.Uint32(raw[index*binary.Size(float32(0)):]))
		sourceError += float64(value-want) * float64(value-want)
	}
	sourceRatio := -10 * math.Log10(sourceError/float64(len(resolved.Source.Pixels)))
	recordedContext, err := os.ReadFile(filepath.Join(filepath.Dir(sourcePixels), "text_context.f32le"))
	if err != nil || len(recordedContext) != len(resolved.Condition.TextContext)*binary.Size(float32(0)) {
		t.Fatalf("the recorded text context holds %d bytes for %d values (%v)", len(recordedContext), len(resolved.Condition.TextContext), err)
	}
	var dot, servedNorm, recordedNorm float64
	for index, value := range resolved.Condition.TextContext {
		want := float64(math.Float32frombits(binary.LittleEndian.Uint32(recordedContext[index*binary.Size(float32(0)):])))
		dot, servedNorm, recordedNorm = dot+float64(value)*want, servedNorm+float64(value)*float64(value), recordedNorm+want*want
	}
	contextCosine := dot / math.Sqrt(servedNorm*recordedNorm)
	t.Logf("resolved request against the earlier run: source frames %.2f dB, text context cosine %.6f", sourceRatio, contextCosine)
	// The clip format keeps the source at 33 dB and the smoothing of its dither
	// at 36 dB; the context differs by rounding alone.
	const sourceFloor, contextFloor = 35, 0.999
	if len(resolved.Source.Pixels) != len(raw)/binary.Size(float32(0)) || sourceRatio < sourceFloor || contextCosine < contextFloor {
		t.Fatalf("the request resolves to source frames at %.2f dB and a text context at cosine %.6f of the earlier run's", sourceRatio, contextCosine)
	}
	output, err := mediacapability.Catalog[recipe.TaskVideoGen].Execute(t.Context(), store, edit, modelrecipetest.CandidateExecution(t, store, program), string(request))
	if err != nil {
		t.Fatal(err)
	}
	video, ok := capabilityruntime.Unwrap(output).(latentvideo.EncodedVideo)
	if !ok || video.Frames != referenceEditFrames || video.Height != referenceEditHeight || video.Width != referenceEditWidth {
		t.Fatalf("served edit = %T of %d frames at %dx%d", output, video.Frames, video.Width, video.Height)
	}
	animation, err := gif.DecodeAll(bytes.NewReader(video.Data))
	if err != nil {
		t.Fatal(err)
	}
	reference, err := media.DecodeVideoFileAt(t.Context(), referenceClip, ffmpeg, referenceEditFPS, referenceEditFrames, referenceEditWidth, referenceEditHeight)
	if err != nil || len(reference) != referenceEditFrames || len(animation.Image) != referenceEditFrames {
		t.Fatalf("the reference clip gave %d frames and the served clip %d (%v), want %d of each", len(reference), len(animation.Image), err, referenceEditFrames)
	}

	bounds := image.Rect(0, 0, referenceEditWidth, referenceEditHeight)
	canvas := image.NewRGBA(bounds)
	var squared, servedDark, servedBright, wantDark, wantBright float64
	blocks := 0
	for frame, paletted := range animation.Image {
		draw.Draw(canvas, paletted.Bounds(), paletted, paletted.Bounds().Min, draw.Over)
		served, want := squareMeans(canvas), squareMeans(reference[frame])
		for index := range served {
			squared += (served[index] - want[index]) * (served[index] - want[index])
		}
		blocks += len(served)
		dark, bright := squareExtremes(served)
		servedDark, servedBright = servedDark+dark/referenceEditFrames, servedBright+bright/referenceEditFrames
		dark, bright = squareExtremes(want)
		wantDark, wantBright = wantDark+dark/referenceEditFrames, wantBright+bright/referenceEditFrames
	}
	const full = math.MaxUint8
	agreement := 20 * math.Log10(full/math.Sqrt(squared/float64(blocks)))
	t.Logf("served edit against the earlier implementation's clip: %.2f dB over %d-pixel squares; darkest squares %.1f against %.1f, brightest %.1f against %.1f of %d",
		agreement, referenceEditBlock, servedDark, wantDark, servedBright, wantBright, full)
	// The runtime alone, given the recorded frames, agrees with this clip at 41
	// dB over the squares. Given the frames through the clip format, smoothed,
	// it agrees at 27 dB; unsmoothed, the editor reads the dither as content and
	// agrees at 17 dB; and an output read in the wrong range lifts the darkest
	// squares by a third of the range.
	const agreementFloor, levelSlack = 23, 16
	if agreement < agreementFloor || math.Abs(servedDark-wantDark) > levelSlack || math.Abs(servedBright-wantBright) > levelSlack {
		t.Fatalf("the served edit agrees with the reference at %.2f dB, with dark squares at %.1f for %.1f and bright ones at %.1f for %.1f", agreement, servedDark, wantDark, servedBright, wantBright)
	}
}

// squareMeans is a frame as the mean of each colour over each square of
// referenceEditBlock pixels, row by row, on the byte scale.
func squareMeans(frame image.Image) []float64 {
	bounds := frame.Bounds()
	columns, rows := bounds.Dx()/referenceEditBlock, bounds.Dy()/referenceEditBlock
	means := make([]float64, media.RGBChannels*rows*columns)
	const scale = math.MaxUint16 / math.MaxUint8
	for y := range rows * referenceEditBlock {
		for x := range columns * referenceEditBlock {
			red, green, blue, _ := frame.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			block := (y/referenceEditBlock*columns + x/referenceEditBlock) * media.RGBChannels
			for channel, value := range []uint32{red, green, blue} {
				means[block+channel] += float64(value) / scale / (referenceEditBlock * referenceEditBlock)
			}
		}
	}
	return means
}

// squareExtremes is the mean of the darkest and of the brightest twentieth
// of a frame's square means.
func squareExtremes(means []float64) (dark, bright float64) {
	sorted := append([]float64(nil), means...)
	slices.Sort(sorted)
	tail := max(len(sorted)/20, 1)
	for index := range tail {
		dark += sorted[index] / float64(tail)
		bright += sorted[len(sorted)-1-index] / float64(tail)
	}
	return dark, bright
}
