//go:build windows

package adaptiveparity

import (
	"encoding/binary"
	"image"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/latentvideo"
	"overgo/internal/media"
	"overgo/internal/modeltest"
	"overgo/internal/testskip"
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
	// over. A served GIF is dithered to a palette and a served MP4 loses
	// detail to its codec, so single pixels differ by those; the mean of a
	// square does not.
	referenceEditBlock = 8
	// The floors, in dB, a request that carries its source as frames is held
	// to. The frames reach the editor at the byte they were written at: the
	// reference's are whole bytes and come back exactly, and rounding to a byte
	// alone leaves 59 dB, so a source below 55 was not carried as it was. The
	// edit of them agrees with the reference clip at 40.1 dB, where the runtime
	// given the recorded frames directly agrees at 41.
	framesSourceFloor, framesAgreementFloor = 55, 38
	// The floor a source carried as a GIF is held to: the clip format keeps it
	// at 33 dB and the smoothing of its dither at 36.
	gifSourceFloor = 35
)

// TestServedEditMatchesTheReferenceEdit holds the served edit path, the one
// a request reaches, to the edit model's reference case. The runtime under
// it is already held to the earlier implementation's latent and clip; this
// is the path around the runtime: a source clip committed to the store, a
// request naming a prompt, a seed and that clip, the text context from the
// base model's text encoder through the edit checkpoint's projection, and
// the edited clip as it is served. Each thing that path does is held to
// what the earlier run recorded: the source frames it resolves, the text
// context it resolves, and the clip it serves, which must agree with the
// clip the earlier implementation drew for the same frames, instruction and
// seed, square by square, and hold its dark and bright levels. The source is
// carried each way a request can carry one: as its frames, which reach the
// editor as they are, as the GIF a video model returns, which reaches it in
// 256 colours a frame and smoothed, and as an MP4, which reaches it through
// FFmpeg cut for the editor. The edit of the frames is served as an MP4.
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
	store, serve := editServing(t, edit, wan, "liveedit-reference-edit")

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
	recordedSource := latentvideo.SourceVideo{
		Pixels: make([]float32, media.RGBChannels*referenceEditFrames*plane), Channels: media.RGBChannels,
		Frames: referenceEditFrames, Height: referenceEditHeight, Width: referenceEditWidth,
	}
	for index := range recordedSource.Pixels {
		recordedSource.Pixels[index] = math.Float32frombits(binary.LittleEndian.Uint32(raw[index*binary.Size(float32(0)):]))
	}
	// The two forms a request carries its source in: the frames as they are,
	// and the clip a video model returns, a GIF.
	frames, err := latentvideo.SourceFramesContent(recordedSource)
	if err != nil {
		t.Fatal(err)
	}
	encoder, err := latentvideo.NewClipEncoder(referenceEditFPS, latentvideo.UnitPixels)
	if err != nil {
		t.Fatal(err)
	}
	for frame := range referenceEditFrames {
		values := make([]float32, media.RGBChannels*plane)
		for channel := range media.RGBChannels {
			copy(values[channel*plane:][:plane], recordedSource.Pixels[(channel*referenceEditFrames+frame)*plane:][:plane])
		}
		if err := encoder.Add(frame, values, referenceEditHeight, referenceEditWidth); err != nil {
			t.Fatal(err)
		}
	}
	sourceVideo, err := encoder.Finish()
	if err != nil {
		t.Fatal(err)
	}
	clip, err := latentvideo.VideoContent(sourceVideo)
	if err != nil {
		t.Fatal(err)
	}
	recording, err := latentvideo.NewMP4Encoder(t.Context(), ffmpeg, referenceEditFPS, latentvideo.UnitPixels)
	if err != nil {
		t.Fatal(err)
	}
	for frame := range referenceEditFrames {
		values := make([]float32, media.RGBChannels*plane)
		for channel := range media.RGBChannels {
			copy(values[channel*plane:][:plane], recordedSource.Pixels[(channel*referenceEditFrames+frame)*plane:][:plane])
		}
		if err := recording.Add(frame, values, referenceEditHeight, referenceEditWidth); err != nil {
			t.Fatal(err)
		}
	}
	encoded, err := recording.Finish()
	if err != nil {
		t.Fatal(err)
	}
	movie, err := latentvideo.VideoContent(encoded)
	if err != nil {
		t.Fatal(err)
	}
	// The frames of this clip are 36 MiB, and the store bounds a batch as it is
	// written, a third larger: they are committed alone.
	for key, content := range map[string]artifact.Content{"test/video-edit-reference": clip, "test/video-edit-reference-frames": frames, "test/video-edit-reference-mp4": movie} {
		if _, err := store.Commit(t.Context(), artifact.Batch{Key: key, Contents: []artifact.Content{content}}); err != nil {
			t.Fatal(err)
		}
	}
	recordedContext, err := os.ReadFile(filepath.Join(filepath.Dir(sourcePixels), "text_context.f32le"))
	if err != nil {
		t.Fatal(err)
	}
	reference, err := media.DecodeVideoFileAt(t.Context(), referenceClip, ffmpeg, referenceEditFPS, referenceEditFrames, referenceEditWidth, referenceEditHeight)
	if err != nil || len(reference) != referenceEditFrames {
		t.Fatalf("the reference clip gave %d frames (%v), want %d", len(reference), err, referenceEditFrames)
	}

	// The runtime alone, given the recorded frames, agrees with the reference
	// clip at 41 dB over the squares. A request that carries the frames reaches
	// the runtime with them at the byte they are written at, and its edit is
	// served and held to the reference. One that carries a GIF keeps its source
	// at 33 dB, and the smoothing of the GIF's dither at 36: its edit agrees at
	// 28 dB, and unsmoothed, with the editor reading the dither as content, at
	// 17. The GIF is held at what it resolves to, which a source left unsmoothed
	// or read in another range fails, and its edit is not served a second time.
	// An MP4 source is held to the GIF's floor, which one read in another range
	// or cut to other frames fails. The edit of the frames is served as an MP4,
	// which keeps every colour the GIF's palette does not.
	for _, carried := range []struct {
		name        string
		source      artifact.ID
		sourceFloor float64
		served      bool
	}{
		{"a GIF", clip.Descriptor.ID, gifSourceFloor, false},
		{"an MP4", movie.Descriptor.ID, gifSourceFloor, false},
		{"frames", frames.Descriptor.ID, framesSourceFloor, true},
	} {
		form := latentvideo.ReferenceEditRequest{Prompt: referenceEditInstruction, Seed: referenceEditSeed, SourceArtifact: carried.source, Format: "mp4"}
		// What the request resolves to, held to what the earlier run recorded: the
		// source frames it fed and the text context it conditioned on. A source
		// read from -1 to 1 misses the first by the whole range, and a context
		// projected by the base model's projection misses the second by a quarter.
		resolved, err := latentvideo.ResolveClipRequest(t.Context(), store, edit, form)
		if err != nil {
			t.Fatal(err)
		}
		if len(resolved.Source.Pixels) != len(recordedSource.Pixels) || len(recordedContext) != len(resolved.Condition.TextContext)*binary.Size(float32(0)) {
			t.Fatalf("source as %s: the request resolves to %d source values and %d of context, the earlier run to %d and %d bytes",
				carried.name, len(resolved.Source.Pixels), len(resolved.Condition.TextContext), len(recordedSource.Pixels), len(recordedContext))
		}
		var sourceError float64
		for index, value := range resolved.Source.Pixels {
			sourceError += float64(value-recordedSource.Pixels[index]) * float64(value-recordedSource.Pixels[index])
		}
		sourceRatio := -10 * math.Log10(sourceError/float64(len(resolved.Source.Pixels)))
		var dot, servedNorm, recordedNorm float64
		for index, value := range resolved.Condition.TextContext {
			want := float64(math.Float32frombits(binary.LittleEndian.Uint32(recordedContext[index*binary.Size(float32(0)):])))
			dot, servedNorm, recordedNorm = dot+float64(value)*want, servedNorm+float64(value)*float64(value), recordedNorm+want*want
		}
		contextCosine := dot / math.Sqrt(servedNorm*recordedNorm)
		t.Logf("source as %s, resolved request against the earlier run: source frames %.2f dB, text context cosine %.6f", carried.name, sourceRatio, contextCosine)
		// The context differs by rounding alone.
		const contextFloor = 0.999
		if sourceRatio < carried.sourceFloor || contextCosine < contextFloor {
			t.Fatalf("source as %s: the request resolves to source frames at %.2f dB and a text context at cosine %.6f of the earlier run's", carried.name, sourceRatio, contextCosine)
		}
		if !carried.served {
			continue
		}
		video := serve(form)
		frames := servedSquareMeans(t, video, referenceEditBlock)
		if video.MediaType != media.MP4MediaType || video.Frames != referenceEditFrames || video.Height != referenceEditHeight || video.Width != referenceEditWidth || len(frames) != referenceEditFrames {
			t.Fatalf("source as %s: served edit as %s of %d frames at %dx%d, %d decoded", carried.name, video.MediaType, video.Frames, video.Width, video.Height, len(frames))
		}
		var squared, servedDark, servedBright, wantDark, wantBright float64
		blocks := 0
		for frame, served := range frames {
			want := squareMeans(reference[frame], referenceEditBlock)
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
		t.Logf("source as %s, served edit against the earlier implementation's clip: %.2f dB over %d-pixel squares; darkest squares %.1f against %.1f, brightest %.1f against %.1f of %d",
			carried.name, agreement, referenceEditBlock, servedDark, wantDark, servedBright, wantBright, full)
		// An output read in the wrong range lifts the darkest squares by a third of the range.
		const levelSlack = 16
		if agreement < framesAgreementFloor || math.Abs(servedDark-wantDark) > levelSlack || math.Abs(servedBright-wantBright) > levelSlack {
			t.Fatalf("source as %s: the served edit agrees with the reference at %.2f dB, with dark squares at %.1f for %.1f and bright ones at %.1f for %.1f",
				carried.name, agreement, servedDark, wantDark, servedBright, wantBright)
		}
	}
}

// squareMeans is a frame as the mean of each colour over each square of side
// pixels, row by row, on the byte scale.
func squareMeans(frame image.Image, side int) []float64 {
	bounds := frame.Bounds()
	columns, rows := bounds.Dx()/side, bounds.Dy()/side
	means := make([]float64, media.RGBChannels*rows*columns)
	const scale = math.MaxUint16 / math.MaxUint8
	for y := range rows * side {
		for x := range columns * side {
			red, green, blue, _ := frame.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			block := (y/side*columns + x/side) * media.RGBChannels
			for channel, value := range []uint32{red, green, blue} {
				means[block+channel] += float64(value) / scale / float64(side*side)
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
