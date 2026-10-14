package latentvideo

import (
	"bytes"
	"context"
	"encoding/json"
	"image"
	"math"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

// movingClip is a clip of smooth colour waves that move a few pixels a frame,
// planar, from 0 to 1: the kind of picture a generator draws, which a palette
// dithers and H.264 keeps.
func movingClip(frames, height, width int) [][]float32 {
	clip := make([][]float32, frames)
	plane := height * width
	for frame := range clip {
		clip[frame] = make([]float32, media.RGBChannels*plane)
		for channel := range media.RGBChannels {
			for at := range plane {
				x, y := float64(at%width+3*frame), float64(at/width)
				wave := math.Sin(2*math.Pi*x/float64(width)+float64(channel)) * math.Cos(math.Pi*y/float64(height)+float64(frame)/8)
				clip[frame][channel*plane+at] = float32(0.5 + 0.4*wave)
			}
		}
	}
	return clip
}

// writeTestClip writes the clip in the format through the encoder a served
// request is written by.
func writeTestClip(t *testing.T, format ClipFormat, fps int, clip [][]float32, height, width int) EncodedVideo {
	t.Helper()
	encoder, err := newServedEncoder(t.Context(), format, fps, UnitPixels)
	if err != nil {
		t.Fatal(err)
	}
	video, err := writeClip(t.Context(), encoder, func(sink VideoFrameSink) error {
		for index, frame := range clip {
			if err := sink(index, frame, height, width); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("format %q: %v", format, err)
	}
	return video
}

// TestMP4ClipContent holds the MP4 a served clip can be written as beside the
// GIF it is written as by default.
//
// A request, an edit condition's record and a session key that name GIF are
// the bytes of ones that name no format, so no answer the store holds for a
// GIF request moves; an MP4 edit's condition record differs, so its answer is
// its own. The same moving frames written both ways come back as the same
// count of frames at the same size, the MP4 nearer the frames and in fewer
// bytes than the GIF. Each clip, read back from what it published as a stored
// answer is, publishes as the same artifact with its frame rate. A frame of an
// odd size, GIF bytes labelled as an MP4, and a format neither GIF nor MP4 are
// refused. An MP4 source clip is cut to the editor's size and to the most
// frames an edit policy admits, and kept whole when the policy admits it all.
func TestMP4ClipContent(t *testing.T) {
	t.Parallel()
	plain := WanRequest{Prompt: "a fox", Seed: 7, Frames: 5, Width: 32, Height: 16}
	named, moving := plain, plain
	named.Format, moving.Format = gifClip, mp4Clip
	plainBytes, err := json.Marshal(plain)
	if err != nil {
		t.Fatal(err)
	}
	namedBytes, err := json.Marshal(named)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(plainBytes, namedBytes) || bytes.Contains(plainBytes, []byte(`"format"`)) {
		t.Fatalf("a request naming GIF is %s, one naming no format %s", namedBytes, plainBytes)
	}
	plainKey, _ := plain.SessionKey()
	movingKey, _ := moving.SessionKey()
	if plainKey != movingKey {
		t.Fatalf("an MP4 request keys session %q, a GIF one %q: one resident generator serves both", movingKey, plainKey)
	}
	condition := ReferenceEditCondition{TextContext: []float32{1}, FramesPerChunk: 3, LocalAttention: 21, Timesteps: []int64{1000}, Sigmas: []float32{1}, Seed: 42}
	namedCondition, mp4Condition := condition, condition
	namedCondition.Format, mp4Condition.Format = gifClip, mp4Clip
	records := make([][]byte, 0, 3)
	for _, each := range []ReferenceEditCondition{condition, namedCondition, mp4Condition} {
		record, err := json.Marshal(each.Record())
		if err != nil {
			t.Fatal(err)
		}
		records = append(records, record)
	}
	if !bytes.Equal(records[0], records[1]) || bytes.Contains(records[0], []byte(`"format"`)) || bytes.Equal(records[0], records[2]) {
		t.Fatalf("edit condition records: none %s, GIF %s, MP4 %s", records[0], records[1], records[2])
	}
	unknown := ClipFormat("webm")
	clipID := testutil.ArtifactID(t, artifact.KindFile, "mp4-clip-content-source")
	if ValidateWanRequestForm(WanRequest{Prompt: "a fox", Format: unknown}) == nil ||
		validateReferenceEditRequestForm(ReferenceEditRequest{Prompt: "a fox", SourceArtifact: clipID, Format: unknown}) == nil {
		t.Fatal("a request for a WebM clip was admitted")
	}
	if _, err := newServedEncoder(t.Context(), unknown, 16, UnitPixels); err == nil {
		t.Fatal("an encoder for a WebM clip was made")
	}

	ffmpeg, err := media.FFmpeg()
	if err != nil {
		testskip.NotApplicable(t, "no FFmpeg to write an MP4 with: "+err.Error())
	}
	const frames, height, width, fps = 9, 96, 128, 16
	clip := movingClip(frames, height, width)
	gifVideo, mp4Video := writeTestClip(t, "", fps, clip, height, width), writeTestClip(t, mp4Clip, fps, clip, height, width)
	if gifVideo.MediaType != media.GIFMediaType || mp4Video.MediaType != media.MP4MediaType {
		t.Fatalf("clips written as %q and %q", gifVideo.MediaType, mp4Video.MediaType)
	}
	gifFrames, err := media.DecodeGIF(bytes.NewReader(gifVideo.Data), 2*frames)
	if err != nil {
		t.Fatal(err)
	}
	mp4Frames, err := media.DecodeEncodedVideo(t.Context(), mp4Video.Data, ffmpeg, fps, 2*frames)
	if err != nil {
		t.Fatal(err)
	}
	ratio := func(decoded []image.Image) float64 {
		var squared float64
		for frame, picture := range decoded {
			bounds := picture.Bounds()
			for at := range height * width {
				red, green, blue, _ := picture.At(bounds.Min.X+at%width, bounds.Min.Y+at/width).RGBA()
				for channel, value := range []uint32{red, green, blue} {
					apart := float64(value>>8) - float64(encodeVideoByte(clip[frame][channel*height*width+at], UnitPixels))
					squared += apart * apart
				}
			}
		}
		return 20 * math.Log10(math.MaxUint8/math.Sqrt(squared/float64(len(decoded)*media.RGBChannels*height*width)))
	}
	for name, decoded := range map[string][]image.Image{"GIF": gifFrames, "MP4": mp4Frames} {
		if len(decoded) != frames || decoded[0].Bounds().Dx() != width || decoded[0].Bounds().Dy() != height {
			t.Fatalf("the %s comes back as %d frames of %v, written as %d of %dx%d", name, len(decoded), decoded[0].Bounds(), frames, width, height)
		}
	}
	gifRatio, mp4Ratio := ratio(gifFrames), ratio(mp4Frames)
	t.Logf("GIF %d bytes at %.2f dB, MP4 %d bytes at %.2f dB", len(gifVideo.Data), gifRatio, len(mp4Video.Data), mp4Ratio)
	if mp4Ratio <= gifRatio || len(mp4Video.Data) >= len(gifVideo.Data) {
		t.Fatalf("the MP4 is %d bytes at %.2f dB and the GIF %d bytes at %.2f dB; the MP4 is to be nearer the frames in fewer bytes",
			len(mp4Video.Data), mp4Ratio, len(gifVideo.Data), gifRatio)
	}

	for _, video := range []EncodedVideo{gifVideo, mp4Video} {
		published, err := VideoContent(video)
		if err != nil {
			t.Fatal(err)
		}
		var replayed EncodedVideo
		if err := replayed.DecodeContent(published); err != nil {
			t.Fatal(err)
		}
		again, err := VideoContent(replayed)
		if err != nil {
			t.Fatalf("a stored %s answer does not publish: %v", video.MediaType, err)
		}
		if again.Descriptor != published.Descriptor || !bytes.Equal(again.Data, published.Data) ||
			replayed.Frames != frames || replayed.Height != height || replayed.Width != width || replayed.FPS != fps {
			t.Fatalf("a %s read back is %+v as %s", video.MediaType, replayed, again.Descriptor.Schema)
		}
	}

	odd, err := newServedEncoder(t.Context(), mp4Clip, fps, UnitPixels)
	if err != nil {
		t.Fatal(err)
	}
	if err := odd.add(t.Context(), 0, make([]float32, media.RGBChannels*15*16), 15, 16); err == nil {
		t.Fatal("an MP4 frame of 16 by 15 pixels was taken")
	}
	if err := odd.Close(); err != nil {
		t.Fatal(err)
	}
	mislabelled := gifVideo
	mislabelled.MediaType = media.MP4MediaType
	if _, err := VideoContent(mislabelled); err == nil {
		t.Fatal("GIF bytes labelled as an MP4 were published")
	}
	var misread EncodedVideo
	if err := misread.DecodeContent(artifact.Content{Descriptor: artifact.Descriptor{MediaType: media.MP4MediaType}, Data: gifVideo.Data}); err == nil {
		t.Fatal("GIF bytes labelled as an MP4 were read back as one")
	}

	// A policy of one latent frame a chunk admits two latent frames when it
	// attends to one before, which is five frames at the stride of four, and nine
	// when it attends to eight: the nine-frame clip is cut to five, and is kept
	// whole.
	source, err := VideoContent(mp4Video)
	if err != nil {
		t.Fatal(err)
	}
	profile := Profile{SampleFPS: fps, Policy: DenoiserPolicy{VAEStride: [3]int{4, 8, 8}}}
	for attended, want := range map[int]int{1: 5, 8: frames} {
		policy := editPolicy{FramesPerChunk: 1, LocalAttentionFrames: attended, SourceWidth: 48, SourceHeight: 64}
		cut, err := sourceVideoFromContent(t.Context(), source, policy, profile)
		if err != nil {
			t.Fatal(err)
		}
		if cut.Frames != want || cut.Width != policy.SourceWidth || cut.Height != policy.SourceHeight || cut.Channels != media.RGBChannels {
			t.Fatalf("attending to %d frames, the MP4 source is cut to %d frames of %dx%d, want %d of %dx%d",
				attended, cut.Frames, cut.Width, cut.Height, want, policy.SourceWidth, policy.SourceHeight)
		}
	}
}

// TestRepeatedGIFRequestReplaysAndPublishes holds a repeated video request to
// its first answer. The first is served by a stand-in generator through the
// served stages and published; the second is answered by the store from the
// clip it holds, read back with the frame rate its delays show. That clip
// publishes as the artifact the first did: a clip read back with no rate is
// refused by the publish check, and a repeated request fails where its first
// was served.
func TestRepeatedGIFRequestReplaysAndPublishes(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := store.Close(); err != nil {
			t.Error(err)
		}
	}()
	profile, err := declaredProfile()
	if err != nil {
		t.Fatal(err)
	}
	profileContent, err := profile.Content()
	if err != nil {
		t.Fatal(err)
	}
	model := testutil.ArtifactID(t, artifact.KindModel, "wan-gif-replay-fixture")
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "test/artifact/" + model.String(), Artifacts: []artifact.Descriptor{{ID: model}}, Contents: []artifact.Content{profileContent},
	}); err != nil {
		t.Fatal(err)
	}
	definition, err := modelrecipe.GenerationDefinition(modelrecipe.ModuleLatentVideoPrepare, model, profile.ID)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	const frames, height, width = 5, 16, 32
	request := WanRequest{Prompt: "a fox", Seed: 7, Frames: frames, Width: width, Height: height, Steps: 2}
	content, err := capabilityruntime.InputContent("overgo.video-gen-input.v1", request)
	if err != nil {
		t.Fatal(err)
	}
	input := definition.Inputs[0]
	inputs := map[recipe.PortName]workflowruntime.Value{input.Name: workflowruntime.ArtifactValue(input.Data, request, content)}
	key := capabilityruntime.RunKey(definition, content)
	clip := movingClip(frames, height, width)
	served, _, err := capabilityruntime.ExecuteMeasured[EncodedVideo](t.Context(), store, model, program, key, inputs, func(runtime *workflowruntime.Runtime) error {
		return registerWanRuntime(runtime, model, func(ctx context.Context, request WanRequest) (EncodedVideo, error) {
			encoder, err := newServedEncoder(ctx, request.Format, profile.SampleFPS, UnitPixels)
			if err != nil {
				return EncodedVideo{}, err
			}
			return writeClip(ctx, encoder, func(sink VideoFrameSink) error {
				for index, frame := range clip {
					if err := sink(index, frame, height, width); err != nil {
						return err
					}
				}
				return nil
			})
		})
	})
	if err != nil {
		t.Fatal(err)
	}
	replayed, _, err := capabilityruntime.ExecuteMeasured[EncodedVideo](t.Context(), store, model, program, key, inputs, nil)
	if err != nil {
		t.Fatal(err)
	}
	if replayed.Data == nil {
		t.Fatal("the store holds no answer to the request it served")
	}
	first, err := VideoContent(served)
	if err != nil {
		t.Fatal(err)
	}
	again, err := VideoContent(replayed)
	if err != nil {
		t.Fatalf("the repeated request's GIF, read back at %d frames a second, does not publish: %v", replayed.FPS, err)
	}
	if again.Descriptor != first.Descriptor || replayed.FPS != profile.SampleFPS || replayed.Frames != frames {
		t.Fatalf("the repeated request publishes %s from %d frames at %d a second; the first published %s from %d at %d",
			again.Descriptor.ID, replayed.Frames, replayed.FPS, first.Descriptor.ID, served.Frames, served.FPS)
	}
}
