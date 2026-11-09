package latentvideo

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/media"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
)

// TestAClipIsSavedAsMP4ByDefault holds a clip asked for in no format to an MP4. The format a page offers declares MP4
// as the default its intake gives a request that names none, one of its choices, and the request so filled records
// "mp4". The clip is H.264 in an MP4 written through FFmpeg, and names what wrote it: the executable's identity, read
// from its bytes, and the arguments it was given, which the serving record reads (media.EncodingOf); a GIF names none.
// The same frames written twice are the same bytes, and they read back nearer the frames the generator handed over
// than the GIF of them does, the format it replaces setting the bar. Where no FFmpeg resolves, an MP4 is refused, the
// refusal naming OVERGO_FFMPEG and "format": "gif".
//
// Not parallel: the refusal is read with OVERGO_FFMPEG naming a file that is not there.
func TestAClipIsSavedAsMP4ByDefault(t *testing.T) {
	var offered ClipFormat
	format := ClipFormat(offered.Default())
	if format != mp4Clip || !slices.Contains(offered.Choices(), string(format)) {
		t.Fatalf("a page offers %q of %v as the default", format, offered.Choices())
	}
	filled := testcheck.Must(json.Marshal(WanRequest{Prompt: "a fox", Seed: 7, Format: format}))(t)
	if !bytes.Contains(filled, []byte(`"format":"mp4"`)) {
		t.Fatalf("a request given the default records %s", filled)
	}

	ffmpeg, err := media.FFmpeg()
	if err != nil {
		testskip.NotApplicable(t, "no FFmpeg to write an MP4 with: "+err.Error())
	}
	const frames, height, width, fps = 9, 96, 128, 16
	clip := movingClip(frames, height, width)
	first, second := writeTestClip(t, format, fps, clip, height, width), writeTestClip(t, format, fps, clip, height, width)
	if first.MediaType != media.MP4MediaType || !media.OpensAsMP4(first.Data) {
		t.Fatalf("a clip in the default format is written as %q", first.MediaType)
	}
	if !bytes.Equal(first.Data, second.Data) {
		t.Fatalf("the same frames were written as %d and %d bytes that differ", len(first.Data), len(second.Data))
	}
	// Two encodes on one machine agree at any thread count, so the encoder's own record of its threads, which x264
	// writes into the stream, is read: one thread and one lookahead thread, the same bytes on any machine.
	for _, threads := range []string{" threads=1 ", " lookahead_threads=1 "} {
		if !bytes.Contains(first.Data, []byte(threads)) {
			t.Fatalf("x264's record in the clip names no%s: the encoder ran on more threads than one", threads)
		}
	}
	tool := testcheck.Must(artifact.IdentifyFile(artifact.KindFile, ffmpeg))(t)
	if recorded := media.EncodingOf(first); recorded.Tool != tool || !slices.Contains(recorded.Args, "libx264") ||
		!slices.Equal(recorded.Args, second.Encoding.Args) {
		t.Fatalf("the clip names the program %+v, want FFmpeg %s writing H.264", recorded, tool)
	}
	gifVideo := writeTestClip(t, "", fps, clip, height, width)
	if !media.EncodingOf(gifVideo).IsZero() {
		t.Fatalf("a GIF names the program %+v", gifVideo.Encoding)
	}
	mp4Ratio, gifRatio := readBackRatio(t, first, ffmpeg, clip), readBackRatio(t, gifVideo, ffmpeg, clip)
	t.Logf("MP4 %d bytes read back at %.2f dB, GIF %d bytes at %.2f dB", len(first.Data), mp4Ratio, len(gifVideo.Data), gifRatio)
	if mp4Ratio <= gifRatio {
		t.Fatalf("the MP4 reads back at %.2f dB and the GIF at %.2f dB; the MP4 is to lie nearer the frames", mp4Ratio, gifRatio)
	}

	t.Setenv("OVERGO_FFMPEG", filepath.Join(t.TempDir(), "ffmpeg.exe"))
	if _, err := newServedEncoder(t.Context(), format, fps, UnitPixels); err == nil ||
		!strings.Contains(err.Error(), "OVERGO_FFMPEG") || !strings.Contains(err.Error(), `"format": "gif"`) {
		t.Fatalf("with no FFmpeg an MP4 is refused with %v, which is to name OVERGO_FFMPEG and \"format\": \"gif\"", err)
	}
}
