package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/evaluation"
	"overgo/internal/gguf"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestCaptureHoldsTheTrainingSplit holds a capture's samples to the training side of the suite they stand beside: no
// sample is one of the suite's clips or of their source videos, and none is captioned as one of a case's wrong answers;
// the samples are as many as the capture asks, the same for its seed and others for another seed.
func TestCaptureHoldsTheTrainingSplit(t *testing.T) {
	t.Parallel()
	const sources, scenes, cases, samples = 24, 2, 4, 10
	clips := t.TempDir()
	var rows []map[string]string
	for source := range sources {
		for scene := range scenes {
			vid := fmt.Sprintf("source%02d-Scene-%04d", source, scene)
			rows = append(rows, map[string]string{"vid": vid, "caption": "a caption of " + vid})
			testcheck.NoError(t, os.WriteFile(filepath.Join(clips, vid+".mp4"), nil, 0o600))
		}
	}
	captions := filepath.Join(t.TempDir(), "captions.json")
	testcheck.NoError(t, os.WriteFile(captions, testcheck.Must(json.Marshal(rows))(t), 0o600))
	spec := captureSpec{Clips: clips, Captions: captions, Cases: cases, SuiteSeed: 17, Samples: samples, Seed: 3}
	suite := testcheck.Must(evaluation.BuildFrameDescriptionMatching(evaluation.CaptionMatchingSpec{Clips: clips, Captions: captions, Cases: cases, Seed: 17}))(t)
	heldSources, heldCaptions := map[string]bool{}, map[string]bool{}
	for _, held := range suite.Cases {
		vid := held.Name[strings.LastIndex(held.Name, "/")+1:]
		source, _, _ := strings.Cut(vid, "-Scene-")
		heldSources[source] = true
		for _, candidate := range held.Candidates {
			heldCaptions[candidate] = true
		}
	}
	picked := testcheck.Must(captureSamples(spec))(t)
	if len(picked) != samples {
		t.Fatalf("the capture picked %d samples, want %d", len(picked), samples)
	}
	for _, sample := range picked {
		if heldSources[sample.Source] || heldCaptions[sample.Caption] {
			t.Errorf("sample %s stands on the held-out side of the suite", sample.Vid)
		}
	}
	again := testcheck.Must(captureSamples(spec))(t)
	spec.Seed++
	other := testcheck.Must(captureSamples(spec))(t)
	if !slices.Equal(again, picked) || slices.Equal(other, picked) {
		t.Error("the capture's samples do not follow its seed")
	}
}

// TestCaptureCacheAdmitsOnlyItsOwnInputs holds the capture's cache to the inputs that made a sample: a sample kept
// under its key is read back whole with its frame tokens' identity, and resumes; another caption, other bytes of the
// clip or other capture inputs give another key, under which it is no hit; and frame tokens cut short or a binding
// never written leave no hit either.
func TestCaptureCacheAdmitsOnlyItsOwnInputs(t *testing.T) {
	t.Parallel()
	cache, clip := t.TempDir(), filepath.Join(t.TempDir(), "clip.mp4")
	testcheck.NoError(t, os.WriteFile(clip, []byte("the clip"), 0o600))
	key := testcheck.Must(sampleKey("the inputs", clip, "a caption"))(t)
	values, tokens := []float32{1, 2, 3}, []float32{4, 5, 6, 7}
	frame := testcheck.Must(keepSample(cache, "clip", key, values, tokens))(t)
	held, heldFrame, kept := cachedSample(cache, "clip", key, len(values))
	if !kept || !slices.Equal(held, values) || heldFrame != frame {
		t.Fatalf("a sample kept under its key read back as %v, %v (%t); want %v and %v", held, heldFrame, kept, values, frame)
	}
	if _, _, kept := cachedSample(cache, "clip", key, len(values)+1); kept {
		t.Error("a sample of another count of values was a hit")
	}
	others := map[string]string{
		"another caption":      testcheck.Must(sampleKey("the inputs", clip, "another caption"))(t),
		"other capture inputs": testcheck.Must(sampleKey("other inputs", clip, "a caption"))(t),
	}
	testcheck.NoError(t, os.WriteFile(clip, []byte("another clip"), 0o600))
	others["other clip bytes"] = testcheck.Must(sampleKey("the inputs", clip, "a caption"))(t)
	for change, other := range others {
		if _, _, kept := cachedSample(cache, "clip", other, len(values)); other == key || kept {
			t.Errorf("with %s the sample kept under the old inputs was a hit", change)
		}
	}
	testcheck.NoError(t, writeFloats(filepath.Join(cache, "clip"+tokensSuffix), tokens[:2]))
	if _, _, kept := cachedSample(cache, "clip", key, len(values)); kept {
		t.Error("frame tokens cut short were a hit")
	}
	testcheck.Must(keepSample(cache, "unbound", key, values, tokens))(t)
	testcheck.NoError(t, os.Remove(filepath.Join(cache, "unbound"+bindingSuffix)))
	if _, _, kept := cachedSample(cache, "unbound", key, len(values)); kept {
		t.Error("a sample whose binding was never written was a hit")
	}
}

// TestACaptureReadsItsTextModelsFile reads a caption's target through a GGUF text model as the capture does: the mean
// of the file's token embedding rows for the caption's tokens, which the file's own tokenizer gives, with the file's
// width and end token beside it. A file whose tokenizer opens every prompt with a beginning token is refused.
func TestACaptureReadsItsTextModelsFile(t *testing.T) {
	t.Parallel()
	const caption, width, vocabulary, beginning, end = "abc", 8, 8, 1, 2
	if _, err := evaluation.OpenCaptionText(testutil.HermeticLlamaGGUFWith(t, 64, testutil.GGUFScalar("tokenizer.ggml.add_bos_token", gguf.ValueTypeBool, true))); err == nil || !strings.Contains(err.Error(), "beginning token") {
		t.Fatalf("a text model that opens every prompt with a beginning token was not refused for it: %v", err)
	}
	path := testutil.HermeticLlamaGGUF(t, 64)
	text := testcheck.Must(evaluation.OpenCaptionText(path))(t)
	defer text.Close()
	ids := testcheck.Must(text.Encode(caption))(t)
	mean := testcheck.Must(captionMean(text, caption))(t)
	// The file's table as the fixture wrote it, a row of its width for each token.
	raw := testcheck.Must(io.ReadAll(testutil.GGUFTensorF32("token_embd.weight", []uint64{width, vocabulary}, 1).Data))(t)
	table := make([]float32, width*vocabulary)
	testcheck.NoError(t, binary.Read(bytes.NewReader(raw), binary.LittleEndian, table))
	want := make([]float32, width)
	for _, id := range ids {
		for column := range width {
			want[column] += table[id*width+column] / float32(len(ids))
		}
	}
	if text.Width != width || text.End != end || len(ids) == 0 || slices.Contains(ids, beginning) || !slices.Equal(mean, want) {
		t.Fatalf("the capture reads %q as tokens %v and the mean row %v, width %d and end %d; want the file's rows' mean %v, width %d and end %d",
			caption, ids, mean, text.Width, text.End, want, width, end)
	}
}
