package evaluation

import (
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/hfconvert"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestACaptionTextReadsADirectoryAsItsConversion opens a Qwen 3.5 Hugging Face directory, its embedding table under
// the multimodal name, as a caption's text model, and the GGUF its conversion writes: through the one loader both give
// the same width, end token and embedding rows, with no file converted for the directory's own read. The fixture's
// tokenizer has no merges, so the rows are read for token ids rather than for an encoded text.
func TestACaptionTextReadsADirectoryAsItsConversion(t *testing.T) {
	t.Parallel()
	ids := []int{1, 5, 31}
	directory := testutil.Qwen35Directory(t)
	converted := filepath.Join(t.TempDir(), "converted.gguf")
	testcheck.Must(hfconvert.Convert(hfconvert.Options{Directory: directory, OutputPath: converted, Name: "fixture"}))(t)
	read := func(path string) (width, end int, rows []float32) {
		text := testcheck.Must(OpenCaptionText(path))(t)
		defer func() { testcheck.NoError(t, text.Close()) }()
		return text.Width, text.End, testcheck.Must(text.Rows(ids))(t)
	}
	width, end, rows := read(directory)
	wantWidth, wantEnd, wantRows := read(converted)
	if width == 0 || len(rows) != len(ids)*width || width != wantWidth || end != wantEnd || !slices.Equal(rows, wantRows) {
		t.Fatalf("the directory reads as width %d, end %d and %d row values; its conversion as width %d, end %d, rows equal %v",
			width, end, len(rows), wantWidth, wantEnd, slices.Equal(rows, wantRows))
	}
}
