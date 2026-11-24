package latentimage

import (
	"bytes"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/media"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// readmeFox is the README's Krea-2-Turbo picture, drawn at 512 by the decoder whose grid krea-degrid takes away.
const readmeFox = "assets/4f08c0fe12295fba0593c4932420b9d49defe20ce73a00b3b59c223f4f1840c3.png"

// TestKreasDecoderGridIsFinishedAway holds krea-degrid on the README's Krea fox: finished as a flow picture whose catalog
// entry declares the Qwen-Image grid, its grid over the whole picture falls from about 0.37 and 0.31 of a level along
// columns and rows to under 0.1, and in its smoothest quarter of 32-pixel blocks from about 0.13 to under 0.08, its
// 1-pixel texture within 1% and no value more than 3 levels (the 2-level limit and a rounding); finished without, it
// keeps its grid. The fox is already rounded to 8 bits, so in its flat blocks, where the grid is under half a level,
// the written picture keeps about half of it: the decoder's unrounded output, which the finish reads in service, is
// held by Krea's frozen generation on the card. Krea's catalog entry declares the grid and MOSS's does not. A
// sharpening of no strength writes the drawn picture's values, both written by one quantizer once the grid is taken
// (SQA-472).
func TestKreasDecoderGridIsFinishedAway(t *testing.T) {
	t.Parallel()
	krea := testcheck.Must(lookupCatalogProfile("Krea2Pipeline"))(t)
	moss := testcheck.Must(lookupCatalogProfile("MOSSImagePipeline"))(t)
	if !krea.QwenImageGrid || moss.QwenImageGrid {
		t.Fatalf("Krea's catalog entry declares the grid %v and MOSS's %v; want Krea's alone", krea.QwenImageGrid, moss.QwenImageGrid)
	}
	encoded := testcheck.Must(os.ReadFile(filepath.Join(testutil.RepoRoot(t), filepath.FromSlash(readmeFox))))(t)
	fox := testcheck.Must(png.Decode(bytes.NewReader(encoded)))(t)
	height, width := fox.Bounds().Dy(), fox.Bounds().Dx()
	// The decoder's output form: normalized HWC RGB in [-1, 1].
	pixels := make([]float32, height*width*media.RGBChannels)
	for row := range height {
		for column := range width {
			r, g, b, _ := fox.At(column, row).RGBA()
			for channel, value := range []uint32{r, g, b} {
				pixels[(row*width+column)*media.RGBChannels+channel] = float32(value>>8)/127.5 - 1
			}
		}
	}
	plain, _, err := finishedPNGs(pixels, height, width, nil, true, false)
	testcheck.NoError(t, err)
	finished, _, err := finishedPNGs(pixels, height, width, nil, true, true)
	testcheck.NoError(t, err)
	before, after := testutil.Luminance(fox), testutil.PNGLuminance(t, finished.Data)
	if columns, rows := testutil.PixelGrid(testutil.PNGLuminance(t, plain.Data), false); math.Abs(columns) < 0.3 || math.Abs(rows) < 0.25 {
		t.Fatalf("the fox finished without the grid declared holds a grid of %.3f and %.3f levels; want its own 0.37 and 0.31", columns, rows)
	}
	if columns, rows := testutil.PixelGrid(after, false); math.Abs(columns) > 0.1 || math.Abs(rows) > 0.1 {
		t.Errorf("the finished fox keeps a grid of %.3f and %.3f levels over the picture; want under 0.1", columns, rows)
	}
	if columns, rows := testutil.PixelGrid(after, true); math.Abs(columns) > 0.08 || math.Abs(rows) > 0.08 {
		t.Errorf("the finished fox keeps a grid of %.3f and %.3f levels in its flat blocks; want under 0.08", columns, rows)
	}
	if loss := 1 - testutil.PixelSteps(after)/testutil.PixelSteps(before); math.Abs(loss) > 0.01 {
		t.Errorf("the finish moved the fox's 1-pixel texture by %.2f%%; want under 1%%", 100*loss)
	}
	beforeRGB, afterRGB := testutil.PNGRGB(t, encoded), testutil.PNGRGB(t, finished.Data)
	for index := range beforeRGB {
		if moved := math.Abs(float64(afterRGB[index]) - float64(beforeRGB[index])); moved > 3 {
			t.Fatalf("a value of the fox moved %v levels; want 3 at most", moved)
		}
	}
	sharpened, drawn, err := finishedPNGs(pixels, height, width, &media.Sharpening{Radius: 1, Sigma: 0.6, Alpha: 0}, true, true)
	testcheck.NoError(t, err)
	if !slices.Equal(testutil.PNGRGB(t, sharpened.Data), testutil.PNGRGB(t, drawn.Data)) || !slices.Equal(testutil.PNGRGB(t, drawn.Data), afterRGB) {
		t.Error("a sharpening of no strength wrote values other than the drawn picture's")
	}
}
