package vitencoder

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// registersGolden is what the reference produced for the synthetic picture
// (testdata/dinov2_registers_golden.json, written by
// scripts/python_golden_references/dinov2_registers_golden.py with torch and
// transformers): a sample of the pixels after torch's bicubic resize and the
// normalization, of the position embeddings fitted to the 16 by 16 grid, and
// of the patch tokens after a final norm without its learned scale and shift.
type registersGolden struct {
	Side            int       `json:"side"`
	Input           int       `json:"input"`
	Tokens          int       `json:"tokens"`
	Hidden          int       `json:"hidden"`
	PixelIndices    []int     `json:"pixel_indices"`
	Pixels          []float32 `json:"pixels"`
	PositionIndices []int     `json:"position_indices"`
	Position        []float32 `json:"position"`
	TokenIndices    []int     `json:"token_indices"`
	TokenValues     []float32 `json:"token_values"`
	TokenMean       float64   `json:"token_mean"`
	TokenAbsMean    float64   `json:"token_abs_mean"`
	Transformers    string    `json:"transformers"`
	Torch           string    `json:"torch"`
}

// TestDinov2WithRegistersMatchesReference holds the with-registers family to
// the reference as a representation autoencoder runs it: torch's bicubic
// resize of a float picture, the position embeddings of a checkpoint trained
// at a 37 by 37 grid fitted to 16 by 16, register tokens that take no
// position, the plain GELU feed-forward, and the patch tokens of a final norm
// without its affine.
func TestDinov2WithRegistersMatchesReference(t *testing.T) {
	t.Parallel()
	var reference registersGolden
	if err := jsonfile.Decode(filepath.Join("testdata", "dinov2_registers_golden.json"), &reference); err != nil {
		t.Fatal(err)
	}
	// The encoder is no registered model of its own: it sits beside the
	// collection whose decoder reads its tokens.
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	directory := roots.ResolveModelPath("dinov2-with-registers-base")
	if _, err := os.Stat(filepath.Join(directory, "model.safetensors")); err != nil {
		testskip.NotApplicable(t, "the with-registers encoder is absent: "+directory)
	}
	encoder, err := Load(directory)
	if err != nil {
		t.Fatal(err)
	}
	input, grid, hidden := encoder.Shape()
	if input != reference.Input || grid*grid != reference.Tokens || hidden != reference.Hidden {
		t.Fatalf("encoder takes %d, a %d grid of %d wide tokens; the reference %d, %d tokens of %d", input, grid, hidden, reference.Input, reference.Tokens, reference.Hidden)
	}
	// The synthetic picture, channel-first in [0, 1]: (x*7 + y*13 + c*29 + (x*y)%17) % 256.
	side := reference.Side
	picture := make([]float32, channels*side*side)
	for c := range channels {
		for y := range side {
			for x := range side {
				picture[(c*side+y)*side+x] = float32((x*7+y*13+c*29+(x*y)%17)%256) / 255
			}
		}
	}
	pixels, err := InterpolateBicubic(picture, channels, side, side, input, input)
	if err != nil {
		t.Fatal(err)
	}
	mean, deviation := encoder.Normalization()
	for c := range channels {
		plane := pixels[c*input*input:][:input*input]
		for index, value := range plane {
			plane[index] = (value - mean[c]) / deviation[c]
		}
	}
	worst := func(got []float32, indices []int, want []float32) float64 {
		var gap float64
		for index, at := range indices {
			gap = max(gap, math.Abs(float64(got[at])-float64(want[index])))
		}
		return gap
	}
	if gap := worst(pixels, reference.PixelIndices, reference.Pixels); gap > 1e-5 {
		t.Fatalf("the resized and normalized pixels differ from the reference by up to %.3g", gap)
	}
	if gap := worst(encoder.position, reference.PositionIndices, reference.Position); gap > 1e-5 {
		t.Fatalf("the fitted position embeddings differ from the reference by up to %.3g", gap)
	}
	tokens, err := encoder.Patches(pixels, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(tokens) != reference.Tokens*reference.Hidden {
		t.Fatalf("%d token values, want %d", len(tokens), reference.Tokens*reference.Hidden)
	}
	var total, magnitude float64
	for _, value := range tokens {
		total += float64(value)
		magnitude += math.Abs(float64(value))
	}
	gap := worst(tokens, reference.TokenIndices, reference.TokenValues)
	t.Logf("patch tokens: max abs diff %.3g over %d samples, mean %.3g (reference %.3g), abs mean %.6f (reference %.6f); reference from torch %s, transformers %s",
		gap, len(reference.TokenIndices), total/float64(len(tokens)), reference.TokenMean, magnitude/float64(len(tokens)), reference.TokenAbsMean,
		reference.Torch, reference.Transformers)
	if gap > 2e-3 || math.Abs(magnitude/float64(len(tokens))-reference.TokenAbsMean) > 1e-4 {
		t.Fatalf("the patch tokens differ from the reference: max abs diff %.3g", gap)
	}
	// With its affine the same forward is the checkpoint's own output, which
	// differs: the two norms are told apart.
	affine, err := encoder.Patches(pixels, true)
	if err != nil {
		t.Fatal(err)
	}
	if worst(affine, reference.TokenIndices, reference.TokenValues) < 1e-2 {
		t.Fatal("the final norm's affine changed nothing")
	}
}
