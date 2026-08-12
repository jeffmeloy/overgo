package seriesforecast

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"overgo/internal/hostmath"
	"overgo/internal/testutil"
)

// goldenFile mirrors fixtures/timesfm_golden.json (schema timesfm_golden/v1),
// produced by the adaptive_new exporter from the transformers reference.
type goldenFile struct {
	Schema    string    `json:"schema"`
	Horizon   int       `json:"horizon"`
	Quantiles []float64 `json:"quantiles"`
	Cases     []struct {
		Context  []float32   `json:"context"`
		Forecast [][]float32 `json:"forecast"`
	} `json:"cases"`
	IntermediateTokenizer struct {
		Case         int         `json:"case"`
		PaddedSeries []float32   `json:"padded_series"`
		Masks        []float32   `json:"masks"`
		TokenizerOut [][]float32 `json:"tokenizer_out"`
		Layer0Out    [][]float32 `json:"layer0_out"`
	} `json:"intermediate_tokenizer"`
}

func readGolden(t *testing.T) *goldenFile {
	t.Helper()
	path := testutil.FixturePath(t, "timesfm_golden.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("UNAVAILABLE: forward golden absent at %s; parity NOT verified", path)
	}
	var g goldenFile
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if g.Schema != "timesfm_golden/v1" {
		t.Fatalf("golden schema %q", g.Schema)
	}
	return &g
}

func maxAbsDiff(got, want []float32) float64 {
	maxAbs := 0.0
	for i := range got {
		if d := math.Abs(float64(got[i]) - float64(want[i])); d > maxAbs {
			maxAbs = d
		}
	}
	return maxAbs
}

func flatten(rows [][]float32) []float32 {
	out := make([]float32, 0, len(rows)*len(rows[0]))
	for _, row := range rows {
		out = append(out, row...)
	}
	return out
}

// embedCase runs pad -> running stats -> RevIN embed for one golden case,
// returning everything later stages need.
func embedCase(t *testing.T, m *Model, context []float32) (padded, masks, hidden []float32, mu, sigma []float64) {
	t.Helper()
	padded, masks, err := padToPatches(context, make([]float32, len(context)), m.Dims.PatchLen)
	if err != nil {
		t.Fatal(err)
	}
	tokens := len(padded) / m.Dims.PatchLen
	mu, sigma = make([]float64, tokens), make([]float64, tokens)
	patchStats(padded, masks, m.Dims.PatchLen, mu, sigma)
	hidden = make([]float32, tokens*m.Dims.Hidden)
	if err := m.patchEmbed(hidden, padded, masks, mu, sigma); err != nil {
		t.Fatal(err)
	}
	return padded, masks, hidden, mu, sigma
}

// TestTokenizerMatchesGolden pins padding, masks, and the RevIN+tokenizer
// embedding against the reference intermediates.
func TestTokenizerMatchesGolden(t *testing.T) {
	if testing.Short() {
		t.Skip("loads ~930MB weights; skipped in -short")
	}
	g := readGolden(t)
	model, err := Load(artifactDir(t))
	if err != nil {
		t.Fatal(err)
	}
	padded, masks, hidden, _, _ := embedCase(t, model, g.Cases[g.IntermediateTokenizer.Case].Context)
	if d := maxAbsDiff(padded, g.IntermediateTokenizer.PaddedSeries); d != 0 {
		t.Fatalf("padded series diff %g", d)
	}
	if d := maxAbsDiff(masks, g.IntermediateTokenizer.Masks); d != 0 {
		t.Fatalf("masks diff %g", d)
	}
	// Tolerance rationale (mirrors the reference test): this host runs f64
	// accumulation against the reference's f32 path through one residual
	// block over a 64-wide input.
	const tol = 2e-3
	if d := maxAbsDiff(hidden, flatten(g.IntermediateTokenizer.TokenizerOut)); d > tol {
		t.Fatalf("tokenizer out max abs diff %g > %g", d, tol)
	}
}

// TestDecoderLayer0MatchesGolden pins one post-norm layer: RoPE-before-norm,
// per-dim softplus query scale, causal attention, sequential feed-forward.
func TestDecoderLayer0MatchesGolden(t *testing.T) {
	if testing.Short() {
		t.Skip("loads ~930MB weights; skipped in -short")
	}
	g := readGolden(t)
	model, err := Load(artifactDir(t))
	if err != nil {
		t.Fatal(err)
	}
	_, _, hidden, _, _ := embedCase(t, model, g.Cases[g.IntermediateTokenizer.Case].Context)
	l, err := model.layerWeights(0)
	if err != nil {
		t.Fatal(err)
	}
	tokens := len(hidden) / model.Dims.Hidden
	model.layerForward(hidden, l, hostmath.RopeInvFreq(model.Dims.RopeTheta, model.Dims.HeadDim), tokens)
	// Accumulated f64-vs-f32 divergence over attention + feed-forward.
	const tol = 5e-3
	if d := maxAbsDiff(hidden, flatten(g.IntermediateTokenizer.Layer0Out)); d > tol {
		t.Fatalf("layer 0 max abs diff %g > %g", d, tol)
	}
}

// TestForecastMatchesGolden is the port's forward parity gate: both golden
// cases through the full 20-layer stack and denormalized quantile head.
func TestForecastMatchesGolden(t *testing.T) {
	if testing.Short() {
		t.Skip("loads ~930MB weights; skipped in -short")
	}
	g := readGolden(t)
	model, err := Load(artifactDir(t))
	if err != nil {
		t.Fatal(err)
	}
	if model.Dims.Horizon != g.Horizon || model.Dims.Quantiles != 1+len(g.Quantiles) {
		t.Fatalf("dims horizon=%d quantiles=%d vs golden %d/%d", model.Dims.Horizon, model.Dims.Quantiles, g.Horizon, 1+len(g.Quantiles))
	}
	for ci, c := range g.Cases {
		got, err := model.Forecast(c.Context)
		if err != nil {
			t.Fatal(err)
		}
		want := flatten(c.Forecast)
		if len(got) != len(want) {
			t.Fatalf("case %d: forecast len %d, want %d", ci, len(got), len(want))
		}
		// Accumulated f64-vs-f32 divergence over 20 layers and the head.
		const tol = 1e-2
		if d := maxAbsDiff(got, want); d > tol {
			t.Fatalf("case %d: forecast max abs diff %g > %g", ci, d, tol)
		}
	}
}
