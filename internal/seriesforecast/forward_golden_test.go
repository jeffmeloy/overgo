package seriesforecast

import (
	"encoding/json"
	"os"
	"testing"

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

func flatten(rows [][]float32) []float32 {
	out := make([]float32, 0, len(rows)*len(rows[0]))
	for _, row := range rows {
		out = append(out, row...)
	}
	return out
}
