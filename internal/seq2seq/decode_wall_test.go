package seq2seq

import (
	"sort"
	"testing"
	"time"
)

// TestDecodeWall: matched-protocol wall measurement against the reference's
// research-tagged oracle test (extmodel TestNeedleIncrementalDecodeMatchesJAXOracle).
// That test's reported elapsed covers artifact resolve+load, encode(src), and
// len(tgt)=8 incremental advances plus parity asserts; the honest match here
// is: load once (reported separately), then median-of-5 for (a) Encode +
// DecodeFull over the oracle pair and (b) an 8-token incremental Generate
// from the oracle src. Numbers land in docs/plan.json (rung6-needle
// performance-leg); this test keeps the protocol reproducible.
const decodeWallRuns = 5

func TestDecodeWall(t *testing.T) {
	oracle := readOracle(t)
	loadStart := time.Now()
	model := loadArtifactModel(t)
	t.Logf("load %.3fs", time.Since(loadStart).Seconds())
	median := func(name string, op func() error) {
		walls := make([]float64, decodeWallRuns)
		for run := range walls {
			start := time.Now()
			if err := op(); err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			walls[run] = time.Since(start).Seconds()
		}
		sort.Float64s(walls)
		t.Logf("%s: median %.4fs over %d runs (min %.4f max %.4f)",
			name, walls[len(walls)/2], decodeWallRuns, walls[0], walls[len(walls)-1])
	}
	median("encode+decode-full", func() error {
		memory, err := model.Encode(oracle.Src)
		if err != nil {
			return err
		}
		_, err = model.DecodeFull(memory, len(oracle.Src), oracle.Tgt)
		return err
	})
	median("generate-8", func() error {
		_, err := executeGenerationStages(model, oracle.Src, 8)
		return err
	})
}
