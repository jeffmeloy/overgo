package tabularicl

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// goldenFile mirrors fixtures/tabfm_predict_golden.json (schema
// tabfm_predict_golden/v1), captured from the adaptive_new server's
// /api/predict against the same artifact.
type goldenFile struct {
	Schema   string `json:"schema"`
	ModelDir string `json:"model_dir"`
	Cases    []struct {
		Name      string    `json:"name"`
		Task      string    `json:"task"`
		Rows      int       `json:"rows"`
		Cols      int       `json:"cols"`
		TrainRows int       `json:"train_rows"`
		X         []float32 `json:"x"`
		Y         []float32 `json:"y"`
		Outputs   []float32 `json:"outputs"`
		OutDim    int       `json:"out_dim"`
	} `json:"cases"`
}

// predictTolerance bounds the compute-path difference between this port and
// the golden's producer. The golden came from the reference host serving
// path (f32 storage, f64-accumulated linears and attention); this port
// reuses hostmath.Linear (f32 accumulation) and a f64 rope angle product
// where the reference rounds the angle product to f32 — accumulation noise
// through 24 residual ICL blocks at width 2048. Measured max abs diff:
// 7.63e-05 (classification logits), 1.29e-05 (regression); 1e-3 is a decade
// of headroom without admitting structural errors.
const predictTolerance = 1e-3

func readGolden(t *testing.T) *goldenFile {
	t.Helper()
	path := testutil.FixturePath(t, "tabfm_predict_golden.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("UNAVAILABLE: predict golden absent at %s; parity NOT verified", path)
	}
	var g goldenFile
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if g.Schema != "tabfm_predict_golden/v1" {
		t.Fatalf("golden schema %q", g.Schema)
	}
	return &g
}

// headDir resolves one task head of the real artifact through the data-root
// contract; absent artifact skips LOUDLY (UNAVAILABLE is never green).
func headDir(t *testing.T, task string) string {
	t.Helper()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(roots.Models, "tabfm-1.0.0-pytorch", task)
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("UNAVAILABLE: tabfm %s head absent at %s; parity NOT verified", task, dir)
	}
	return dir
}

// TestPredictParity loads one 6.5GB head at a time (both resident at once
// would double peak memory), asserts the derived dims against the artifact,
// and compares every golden case's outputs.
func TestPredictParity(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration + ": loads ~6.5GB weights per head")
	}
	golden := readGolden(t)
	wantDims := map[string]Dims{
		TaskClassification: {
			EmbedDim: 256, NumFreq: 32, FeatureGroupSize: 3, NumCLS: 8, NumInds: 256,
			ColHeads: 4, RowHeads: 8, ICLHeads: 8, ColBlocks: 3, RowBlocks: 3, ICLBlocks: 24,
			MaxClasses: 10, OutDim: 10, IsClassifier: true,
		},
		TaskRegression: {
			EmbedDim: 256, NumFreq: 32, FeatureGroupSize: 3, NumCLS: 8, NumInds: 256,
			ColHeads: 4, RowHeads: 8, ICLHeads: 8, ColBlocks: 3, RowBlocks: 3, ICLBlocks: 24,
			MaxClasses: 0, OutDim: 1, IsClassifier: false,
		},
	}
	for _, task := range Tasks() {
		head, err := LoadHead(headDir(t, task))
		if err != nil {
			t.Fatal(err)
		}
		if head.Dims != wantDims[task] {
			t.Fatalf("%s derived dims = %+v, want %+v", task, head.Dims, wantDims[task])
		}
		for _, c := range golden.Cases {
			if c.Task != task {
				continue
			}
			got, err := head.Predict(c.X, c.Y, c.Rows, c.Cols, c.TrainRows, nil)
			if err != nil {
				t.Fatalf("%s: %v", c.Name, err)
			}
			if len(got) != len(c.Outputs) || len(got) != c.Rows*c.OutDim || head.Dims.OutDim != c.OutDim {
				t.Fatalf("%s: outputs len %d out_dim %d, want len %d out_dim %d",
					c.Name, len(got), head.Dims.OutDim, len(c.Outputs), c.OutDim)
			}
			diff := testutil.MaxAbsDiff(got, c.Outputs)
			t.Logf("%s: max abs diff %.3g over %d outputs", c.Name, diff, len(got))
			if diff > predictTolerance {
				t.Fatalf("%s: max abs diff %g exceeds %g", c.Name, diff, predictTolerance)
			}
		}
		head = nil
		runtime.GC() // Release this head before loading the next 6.5GB.
	}
}
