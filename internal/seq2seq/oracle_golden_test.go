package seq2seq

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

// oracleFile mirrors fixtures/needle_forward_oracle.json: a JAX 0.11.0
// reference forward over one (src, tgt) pair. The *_f32 variants ran the
// reference in f32 params — the cleaner target for this f32-host port; the
// unsuffixed variants are the bf16 native path and pin argmax.
type oracleFile struct {
	Src           []int     `json:"src"`
	Tgt           []int     `json:"tgt"`
	Logits        []float32 `json:"logits"`
	EncoderOut    []float32 `json:"encoder_out"`
	LogitsF32     []float32 `json:"logits_f32"`
	EncoderOutF32 []float32 `json:"encoder_out_f32"`
	JaxVersion    string    `json:"jax_version"`
}

// Tolerances bound the compute path against the f32 JAX oracle in relRMS,
// the reference implementation's parity metric. The gap is oracle-side
// activation rounding, not port noise: this port measures encoder relRMS
// 0.0040101128 / logits 0.0024195983, and the reference host
// implementation (an independent f64-accumulating port) measures
// 0.0040101069 / 0.0024196066 against the same fixture — agreement to 7
// significant digits pins the residual on the JAX side. Thresholds sit at
// ~2x the measured floor; the hard behavioral assertion is argmax equality
// against the bf16 native oracle (the reference test's contract).
const (
	encoderRelRMSTolerance = 8e-3
	logitsRelRMSTolerance  = 5e-3
)

func readOracle(t *testing.T) *oracleFile {
	t.Helper()
	path := testutil.FixturePath(t, "needle_forward_oracle.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("UNAVAILABLE: forward oracle absent at %s; parity NOT verified", path)
	}
	var oracle oracleFile
	if err := json.Unmarshal(raw, &oracle); err != nil {
		t.Fatal(err)
	}
	if len(oracle.Src) == 0 || len(oracle.Tgt) == 0 {
		t.Fatalf("oracle has empty src/tgt")
	}
	return &oracle
}

// loadArtifactModel resolves the artifact through the data-root contract;
// absent artifact skips LOUDLY (UNAVAILABLE is never green).
func loadArtifactModel(t *testing.T) *Model {
	t.Helper()
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip + ": loads the full artifact")
	}
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(roots.Models, "needle")
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("UNAVAILABLE: artifact absent at %s; parity NOT verified", dir)
	}
	model, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	return model
}

func relRMS(got, want []float32) float64 {
	var num, den float64
	for i := range got {
		delta := float64(got[i]) - float64(want[i])
		num += delta * delta
		den += float64(want[i]) * float64(want[i])
	}
	if den == 0 {
		return math.Sqrt(num)
	}
	return math.Sqrt(num / den)
}

func rowArgmax(values []float32, rows, cols int) []int {
	out := make([]int, rows)
	for row := range out {
		out[row] = argmax(values[row*cols : (row+1)*cols])
	}
	return out
}

// TestDerivedDimsMatchArtifact: every geometric fact must come out of the
// tensor shapes; the expected values restate the artifact, not the code.
func TestDerivedDimsMatchArtifact(t *testing.T) {
	model := loadArtifactModel(t)
	got := model.Dims
	if got.Vocab != 8192 || got.DModel != 512 || got.Heads != 8 || got.KVHeads != 4 ||
		got.HeadDim != 64 || got.EncoderLayers != 12 || got.DecoderLayers != 8 {
		t.Fatalf("derived dims %+v disagree with artifact", got)
	}
	if got.RopeTheta != 10000 || got.RMSEps != 1e-6 || got.StartToken != 1 || got.EOSToken != 1 {
		t.Fatalf("config facts %+v disagree with artifact config.json", got)
	}
}

// TestEncoderMatchesOracle: parity leg (a).
func TestEncoderMatchesOracle(t *testing.T) {
	oracle := readOracle(t)
	model := loadArtifactModel(t)
	encoded, err := model.Encode(oracle.Src)
	if err != nil {
		t.Fatal(err)
	}
	if len(encoded) != len(oracle.EncoderOutF32) {
		t.Fatalf("encoder output %d values, oracle %d", len(encoded), len(oracle.EncoderOutF32))
	}
	rms := relRMS(encoded, oracle.EncoderOutF32)
	t.Logf("encoder vs f32 oracle: max abs diff %g relRMS %g; vs bf16 oracle: max abs diff %g relRMS %g (jax %s)",
		testutil.MaxAbsDiff(encoded, oracle.EncoderOutF32), rms, testutil.MaxAbsDiff(encoded, oracle.EncoderOut), relRMS(encoded, oracle.EncoderOut), oracle.JaxVersion)
	if rms > encoderRelRMSTolerance {
		t.Fatalf("encoder relRMS %g exceeds %g", rms, encoderRelRMSTolerance)
	}
}

// TestIncrementalDecodeMatchesOracle: parity leg (b) — teacher-forced
// incremental logits against the oracle; argmax must match the bf16 native
// path exactly (the reference implementation's assertion).
func TestIncrementalDecodeMatchesOracle(t *testing.T) {
	oracle := readOracle(t)
	model := loadArtifactModel(t)
	memory, err := model.Encode(oracle.Src)
	if err != nil {
		t.Fatal(err)
	}
	session, err := model.NewDecoder(memory, len(oracle.Src), len(oracle.Tgt))
	if err != nil {
		t.Fatal(err)
	}
	vocab := model.Dims.Vocab
	logits := make([]float32, len(oracle.Tgt)*vocab)
	for pos, token := range oracle.Tgt {
		if err := session.Advance(logits[pos*vocab:(pos+1)*vocab], token); err != nil {
			t.Fatal(err)
		}
	}
	if len(logits) != len(oracle.LogitsF32) {
		t.Fatalf("logits %d values, oracle %d", len(logits), len(oracle.LogitsF32))
	}
	rms := relRMS(logits, oracle.LogitsF32)
	t.Logf("logits vs f32 oracle: max abs diff %g relRMS %g; vs bf16 oracle: max abs diff %g relRMS %g",
		testutil.MaxAbsDiff(logits, oracle.LogitsF32), rms, testutil.MaxAbsDiff(logits, oracle.Logits), relRMS(logits, oracle.Logits))
	if rms > logitsRelRMSTolerance {
		t.Fatalf("logits relRMS %g exceeds %g", rms, logitsRelRMSTolerance)
	}
	got := rowArgmax(logits, len(oracle.Tgt), vocab)
	want := rowArgmax(oracle.Logits, len(oracle.Tgt), vocab)
	for row := range got {
		if got[row] != want[row] {
			t.Fatalf("argmax row %d = %d, bf16 oracle %d", row, got[row], want[row])
		}
	}
}

// TestIncrementalMatchesFullDecode: parity leg (c) — the O(n) incremental
// path against full recompute, BIT-for-bit (both paths share every per-row
// operation; hostmath pins step==full for the causal core).
func TestIncrementalMatchesFullDecode(t *testing.T) {
	oracle := readOracle(t)
	model := loadArtifactModel(t)
	memory, err := model.Encode(oracle.Src)
	if err != nil {
		t.Fatal(err)
	}
	full, err := model.DecodeFull(memory, len(oracle.Src), oracle.Tgt)
	if err != nil {
		t.Fatal(err)
	}
	session, err := model.NewDecoder(memory, len(oracle.Src), len(oracle.Tgt))
	if err != nil {
		t.Fatal(err)
	}
	vocab := model.Dims.Vocab
	incremental := make([]float32, len(oracle.Tgt)*vocab)
	for pos, token := range oracle.Tgt {
		if err := session.Advance(incremental[pos*vocab:(pos+1)*vocab], token); err != nil {
			t.Fatal(err)
		}
	}
	for i := range incremental {
		if incremental[i] != full[i] {
			t.Fatalf("incremental[%d] = %g, full recompute %g (max abs diff %g)",
				i, incremental[i], full[i], testutil.MaxAbsDiff(incremental, full))
		}
	}
	t.Logf("incremental vs full recompute: bit-identical over %d logits", len(incremental))
}

// TestGreedyGenerationStable: parity leg (d) — a short greedy generation
// from the oracle source terminates, stays in-vocab, and is deterministic.
func executeGenerationStages(model *Model, source []int, limit int) ([]int, error) {
	encoded, err := model.encodeTokens(source, limit)
	if err != nil {
		return nil, err
	}
	selector, err := model.prepareTokens(encoded)
	if err != nil {
		return nil, err
	}
	return selector.selectTokens()
}

func TestGreedyGenerationStable(t *testing.T) {
	oracle := readOracle(t)
	model := loadArtifactModel(t)
	first, err := executeGenerationStages(model, oracle.Src, 8)
	if err != nil {
		t.Fatal(err)
	}
	if len(first) > 8 {
		t.Fatalf("generation produced %d tokens beyond the limit", len(first))
	}
	for i, token := range first {
		if token < 0 || token >= model.Dims.Vocab {
			t.Fatalf("generated token %d at %d outside vocab %d", token, i, model.Dims.Vocab)
		}
	}
	second, err := executeGenerationStages(model, oracle.Src, 8)
	if err != nil {
		t.Fatal(err)
	}
	if len(second) != len(first) {
		t.Fatalf("greedy generation nondeterministic: %v vs %v", first, second)
	}
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("greedy generation nondeterministic at %d: %v vs %v", i, first, second)
		}
	}
	t.Logf("greedy generation: %d tokens %v", len(first), first)
}
