//go:build modeltest

package thoughtbank

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/hfbpe"
	"overgo/internal/testutil"
)

func TestFractalePrefillLeadership(t *testing.T) {
	if os.Getenv("OVERGO_FRACTALE_BASELINE") != "1" {
		t.Skip("set OVERGO_FRACTALE_BASELINE=1 for matched Fractale leadership")
	}
	modelDirectory := os.Getenv("OVERGO_THOUGHTBANK_MODEL")
	if modelDirectory == "" {
		modelDirectory = `C:\Users\jeffm\adaptive_new\models\Fractale-350M-base`
	}
	checkpoint := filepath.Join(modelDirectory, "model.pt")
	if _, err := os.Stat(checkpoint); err != nil {
		t.Skipf("Fractale checkpoint unavailable: %v", err)
	}
	weights, config, err := LoadCheckpoint(checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	tokenizer, err := hfbpe.Load(modelDirectory)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := tokenizer.Encode("def fibonacci(n):")
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]int32, len(encoded))
	for index, token := range encoded {
		ids[index] = int32(token)
	}
	memory := make([]float32, config.MemSeedSlots*config.MemDim)

	start := time.Now()
	state, logits, err := FastWeightBankLMDecodeInit(weights, ids, memory, config.MemSeedSlots)
	prefillWall := time.Since(start)
	if err != nil {
		t.Fatal(err)
	}
	start = time.Now()
	legacyState, legacyLogits, err := tokenPrefillReference(weights, ids, memory, config.MemSeedSlots)
	legacyWall := time.Since(start)
	if err != nil {
		t.Fatal(err)
	}
	start = time.Now()
	full, err := FastWeightBankLMForward(ids, memory, config.MemSeedSlots, weights)
	fullWall := time.Since(start)
	if err != nil {
		t.Fatal(err)
	}
	fullLogits := full.Logits[(len(ids)-1)*weights.VocabSize:]
	if delta := max(testutil.MaxAbsDiff(logits, legacyLogits), testutil.MaxAbsDiff(logits, fullLogits)); delta != 0 {
		t.Fatalf("prefill max logit delta %.3e", delta)
	}

	next := int32(argmaxLogit(logits))
	cachedNext, err := FastWeightBankLMDecodeStep(state, next)
	if err != nil {
		t.Fatal(err)
	}
	legacyNext, err := FastWeightBankLMDecodeStep(legacyState, next)
	if err != nil {
		t.Fatal(err)
	}
	continued := append(append([]int32(nil), ids...), next)
	full, err = FastWeightBankLMForward(continued, memory, config.MemSeedSlots, weights)
	if err != nil {
		t.Fatal(err)
	}
	fullNext := full.Logits[(len(continued)-1)*weights.VocabSize:]
	continuationDelta := max(testutil.MaxAbsDiff(cachedNext, legacyNext), testutil.MaxAbsDiff(cachedNext, fullNext))
	speedup := float64(legacyWall) / float64(prefillWall)
	t.Logf("Fractale matched prefill: prompt=%d full-cache=%s token-cache=%s full-only=%s cache-speedup=%.2fx continuation_delta=%.3e",
		len(ids), prefillWall, legacyWall, fullWall, speedup, continuationDelta)
	if continuationDelta != 0 {
		t.Fatalf("prefilled cache continuation delta %.3e", continuationDelta)
	}
	if speedup < 1.5 {
		t.Fatalf("full prefill speedup %.2fx is below 1.5x", speedup)
	}
	if prefillWall > 2*fullWall {
		t.Fatalf("cache-building prefill %s exceeds 2x full-only %s", prefillWall, fullWall)
	}
}

func tokenPrefillReference(weights *FastWeightBankLMWeights, ids []int32, memory []float32, slots int) (*FastWeightBankLMDecodeState, []float32, error) {
	state := &FastWeightBankLMDecodeState{
		w: weights, bank: memory, slots: slots,
		layers: make([]*fwbAttnLayerCache, len(weights.Blocks)),
	}
	for layer, block := range weights.Blocks {
		state.layers[layer] = newFwbAttnLayerCache(block.Attn)
	}
	var logits []float32
	for _, id := range ids {
		var err error
		logits, err = state.stepToken(id)
		if err != nil {
			return nil, nil, err
		}
	}
	return state, logits, nil
}

func TestFractaleIncrementalLeadership(t *testing.T) {
	if os.Getenv("OVERGO_FRACTALE_BASELINE") != "1" {
		t.Skip("set OVERGO_FRACTALE_BASELINE=1 for matched Fractale leadership")
	}
	modelDirectory := os.Getenv("OVERGO_THOUGHTBANK_MODEL")
	if modelDirectory == "" {
		modelDirectory = `C:\Users\jeffm\adaptive_new\models\Fractale-350M-base`
	}
	checkpoint := filepath.Join(modelDirectory, "model.pt")
	if _, err := os.Stat(checkpoint); err != nil {
		t.Skipf("Fractale checkpoint unavailable: %v", err)
	}
	weights, config, err := LoadCheckpoint(checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	tokenizer, err := hfbpe.Load(modelDirectory)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := tokenizer.Encode("def fibonacci(n):")
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]int32, len(encoded))
	for index, token := range encoded {
		ids[index] = int32(token)
	}
	memory := make([]float32, config.MemSeedSlots*config.MemDim)

	candidatePrefillStart := time.Now()
	state, candidateLogits, err := FastWeightBankLMDecodeInit(weights, ids, memory, config.MemSeedSlots)
	if err != nil {
		t.Fatal(err)
	}
	candidatePrefill := time.Since(candidatePrefillStart)
	baselinePrefillStart := time.Now()
	baseline, err := FastWeightBankLMForward(ids, memory, config.MemSeedSlots, weights)
	if err != nil {
		t.Fatal(err)
	}
	baselinePrefill := time.Since(baselinePrefillStart)
	baselineLogits := baseline.Logits[(len(ids)-1)*weights.VocabSize:]
	if delta := testutil.MaxAbsDiff(candidateLogits, baselineLogits); delta > 2e-5 {
		t.Fatalf("prefill max logit delta %.3e", delta)
	}

	const steps = 6
	var candidateWall, baselineWall time.Duration
	var worst float64
	generated := make([]int, 0, steps)
	for range steps {
		next := argmaxLogit(candidateLogits)
		if next != argmaxLogit(baselineLogits) {
			t.Fatalf("argmax differs at generated position %d", len(generated))
		}
		generated = append(generated, next)
		ids = append(ids, int32(next))

		candidateStart := time.Now()
		candidateLogits, err = FastWeightBankLMDecodeStep(state, int32(next))
		candidateWall += time.Since(candidateStart)
		if err != nil {
			t.Fatal(err)
		}
		baselineStart := time.Now()
		baseline, err = FastWeightBankLMForward(ids, memory, config.MemSeedSlots, weights)
		baselineWall += time.Since(baselineStart)
		if err != nil {
			t.Fatal(err)
		}
		baselineLogits = baseline.Logits[(len(ids)-1)*weights.VocabSize:]
		worst = max(worst, testutil.MaxAbsDiff(candidateLogits, baselineLogits))
	}
	completion := tokenizer.Decode(generated)
	speedup := float64(baselineWall) / float64(candidateWall)
	t.Logf("Fractale matched decode: prompt=%d steps=%d completion=%q prefill incremental=%s full=%s decode incremental=%s repeated-full=%s speedup=%.2fx max_delta=%.3e",
		len(encoded), steps, completion, candidatePrefill, baselinePrefill, candidateWall, baselineWall, speedup, worst)
	if worst > 2e-5 {
		t.Fatalf("decode max logit delta %.3e exceeds 2e-5", worst)
	}
	if speedup < 1.25 {
		t.Fatalf("incremental decode speedup %.2fx is below 1.25x", speedup)
	}
}
