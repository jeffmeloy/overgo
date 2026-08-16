//go:build modeltest

package thoughtbank

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/hfbpe"
)

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
	if delta := maxAbsDiff(candidateLogits, baselineLogits); delta > 2e-5 {
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
		worst = max(worst, maxAbsDiff(candidateLogits, baselineLogits))
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
