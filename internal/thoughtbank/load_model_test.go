//go:build modeltest

package thoughtbank

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/hfbpe"
)

func TestRealCheckpointCodeCompletion(t *testing.T) {
	modelDir := os.Getenv("OVERGO_THOUGHTBANK_MODEL")
	if modelDir == "" {
		modelDir = `C:\Users\jeffm\adaptive_new\models\Fractale-350M-base`
	}
	checkpoint := filepath.Join(modelDir, "model.pt")
	if _, err := os.Stat(checkpoint); err != nil {
		t.Skipf("Fractale checkpoint unavailable: %v", err)
	}
	weights, config, err := LoadCheckpoint(checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	tokenizer, err := hfbpe.Load(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := tokenizer.Encode("def fibonacci(n):")
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]int32, len(encoded))
	for index, id := range encoded {
		ids[index] = int32(id)
	}
	seedSlots := config.MemSeedSlots
	state, logits, err := FastWeightBankLMDecodeInit(
		weights, ids, make([]float32, seedSlots*config.MemDim), seedSlots,
	)
	if err != nil {
		t.Fatal(err)
	}
	generated := make([]int, 0, 8)
	for range 8 {
		token := argmax(logits)
		generated = append(generated, token)
		logits, err = FastWeightBankLMDecodeStep(state, int32(token))
		if err != nil {
			t.Fatal(err)
		}
	}
	completion := tokenizer.Decode(generated)
	const reference = "\n    \"\"\"\n    Returns the number of iterations"
	if completion != reference {
		t.Fatalf("completion %q does not match adaptive_new result %q", completion, reference)
	}
	t.Logf("real 386M completion: %q -> %q", "def fibonacci(n):", completion)
}

func argmax(values []float32) int {
	best := 0
	for index := 1; index < len(values); index++ {
		if values[index] > values[best] {
			best = index
		}
	}
	return best
}
