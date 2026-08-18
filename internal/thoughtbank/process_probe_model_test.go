//go:build modeltest

package thoughtbank

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testutil"
)

func TestFractaleTwentyTokenProcessProbe(t *testing.T) {
	modelDirectory := os.Getenv("OVERGO_THOUGHTBANK_MODEL")
	if modelDirectory == "" {
		modelDirectory = `C:\Users\jeffm\adaptive_new\models\Fractale-350M-base`
	}
	weights, config, err := LoadCheckpoint(filepath.Join(modelDirectory, "model.pt"))
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]int32, 20)
	for index := range ids {
		ids[index] = int32((index * 1237) % config.Vocab)
	}
	memory := make([]float32, 4*config.MemDim)
	reference, err := FastWeightBankLMForward(ids, memory, 4, weights)
	if err != nil {
		t.Fatal(err)
	}
	prompt := len(ids) / 2
	_, promptLogits, err := FastWeightBankLMDecodeInit(weights, ids[:prompt], memory, 4)
	if err != nil {
		t.Fatal(err)
	}
	if delta := testutil.MaxAbsDiff(promptLogits, reference.Logits[(prompt-1)*config.Vocab:prompt*config.Vocab]); delta != 0 {
		t.Fatalf("prompt delta %.3e", delta)
	}
	state, logits, err := FastWeightBankLMDecodeInit(weights, ids[:1], memory, 4)
	if err != nil {
		t.Fatal(err)
	}
	for position := 1; position < len(ids); position++ {
		logits, err = FastWeightBankLMDecodeStep(state, ids[position])
		if err != nil {
			t.Fatal(err)
		}
		if delta := testutil.MaxAbsDiff(logits, reference.Logits[position*config.Vocab:(position+1)*config.Vocab]); delta != 0 {
			t.Fatalf("position %d delta %.3e", position, delta)
		}
	}
	bank, slots, err := state.MemBank()
	if err != nil {
		t.Fatal(err)
	}
	if slots != reference.Slots || testutil.MaxAbsDiff(bank, reference.MemBank) != 0 {
		t.Fatal("carried bank differs")
	}
	fmt.Printf("FRACTALE_PROCESS_PROBE tokens=%d vocab=%d\n", len(ids), config.Vocab)
}
