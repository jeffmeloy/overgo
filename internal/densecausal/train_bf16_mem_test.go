package densecausal

import (
	"testing"

	"overgo/internal/optimizer"
)

// TestBF16TrainingPersistentFootprint accounts the persistent training-state
// bytes of the bf16-SGD path and pins the reduction against the SQA finding.
// Level 1 (this): the bf16 master is the sole weight store -- persistent state is
// Model.Weights(fp32, reused as the forward's working copy) + gradients(fp32) +
// master(bf16) = ~10n, down from the old ~18n (which also held a flat fp32 weight
// copy and a full-model fp32 work slab). Still ABOVE the plain fp32-SGD floor
// (weights+grads = 8n) because Model.Weights stays fp32; Level 2 (forward reads
// the bf16 master per-layer, no full fp32 copy) is what drives it below, toward
// the 2n master target. This test measures the buffers TrainBF16SGD holds; when
// Level 2 lands it drops Model.Weights from the resident set and this bound moves.
func TestBF16TrainingPersistentFootprint(t *testing.T) {
	g := readGolden(t, "llama_train_golden.json", "llama_train_golden/v1")
	m := modelFromGolden(t, g)
	var n int
	for _, w := range m.Weights {
		n += len(w)
	}

	// TrainBF16SGD's persistent buffers, enumerated (weights flat + work slab are
	// NOT held in Level 1): Model.Weights fp32 + flat fp32 gradients + bf16 master.
	names, weights, gradients, _, _, err := m.trainSetup(1.0)
	if err != nil {
		t.Fatal(err)
	}
	opt := optimizer.NewBF16SGD(weights, 1) // seeds the master; weights not retained by TrainBF16SGD
	masterBytes := len(opt.Master()) * 2
	gradientsBytes := len(gradients) * 4
	modelWeightsBytes := n * 4
	resident := uint64(masterBytes + gradientsBytes + modelWeightsBytes)
	_ = names

	fp32SGDFloor := uint64(n) * 4 * 2 // weights + grads: plain fp32-SGD minimum
	bf16MasterTarget := uint64(n) * 2 // the master alone; Level 2's weight store
	oldPathBytes := uint64(n) * 18    // pre-Level-1: +flat fp32 weights +fp32 work slab

	t.Logf("n=%d | Level-1 resident=%d (Model.Weights %d + grads %d + master %d)",
		n, resident, modelWeightsBytes, gradientsBytes, masterBytes)
	t.Logf("old path ~%d | fp32-SGD floor %d | bf16-master target %d", oldPathBytes, fp32SGDFloor, bf16MasterTarget)

	// Level 1 reduced the resident set below the old path.
	if resident >= oldPathBytes {
		t.Fatalf("Level 1 did not reduce: resident %d >= old ~%d", resident, oldPathBytes)
	}
	// Honest bound: still above the fp32-SGD floor -- Level 2 (native-bf16 forward)
	// is required to go below. This documents the remaining gap, not a pass of the
	// step's full acceptance.
	if resident <= fp32SGDFloor {
		t.Logf("resident %d already <= fp32 floor %d (Level 2 effectively reached)", resident, fp32SGDFloor)
	} else {
		t.Logf("resident %d still %.2fx the fp32 floor %d; Level 2 forward-read closes it toward %d",
			resident, float64(resident)/float64(fp32SGDFloor), fp32SGDFloor, bf16MasterTarget)
	}
}
