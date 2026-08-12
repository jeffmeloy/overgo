package densecausal

import (
	"runtime"
	"testing"

	"overgo/internal/optimizer"
)

// TestBF16TrainingPersistentFootprint measures the persistent training-state
// bytes of the current bf16-SGD path and pins the SQA finding numerically: it
// does NOT reduce memory. The plain fp32-SGD floor is weights+grads = 2*n*4; the
// bf16-master TARGET is ~n*2 (master) + streamed grads. The current path sits
// ABOVE the fp32 floor (Model.Weights fp32 + flat fp32 weights + fp32 grads +
// fp32 work slab + bf16 master), so this asserts the gap that the native-bf16
// forward (reading the master directly, no full fp32 copy) must close.
func TestBF16TrainingPersistentFootprint(t *testing.T) {
	g := readGolden(t, "llama_train_golden.json", "llama_train_golden/v1")
	m := modelFromGolden(t, g)
	var n int
	for _, w := range m.Weights {
		n += len(w)
	}
	fp32SGDFloor := uint64(n) * 4 * 2 // weights + grads: the minimum plain fp32-SGD needs
	bf16MasterTarget := uint64(n) * 2 // the master alone; the real path's weight store

	// Measure the heap the current bf16 path holds resident for its persistent
	// buffers (flat weights + grads + work slab + bf16 master), on top of the
	// model's own fp32 Model.Weights.
	runtime.GC()
	var before runtime.MemStats
	runtime.ReadMemStats(&before)
	names, weights, gradients, _, _, err := m.trainSetup(1.0)
	if err != nil {
		t.Fatal(err)
	}
	opt := optimizer.NewBF16SGD(weights, 1)
	work := make([]float32, len(weights))
	var after runtime.MemStats
	runtime.ReadMemStats(&after)
	persistent := after.HeapAlloc - before.HeapAlloc
	runtime.KeepAlive(names)
	runtime.KeepAlive(weights)
	runtime.KeepAlive(gradients)
	runtime.KeepAlive(opt)
	runtime.KeepAlive(work)

	modelWeights := uint64(n) * 4 // Model.Weights, retained fp32 throughout training
	t.Logf("n=%d | Model.Weights(fp32)=%d | bf16-path persistent buffers=%d | total>=%d",
		n, modelWeights, persistent, modelWeights+persistent)
	t.Logf("fp32-SGD floor (weights+grads)=%d | bf16-master TARGET=%d", fp32SGDFloor, bf16MasterTarget)

	// The finding, measured: the current path's resident state (Model.Weights +
	// its buffers) exceeds the fp32-SGD floor -- it is not a reduction.
	total := modelWeights + persistent
	if total <= fp32SGDFloor {
		t.Fatalf("expected current bf16 path to EXCEED the fp32-SGD floor (finding); total=%d floor=%d", total, fp32SGDFloor)
	}
	t.Logf("CONFIRMED: current bf16 path resident %d > fp32-SGD floor %d (%.2fx); native-bf16 forward must drive it toward the %d master target",
		total, fp32SGDFloor, float64(total)/float64(fp32SGDFloor), bf16MasterTarget)
}
