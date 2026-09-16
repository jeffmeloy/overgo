package audioparity

import (
	"testing"
	"time"
)

// adapterUpdateBudget bounds one CPU adapter update on the pinned real
// encoder. Measured 2026-09-14: 42 s before the host linear kernel read each
// weight row once per input row and the Newton-Schulz orthogonalizer fanned
// out, about 1 s after; the budget keeps a tenfold margin. The package's
// short suite fell from 590 s to 66 s on that update and the parallel marks.
const adapterUpdateBudget = 15 * time.Second

// TestAdapterUpdateWall holds the cost of the primitive every adapter
// acceptance repeats: one exact update of the linear CTC adapter. It runs
// before the parallel tests so the measurement is not theirs to contend.
func TestAdapterUpdateWall(t *testing.T) {
	l := newAdapterLifecycle(t)
	batcher := l.batcher(t, nil)
	started := time.Now()
	receipt := l.update(t, batcher)
	elapsed := time.Since(started)
	t.Logf("one adapter update: %s (loss %.4f, stream position %d)", elapsed.Round(time.Millisecond), receipt.Loss, receipt.Stream.Position)
	if elapsed > adapterUpdateBudget {
		t.Fatalf("adapter update took %s, over %s", elapsed.Round(time.Millisecond), adapterUpdateBudget)
	}
}
