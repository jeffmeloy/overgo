package hybridtrain

import "testing"

// TestResidentMatrixLayout is the device-free guard for the no-weight-motion loop:
// it builds the hybrid model and asserts the per-layer MATRIX-weight plans the
// resident loop turns into device pointers (a) resolve every expected matrix, (b)
// exactly tile the flat matW buffer (no gaps/overlaps), and (c) size each slot as
// rows*cols. If any offset were wrong, the resident weight pointers would address
// the wrong bytes; this catches that in CI without a GPU (mirrors how the CUDA
// parity test skips honestly when OVERGO_CUDA_TEST is unset).
func TestResidentMatrixLayout(t *testing.T) {
	cfg := smallHybrid()
	m, err := BuildModel(cfg, 20250812)
	if err != nil {
		t.Fatal(err)
	}
	plans, err := m.matrixPlans()
	if err != nil {
		t.Fatal(err)
	}
	if len(plans) != len(cfg.Types) {
		t.Fatalf("plans=%d, want %d layers", len(plans), len(cfg.Types))
	}
	if err := validateMatrixTiling(plans, m.MatrixParamCount()); err != nil {
		t.Fatalf("matrix tiling invalid: %v", err)
	}

	// Each slot's size must match its matrix geometry (rows*cols) from mats.
	byName := map[string]matDesc{}
	for _, md := range m.mats {
		byName[md.name] = md
	}
	total := 0
	for _, md := range m.mats {
		if md.size != md.rows*md.cols {
			t.Errorf("matrix %q size=%d != rows*cols=%d", md.name, md.size, md.rows*md.cols)
		}
		total += md.size
	}
	if total != m.MatrixParamCount() {
		t.Errorf("sum of matrix sizes=%d != matW len=%d", total, m.MatrixParamCount())
	}

	// The plans must reference exactly the mats set (no matrix left unmapped).
	mapped := 0
	for _, p := range plans {
		mapped += len(p.slots())
	}
	if mapped != len(m.mats) {
		t.Errorf("plans map %d matrices, but model has %d", mapped, len(m.mats))
	}
	t.Logf("layout OK: %d layers, %d matrices tiling %d matW elements", len(plans), len(m.mats), m.MatrixParamCount())
}
