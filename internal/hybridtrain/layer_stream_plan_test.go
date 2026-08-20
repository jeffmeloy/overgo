package hybridtrain

import "testing"

// TestLayerStreamPlansPartitionSlabs proves the per-layer stream plans are an
// exact partition of the compiled matrix and vector plans: contiguous windows
// in layer order, no gap, no overlap, and every group claimed by exactly one
// layer. This is the layout contract the layer-streamed training lane steps
// through, so a binding-order drift fails here before it can corrupt a run.
func TestLayerStreamPlansPartitionSlabs(t *testing.T) {
	cfg := StackConfig{
		Types:  []LayerKind{FullAttention, LinearAttention, LinearAttention, FullAttention},
		Tokens: 3, Hidden: 8, Inter: 12, Eps: 1e-6,
		Heads: 2, KVHeads: 1, HeadDim: 4, RopeDim: 4, RopeTheta: 10000,
		GDNKeyHeads: 2, GDNValueHeads: 2, GDNHeadDim: 4, GDNConvK: 3,
	}
	m, err := BuildModel(cfg, 7)
	if err != nil {
		t.Fatal(err)
	}
	plans, err := m.layerStreamPlans()
	if err != nil {
		t.Fatal(err)
	}
	if len(plans) != len(cfg.Types) {
		t.Fatalf("plans cover %d layers, want %d", len(plans), len(cfg.Types))
	}
	matNext, vecNext := 0, 0
	for li, p := range plans {
		if p.matOff != matNext || p.matSize <= 0 {
			t.Fatalf("layer %d matrix window [%d,+%d) does not continue at %d", li, p.matOff, p.matSize, matNext)
		}
		if p.vecOff != vecNext || p.vecSize <= 0 {
			t.Fatalf("layer %d vector window [%d,+%d) does not continue at %d", li, p.vecOff, p.vecSize, vecNext)
		}
		if p.plan.ParameterCount() != p.matSize {
			t.Fatalf("layer %d sub-plan covers %d of %d window elements", li, p.plan.ParameterCount(), p.matSize)
		}
		matNext += p.matSize
		vecNext += p.vecSize
	}
	if matNext != m.MatrixParamCount() {
		t.Fatalf("matrix windows cover %d of %d elements", matNext, m.MatrixParamCount())
	}
	if vecNext != m.VectorParamCount() {
		t.Fatalf("vector windows cover %d of %d elements", vecNext, m.VectorParamCount())
	}

	budget, err := m.LayerStreamedBudget()
	if err != nil {
		t.Fatal(err)
	}
	maxLayer := 0
	for _, p := range plans {
		maxLayer = max(maxLayer, p.matSize)
	}
	if budget.ScratchElems != maxLayer || budget.ScratchElems >= budget.MasterElems {
		t.Fatalf("scratch bound %d, want largest layer %d strictly below masters %d",
			budget.ScratchElems, maxLayer, budget.MasterElems)
	}
	if budget.MasterElems != m.MatrixParamCount() || budget.MomentumElems != m.MatrixParamCount() {
		t.Fatalf("budget masters/momentum %d/%d, want %d", budget.MasterElems, budget.MomentumElems, m.MatrixParamCount())
	}
}
