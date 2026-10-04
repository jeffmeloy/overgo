package optimizer

import (
	"slices"
	"strings"
	"testing"
)

func TestTensorPackBindsDeterministicallyAndReplacesGradients(t *testing.T) {
	const (
		matrixRows = 2
		matrixCols = 2
		vectorRows = 2
		vectorCols = 1
	)
	matrix := []float32{1, 2, 3, 4}
	vector := []float32{5, 6}
	pack, err := NewTensorPack(
		map[string][]float32{"vector": vector, "matrix": matrix},
		MatrixGeometry(map[string][2]int{
			"matrix": {matrixRows, matrixCols},
			"vector": {vectorRows, vectorCols},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	first, _ := pack.plan.Group(0)
	second, _ := pack.plan.Group(1)
	if first.Name != "matrix" || second.Name != "vector" {
		t.Fatalf("group order = %q, %q", first.Name, second.Name)
	}
	copy(pack.weights, []float32{7, 8, 9, 10, 11, 12})
	pack.Scatter()
	if !slices.Equal(matrix, []float32{7, 8, 9, 10}) || !slices.Equal(vector, []float32{11, 12}) {
		t.Fatalf("scattered matrix=%v vector=%v", matrix, vector)
	}

	for index := range pack.gradients {
		pack.gradients[index] = 99
	}
	if err := pack.GatherGradients(map[string][]float32{"matrix": {1, 2, 3, 4}}); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(pack.gradients, []float32{1, 2, 3, 4, 0, 0}) {
		t.Fatalf("gradients = %v", pack.gradients)
	}
	if err := pack.GatherGradients(map[string][]float32{"matrix": {1}}); err == nil ||
		!strings.Contains(err.Error(), "gradient length") {
		t.Fatalf("mismatched gradient error = %v", err)
	}
	plan, err := TensorPlan(map[string][]float32{"vector": vector, "matrix": matrix}, MatrixGeometry(map[string][2]int{
		"matrix": {matrixRows, matrixCols},
		"vector": {vectorRows, vectorCols},
	}))
	if err != nil || plan.Identity() != pack.Plan().Identity() {
		t.Fatalf("storage-free plan identity = %q, %v; want the pack's %q", plan.Identity(), err, pack.Plan().Identity())
	}
}

// TestSectionPackKeepsDeclarationOrderAndDecodesBF16Masters packs sections
// out of name order, one from BF16 storage, and reads each group back
// through its spans.
func TestSectionPackKeepsDeclarationOrderAndDecodesBF16Masters(t *testing.T) {
	const (
		oneBF16      = 0x3f80 // 1.0
		minusTwoBF16 = 0xc000 // -2.0
	)
	pack, err := NewSectionPack([]TensorSection{
		{Name: "z.bias", Rows: 1, Cols: 2, Values: []float32{3, 4}},
		{Name: "a.weight", Rows: 1, Cols: 2, BF16: []uint16{oneBF16, minusTwoBF16}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if first, _ := pack.Plan().Group(0); first.Name != "z.bias" {
		t.Fatalf("first group = %q, want declaration order", first.Name)
	}
	if !slices.Equal(pack.Weights(0), []float32{3, 4}) || !slices.Equal(pack.Weights(1), []float32{1, -2}) {
		t.Fatalf("weights = %v, %v", pack.Weights(0), pack.Weights(1))
	}
	copy(pack.Gradients(1), []float32{3, 4})
	if weights, gradients := pack.Slabs(); len(weights) != pack.ParameterCount() || !slices.Equal(gradients, []float32{0, 0, 3, 4}) {
		t.Fatalf("gradient slab = %v", gradients)
	}
	if norm := pack.GradientL2(); norm != 5 {
		t.Fatalf("gradient L2 = %g, want 5", norm)
	}
	pack.ClearGradients()
	if pack.GradientL2() != 0 {
		t.Fatal("cleared gradients kept a norm")
	}
	if _, err := NewSectionPack([]TensorSection{{Name: "short", Rows: 2, Cols: 2, Values: []float32{1}}}); err == nil {
		t.Fatal("section shorter than its geometry admitted")
	}
}
