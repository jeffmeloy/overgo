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
}
