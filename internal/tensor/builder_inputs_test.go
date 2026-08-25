package tensor

import (
	"testing"

	"overgo/internal/tensor/dtype"
)

func TestWeightInputsSelectsCompiledMatrixStorage(t *testing.T) {
	const (
		inputWidth  = 3
		outputWidth = 5
	)
	builder := NewBuilder()
	var bindings WeightBindings
	weights := WeightInputs{Builder: builder, Bindings: &bindings, MatrixType: dtype.BF16}
	vector := weights.Input("vector", inputWidth)
	matrix := weights.Input("matrix", inputWidth, outputWidth)
	if vector.Type != dtype.F32 || matrix.Type != dtype.BF16 || bindings.Node(vector.Name) != vector || bindings.Node(matrix.Name) != matrix {
		t.Fatalf("weight inputs = vector %s matrix %s catalog %d", vector.Type, matrix.Type, len(bindings))
	}
}
