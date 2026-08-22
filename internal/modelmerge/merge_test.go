package modelmerge

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/tensor"
	"overgo/internal/testutil"
)

func TestCompatiblePassthroughAndTaskArithmetic(t *testing.T) {
	compiler := Compiler{}
	definition := testutil.ArtifactID(t, artifact.KindModelDefinition, "compatible-definition")
	first, err := compiler.Seal(definition, artifact.ID{}, map[string]Weight{
		"layer.0": {Layout: tensor.MustShape(2, 1), Values: []float32{1, 2}},
		"layer.1": {Layout: tensor.MustShape(2, 1), Values: []float32{3, 4}},
	})
	if err != nil {
		t.Fatal(err)
	}
	second, err := compiler.Seal(definition, artifact.ID{}, map[string]Weight{
		"layer.0": {Layout: tensor.MustShape(2, 1), Values: []float32{10, 20}},
		"layer.1": {Layout: tensor.MustShape(2, 1), Values: []float32{30, 40}},
	})
	if err != nil {
		t.Fatal(err)
	}
	passthrough, err := compiler.Passthrough([]Snapshot{first, second}, []Selection{
		{Tensor: "layer.0", Source: first.ID},
		{Tensor: "layer.1", Source: second.ID},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(passthrough.Model.Tensors["layer.0"].Values, []float32{1, 2}) ||
		!slices.Equal(passthrough.Model.Tensors["layer.1"].Values, []float32{30, 40}) {
		t.Fatalf("passthrough tensors=%v", passthrough.Model.Tensors)
	}

	task, err := compiler.Seal(definition, first.ID, map[string]Weight{
		"layer.0": {Layout: tensor.MustShape(2, 1), Values: []float32{3, 6}},
		"layer.1": {Layout: tensor.MustShape(2, 1), Values: []float32{7, 10}},
	})
	if err != nil {
		t.Fatal(err)
	}
	arithmetic, err := compiler.TaskArithmetic(first, []TaskDelta{{Model: task, Scale: 0.5}})
	if err != nil {
		t.Fatal(err)
	}
	if arithmetic.Model.Base != first.ID ||
		!slices.Equal(arithmetic.Model.Tensors["layer.0"].Values, []float32{2, 4}) ||
		!slices.Equal(arithmetic.Model.Tensors["layer.1"].Values, []float32{5, 7}) {
		t.Fatalf("task arithmetic=%+v", arithmetic.Model)
	}

	foreignDefinition := testutil.ArtifactID(t, artifact.KindModelDefinition, "foreign-definition")
	foreign, err := compiler.Seal(foreignDefinition, artifact.ID{}, first.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := compiler.Passthrough([]Snapshot{first, foreign}, []Selection{
		{Tensor: "layer.0", Source: first.ID}, {Tensor: "layer.1", Source: foreign.ID},
	}); err == nil {
		t.Fatal("cross-definition passthrough accepted")
	}
	wrongBase, err := compiler.Seal(definition, second.ID, task.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := compiler.TaskArithmetic(first, []TaskDelta{{Model: wrongBase, Scale: 1}}); err == nil {
		t.Fatal("task arithmetic from a different base accepted")
	}
}
