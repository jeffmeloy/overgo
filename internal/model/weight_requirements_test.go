package model

import (
	"strings"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/tensor/dtype"
)

func tensorBindingFixtures(
	prefix string,
	requirements []tensorBinding,
	optional map[string]bool,
) []gguf.TensorInfo {
	fixtures := make([]gguf.TensorInfo, 0, len(requirements))
	for _, requirement := range requirements {
		if requirement.optional && !optional[requirement.name] {
			continue
		}
		fixtures = append(fixtures, tensorInfo(prefix+requirement.name, requirement.shapes[0]...))
	}
	return fixtures
}

// TestTensorCatalogProgramIsAuthoritative checks compiled atomic binding.
func TestTensorCatalogProgramIsAuthoritative(t *testing.T) {
	const (
		firstTensor  = "first"
		secondTensor = "second"
		thirdTensor  = "third"
	)
	catalog, err := newWeightCatalog(&gguf.File{Tensors: []gguf.TensorInfo{
		{Name: firstTensor}, {Name: thirdTensor},
	}})
	if err != nil {
		t.Fatal(err)
	}
	var first, second, third gguf.TensorInfo
	err = bindTensorProgram(catalog, "", []tensorBinding{
		requiredTensor(firstTensor, &first),
		requiredTensor(secondTensor, &second),
		requiredTensor(thirdTensor, &third),
	})
	if err == nil || !strings.Contains(err.Error(), secondTensor) || first.Name != "" || third.Name != "" {
		t.Fatalf("error/bindings = %v first=%q third=%q", err, first.Name, third.Name)
	}
	catalog, err = newWeightCatalog(&gguf.File{Tensors: []gguf.TensorInfo{
		{Name: thirdTensor}, {Name: firstTensor}, {Name: secondTensor},
	}})
	if err != nil {
		t.Fatal(err)
	}
	var firstView, secondView, thirdView *gguf.TensorInfo
	err = bindTensorProgram(catalog, "", []tensorBinding{
		requiredTensorPointer(firstTensor, &firstView),
		requiredTensorPointer(secondTensor, &secondView),
		requiredTensorPointer(thirdTensor, &thirdView),
	})
	if err != nil {
		t.Fatal(err)
	}
	if firstView != &catalog.items[catalog.tensors[firstTensor]] ||
		secondView != &catalog.items[catalog.tensors[secondTensor]] ||
		thirdView != &catalog.items[catalog.tensors[thirdTensor]] {
		t.Fatal("bindings do not reference compiled catalog slots")
	}
}

func TestLoadTensorRequirementsEnforcesStorage(t *testing.T) {
	item := tensorInfo("bias", 2)
	item.Type = dtype.F16
	catalog, err := newWeightCatalog(&gguf.File{Tensors: []gguf.TensorInfo{item}})
	if err != nil {
		t.Fatal(err)
	}
	var destination *gguf.TensorInfo
	err = bindTensorProgram(
		catalog, "",
		[]tensorBinding{requiredF32TensorPointer("bias", &destination, 2)},
	)
	if err == nil || destination != nil {
		t.Fatalf("storage constraint = %v destination %#v", err, destination)
	}
}
