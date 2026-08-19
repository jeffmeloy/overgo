package model

import (
	"strings"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/tensor/dtype"
)

func TestLoadTensorRequirementsBindsPresentAndSkipsMissing(t *testing.T) {
	const (
		directRows       = 2
		directColumns    = 3
		optionalElements = 5
	)
	var direct gguf.TensorInfo
	var optionalPresent, missing *gguf.TensorInfo
	requirements := []tensorBinding{
		requiredTensor("direct", &direct, directRows, directColumns),
		optionalTensorPointer("missing", &missing, optionalElements),
		optionalTensorPointer("optional", &optionalPresent, optionalElements),
	}
	catalog, err := newWeightCatalog(&gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("blk.direct", directRows, directColumns),
		tensorInfo("blk.optional", optionalElements),
	}})
	if err != nil {
		t.Fatal(err)
	}
	if err = bindTensorProgram(catalog, "blk.", requirements); err != nil {
		t.Fatal(err)
	}
	if direct.Name != "blk.direct" || optionalPresent == nil || optionalPresent.Name != "blk.optional" || missing != nil {
		t.Fatalf("bindings = direct %q optional %#v missing %#v", direct.Name, optionalPresent, missing)
	}
}

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

func TestRelationalProfileOwnsCompilationAndBindings(t *testing.T) {
	const (
		firstTensor  = "first"
		secondTensor = "second"
		thirdTensor  = "third"
	)
	registry, err := parseArchitectureRegistry(architectureProfileCatalog)
	if err != nil {
		t.Fatal(err)
	}
	if len(registry) != len(architectureRegistry) {
		t.Fatalf("compiled profiles=%d want %d", len(registry), len(architectureRegistry))
	}
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
