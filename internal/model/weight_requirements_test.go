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
	requirements := []tensorRequirement{
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
	if err = loadTensorRequirements(catalog, "blk.", requirements); err != nil {
		t.Fatal(err)
	}
	if direct.Name != "blk.direct" || optionalPresent == nil || optionalPresent.Name != "blk.optional" || missing != nil {
		t.Fatalf("bindings = direct %q optional %#v missing %#v", direct.Name, optionalPresent, missing)
	}
}

func tensorRequirementFixtures(
	prefix string,
	requirements []tensorRequirement,
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

func TestLoadTensorRequirementsStopsAtFirstError(t *testing.T) {
	catalog, err := newWeightCatalog(&gguf.File{Tensors: []gguf.TensorInfo{
		{Name: "first"}, {Name: "third"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	var first, second, third gguf.TensorInfo
	err = loadTensorRequirements(catalog, "", []tensorRequirement{
		requiredTensor("first", &first),
		requiredTensor("second", &second),
		requiredTensor("third", &third),
	})
	if err == nil || !strings.Contains(err.Error(), `"second"`) || first.Name != "first" || third.Name != "" {
		t.Fatalf("error/bindings = %v first=%q third=%q", err, first.Name, third.Name)
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
	err = loadTensorRequirements(
		catalog, "",
		[]tensorRequirement{requiredF32TensorPointer("bias", &destination, 2)},
	)
	if err == nil || destination != nil {
		t.Fatalf("storage constraint = %v destination %#v", err, destination)
	}
}
