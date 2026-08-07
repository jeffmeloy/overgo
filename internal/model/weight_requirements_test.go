package model

import (
	"errors"
	"reflect"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/tensor/dtype"
)

func TestLoadTensorRequirementsPreservesOrderAndBindings(t *testing.T) {
	var direct gguf.TensorInfo
	var pointer, missing **gguf.TensorInfo
	var optionalPresent *gguf.TensorInfo
	pointer = &optionalPresent
	missingValue := (*gguf.TensorInfo)(nil)
	missing = &missingValue
	requirements := []tensorRequirement{
		requiredTensor("direct", &direct, 2, 3),
		optionalTensorPointer("missing", missing, 4),
		optionalTensorPointer("optional", pointer, 5),
	}
	fixtures := tensorRequirementFixtures("blk.", requirements, map[string]bool{"optional": true})
	catalog := make(map[string]gguf.TensorInfo, len(fixtures))
	for _, fixture := range fixtures {
		catalog[fixture.Name] = fixture
	}
	var names []string
	load := func(name string, shape ...uint64) (gguf.TensorInfo, error) {
		names = append(names, name)
		item, ok := catalog[name]
		if !ok {
			return gguf.TensorInfo{}, errors.New("fixture missing")
		}
		return item, nil
	}
	err := loadTensorRequirements(load, catalog, "blk.", requirements)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(names, []string{"blk.direct", "blk.optional"}) {
		t.Fatalf("load order = %v", names)
	}
	if direct.Name != "blk.direct" || optionalPresent == nil || optionalPresent.Name != "blk.optional" || missingValue != nil {
		t.Fatalf("bindings = direct %q optional %#v missing %#v", direct.Name, optionalPresent, missingValue)
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
		fixtures = append(fixtures, tensorInfo(prefix+requirement.name, requirement.shape...))
	}
	return fixtures
}

func TestLoadTensorRequirementsStopsAtFirstError(t *testing.T) {
	want := errors.New("stop")
	var names []string
	load := func(name string, _ ...uint64) (gguf.TensorInfo, error) {
		names = append(names, name)
		if name == "second" {
			return gguf.TensorInfo{}, want
		}
		return gguf.TensorInfo{Name: name}, nil
	}
	var first, second, third gguf.TensorInfo
	err := loadTensorRequirements(load, nil, "", []tensorRequirement{
		requiredTensor("first", &first),
		requiredTensor("second", &second),
		requiredTensor("third", &third),
	})
	if !errors.Is(err, want) || !reflect.DeepEqual(names, []string{"first", "second"}) {
		t.Fatalf("error/order = %v %v", err, names)
	}
}

func TestLoadTensorRequirementsEnforcesStorage(t *testing.T) {
	item := tensorInfo("bias", 2)
	item.Type = dtype.F16
	var destination *gguf.TensorInfo
	err := loadTensorRequirements(
		func(_ string, _ ...uint64) (gguf.TensorInfo, error) { return item, nil },
		map[string]gguf.TensorInfo{"bias": item}, "",
		[]tensorRequirement{requiredF32TensorPointer("bias", &destination, 2)},
	)
	if err == nil || destination != nil {
		t.Fatalf("storage constraint = %v destination %#v", err, destination)
	}
}
