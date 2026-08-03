package model

import (
	"errors"
	"reflect"
	"testing"

	"llamacpp2go/internal/gguf"
)

func TestLoadTensorRequirementsPreservesOrderAndBindings(t *testing.T) {
	var direct gguf.TensorInfo
	var pointer, missing **gguf.TensorInfo
	var optionalPresent *gguf.TensorInfo
	pointer = &optionalPresent
	missingValue := (*gguf.TensorInfo)(nil)
	missing = &missingValue
	var names []string
	load := func(name string, shape ...uint64) (gguf.TensorInfo, error) {
		names = append(names, name)
		item := gguf.TensorInfo{Name: name, Dimensions: uint32(len(shape))}
		copy(item.Shape[:], shape)
		return item, nil
	}
	err := loadTensorRequirements(load, map[string]gguf.TensorInfo{"blk.optional": {}}, "blk.", []tensorRequirement{
		requiredTensor("direct", &direct, 2, 3),
		optionalTensorPointer("missing", missing, 4),
		optionalTensorPointer("optional", pointer, 5),
	})
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
