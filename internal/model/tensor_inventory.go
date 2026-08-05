package model

import (
	"reflect"

	"llamacpp2go/internal/gguf"
)

var tensorInfoType = reflect.TypeOf(gguf.TensorInfo{})

// TensorInfos returns every populated tensor in the weight catalog.
func (w Weights) TensorInfos() []gguf.TensorInfo {
	result := make([]gguf.TensorInfo, 0)
	seen := make(map[string]struct{})
	collectTensorInfos(reflect.ValueOf(w), seen, &result)
	return result
}

func collectTensorInfos(value reflect.Value, seen map[string]struct{}, result *[]gguf.TensorInfo) {
	if !value.IsValid() {
		return
	}
	if value.Kind() == reflect.Pointer {
		if value.IsNil() {
			return
		}
		collectTensorInfos(value.Elem(), seen, result)
		return
	}
	if value.Type() == tensorInfoType {
		info := value.Interface().(gguf.TensorInfo)
		if info.Name == "" {
			return
		}
		if _, ok := seen[info.Name]; ok {
			return
		}
		seen[info.Name] = struct{}{}
		*result = append(*result, info)
		return
	}
	switch value.Kind() {
	case reflect.Struct:
		for index := 0; index < value.NumField(); index++ {
			collectTensorInfos(value.Field(index), seen, result)
		}
	case reflect.Slice, reflect.Array:
		for index := 0; index < value.Len(); index++ {
			collectTensorInfos(value.Index(index), seen, result)
		}
	}
}
