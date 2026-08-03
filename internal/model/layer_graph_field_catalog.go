package model

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"unicode"

	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
)

type layerGraphField struct {
	name       string
	inputName  string
	optional   bool
	hostAuto   bool
	infoIndex  []int
	hostIndex  []int
	graphIndex []int
}

var layerGraphFields = compileLayerGraphFields()

func loadHostLayerGraphFields(
	ctx context.Context,
	file *gguf.File,
	info *LayerWeights,
	result *HostLayer,
) error {
	infoValue := reflect.ValueOf(info).Elem()
	hostValue := reflect.ValueOf(result).Elem()
	for _, field := range layerGraphFields {
		tensorField := infoValue.FieldByIndex(field.infoIndex)
		var tensorInfo *gguf.TensorInfo
		if field.optional {
			if tensorField.IsNil() {
				continue
			}
			tensorInfo = tensorField.Interface().(*gguf.TensorInfo)
		} else {
			value := tensorField.Interface().(gguf.TensorInfo)
			if value.Name == "" {
				continue
			}
			tensorInfo = &value
		}
		value, err := LoadHostTensor(ctx, file, *tensorInfo)
		if err != nil {
			return err
		}
		if field.optional {
			hostValue.FieldByIndex(field.hostIndex).Set(reflect.ValueOf(&value))
		} else {
			hostValue.FieldByIndex(field.hostIndex).Set(reflect.ValueOf(value))
		}
	}
	return nil
}

func compileLayerGraphFields() []layerGraphField {
	infoType := reflect.TypeOf(LayerWeights{})
	hostType := reflect.TypeOf(HostLayer{})
	graphType := reflect.TypeOf(LayerGraphWeights{})
	infoPointer := reflect.TypeOf((*gguf.TensorInfo)(nil))
	infoValue := reflect.TypeOf(gguf.TensorInfo{})
	hostPointer := reflect.TypeOf((*reference.Value)(nil))
	hostValue := reflect.TypeOf(reference.Value{})
	graphPointer := reflect.TypeOf((*tensor.Tensor)(nil))
	fields := make([]layerGraphField, 0, infoType.NumField())
	for index := range infoType.NumField() {
		infoField := infoType.Field(index)
		hostField, hostOK := hostType.FieldByName(infoField.Name)
		graphField, graphOK := graphType.FieldByName(infoField.Name)
		if !hostOK || !graphOK {
			continue
		}
		optional := infoField.Type == infoPointer && hostField.Type == hostPointer
		required := infoField.Type == infoValue && hostField.Type == hostValue
		if (!optional && !required) || graphField.Type != graphPointer {
			panic(fmt.Sprintf("layer graph field %s has incompatible mirror types", infoField.Name))
		}
		fields = append(fields, layerGraphField{
			name: infoField.Name, inputName: graphInputName(infoField.Name),
			optional: optional, hostAuto: optional && hostAutoGraphField(infoField.Name),
			infoIndex: infoField.Index, hostIndex: hostField.Index, graphIndex: graphField.Index,
		})
	}
	return fields
}

func hostAutoGraphField(name string) bool {
	switch name {
	case "AttentionNormBias", "AttentionNorm2", "AttentionNorm2Bias", "FeedForwardNormBias",
		"AttentionQKV", "AttentionGate":
		return false
	}
	return !strings.HasPrefix(name, "SSM") && !strings.HasPrefix(name, "TimeMix") &&
		!strings.HasPrefix(name, "ChannelMix") && !strings.HasPrefix(name, "ShortConv")
}

func graphInputName(name string) string {
	items := []rune(name)
	var output strings.Builder
	for index, item := range items {
		if unicode.IsUpper(item) && index > 0 &&
			(unicode.IsLower(items[index-1]) ||
				(index+1 < len(items) && unicode.IsLower(items[index+1]))) {
			output.WriteByte('_')
		}
		output.WriteRune(unicode.ToLower(item))
	}
	return output.String()
}

func bindHostLayerGraphFields(
	builder *tensor.Builder,
	prefix string,
	layer *HostLayer,
	result *LayerGraphWeights,
	feeds map[*tensor.Tensor]reference.Value,
) {
	hostValue := reflect.ValueOf(layer).Elem()
	graphValue := reflect.ValueOf(result).Elem()
	for _, field := range layerGraphFields {
		if !field.hostAuto {
			continue
		}
		value := hostValue.FieldByIndex(field.hostIndex)
		if value.IsNil() {
			continue
		}
		host := value.Interface().(*reference.Value)
		node := builder.Input(prefix+field.inputName, dtype.F32, host.Shape)
		if node != nil {
			feeds[node] = *host
		}
		graphValue.FieldByIndex(field.graphIndex).Set(reflect.ValueOf(node))
	}
}

func bindDeviceLayerGraphFields(
	weights *DeviceF32Weights,
	builder *tensor.Builder,
	info *LayerWeights,
	result *LayerGraphWeights,
	feeds map[*tensor.Tensor]driver.DevicePtr,
) error {
	infoValue := reflect.ValueOf(info).Elem()
	graphValue := reflect.ValueOf(result).Elem()
	for _, field := range layerGraphFields {
		value := infoValue.FieldByIndex(field.infoIndex)
		var tensorInfo *gguf.TensorInfo
		if field.optional {
			if value.IsNil() {
				continue
			}
			tensorInfo = value.Interface().(*gguf.TensorInfo)
		} else {
			item := value.Interface().(gguf.TensorInfo)
			if item.Name == "" {
				continue
			}
			tensorInfo = &item
		}
		node, pointer, err := weights.Input(builder, tensorInfo.Name)
		if err != nil {
			return err
		}
		feeds[node] = pointer
		graphValue.FieldByIndex(field.graphIndex).Set(reflect.ValueOf(node))
	}
	return nil
}
