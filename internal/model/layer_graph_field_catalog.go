package model

import (
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
	hostAuto   bool
	infoIndex  []int
	hostIndex  []int
	graphIndex []int
}

var layerGraphFields = compileLayerGraphFields()

func compileLayerGraphFields() []layerGraphField {
	infoType := reflect.TypeOf(LayerWeights{})
	hostType := reflect.TypeOf(HostLayer{})
	graphType := reflect.TypeOf(LayerGraphWeights{})
	infoPointer := reflect.TypeOf((*gguf.TensorInfo)(nil))
	hostPointer := reflect.TypeOf((*reference.Value)(nil))
	graphPointer := reflect.TypeOf((*tensor.Tensor)(nil))
	fields := make([]layerGraphField, 0, infoType.NumField())
	for index := range infoType.NumField() {
		infoField := infoType.Field(index)
		if infoField.Type != infoPointer {
			continue
		}
		hostField, hostOK := hostType.FieldByName(infoField.Name)
		graphField, graphOK := graphType.FieldByName(infoField.Name)
		if !hostOK || !graphOK {
			continue
		}
		if hostField.Type != hostPointer || graphField.Type != graphPointer {
			panic(fmt.Sprintf("layer graph field %s has incompatible mirror types", infoField.Name))
		}
		fields = append(fields, layerGraphField{
			name: infoField.Name, inputName: graphInputName(infoField.Name),
			hostAuto:  hostAutoGraphField(infoField.Name),
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
		if value.IsNil() {
			continue
		}
		tensorInfo := value.Interface().(*gguf.TensorInfo)
		node, pointer, err := weights.Input(builder, tensorInfo.Name)
		if err != nil {
			return err
		}
		feeds[node] = pointer
		graphValue.FieldByIndex(field.graphIndex).Set(reflect.ValueOf(node))
	}
	return nil
}
