package model

import (
	"errors"
	"reflect"

	"overgo/internal/gguf"
	"overgo/internal/tensor"
)

var graphTensorPointerType = reflect.TypeOf((*tensor.Tensor)(nil))

type graphWeightField struct {
	name          string
	source, graph int
	optional      bool
}

var (
	sequenceOutputGraphFields = compileGraphWeightFields(
		reflect.TypeOf(AudioDecoderWeights{}), reflect.TypeOf(SequenceOutputGraphWeights{}),
	)
	sequenceResidualGraphFields = compileGraphWeightFields(
		reflect.TypeOf(WavPosNetWeights{}), reflect.TypeOf(SequenceResidualGraphWeights{}),
	)
	sequenceConvGraphFields = compileGraphWeightFields(
		reflect.TypeOf(WavConvNextWeights{}), reflect.TypeOf(SequenceConvGraphWeights{}),
	)
)

// BindSequenceOutputGraphWeights binds the compiled sequence-output catalog.
func BindSequenceOutputGraphWeights(
	info *AudioDecoderWeights,
	bind func(gguf.TensorInfo) (*tensor.Tensor, error),
) (SequenceOutputGraphWeights, error) {
	if info == nil {
		return SequenceOutputGraphWeights{}, errors.New("sequence-output weights are missing")
	}
	if bind == nil {
		return SequenceOutputGraphWeights{}, errors.New("sequence-output tensor binder is nil")
	}
	var result SequenceOutputGraphWeights
	if err := bindGraphWeights(
		reflect.ValueOf(*info), reflect.ValueOf(&result).Elem(), sequenceOutputGraphFields, bind,
	); err != nil {
		return SequenceOutputGraphWeights{}, err
	}
	result.Residual = make([]SequenceResidualGraphWeights, len(info.PosNet))
	for index := range info.PosNet {
		if err := bindGraphWeights(
			reflect.ValueOf(info.PosNet[index]), reflect.ValueOf(&result.Residual[index]).Elem(),
			sequenceResidualGraphFields, bind,
		); err != nil {
			return SequenceOutputGraphWeights{}, err
		}
	}
	result.Convolution = make([]SequenceConvGraphWeights, len(info.ConvNext))
	for index := range info.ConvNext {
		if err := bindGraphWeights(
			reflect.ValueOf(info.ConvNext[index]), reflect.ValueOf(&result.Convolution[index]).Elem(),
			sequenceConvGraphFields, bind,
		); err != nil {
			return SequenceOutputGraphWeights{}, err
		}
	}
	return result, nil
}

func compileGraphWeightFields(source, destination reflect.Type) []graphWeightField {
	fields := make([]graphWeightField, 0, source.NumField())
	for index := range source.NumField() {
		sourceField := source.Field(index)
		destinationField, ok := destination.FieldByName(sourceField.Name)
		if !ok {
			continue
		}
		optional := sourceField.Type == tensorInfoPointerType
		if (!optional && sourceField.Type != tensorInfoType) || destinationField.Type != graphTensorPointerType {
			panic("model: incompatible graph weight schema")
		}
		fields = append(fields, graphWeightField{
			name: sourceField.Name, source: index,
			graph: destinationField.Index[0], optional: optional,
		})
	}
	return fields
}

func bindGraphWeights(
	source reflect.Value,
	destination reflect.Value,
	fields []graphWeightField,
	bind func(gguf.TensorInfo) (*tensor.Tensor, error),
) error {
	for _, field := range fields {
		info := source.Field(field.source).Interface().(gguf.TensorInfo)
		if info.Name == "" {
			continue
		}
		node, err := bind(info)
		if err != nil {
			return err
		}
		destination.Field(field.graph).Set(reflect.ValueOf(node))
	}
	return nil
}
