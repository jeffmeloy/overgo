package model

import (
	"errors"
	"fmt"
	"reflect"

	"overgo/internal/gguf"
	"overgo/internal/tensor"
)

var graphTensorPointerType = reflect.TypeOf((*tensor.Tensor)(nil))

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
	if err := bindNamedGraphWeights(reflect.ValueOf(*info), reflect.ValueOf(&result).Elem(), bind); err != nil {
		return SequenceOutputGraphWeights{}, err
	}
	result.Residual = make([]SequenceResidualGraphWeights, len(info.PosNet))
	for index := range info.PosNet {
		if err := bindNamedGraphWeights(
			reflect.ValueOf(info.PosNet[index]), reflect.ValueOf(&result.Residual[index]).Elem(), bind,
		); err != nil {
			return SequenceOutputGraphWeights{}, err
		}
	}
	result.Convolution = make([]SequenceConvGraphWeights, len(info.ConvNext))
	for index := range info.ConvNext {
		if err := bindNamedGraphWeights(
			reflect.ValueOf(info.ConvNext[index]), reflect.ValueOf(&result.Convolution[index]).Elem(), bind,
		); err != nil {
			return SequenceOutputGraphWeights{}, err
		}
	}
	return result, nil
}

func bindNamedGraphWeights(
	source reflect.Value,
	destination reflect.Value,
	bind func(gguf.TensorInfo) (*tensor.Tensor, error),
) error {
	for index := range source.NumField() {
		sourceField := source.Field(index)
		destinationField := destination.FieldByName(source.Type().Field(index).Name)
		if !destinationField.IsValid() {
			continue
		}
		if sourceField.Type() == tensorInfoType && destinationField.Type() == graphTensorPointerType {
			info := sourceField.Interface().(gguf.TensorInfo)
			if info.Name == "" {
				continue
			}
			node, err := bind(info)
			if err != nil {
				return err
			}
			destinationField.Set(reflect.ValueOf(node))
			continue
		}
		if sourceField.Kind() == reflect.Struct && destinationField.Kind() == reflect.Struct {
			if err := bindNamedGraphWeights(sourceField, destinationField, bind); err != nil {
				return err
			}
			continue
		}
		return fmt.Errorf(
			"graph weight field %s cannot bind %s to %s",
			source.Type().Field(index).Name, sourceField.Type(), destinationField.Type(),
		)
	}
	return nil
}
