package model

import (
	"errors"
	"fmt"

	"overgo/internal/tensor"
)

type graphWeight struct {
	name  string
	value *tensor.Tensor
}

type graphWeights []graphWeight

func requireGraphWeight(name string, value *tensor.Tensor) graphWeight {
	return graphWeight{name: name, value: value}
}

func (w *graphWeights) add(name string, value *tensor.Tensor) {
	*w = append(*w, requireGraphWeight(name, value))
}

func (w graphWeights) validate(scope string) error {
	for _, weight := range w {
		if weight.value == nil {
			return fmt.Errorf("%s %s weight is nil", scope, weight.name)
		}
	}
	return nil
}

func requireTensorPair(first, second *tensor.Tensor, message string) error {
	if (first == nil) != (second == nil) {
		return errors.New(message)
	}
	return nil
}
