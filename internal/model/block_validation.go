package model

import (
	"errors"
	"fmt"

	"llamacpp2go/internal/tensor"
)

func requireBlockWeights(scope string, weights map[string]*tensor.Tensor) error {
	for name, weight := range weights {
		if weight == nil {
			return fmt.Errorf("%s %s weight is nil", scope, name)
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
