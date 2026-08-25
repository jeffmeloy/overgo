package model

import (
	"errors"
	"fmt"

	"overgo/internal/tensor"
)

type graphWeights []*tensor.Tensor

func (w graphWeights) validate(scope string) error {
	for slot, weight := range w {
		if weight == nil {
			return fmt.Errorf("%s weight slot %d is nil", scope, slot)
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
