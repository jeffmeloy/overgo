package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"

	"llamacpp2go/internal/inference"
)

const projectedInputsJSONLimit = 1 << 30

func readProjectedInputs(path string) (inference.ProjectedInputs, error) {
	file, err := os.Open(path)
	if err != nil {
		return inference.ProjectedInputs{}, fmt.Errorf("generate: open projected inputs: %w", err)
	}
	defer file.Close()
	decoder := json.NewDecoder(io.LimitReader(file, projectedInputsJSONLimit+1))
	decoder.DisallowUnknownFields()
	var document inference.ProjectedInputsJSON
	if err := decoder.Decode(&document); err != nil {
		return inference.ProjectedInputs{}, fmt.Errorf("generate: decode projected inputs: %w", err)
	}
	if err := requireJSONEOF(decoder); err != nil {
		return inference.ProjectedInputs{}, fmt.Errorf("generate: decode projected inputs: %w", err)
	}
	result, err := document.ProjectedInputs()
	if err != nil {
		return inference.ProjectedInputs{}, fmt.Errorf("generate: %w", err)
	}
	return result, nil
}

func requireJSONEOF(decoder *json.Decoder) error {
	var trailing any
	if err := decoder.Decode(&trailing); errors.Is(err, io.EOF) {
		return nil
	} else if err != nil {
		return err
	}
	return errors.New("trailing JSON value")
}
