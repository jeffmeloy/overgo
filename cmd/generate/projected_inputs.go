package main

import (
	"fmt"
	"os"

	"llamacpp2go/internal/inference"
	"llamacpp2go/internal/strictjson"
)

const projectedInputsJSONLimit = 1 << 30

func readProjectedInputs(path string) (inference.ProjectedInputs, error) {
	file, err := os.Open(path)
	if err != nil {
		return inference.ProjectedInputs{}, fmt.Errorf("generate: open projected inputs: %w", err)
	}
	defer file.Close()
	var document inference.ProjectedInputsJSON
	if err := strictjson.DecodeBounded(file, projectedInputsJSONLimit, &document); err != nil {
		return inference.ProjectedInputs{}, fmt.Errorf("generate: decode projected inputs: %w", err)
	}
	result, err := document.ProjectedInputs()
	if err != nil {
		return inference.ProjectedInputs{}, fmt.Errorf("generate: %w", err)
	}
	return result, nil
}
