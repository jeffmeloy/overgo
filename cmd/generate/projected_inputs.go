package main

import (
	"fmt"
	"os"

	"overgo/internal/inference"
	"overgo/internal/strictjson"
)

func readProjectedInputs(path string) (inference.ProjectedInputs, error) {
	file, err := os.Open(path)
	if err != nil {
		return inference.ProjectedInputs{}, fmt.Errorf("generate: open projected inputs: %w", err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return inference.ProjectedInputs{}, fmt.Errorf("generate: stat projected inputs: %w", err)
	}
	var document inference.ProjectedInputsJSON
	if err := strictjson.DecodeBounded(file, info.Size(), &document); err != nil {
		return inference.ProjectedInputs{}, fmt.Errorf("generate: decode projected inputs: %w", err)
	}
	result, err := document.ProjectedInputs()
	if err != nil {
		return inference.ProjectedInputs{}, fmt.Errorf("generate: %w", err)
	}
	return result, nil
}
