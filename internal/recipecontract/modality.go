package recipecontract

import (
	"errors"
	"fmt"

	"overgo/internal/recipe"
)

type Modality string

const (
	ModalityText       Modality = "text"
	ModalityImage      Modality = "image"
	ModalityAudio      Modality = "audio"
	ModalityVideo      Modality = "video"
	ModalityTimeSeries Modality = "time-series"
	ModalityTable      Modality = "table"
)

type ModalitySignature struct {
	Inputs  []Modality
	Outputs []Modality
}

// CompileModalitySignature derives semantic I/O from ordered recipe ports.
func CompileModalitySignature(definition recipe.Definition) (ModalitySignature, error) {
	if !definition.ID.Valid() || len(definition.Inputs) == 0 || len(definition.Outputs) == 0 {
		return ModalitySignature{}, errors.New("recipe: incomplete modality definition")
	}
	placement := recipe.Placement("")
	for _, node := range definition.Nodes {
		if placement == "" {
			placement = node.Placement
		}
		if node.Placement != placement && definition.Task != recipe.TaskVQA {
			return ModalitySignature{}, errors.New("recipe: mixed placement has no modality policy")
		}
	}
	signature := ModalitySignature{Inputs: make([]Modality, len(definition.Inputs)), Outputs: make([]Modality, len(definition.Outputs))}
	for index, input := range definition.Inputs {
		modality, err := semanticModality(definition.Task, placement, true, input.Data)
		if err != nil {
			return ModalitySignature{}, fmt.Errorf("recipe: input %q: %w", input.Name, err)
		}
		signature.Inputs[index] = modality
	}
	for index, output := range definition.Outputs {
		modality, err := semanticModality(definition.Task, placement, false, output.Data)
		if err != nil {
			return ModalitySignature{}, fmt.Errorf("recipe: output %q: %w", output.Name, err)
		}
		signature.Outputs[index] = modality
	}
	return signature, nil
}

func semanticModality(task recipe.Task, placement recipe.Placement, input bool, data recipe.DataKind) (Modality, error) {
	switch data {
	case recipe.DataText, recipe.DataTokens, recipe.DataEmbeddings, recipe.DataLogits:
		return ModalityText, nil
	case recipe.DataImage:
		return ModalityImage, nil
	case recipe.DataAudio:
		return ModalityAudio, nil
	case recipe.DataVideo:
		return ModalityVideo, nil
	case recipe.DataTensor:
		switch task {
		case recipe.TaskForecast:
			return ModalityTimeSeries, nil
		case recipe.TaskTabular:
			return ModalityTable, nil
		case recipe.TaskImageGen:
			if input && placement == recipe.PlacementHybrid {
				return ModalityText, nil
			}
			if input {
				return ModalityTable, nil
			}
		}
	}
	return "", fmt.Errorf("data kind %q has no semantic modality", data)
}

func ValidModality(modality Modality) bool {
	switch modality {
	case ModalityText, ModalityImage, ModalityAudio, ModalityVideo, ModalityTimeSeries, ModalityTable:
		return true
	default:
		return false
	}
}
