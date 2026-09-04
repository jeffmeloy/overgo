package recipecontract

import (
	"errors"
	"fmt"
	"slices"

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
	Inputs  []Modality `json:"inputs"`
	Outputs []Modality `json:"outputs"`
}

func (signature ModalitySignature) Validate() error {
	if len(signature.Inputs) == 0 || len(signature.Outputs) == 0 {
		return errors.New("empty modality signature")
	}
	for _, modalities := range [][]Modality{signature.Inputs, signature.Outputs} {
		for _, modality := range modalities {
			if !ValidModality(modality) {
				return fmt.Errorf("invalid modality %q", modality)
			}
		}
	}
	return nil
}

func (signature ModalitySignature) Clone() ModalitySignature {
	signature.Inputs = slices.Clone(signature.Inputs)
	signature.Outputs = slices.Clone(signature.Outputs)
	return signature
}

// CompileModalitySignature derives semantic I/O from ordered recipe ports.
func CompileModalitySignature(definition recipe.Definition) (ModalitySignature, error) {
	if !definition.ID.Valid() || len(definition.Inputs) == 0 || len(definition.Outputs) == 0 {
		return ModalitySignature{}, errors.New("recipe: incomplete modality definition")
	}
	signature := ModalitySignature{Inputs: make([]Modality, len(definition.Inputs)), Outputs: make([]Modality, len(definition.Outputs))}
	for index, input := range definition.Inputs {
		modality, err := semanticModality(definition.Task, input.Data)
		if err != nil {
			return ModalitySignature{}, fmt.Errorf("recipe: input %q: %w", input.Name, err)
		}
		signature.Inputs[index] = modality
	}
	for index, output := range definition.Outputs {
		modality, err := semanticModality(definition.Task, output.Data)
		if err != nil {
			return ModalitySignature{}, fmt.Errorf("recipe: output %q: %w", output.Name, err)
		}
		signature.Outputs[index] = modality
	}
	return signature, nil
}

func semanticModality(task recipe.Task, data recipe.DataKind) (Modality, error) {
	switch data {
	case recipe.DataText, recipe.DataTokens, recipe.DataEmbeddings, recipe.DataLogits,
		recipe.DataPromptConditioning, recipe.DataTranscription:
		return ModalityText, nil
	case recipe.DataClassConditioning:
		return ModalityTable, nil
	case recipe.DataImage:
		return ModalityImage, nil
	case recipe.DataAudio, recipe.DataTimestampedAlignment, recipe.DataSpeechTurns,
		recipe.DataActivitySegments, recipe.DataConvertedAudio, recipe.DataGeneratedAudio:
		return ModalityAudio, nil
	case recipe.DataVideo:
		return ModalityVideo, nil
	case recipe.DataTensor:
		switch task {
		case recipe.TaskForecast:
			return ModalityTimeSeries, nil
		case recipe.TaskTabular:
			return ModalityTable, nil
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
