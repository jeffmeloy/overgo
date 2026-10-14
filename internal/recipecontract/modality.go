package recipecontract

import (
	"errors"
	"fmt"
	"slices"
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

func ValidModality(modality Modality) bool {
	switch modality {
	case ModalityText, ModalityImage, ModalityAudio, ModalityVideo, ModalityTimeSeries, ModalityTable:
		return true
	default:
		return false
	}
}
