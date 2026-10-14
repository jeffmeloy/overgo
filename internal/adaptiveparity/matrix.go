package adaptiveparity

import (
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
)

type Contract struct {
	Task recipe.Task
	Signature
}

type ContractStatus string

const (
	ContractSupported ContractStatus = "supported"
	ContractRefused   ContractStatus = "refused"
)

type ContractRow struct {
	Contract
	Recipe artifact.ID
	Status ContractStatus
	Reason string
}

// InferenceModalityMatrix derives supported rows from compiled recipe ports.
// Required rows without an active recipe become explicit refusals.
func InferenceModalityMatrix(active []recipe.Definition, required []Contract) ([]ContractRow, error) {
	rows := make(map[string]ContractRow, len(active)+len(required))
	for _, definition := range active {
		contract, err := modalityContract(definition)
		if err != nil {
			return nil, err
		}
		key := contractKey(contract)
		if _, exists := rows[key]; exists {
			return nil, fmt.Errorf("adaptive parity: duplicate active modality contract %s", key)
		}
		rows[key] = ContractRow{Contract: contract, Recipe: definition.ID, Status: ContractSupported}
	}
	for _, contract := range required {
		if err := validateContract(contract); err != nil {
			return nil, err
		}
		key := contractKey(contract)
		if _, exists := rows[key]; !exists {
			rows[key] = ContractRow{
				Contract: cloneContract(contract), Status: ContractRefused,
				Reason: "no active recipe owns this ordered modality contract",
			}
		}
	}
	keys := make([]string, 0, len(rows))
	for key := range rows {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	result := make([]ContractRow, len(keys))
	for index, key := range keys {
		result[index] = rows[key]
	}
	return result, nil
}

func modalityContract(definition recipe.Definition) (Contract, error) {
	signature, err := compileModalitySignature(definition)
	if err != nil {
		return Contract{}, fmt.Errorf("adaptive parity: recipe %s: %w", definition.ID, err)
	}
	return Contract{Task: definition.Task, Signature: signature}, nil
}

// compileModalitySignature derives semantic I/O from ordered recipe ports.
func compileModalitySignature(definition recipe.Definition) (Signature, error) {
	if !definition.ID.Valid() || len(definition.Inputs) == 0 || len(definition.Outputs) == 0 {
		return Signature{}, errors.New("recipe: incomplete modality definition")
	}
	signature := Signature{Inputs: make([]recipecontract.Modality, len(definition.Inputs)), Outputs: make([]recipecontract.Modality, len(definition.Outputs))}
	for index, input := range definition.Inputs {
		modality, err := semanticModality(definition.Task, input.Data)
		if err != nil {
			return Signature{}, fmt.Errorf("recipe: input %q: %w", input.Name, err)
		}
		signature.Inputs[index] = modality
	}
	for index, output := range definition.Outputs {
		modality, err := semanticModality(definition.Task, output.Data)
		if err != nil {
			return Signature{}, fmt.Errorf("recipe: output %q: %w", output.Name, err)
		}
		signature.Outputs[index] = modality
	}
	return signature, nil
}

func semanticModality(task recipe.Task, data recipe.DataKind) (recipecontract.Modality, error) {
	if data == recipe.DataEmbeddings && task == recipe.TaskImageEmbedding {
		return recipecontract.ModalityImage, nil
	}
	switch data {
	case recipe.DataText, recipe.DataTokens, recipe.DataEmbeddings, recipe.DataLogits,
		recipe.DataPromptConditioning, recipe.DataTranscription:
		return recipecontract.ModalityText, nil
	case recipe.DataClassConditioning:
		return recipecontract.ModalityTable, nil
	case recipe.DataImage:
		return recipecontract.ModalityImage, nil
	case recipe.DataAudio, recipe.DataTimestampedAlignment, recipe.DataSpeechTurns,
		recipe.DataActivitySegments, recipe.DataConvertedAudio, recipe.DataGeneratedAudio:
		return recipecontract.ModalityAudio, nil
	case recipe.DataVideo:
		return recipecontract.ModalityVideo, nil
	case recipe.DataTensor:
		switch task {
		case recipe.TaskForecast:
			return recipecontract.ModalityTimeSeries, nil
		case recipe.TaskTabular:
			return recipecontract.ModalityTable, nil
		}
	}
	return "", fmt.Errorf("data kind %q has no semantic modality", data)
}

func validateContract(contract Contract) error {
	if contract.Task == "" {
		return fmt.Errorf("adaptive parity: invalid required modality contract")
	}
	if err := contract.Signature.Validate(); err != nil {
		return fmt.Errorf("adaptive parity: invalid required modality contract: %w", err)
	}
	return nil
}

func contractKey(contract Contract) string {
	inputs := make([]string, len(contract.Inputs))
	for index, modality := range contract.Inputs {
		inputs[index] = string(modality)
	}
	outputs := make([]string, len(contract.Outputs))
	for index, modality := range contract.Outputs {
		outputs[index] = string(modality)
	}
	return string(contract.Task) + ":" + strings.Join(inputs, "+") + "->" + strings.Join(outputs, "+")
}

func cloneContract(contract Contract) Contract {
	contract.Signature = contract.Signature.Clone()
	return contract
}
