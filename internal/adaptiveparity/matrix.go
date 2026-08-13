package adaptiveparity

import (
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
	signature, err := recipecontract.CompileModalitySignature(definition)
	if err != nil {
		return Contract{}, fmt.Errorf("adaptive parity: recipe %s: %w", definition.ID, err)
	}
	return Contract{Task: definition.Task, Signature: signature}, nil
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
