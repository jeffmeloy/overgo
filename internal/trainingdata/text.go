package trainingdata

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"overgo/internal/recipecontract"
)

const EncodingUTF8 = "utf-8"

// JSONTextPairProcessor maps two named JSON string fields to a text pair.
func JSONTextPairProcessor(inputField, targetField string) (Processor, error) {
	inputField = strings.TrimSpace(inputField)
	targetField = strings.TrimSpace(targetField)
	if inputField == "" || targetField == "" || inputField == targetField {
		return nil, errors.New("training data: invalid JSON text-pair fields")
	}
	return func(_ context.Context, record RawRecord) (Example, error) {
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(record.Data, &fields); err != nil {
			return Example{}, fmt.Errorf("training data: decode JSON text pair: %w", err)
		}
		read := func(name string) (string, error) {
			raw, ok := fields[name]
			if !ok {
				return "", fmt.Errorf("training data: JSON field %q absent", name)
			}
			var value string
			if err := json.Unmarshal(raw, &value); err != nil || strings.TrimSpace(value) == "" {
				return "", fmt.Errorf("training data: JSON field %q is not nonempty text", name)
			}
			return value, nil
		}
		input, err := read(inputField)
		if err != nil {
			return Example{}, err
		}
		target, err := read(targetField)
		if err != nil {
			return Example{}, err
		}
		return Example{ID: record.ID, Group: record.Group, Values: []Value{
			{Role: RoleInput, Modality: recipecontract.ModalityText, Encoding: EncodingUTF8, Data: []byte(input)},
			{Role: RoleTarget, Modality: recipecontract.ModalityText, Encoding: EncodingUTF8, Data: []byte(target)},
		}}, nil
	}, nil
}

// TextPair extracts one UTF-8 input and target from a typed example.
func TextPair(example Example) (string, string, error) {
	var input, target string
	for _, value := range example.Values {
		if value.Modality != recipecontract.ModalityText || value.Encoding != EncodingUTF8 || len(value.Data) == 0 {
			return "", "", errors.New("training data: text pair has an incompatible value")
		}
		switch value.Role {
		case RoleInput:
			if input != "" {
				return "", "", errors.New("training data: text pair has multiple inputs")
			}
			input = string(value.Data)
		case RoleTarget:
			if target != "" {
				return "", "", errors.New("training data: text pair has multiple targets")
			}
			target = string(value.Data)
		default:
			return "", "", errors.New("training data: text pair has an invalid role")
		}
	}
	if strings.TrimSpace(input) == "" || strings.TrimSpace(target) == "" {
		return "", "", errors.New("training data: text input and target required")
	}
	return input, target, nil
}
