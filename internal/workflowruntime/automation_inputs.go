package workflowruntime

import (
	"encoding/json"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/strictjson"
)

const (
	automationInputMediaType = "application/json"
	automationInputSchema    = "overgo/automation-input/v1"
)

// BindAutomationInputs converts one exact JSON object into the typed values
// declared by a recipe. Each canonical field value is content-addressed once
// so a copied stage output can revive the same typed value after restart.
func BindAutomationInputs(
	definition recipe.Definition,
	raw map[string]json.RawMessage,
) (map[recipe.PortName]Value, error) {
	if definition.ValidateIdentity() != nil || len(raw) != len(definition.Inputs) {
		return nil, errors.New("workflow runtime: automation input set differs")
	}
	values := make(map[recipe.PortName]Value, len(raw))
	for _, input := range definition.Inputs {
		encoded, found := raw[string(input.Name)]
		if !found {
			return nil, errors.New("workflow runtime: required automation input is absent")
		}
		var decoded any
		if err := strictjson.DecodeBytes(encoded, &decoded); err != nil {
			return nil, err
		}
		canonical, err := json.Marshal(decoded)
		if err != nil {
			return nil, err
		}
		id, err := artifact.IdentifyBytes(artifact.KindFile, canonical)
		if err != nil {
			return nil, err
		}
		content := artifact.Content{Descriptor: artifact.Descriptor{
			ID: id, Size: uint64(len(canonical)), MediaType: automationInputMediaType, Schema: automationInputSchema,
		}, Data: canonical}
		values[input.Name] = Value{Kind: input.Data, Items: []Datum{{
			Artifact: content.Descriptor, Content: &content, Value: decoded,
		}}}
	}
	return values, nil
}
