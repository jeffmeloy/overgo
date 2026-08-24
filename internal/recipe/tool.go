package recipe

import (
	"encoding/json"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

const (
	// ToolCallPort names the typed tool input.
	ToolCallPort PortName = "call"
	// ToolResultPort names the typed tool output.
	ToolResultPort   PortName = "result"
	toolCallSchema            = "overgo.tool-call.v1"
	toolResultSchema          = "overgo.tool-result.v1"
)

// ToolCall binds one invocation to its recipe module.
type ToolCall struct {
	ID        string          `json:"id"`
	Module    ModuleID        `json:"module"`
	Arguments json.RawMessage `json:"arguments"`
}

// NewToolCall validates strict object arguments at the protocol boundary.
func NewToolCall(id string, module ModuleID, arguments json.RawMessage) (ToolCall, error) {
	call := ToolCall{ID: id, Module: module, Arguments: arguments}
	if err := call.Validate(); err != nil {
		return ToolCall{}, err
	}
	call.Arguments = append(json.RawMessage(nil), arguments...)
	return call, nil
}

// Validate checks tool identity and strict object arguments.
func (call ToolCall) Validate() error {
	if call.ID == "" || !validName(string(call.Module)) {
		return errors.New("recipe: invalid tool call identity")
	}
	var object map[string]json.RawMessage
	if err := strictjson.DecodeBytes(call.Arguments, &object); err != nil || object == nil {
		return errors.New("recipe: tool arguments must be a strict JSON object")
	}
	return nil
}

// ArtifactContent returns the durable call boundary.
func (call ToolCall) ArtifactContent() (artifact.Content, error) {
	if err := call.Validate(); err != nil {
		return artifact.Content{}, err
	}
	return artifact.JSONContent(ToolCallDocumentContract(), call)
}

// ToolCallDocumentContract identifies durable tool calls.
func ToolCallDocumentContract() artifact.DocumentContract {
	return artifact.JSONContract(artifact.KindEvidence, toolCallSchema)
}

// ToolResult binds one typed result to its originating call.
type ToolResult struct {
	CallID string          `json:"call_id"`
	Module ModuleID        `json:"module"`
	Output json.RawMessage `json:"output"`
}

// ArtifactContent returns the durable result boundary.
func (result ToolResult) ArtifactContent() (artifact.Content, error) {
	if result.CallID == "" || !validName(string(result.Module)) || !json.Valid(result.Output) {
		return artifact.Content{}, errors.New("recipe: invalid tool result")
	}
	return artifact.JSONContent(ToolResultDocumentContract(), result)
}

// ToolResultDocumentContract identifies durable tool results.
func ToolResultDocumentContract() artifact.DocumentContract {
	return artifact.JSONContract(artifact.KindOutput, toolResultSchema)
}

// ToolModule returns the scalar module contract used by tool recipes.
func ToolModule(id ModuleID, placements ...Placement) Module {
	return Module{
		ID: id, Tasks: []Task{TaskInference}, Placements: placements,
		Inputs:  []Port{{Name: ToolCallPort, Data: DataToolCall, Cardinality: CardinalityOne}},
		Outputs: []Port{{Name: ToolResultPort, Data: DataToolResult, Cardinality: CardinalityOne}},
	}
}

// ToolAdmission proves one compiled recipe is an exact tool invocation.
type ToolAdmission struct {
	Recipe artifact.ID
	Model  artifact.ID
	Module ModuleID
}

// AdmitTool validates the sole executable node and its external contract.
func AdmitTool(program Program) (ToolAdmission, error) {
	definition, stages := program.Definition(), program.Stages()
	if definition.Task != TaskInference || len(stages) != 1 ||
		len(definition.Inputs) != 1 || len(definition.Outputs) != 1 {
		return ToolAdmission{}, errors.New("recipe: tool recipe must contain one inference stage")
	}
	stage, input, output := stages[0], definition.Inputs[0], definition.Outputs[0]
	if err := validateToolModule(stage.Module); err != nil {
		return ToolAdmission{}, err
	}
	if input.Name != ToolCallPort || input.Data != DataToolCall || input.Target != (Endpoint{Node: stage.Node.ID, Port: ToolCallPort}) ||
		output.Name != ToolResultPort || output.Data != DataToolResult || output.Source != (Endpoint{Node: stage.Node.ID, Port: ToolResultPort}) {
		return ToolAdmission{}, errors.New("recipe: tool recipe boundary differs from module contract")
	}
	return ToolAdmission{Recipe: definition.ID, Model: definition.Model, Module: stage.Module.ID}, nil
}

func validateToolModule(module Module) error {
	want := ToolModule(module.ID, module.Placements...)
	canonical, err := canonicalModule(want)
	if err != nil {
		return err
	}
	actual, err := canonicalModule(module)
	if err != nil {
		return err
	}
	if actual.ID != canonical.ID || !slices.Equal(actual.Tasks, canonical.Tasks) ||
		!slices.Equal(actual.Placements, canonical.Placements) ||
		!slices.Equal(actual.Inputs, canonical.Inputs) || !slices.Equal(actual.Outputs, canonical.Outputs) {
		return errors.New("recipe: module does not satisfy the tool contract")
	}
	return nil
}
