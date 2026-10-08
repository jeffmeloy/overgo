package server

import (
	"encoding/json"
	"reflect"
	"testing"

	"overgo/internal/inference"
)

// TestToolSchemasKeepTheirKeyOrderAcrossProtocols holds the two protocols
// that write a tool in a shape of their own to handing its schema on as the
// request wrote it. A model's template writes a tool's schema out in the
// order of its keys, so a schema read into a Go map on the way, which has no
// order, would reach the model rearranged; the tool each protocol makes must
// be the one made from the request's own bytes.
func TestToolSchemasKeepTheirKeyOrderAcrossProtocols(t *testing.T) {
	t.Parallel()
	const schema = `{"type": "object", "required": ["zeta"], "properties": {"zeta": {"type": "string"}, "alpha": {"type": "integer"}}}`
	want, err := inference.FunctionTool("lookup", "Look it up.", json.RawMessage(schema))
	if err != nil {
		t.Fatal(err)
	}
	anthropic, err := selectAnthropicTools(json.RawMessage(`[{"name": "lookup", "description": "Look it up.", "input_schema": `+schema+`}]`), nil)
	if err != nil {
		t.Fatal(err)
	}
	responses, err := selectResponsesTools(json.RawMessage(`[{"type": "function", "name": "lookup", "description": "Look it up.", "parameters": `+schema+`}]`), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	for protocol, selection := range map[string]chatToolSelection{"Anthropic": anthropic, "Responses": responses} {
		if len(selection.prompt) != 1 || !reflect.DeepEqual(selection.prompt[0], want) {
			t.Errorf("the %s protocol hands on %+v, and the tool made from the request's bytes is %+v", protocol, selection.prompt, want)
		}
	}
	for protocol, refused := range map[string]func() error{
		"Anthropic": func() error {
			_, err := selectAnthropicTools(json.RawMessage(`[{"name": "lookup", "input_schema": ["not", "an", "object"]}]`), nil)
			return err
		},
		"Responses": func() error {
			_, err := selectResponsesTools(json.RawMessage(`[{"type": "function", "name": "lookup", "parameters": "text"}]`), nil, nil)
			return err
		},
	} {
		if refused() == nil {
			t.Errorf("the %s protocol took a schema that is not an object", protocol)
		}
	}
}
