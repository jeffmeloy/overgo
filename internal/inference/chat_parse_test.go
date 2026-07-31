package inference

import (
	"strings"
	"testing"
)

func TestParseQwenJSONToolCalls(t *testing.T) {
	message, err := parseChatOutput(
		`{{ tool_call.arguments | tojson }}`,
		"<think>\ncheck weather\n</think>\n\nI will check.\n"+
			"<tool_call>\n"+
			`{"name":"weather","arguments":{"city":"Paris"}}`+
			"\n</tool_call>\n"+
			"<tool_call>\n"+
			`{"name":"time","arguments":{"zone":"UTC"}}`+
			"\n</tool_call>",
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if message.Role != "assistant" ||
		message.ReasoningContent != "check weather" ||
		message.Content != "I will check." ||
		len(message.ToolCalls) != 2 ||
		message.ToolCalls[0].Function.Name != "weather" ||
		message.ToolCalls[0].Function.Arguments != `{"city":"Paris"}` ||
		message.ToolCalls[1].Function.Name != "time" {
		t.Fatalf("parsed message = %+v", message)
	}
}

func TestParseHermesToolCallCoercesSchemaTypes(t *testing.T) {
	tools := []ChatTool{{
		Type: "function",
		Function: ChatToolDefinition{
			Name: "weather",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city":    map[string]any{"type": "string"},
					"days":    map[string]any{"type": "integer"},
					"metric":  map[string]any{"type": "boolean"},
					"options": map[string]any{"type": "object"},
				},
			},
		},
	}}
	message, err := parseChatOutput(
		`<function=example_function_name>`,
		"look it up\n</think>\n\n"+
			"<tool_call>\n<function=weather>\n"+
			"<parameter=city>\nParis\n</parameter>\n"+
			"<parameter=days>\n3\n</parameter>\n"+
			"<parameter=metric>\ntrue\n</parameter>\n"+
			"<parameter=options>\n{\"hourly\":true}\n</parameter>\n"+
			"</function>\n</tool_call>",
		tools,
	)
	if err != nil {
		t.Fatal(err)
	}
	if message.ReasoningContent != "look it up" ||
		message.Content != "" ||
		len(message.ToolCalls) != 1 ||
		message.ToolCalls[0].Function.Name != "weather" ||
		message.ToolCalls[0].Function.Arguments !=
			`{"city":"Paris","days":3,"metric":true,"options":{"hourly":true}}` {
		t.Fatalf("parsed message = %+v", message)
	}
}

func TestParseChatOutputRejectsMalformedTagsAndPayloads(t *testing.T) {
	cases := []string{
		`</tool_call>`,
		`<tool_call>{"name":"x","arguments":{}}`,
		`<tool_call>{"name":"x","arguments":{}}</tool_call>tail`,
		`<tool_call>{"name":"","arguments":{}}</tool_call>`,
		`<think>x`,
		`before<think>x</think>`,
	}
	for _, output := range cases {
		if _, err := parseChatOutput("", output, nil); err == nil {
			t.Fatalf("output %q was accepted", output)
		}
	}
	_, err := parseChatOutput(
		`<function=example_function_name>`,
		"<tool_call><function=x><parameter=a>1</parameter>"+
			"<parameter=a>2</parameter></function></tool_call>",
		nil,
	)
	if err == nil || !strings.Contains(err.Error(), "duplicated") {
		t.Fatalf("duplicate parameter error = %v", err)
	}
	_, err = parseChatOutput(
		`{{ tool_call.arguments | tojson }}`,
		`<tool_call>{"name":"unknown","arguments":{}}</tool_call>`,
		[]ChatTool{toolWeatherDefinition()},
	)
	if err == nil || !strings.Contains(err.Error(), "not present") {
		t.Fatalf("unknown tool error = %v", err)
	}
	_, err = parseChatOutput(
		`{{ tool_call.arguments | tojson }}`,
		`<tool_call>{"name":"weather","arguments":{}}</tool_call>`,
		[]ChatTool{toolWeatherDefinition()},
	)
	if err == nil || !strings.Contains(err.Error(), "required") {
		t.Fatalf("missing required argument error = %v", err)
	}
}

func TestParseChatOutputLeavesPlainContentUntouched(t *testing.T) {
	const output = "  ordinary answer\n"
	message, err := parseChatOutput("", output, nil)
	if err != nil {
		t.Fatal(err)
	}
	if message.Content != output || len(message.ToolCalls) != 0 {
		t.Fatalf("parsed message = %+v", message)
	}
}
