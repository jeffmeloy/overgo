package inference

import "testing"

func TestJSONChatOutputStreamDeltas(t *testing.T) {
	stream := NewChatOutputStream(
		`{{ tool_call.arguments | tojson }}`,
		nil,
	)
	pieces := []string{
		`<tool_call>{"name":"weather","arguments":`,
		`{"city":"Pa`,
		`ris"}}</tool_call>`,
	}
	want := []ChatToolCallDelta{
		{Index: 0, Name: "weather", Started: true},
		{Index: 0, Name: "weather", Arguments: `{"city":"Pa`},
		{Index: 0, Name: "weather", Arguments: `ris"}`},
	}
	assertChatStreamDeltas(t, stream, pieces, want)
}

func TestHermesChatOutputStreamDeltas(t *testing.T) {
	tools := []ChatTool{{
		Type: "function",
		Function: ChatToolDefinition{
			Name: "weather",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{"type": "string"},
				},
			},
		},
	}}
	stream := NewChatOutputStream(
		`<function=example_function_name>`,
		tools,
	)
	pieces := []string{
		"<tool_call><function=weather><parameter=city>\n",
		`Pa`,
		"ris\n</parameter></function></tool_call>",
	}
	want := []ChatToolCallDelta{
		{Index: 0, Name: "weather", Arguments: `{"city":"`, Started: true},
		{Index: 0, Name: "weather", Arguments: `Pa`},
		{Index: 0, Name: "weather", Arguments: `ris"}`},
	}
	assertChatStreamDeltas(t, stream, pieces, want)
}

func assertChatStreamDeltas(
	t *testing.T,
	stream ChatOutputStream,
	pieces []string,
	want []ChatToolCallDelta,
) {
	t.Helper()
	var got []ChatToolCallDelta
	for _, piece := range pieces {
		deltas, err := stream.Accept(piece)
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, deltas...)
	}
	if len(got) != len(want) {
		t.Fatalf("delta count = %d, want %d: %#v", len(got), len(want), got)
	}
	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("delta %d = %#v, want %#v", index, got[index], want[index])
		}
	}
	message, err := stream.Finish()
	if err != nil {
		t.Fatal(err)
	}
	if len(message.ToolCalls) != 1 || message.ToolCalls[0].Function.Name != "weather" {
		t.Fatalf("message = %#v", message)
	}
}
