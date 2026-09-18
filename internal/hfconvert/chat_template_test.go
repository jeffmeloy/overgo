package hfconvert

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestConvertedChatTemplateSources(t *testing.T) {
	const basic = "  {{ bos_token }}\n{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}assistant: "
	const tools = " tools: {{ tools | tojson }}\n"
	for _, test := range []struct {
		name    string
		config  any
		files   map[string]string
		want    map[string]string
		invalid bool
	}{
		{name: "absent"},
		{name: "config_string", config: basic, want: map[string]string{"tokenizer.chat_template": basic}},
		{name: "config_named", config: []any{map[string]string{"name": "default", "template": basic}, map[string]string{"name": "tool_use", "template": tools}}, want: map[string]string{"tokenizer.chat_template": basic, "tokenizer.chat_template.tool_use": tools}},
		{name: "standalone_precedence", config: basic, files: map[string]string{"chat_template.jinja": " standalone\n"}, want: map[string]string{"tokenizer.chat_template": " standalone\n"}},
		{name: "standalone_keeps_named", config: []any{map[string]string{"name": "default", "template": basic}, map[string]string{"name": "tool_use", "template": tools}}, files: map[string]string{"chat_template.jinja": "replacement"}, want: map[string]string{"tokenizer.chat_template": "replacement", "tokenizer.chat_template.tool_use": tools}},
		{name: "no_default", config: []any{map[string]string{"name": "tool_use", "template": tools}}, invalid: true},
		{name: "duplicate", config: []any{map[string]string{"name": "default", "template": basic}, map[string]string{"name": "default", "template": tools}}, invalid: true},
		{name: "missing_template", config: []any{map[string]string{"name": "default"}}, invalid: true},
		{name: "invalid_type", config: 3, invalid: true},
		{name: "empty_list", config: []any{}, invalid: true},
		{name: "null_entry", config: []any{nil}, invalid: true},
		{name: "null_template", config: []any{map[string]any{"name": "default", "template": nil}}, invalid: true},
		{name: "missing_name", config: []any{map[string]string{"template": basic}}, invalid: true},
		{name: "empty", config: "", invalid: true},
		{name: "blank", config: " \n\t", invalid: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			directory := t.TempDir()
			if test.config != nil {
				data, err := json.Marshal(map[string]any{"chat_template": test.config})
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(directory, "tokenizer_config.json"), data, 0600); err != nil {
					t.Fatal(err)
				}
			}
			for name, content := range test.files {
				path := filepath.Join(directory, filepath.FromSlash(name))
				if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, []byte(content), 0600); err != nil {
					t.Fatal(err)
				}
			}
			metadata, err := chatTemplateMetadata(directory)
			if test.invalid {
				if err == nil {
					t.Fatal("invalid template declaration was accepted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(metadata) != len(test.want) {
				t.Fatalf("metadata count=%d, want %d", len(metadata), len(test.want))
			}
			for _, item := range metadata {
				if want, ok := test.want[item.Key]; !ok || item.Value.Data != want {
					t.Fatalf("metadata %s=%q; want %q", item.Key, item.Value.Data, want)
				}
			}
		})
	}
}
