package hfconvert

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
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
		names   []string
		invalid bool
	}{
		{name: "absent"},
		{name: "config_string", config: basic, want: map[string]string{"tokenizer.chat_template": basic}},
		{name: "config_named", config: []any{map[string]string{"name": "default", "template": basic}, map[string]string{"name": "tool_use", "template": tools}}, want: map[string]string{"tokenizer.chat_template": basic, "tokenizer.chat_template.tool_use": tools}, names: []string{"tool_use"}},
		{name: "named_inventory", config: []any{map[string]string{"name": "tool_use", "template": tools}, map[string]string{"name": "default", "template": basic}, map[string]string{"name": "retrieval", "template": "context"}}, want: map[string]string{"tokenizer.chat_template": basic, "tokenizer.chat_template.tool_use": tools, "tokenizer.chat_template.retrieval": "context"}, names: []string{"retrieval", "tool_use"}},
		{name: "standalone_precedence", config: basic, files: map[string]string{"chat_template.jinja": " standalone\n"}, want: map[string]string{"tokenizer.chat_template": " standalone\n"}},
		{name: "standalone_keeps_named", config: []any{map[string]string{"name": "default", "template": basic}, map[string]string{"name": "tool_use", "template": tools}}, files: map[string]string{"chat_template.jinja": "replacement"}, want: map[string]string{"tokenizer.chat_template": "replacement", "tokenizer.chat_template.tool_use": tools}, names: []string{"tool_use"}},
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
			count := len(test.want)
			if len(test.names) != 0 {
				count++
			}
			if len(metadata) != count {
				t.Fatalf("metadata count=%d, want %d", len(metadata), count)
			}
			seen := make(map[string]bool)
			for _, item := range metadata {
				if seen[item.Key] {
					t.Fatalf("duplicate metadata key %s", item.Key)
				}
				seen[item.Key] = true
				if item.Key == "tokenizer.chat_templates" {
					names, ok := item.Value.Data.([]string)
					if !ok || !slices.Equal(names, test.names) {
						t.Fatalf("template names=%v, want %v", item.Value.Data, test.names)
					}
					continue
				}
				if want, ok := test.want[item.Key]; !ok || item.Value.Data != want {
					t.Fatalf("metadata %s=%q; want %q", item.Key, item.Value.Data, want)
				}
			}
		})
	}
}
