package inference

import (
	"encoding/json"
	"path/filepath"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/hfconvert"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

func TestConvertedChatTemplateRendering(t *testing.T) {
	const source = "{{ bos_token }}{% for message in messages %}[{{ message['role'] }}] {{ message['content'] }}{{ eos_token }}\n{% endfor %}{% if add_generation_prompt %}[assistant]{% if enable_thinking %}<think>{% endif %}{% endif %}"
	const toolSource = "{{ tools[0].function.name }}\n" + source
	for _, test := range []struct {
		name        string
		declaration any
		standalone  string
	}{
		{"embedded_string", source, ""},
		{"embedded_named", []map[string]string{{"name": "default", "template": source}, {"name": "tool_use", "template": toolSource}}, ""},
		{"standalone", "incorrect default", source},
	} {
		t.Run(test.name, func(t *testing.T) {
			directory := t.TempDir()
			// Small shared fixture catalog exercises public conversion, not model quality.
			spec := testutil.DenseCausalSpec{Vocab: 3, Hidden: 2, Heads: 1, HeadDim: 2, KVHeads: 1, Intermediate: 2, Layers: 1}
			weights, shapes := testutil.DenseCausalWeights(t, spec)
			if err := safetensors.Save(filepath.Join(directory, "model.safetensors"), weights, shapes, nil); err != nil {
				t.Fatal(err)
			}
			config := map[string]any{"model_type": "llama", "hidden_size": spec.Hidden, "intermediate_size": spec.Intermediate, "num_attention_heads": spec.Heads, "num_key_value_heads": spec.KVHeads, "head_dim": spec.HeadDim, "num_hidden_layers": spec.Layers, "vocab_size": spec.Vocab, "max_position_embeddings": 16, "rms_norm_eps": 1e-5, "rope_theta": 10000, "tie_word_embeddings": true}
			encoded, err := json.Marshal(config)
			if err != nil {
				t.Fatal(err)
			}
			testutil.WriteTextFile(t, directory, "config.json", string(encoded))
			testutil.WriteTextFile(t, directory, "tokenizer.json", `{"model":{"vocab":{"a":0,"<s>":1,"</s>":2},"merges":[]},"added_tokens":[{"id":1,"content":"<s>","special":true},{"id":2,"content":"</s>","special":true}]}`)
			testutil.WriteTextFile(t, directory, "generation_config.json", `{"bos_token_id":1,"eos_token_id":2}`)
			encoded, err = json.Marshal(map[string]any{"chat_template": test.declaration})
			if err != nil {
				t.Fatal(err)
			}
			testutil.WriteTextFile(t, directory, "tokenizer_config.json", string(encoded))
			if test.standalone != "" {
				testutil.WriteTextFile(t, directory, "chat_template.jinja", test.standalone)
			}
			path := filepath.Join(directory, "model.gguf")
			if _, err := hfconvert.Convert(hfconvert.Options{Directory: directory, OutputPath: path}); err != nil {
				t.Fatal(err)
			}
			file, err := gguf.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer file.Close()
			vocab, err := tokenizer.Load(file)
			if err != nil {
				t.Fatal(err)
			}
			if vocab.BOS != 1 || vocab.EOS != 2 {
				t.Fatalf("special IDs changed: BOS=%d EOS=%d", vocab.BOS, vocab.EOS)
			}
			runner := &Runner{preparedModel: preparedModel{file: file, vocab: vocab}}
			messages := []ChatMessage{{Role: "system", Content: "rules"}, {Role: "user", Content: "hello"}}
			for _, render := range []struct {
				name    string
				options ChatFormatOptions
				want    string
			}{
				{"generation", ChatFormatOptions{AddGenerationPrompt: true}, "<s>[system] rules</s>\n[user] hello</s>\n[assistant]"},
				{"thinking", ChatFormatOptions{AddGenerationPrompt: true, EnableThinking: true}, "<s>[system] rules</s>\n[user] hello</s>\n[assistant]<think>"},
				{"history", ChatFormatOptions{}, "<s>[system] rules</s>\n[user] hello</s>\n"},
			} {
				t.Run(render.name, func(t *testing.T) {
					got, err := runner.FormatChatWithOptions(messages, render.options)
					if err != nil {
						t.Fatal(err)
					}
					if got != render.want {
						t.Fatalf("prompt=%q, want %q", got, render.want)
					}
				})
			}
			options := ChatFormatOptions{AddGenerationPrompt: true, Tools: []ChatTool{{Type: "function", Function: ChatToolDefinition{Name: "weather", Parameters: map[string]any{"type": "object", "properties": map[string]any{}}}}}}
			want := "<s>[system] rules</s>\n[user] hello</s>\n[assistant]"
			if test.name == "embedded_named" {
				want = "weather\n" + want
			}
			got, err := runner.FormatChatWithOptions(messages, options)
			if err != nil {
				t.Fatal(err)
			}
			if got != want {
				t.Fatalf("tool prompt=%q, want %q", got, want)
			}
		})
	}
}
