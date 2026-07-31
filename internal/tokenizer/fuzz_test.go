package tokenizer

import (
	"testing"

	"llamacpp2go/internal/gguf"
)

func FuzzEncodeDecodeNeverPanics(f *testing.F) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		scalar("tokenizer.ggml.model", gguf.ValueTypeString, "gpt2"),
		scalar("tokenizer.ggml.pre", gguf.ValueTypeString, "qwen2"),
		array("tokenizer.ggml.tokens", gguf.ValueTypeString, []string{
			"a", "b", "ab", "<eos>", "<0x00>", "<0xFF>",
		}),
		array("tokenizer.ggml.token_type", gguf.ValueTypeInt32, []int32{1, 1, 1, 3, 6, 6}),
		array("tokenizer.ggml.merges", gguf.ValueTypeString, []string{"a b"}),
		scalar("tokenizer.ggml.eos_token_id", gguf.ValueTypeUint32, uint32(3)),
	}}
	vocab, err := Load(file)
	if err != nil {
		f.Fatal(err)
	}
	f.Add([]byte("ab"), true, true)
	f.Add([]byte{0, 0xff}, false, false)
	f.Fuzz(func(t *testing.T, data []byte, addSpecial, parseSpecial bool) {
		if len(data) > 1<<20 {
			t.Skip()
		}
		ids, encodeErr := vocab.Encode(string(data), EncodeOptions{
			AddSpecial:   addSpecial,
			ParseSpecial: parseSpecial,
		})
		if encodeErr == nil {
			_, _ = vocab.Decode(ids, true)
		}
	})
}
