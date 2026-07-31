package tokenizer

import (
	"reflect"
	"testing"

	"llamacpp2go/internal/gguf"
)

func TestByteCodecRoundTrip(t *testing.T) {
	input := []byte{0, '\n', ' ', '!', 0xa0, 0xff}
	encoded := encodeBytes(input)
	decoded, err := decodeBytes(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decoded, input) {
		t.Fatalf("decoded = %v, want %v", decoded, input)
	}
	if encoded != "ĀĊĠ!łÿ" {
		t.Fatalf("encoded = %q, want %q", encoded, "ĀĊĠ!łÿ")
	}
}

func TestEndOfGenerationRecognizesPromotedTokenText(t *testing.T) {
	vocab := &Vocab{
		Tokens: []Token{{Text: "ordinary"}, {Text: "<eos>"}, {Text: "<end_of_turn>"}},
		EOS:    NullToken,
		EOT:    NullToken,
		EOM:    NullToken,
	}
	if vocab.IsEOG(0) || !vocab.IsEOG(1) || !vocab.IsEOG(2) {
		t.Fatalf("unexpected EOG classification: %v", vocab.EOGTokens())
	}
	if want := []TokenID{1, 2}; !reflect.DeepEqual(vocab.EOGTokens(), want) {
		t.Fatalf("EOG tokens = %v, want %v", vocab.EOGTokens(), want)
	}
}

func TestPreTokenizeQwen2(t *testing.T) {
	tests := []struct {
		text string
		want []string
	}{
		{"Hello world!", []string{"Hello", " world", "!"}},
		{"WE'RE ready", []string{"WE", "'RE", " ready"}},
		{"123", []string{"1", "2", "3"}},
		{"foo  bar", []string{"foo", " ", " bar"}},
		{"a \n  b", []string{"a", " \n", " ", " b"}},
		{"你好，world", []string{"你好", "，world"}},
	}
	for _, test := range tests {
		if got := preTokenizeQwen2(test.text); !reflect.DeepEqual(got, test.want) {
			t.Errorf("preTokenizeQwen2(%q) = %#v, want %#v", test.text, got, test.want)
		}
	}
}

func TestBailingPreTokenizersUseQwen2Segmentation(t *testing.T) {
	for _, pre := range []string{"bailingmoe", "bailingmoe2"} {
		if !supportedPreTokenizer(pre) || !isQwen2Pre(pre) {
			t.Fatalf("pre-tokenizer %q is not mapped to Bailing segmentation", pre)
		}
	}
}

func TestDBRXPreTokenizerUsesLlama3Segmentation(t *testing.T) {
	if !supportedPreTokenizer("dbrx") || !isLlama3Pre("dbrx") {
		t.Fatal("DBRX pre-tokenizer is not mapped to Llama 3 segmentation")
	}
}

func TestPreTokenizeDeepSeekLLM(t *testing.T) {
	tests := []struct {
		text string
		want []string
	}{
		{"Hello world!", []string{"Hello", " world", "!"}},
		{"foo  bar", []string{"foo", " ", " bar"}},
		{"a\n b", []string{"a", "\n", " b"}},
		{"我想在apple工作1314151天～", []string{"我想在", "apple", "工作", "1314151", "天", "～"}},
		{"Cửa Việt", []string{"C", "ử", "a", " Vi", "ệ", "t"}},
		{"x  ", []string{"x", "  "}},
	}
	for _, test := range tests {
		if got := preTokenizeDeepSeekLLM(test.text); !reflect.DeepEqual(got, test.want) {
			t.Errorf("preTokenizeDeepSeekLLM(%q) = %#v, want %#v", test.text, got, test.want)
		}
	}
}

func TestPreTokenizeGPT2(t *testing.T) {
	tests := []struct {
		text string
		want []string
	}{
		{"Hello world!", []string{"Hello", " world", "!"}},
		{"we're 123", []string{"we", "'re", " 123"}},
		{"foo  bar", []string{"foo", " ", " bar"}},
	}
	for _, test := range tests {
		if got := preTokenizeGPT2(test.text); !reflect.DeepEqual(got, test.want) {
			t.Errorf("preTokenizeGPT2(%q) = %#v, want %#v", test.text, got, test.want)
		}
	}
}

func TestLoadEncodeDecodeAndSpecialTokens(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		scalar("tokenizer.ggml.model", gguf.ValueTypeString, "gpt2"),
		scalar("tokenizer.ggml.pre", gguf.ValueTypeString, "qwen2"),
		array("tokenizer.ggml.tokens", gguf.ValueTypeString, []string{
			"h", "e", "l", "o", "he", "hel", "hell", "hello", "<eos>", "1",
		}),
		array("tokenizer.ggml.token_type", gguf.ValueTypeInt32, []int32{
			1, 1, 1, 1, 1, 1, 1, 1, 3, 1,
		}),
		array("tokenizer.ggml.merges", gguf.ValueTypeString, []string{
			"h e", "he l", "hel l", "hell o",
		}),
		scalar("tokenizer.ggml.eos_token_id", gguf.ValueTypeUint32, uint32(8)),
		scalar("tokenizer.ggml.add_eos_token", gguf.ValueTypeBool, true),
	}}
	vocab, err := Load(file)
	if err != nil {
		t.Fatal(err)
	}
	if vocab.Len() != 10 || vocab.MergeCount() != 4 {
		t.Fatalf("vocab sizes = %d/%d, want 10/4", vocab.Len(), vocab.MergeCount())
	}

	ids, err := vocab.Encode("hello", EncodeOptions{AddSpecial: true})
	if err != nil {
		t.Fatal(err)
	}
	if want := []TokenID{7, 8}; !reflect.DeepEqual(ids, want) {
		t.Fatalf("Encode = %v, want %v", ids, want)
	}
	decoded, err := vocab.Decode(ids, false)
	if err != nil {
		t.Fatal(err)
	}
	if decoded != "hello" {
		t.Fatalf("Decode = %q, want hello", decoded)
	}

	ids, err = vocab.Encode("<eos>hello", EncodeOptions{ParseSpecial: true})
	if err != nil {
		t.Fatal(err)
	}
	if want := []TokenID{8, 7}; !reflect.DeepEqual(ids, want) {
		t.Fatalf("special Encode = %v, want %v", ids, want)
	}
}

func TestLoadRejectsDuplicateTokens(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		scalar("tokenizer.ggml.model", gguf.ValueTypeString, "gpt2"),
		array("tokenizer.ggml.tokens", gguf.ValueTypeString, []string{"x", "x"}),
		array("tokenizer.ggml.merges", gguf.ValueTypeString, []string{"x x"}),
	}}
	if _, err := Load(file); err == nil {
		t.Fatal("Load accepted duplicate tokens")
	}
}

func TestLoadFIMMetadataFallbacksAndEOGPromotion(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		scalar("tokenizer.ggml.model", gguf.ValueTypeString, "gpt2"),
		array("tokenizer.ggml.tokens", gguf.ValueTypeString, []string{
			"x",
			"<|fim_prefix|>",
			"<|fim_suffix|>",
			"<|fim_middle|>",
			"<|fim_pad|>",
			"<|fim_repo|>",
			"<|file_sep|>",
			"<legacy-prefix>",
		}),
		array("tokenizer.ggml.token_type", gguf.ValueTypeInt32, []int32{
			1, 1, 1, 1, 1, 1, 1, 1,
		}),
		array("tokenizer.ggml.merges", gguf.ValueTypeString, []string{}),
		scalar("tokenizer.ggml.fim_pre_token_id", gguf.ValueTypeUint32, uint32(1)),
		scalar("tokenizer.ggml.prefix_token_id", gguf.ValueTypeUint32, uint32(7)),
	}}
	vocab, err := Load(file)
	if err != nil {
		t.Fatal(err)
	}
	if vocab.FIMPre != 7 ||
		vocab.FIMSuf != 2 ||
		vocab.FIMMid != 3 ||
		vocab.FIMPad != 4 ||
		vocab.FIMRep != 5 ||
		vocab.FIMSep != 6 {
		t.Fatalf(
			"FIM IDs = pre:%d suf:%d mid:%d pad:%d rep:%d sep:%d",
			vocab.FIMPre,
			vocab.FIMSuf,
			vocab.FIMMid,
			vocab.FIMPad,
			vocab.FIMRep,
			vocab.FIMSep,
		)
	}
	if !vocab.IsEOG(4) || !vocab.IsEOG(5) || !vocab.IsEOG(6) {
		t.Fatal("FIM pad/repository/separator tokens were not promoted to EOG")
	}
	for _, id := range []TokenID{2, 3, 4, 5, 6} {
		if vocab.Tokens[id].Type != TokenControl {
			t.Fatalf("auto-detected FIM token %d was not promoted to control", id)
		}
	}
}

func scalar(key string, kind gguf.ValueType, data any) gguf.Metadata {
	return gguf.Metadata{Key: key, Value: gguf.Value{Type: kind, Data: data}}
}

func array(key string, kind gguf.ValueType, data any) gguf.Metadata {
	return gguf.Metadata{
		Key: key,
		Value: gguf.Value{
			Type:      gguf.ValueTypeArray,
			ArrayType: kind,
			Data:      data,
		},
	}
}
