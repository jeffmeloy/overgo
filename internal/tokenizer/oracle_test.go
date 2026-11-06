package tokenizer

import (
	"bufio"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
)

// TestUpstreamTokenizerCorpus compares against checked-in llama.cpp
// tokenizer oracle when OVERGO_LLAMA_CPP points at source checkout
func TestUpstreamTokenizerCorpus(t *testing.T) {
	testskip.Short(t, "requires pinned llama.cpp tokenizer corpus")
	source := os.Getenv("OVERGO_LLAMA_CPP")
	if source == "" {
		testskip.NotApplicable(t, "OVERGO_LLAMA_CPP is not set")
	}
	for _, name := range []string{"qwen2", "qwen35", "deepseek-llm", "gpt-2", "llama-spm", "llama-bpe", "bert-bge"} {
		t.Run(name, func(t *testing.T) {
			modelPath := filepath.Join(source, "models", "ggml-vocab-"+name+".gguf")
			file := testcheck.Must(gguf.Open(modelPath))(t)
			defer file.Close()
			vocab := testcheck.Must(Load(file))(t)
			inputData := testcheck.Must(os.ReadFile(modelPath + ".inp"))(t)
			expected := testcheck.Must(readExpectedTokenIDs(modelPath + ".out"))(t)
			// oracle: C++ text-mode file; Normalize its Windows CRLF
			// representation to bytes observed by std::ifstream
			inputText := strings.ReplaceAll(string(inputData), "\r\n", "\n")
			const separator = "\n__ggml_vocab_test__\n"
			inputs := strings.Split(inputText, separator)
			if strings.HasSuffix(inputText, separator) {
				inputs = inputs[:len(inputs)-1]
			}
			if len(inputs) != len(expected) {
				t.Fatalf("corpus has %d inputs and %d outputs", len(inputs), len(expected))
			}
			for i, input := range inputs {
				got, encodeErr := vocab.Encode(input, EncodeOptions{})
				if encodeErr != nil {
					t.Fatalf("case %d %q: %v", i, input, encodeErr)
				}
				if !reflect.DeepEqual(got, expected[i]) {
					pieces := make([]string, len(expected[i]))
					for index, id := range expected[i] {
						token, _ := vocab.Token(id)
						pieces[index] = token.Text
					}
					t.Fatalf(
						"case %d %q: got %v, want %v (%q)",
						i,
						input,
						got,
						expected[i],
						pieces,
					)
				}
			}
		})
	}
}

func readExpectedTokenIDs(path string) ([][]TokenID, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	var result [][]TokenID
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		ids := make([]TokenID, len(fields))
		for i, field := range fields {
			value, parseErr := ParseTokenID(field)
			if parseErr != nil {
				return nil, parseErr
			}
			ids[i] = value
		}
		result = append(result, ids)
	}
	return result, scanner.Err()
}
