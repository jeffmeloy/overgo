package hfbpe

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/binaryschema"
	"overgo/internal/modeltest"
)

// A byte-complete vocabulary exposes the normalized input directly as IDs;
// fixed Unicode spellings provide the expected bytes independently of BPE.
func TestDeclaredNFCNormalization(t *testing.T) {
	nfc := loadTokenizerFixture(t, json.RawMessage(`{"type":"NFC"}`), nil, nil)
	plain := loadTokenizerFixture(t, nil, nil, nil)
	for _, input := range []string{"caf\u00e9", "cafe\u0301"} {
		got, err := nfc.Encode(input)
		if err != nil || !slices.Equal(got, byteIDs("caf\u00e9")) {
			t.Fatalf("declared NFC input=%q ids=%v error=%v", input, got, err)
		}
	}
	got, err := plain.Encode("cafe\u0301")
	if err != nil || !slices.Equal(got, byteIDs("cafe\u0301")) {
		t.Fatalf("undeclared normalization changed IDs: %v %v", got, err)
	}
	if _, err := nfc.Encode("x\xff"); err == nil {
		t.Fatal("NFC encoding accepted non-UTF-8 input")
	}

	// Raw added tokens are extracted before normalization, including ordinary
	// tokens; composition must not cross their boundary.
	for _, special := range []bool{false, true} {
		added := []map[string]any{{"id": binaryschema.ByteValueCount, "content": "e", "special": special, "normalized": false}}
		tokenizer := loadTokenizerFixture(t, json.RawMessage(`{"type":"NFC"}`), nil, added)
		want := append([]int{binaryschema.ByteValueCount}, byteIDs("\u0301")...)
		got, err := tokenizer.Encode("e\u0301")
		if err != nil || !slices.Equal(got, want) {
			t.Fatalf("raw added-token boundary: %v %v", got, err)
		}
	}
	for _, flag := range []string{"normalized", "single_word", "lstrip", "rstrip"} {
		added := map[string]any{"id": binaryschema.ByteValueCount, "content": "token", "normalized": false}
		added[flag] = true
		tokenizer := loadTokenizerFixture(t, json.RawMessage(`{"type":"NFC"}`), nil, []map[string]any{added})
		if _, err := tokenizer.Encode("token"); err == nil {
			t.Fatalf("silently ignored NFC added-token %s", flag)
		}
		if got := tokenizer.Decode([]int{binaryschema.ByteValueCount}); got != "token" {
			t.Fatalf("decoding depended on encoding declaration: %q", got)
		}
	}
	unspecified := loadTokenizerFixture(t, json.RawMessage(`{"type":"NFC"}`), nil, []map[string]any{{"id": binaryschema.ByteValueCount, "content": "token"}})
	if _, err := unspecified.Encode("token"); err == nil {
		t.Fatal("NFC added-token matching accepted unspecified normalization")
	}
	for _, declaration := range []string{`{"type":"Lowercase"}`, `{"type":"UnknownNormalization"}`, `{"type":"Sequence","normalizers":[{"type":"NFC"}]}`, `{"type":"Sequence"}`, `{"type":"Replace","pattern":{"Regex":" +"},"content":"_"}`} {
		tokenizer := loadTokenizerFixture(t, json.RawMessage(declaration), nil, nil)
		if _, err := tokenizer.Encode("A"); err == nil {
			t.Fatalf("silently ignored %s", declaration)
		}
		if got, err := tokenizer.DecodeStrict(byteIDs("A")); err != nil || got != "A" {
			t.Fatalf("decode-only use failed: %q %v", got, err)
		}
	}
	// Existing literal space replacement still selects the marker scheme.
	directory := t.TempDir()
	fixture := `{"model":{"type":"BPE","vocab":{"a":0,"▁":1,"b":2},"merges":[]},"normalizer":{"type":"Replace","pattern":{"String":" "},"content":"▁"}}`
	if err := os.WriteFile(filepath.Join(directory, "tokenizer.json"), []byte(fixture), 0600); err != nil {
		t.Fatal(err)
	}
	marker, err := Load(directory)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := marker.Encode("a b"); err != nil || !slices.Equal(got, []int{0, 1, 2}) {
		t.Fatalf("legacy marker encoding changed: %v %v", got, err)
	}
}

// RxBrain declares an empty Sequence. It must preserve decomposed input, not
// apply NFC and not refuse an otherwise valid encoding request.
func TestRxBrainEmptyNormalizerSequence(t *testing.T) {
	identity := loadTokenizerFixture(t, json.RawMessage(`{"type":"Sequence","normalizers":[]}`), nil, nil)
	input := "cafe\u0301"
	got, err := identity.Encode(input)
	if err != nil || !slices.Equal(got, byteIDs(input)) {
		t.Fatalf("empty Sequence changed input: ids=%v error=%v", got, err)
	}
}

// These IDs were frozen against the pinned Krea artifact before NFC was
// reactivated. The decomposed spelling catches omission of declared NFC.
func TestKreaDeclaredNFCParity(t *testing.T) {
	dir := modeltest.Directory(t, "Krea-2-Turbo")
	tokenizerPath := filepath.Join(dir, "tokenizer", "tokenizer.json")
	data, err := os.ReadFile(tokenizerPath)
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506" {
		t.Fatalf("Krea tokenizer drifted: %s", got)
	}
	tokenizer, err := Load(filepath.Dir(tokenizerPath))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		input string
		ids   []int
	}{
		{"a red fox eating ice cream, studio photograph", []int{64, 2518, 38835, 12182, 9853, 12644, 11, 14029, 10300}},
		{"cafe\u0301", []int{924, 58858}},
		{"\n \n  a", []int{26809, 220, 264}},
	} {
		got, err := tokenizer.Encode(tc.input)
		if err != nil || !slices.Equal(got, tc.ids) {
			t.Fatalf("Krea %q: ids=%v want=%v err=%v", tc.input, got, tc.ids, err)
		}
	}
}

func byteIDs(text string) []int {
	result := make([]int, len(text))
	for index := range len(text) {
		result[index] = int(text[index])
	}
	return result
}

func loadTokenizerFixture(t *testing.T, normalizer, pre any, added []map[string]any) *Tokenizer {
	t.Helper()
	var alphabet Tokenizer
	alphabet.buildByteAlphabet()
	vocabulary := map[string]int{}
	for value, character := range alphabet.b2u {
		vocabulary[string(character)] = value
	}
	fixture := map[string]any{
		"model":      map[string]any{"type": "BPE", "vocab": vocabulary, "merges": []string{}},
		"normalizer": normalizer, "pre_tokenizer": pre, "added_tokens": added,
	}
	data, err := json.Marshal(fixture)
	if err != nil {
		t.Fatal(err)
	}
	directory := t.TempDir()
	if err := os.WriteFile(filepath.Join(directory, "tokenizer.json"), data, 0600); err != nil {
		t.Fatal(err)
	}
	tokenizer, err := Load(directory)
	if err != nil {
		t.Fatal(err)
	}
	return tokenizer
}
