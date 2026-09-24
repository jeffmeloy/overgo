package hfbpe

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestKimiTinyIgnoreMerges uses Colibri c/tests/tok_kimi_tiny.json at
// f028d26b422144ed4a69ad9aeaee2553ce0f9572 (Apache-2.0). Its C test
// independently asserts the Han/Latin token counts; the pinned vocabulary
// supplies exact IDs. This tiny fixture does not establish real Kimi IDs.
func TestKimiTinyIgnoreMerges(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "kimi_tiny.json"))
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(raw)); got != "99b5ff4e6cc9f14bcdc9c4e1d854c35d4afc087af8e825f3264b1ebbc10132eb" {
		t.Fatalf("Kimi source fixture changed: %s", got)
	}
	load := func(source []byte) *Tokenizer {
		t.Helper()
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), source, 0600); err != nil {
			t.Fatal(err)
		}
		encoder, err := Load(dir)
		if err != nil {
			t.Fatal(err)
		}
		return encoder
	}
	encoder := load(raw)
	for _, sample := range []struct {
		text string
		ids  []int
	}{
		{"中文", []int{276}},
		{"中文abc", []int{276, 269, 99}},
		{"龥a", []int{279, 97}},
		{"abc", []int{269, 99}},
		{"hello", []int{259}},
		{"1234", []int{268, 52}},
		{"./", []int{272}},
		{"\n\n", []int{273}},
	} {
		got, err := encoder.Encode(sample.text)
		if err != nil || !slices.Equal(got, sample.ids) {
			t.Errorf("Encode(%q) = %v, %v; want %v", sample.text, got, err, sample.ids)
		}
	}
	// The source fixture aliases six special IDs with ordinary vocabulary
	// IDs. Decode only unambiguous IDs; full decode parity needs a real vocab.
	for _, sample := range []struct {
		id   int
		text string
	}{{259, "hello"}, {268, "123"}, {272, "./"}, {273, "\n\n"}} {
		got, err := encoder.DecodeStrict([]int{sample.id})
		if err != nil || got != sample.text {
			t.Errorf("DecodeStrict(%d) = %q, %v; want %q", sample.id, got, err, sample.text)
		}
	}
	withoutIgnore := strings.Replace(string(raw), `"ignore_merges": true`, `"ignore_merges": false`, 1)
	if withoutIgnore == string(raw) {
		t.Fatal("Kimi source no longer declares ignore_merges")
	}
	control, err := load([]byte(withoutIgnore)).Encode("中文")
	if err != nil || slices.Equal(control, []int{276}) {
		t.Fatalf("ignore_merges control ids=%v error=%v; expected byte-level BPE", control, err)
	}
}
