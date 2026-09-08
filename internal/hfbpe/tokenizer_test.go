package hfbpe

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDecodeStrictRejectsUnknownToken(t *testing.T) {
	tokenizer := &Tokenizer{vocab: map[string]int{"x": 3}, special: map[string]int{}}
	tokenizer.buildByteAlphabet()
	tokenizer.buildID2Vocab()
	if decoded, err := tokenizer.DecodeStrict([]int{4}); err == nil || decoded != "" {
		t.Fatalf("strict decode = %q, %v", decoded, err)
	}
	if decoded, err := tokenizer.DecodeStrict([]int{3}); err != nil || decoded != "x" {
		t.Fatalf("strict decode = %q, %v", decoded, err)
	}
}

func TestDeclaredMetaspaceDecoder(t *testing.T) {
	fixture := `{"model":{"type":"BPE","vocab":{"▁":0,"Hello":1,"▁world":2,"!":3,"<locale>":4,"▁added":5},"merges":[]},"normalizer":{"type":"Sequence","normalizers":[]},"decoder":{"type":"Metaspace","replacement":"▁","prepend_scheme":"SCHEME","split":true},"added_tokens":[{"id":4,"content":"<locale>","special":true},{"id":5,"content":"▁added","special":false}]}`
	for _, scheme := range []string{"always", "first", "never"} {
		t.Run(scheme, func(t *testing.T) {
			directory := t.TempDir()
			if err := os.WriteFile(filepath.Join(directory, "tokenizer.json"), []byte(strings.ReplaceAll(fixture, "SCHEME", scheme)), 0600); err != nil {
				t.Fatal(err)
			}
			tokenizer, err := Load(directory)
			if err != nil {
				t.Fatal(err)
			}
			prefix := ""
			if scheme == "never" {
				prefix = " "
			}
			want := prefix + "Hello world! added "
			ids := []int{0, 1, 2, 3, 5, 0, 4}
			text, err := tokenizer.DecodeText(ids)
			if err != nil || text != want {
				t.Fatalf("text=%q want=%q error=%v", text, want, err)
			}
			for size := 1; size <= len(ids); size++ {
				var state DecodeState
				var result strings.Builder
				for start := 0; start < len(ids); start += size {
					end := min(start+size, len(ids))
					part, next, err := tokenizer.DecodeTextChunk(ids[start:end], state, end == len(ids))
					if err != nil {
						t.Fatal(err)
					}
					result.WriteString(part)
					state = next
				}
				if result.String() != want || len(state.Pending) != 0 {
					t.Fatalf("chunk=%d text=%q want=%q", size, result.String(), want)
				}
			}
			literal, err := tokenizer.DecodeStrict(ids)
			if err != nil || literal != want+"<locale>" {
				t.Fatalf("literal=%q error=%v", literal, err)
			}
			if _, err := tokenizer.DecodeText([]int{999}); err == nil {
				t.Fatal("unknown ID was silently dropped")
			}
			if _, err := tokenizer.Encode("Hello world"); err == nil {
				t.Fatal("claimed unsupported normalization/encoding")
			}
		})
	}
	for _, change := range []string{"unknown", ""} {
		directory := t.TempDir()
		if err := os.WriteFile(filepath.Join(directory, "tokenizer.json"), []byte(strings.ReplaceAll(fixture, "SCHEME", change)), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := Load(directory); err == nil {
			t.Fatal("accepted undefined Metaspace prepend policy")
		}
	}
}
