package latentimage

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"overgo/internal/hfbpe"
	"path/filepath"
	"testing"
)

func TestMOSSConditioningGolden(t *testing.T) {
	source := mossGolden(t, "tokenizer")
	var prompts, files map[string]string
	for _, metadata := range source.Metadata {
		if err := json.Unmarshal([]byte(metadata["prompts"]), &prompts); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal([]byte(metadata["tokenizer_files"]), &files); err != nil {
			t.Fatal(err)
		}
	}
	if len(prompts) != 6 || len(files) != 3 {
		t.Fatal("native prompt fixture coverage")
	}
	packed := mossFixtureBytes(t, "tokenizer-source.tar.gz", "d45ccb489f4c3bcd47e23947ffe0fc0536e416cb2931fb40b3df7785850d2865")
	gz, err := gzip.NewReader(bytes.NewReader(packed))
	if err != nil {
		t.Fatal(err)
	}
	defer gz.Close()
	reader := tar.NewReader(gz)
	root := t.TempDir()
	seen := map[string]bool{}
	for {
		header, err := reader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		digest, found := files[header.Name]
		if !found || filepath.Base(header.Name) != header.Name || seen[header.Name] {
			t.Fatal("unexpected tokenizer source", header.Name)
		}
		data, err := io.ReadAll(reader)
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256(data)) != digest {
			t.Fatal("tokenizer source identity differs", header.Name)
		}
		if err = os.WriteFile(filepath.Join(root, header.Name), data, 0600); err != nil {
			t.Fatal(err)
		}
		seen[header.Name] = true
	}
	if len(seen) != len(files) {
		t.Fatal("missing tokenizer sources")
	}
	tok, err := hfbpe.Load(root)
	if err != nil {
		t.Fatal(err)
	}
	template, err := os.ReadFile(filepath.Join(root, "chat_template.jinja"))
	if err != nil {
		t.Fatal(err)
	}
	policy := textTemplate{ChatTemplate: string(template), EnableThinking: true, MaxTokens: 4096}
	for name, prompt := range prompts {
		t.Run(name, func(t *testing.T) {
			text, err := renderTextInput(tok, prompt, policy)
			if err != nil {
				t.Fatal(err)
			}
			for suffix, values := range map[string][]int{"input_ids": text.IDs, "attention_mask": maskInts(text.Mask)} {
				item := source.Tensors[name+"."+suffix]
				want, err := io.ReadAll(item.Reader())
				if err != nil {
					t.Fatal(err)
				}
				if item.DType != "I64" || len(want) != len(values)*8 {
					t.Fatal("native token shape differs")
				}
				for i, value := range values {
					if int64(binary.LittleEndian.Uint64(want[i*8:])) != int64(value) {
						t.Fatalf("native %s differs at %d", suffix, i)
					}
				}
			}
			if text.PromptRows != len(text.IDs) {
				t.Fatal("unexpected padded prompt")
			}
			t.Logf("tokens=%d IDs and unpadded mask exact", len(text.IDs))
		})
	}
	for _, invalid := range []textTemplate{{ChatTemplate: string(template)}, {ChatTemplate: "{% invalid %}", MaxTokens: 1}, {ChatTemplate: "", MaxTokens: 1}} {
		if _, err := renderTextInput(tok, "", invalid); err == nil {
			t.Fatal("invalid prompt policy accepted")
		}
	}
}
