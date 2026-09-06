package modelartifact

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestModelConfigImageAttentionDeclaration(t *testing.T) {
	for _, test := range []struct {
		name, content, want string
		refuse              bool
	}{
		{"absent", `{}`, "", false},
		{"explicit causal", `{"text_config":{"use_bidirectional_attention":null}}`, "causal", false},
		{"vision", `{"text_config":{"use_bidirectional_attention":"vision"}}`, "vision", false},
		{"flat", `{"use_bidirectional_attention":"vision"}`, "vision", false},
		{"equivalent declarations", `{"use_bidirectional_attention":"vision","text_config":{"use_bidirectional_attention":"\u0076ision"}}`, "vision", false},
		{"empty", `{"use_bidirectional_attention":""}`, "", true},
		{"unknown", `{"use_bidirectional_attention":"future"}`, "", true},
		{"unsupported all", `{"use_bidirectional_attention":"all"}`, "", true},
		{"boolean", `{"use_bidirectional_attention":false}`, "", true},
		{"contradictory", `{"use_bidirectional_attention":"vision","text_config":{"use_bidirectional_attention":null}}`, "", true},
		{"malformed nested", `{"text_config":false}`, "", true},
	} {
		t.Run(test.name, func(t *testing.T) {
			directory := t.TempDir()
			if err := os.WriteFile(filepath.Join(directory, "config.json"), []byte(test.content), 0600); err != nil {
				t.Fatal(err)
			}
			_, generation, sources, err := ReadModelConfigComponents(directory)
			if test.refuse {
				if err == nil {
					t.Fatal("unsupported declaration accepted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			got := ""
			if generation != nil {
				got = generation.ImageAttention
			}
			if got != test.want {
				t.Fatalf("attention=%q want %q", got, test.want)
			}
			if len(sources) != 1 || sources[0].Name != "config.json" || sources[0].SHA256 != fmt.Sprintf("%x", sha256.Sum256([]byte(test.content))) {
				t.Fatalf("declaration lost exact source provenance: %+v", sources)
			}
		})
	}
}
