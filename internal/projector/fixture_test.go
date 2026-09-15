package projector

import (
	"strings"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

func classifyProjectorIntegration(t *testing.T) {
	t.Helper()
	if testing.Short() {
		t.Skip(testskip.ShortIntegration)
	}
}

func writeProjectorFixture(
	t *testing.T,
	name string,
	metadata []gguf.Metadata,
	tensors []gguf.TensorData,
) string {
	t.Helper()
	return testutil.TempGGUF(t, name, metadata, tensors)
}

func rewriteProjectorFixture(
	t *testing.T,
	path string,
	metadata []gguf.Metadata,
	tensors []gguf.TensorData,
) {
	t.Helper()
	testutil.WriteGGUF(t, path, metadata, tensors)
}

// imagePromptTokenizer preserves each fixture's image marker and token ID.
type imagePromptTokenizer struct {
	marker string
	token  tokenizer.TokenID
}

func (fixture imagePromptTokenizer) TokenizeText(text string, _, _ bool) ([]tokenizer.TokenID, error) {
	const placeholder = "\x00"
	text = strings.ReplaceAll(text, fixture.marker, placeholder)
	ids := make([]tokenizer.TokenID, 0, len(text))
	for _, value := range text {
		if value == 0 {
			ids = append(ids, fixture.token)
		} else {
			ids = append(ids, tokenizer.TokenID(value))
		}
	}
	return ids, nil
}
