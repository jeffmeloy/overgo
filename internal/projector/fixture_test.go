package projector

import (
	"strings"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

func classifyProjectorIntegration(t *testing.T) {
	t.Helper()
	testskip.Short(t)
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

// recordingPromptTokenizer records each text it tokenizes and gives token 8 for each image pad in it and 1 for each
// other byte.
type recordingPromptTokenizer struct {
	pad   string
	texts []string
}

func (recorder *recordingPromptTokenizer) TokenizeText(text string, _, _ bool) ([]tokenizer.TokenID, error) {
	recorder.texts = append(recorder.texts, text)
	ids := make([]tokenizer.TokenID, 0, len(text))
	for len(text) > 0 {
		if strings.HasPrefix(text, recorder.pad) {
			ids, text = append(ids, 8), text[len(recorder.pad):]
		} else {
			ids, text = append(ids, 1), text[1:]
		}
	}
	return ids, nil
}

// requireOpensAs opens the projector written at path and requires it to be a Runner.
func requireOpensAs[Runner any](t *testing.T, path string) {
	t.Helper()
	projector := testcheck.Must(OpenAs[Projector](t.Context(), path, OpenOptions{}))(t)
	defer projector.Close()
	if _, ok := projector.(Runner); !ok {
		t.Fatalf("projector type = %T", projector)
	}
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
