package projector

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/dataroot"
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

// The converted projectors the real tests read: Gemma 4 12B's, and Qwen3.5 4B's with its facts.
const (
	gemma4Projector = "gemma-4-12B-it-fp8-mmproj-bf16.gguf"
	qwen35Projector = "Qwen3.5-4B-mmproj-bf16-facts-v2.gguf"
)

// realProjectorFile is a real file a projector test reads, which skips the test as inapplicable where this machine
// holds none: a converted projector under the checkpoints' overgo-hfconvert, else a fixture beside the models, as
// adaptiveparity's TestMultimodalInputMatrix finds them. Each had been named by a variable no lane set, so the tests
// ran nowhere (opt-in-tests/run-or-remove).
func realProjectorFile(t *testing.T, name string) string {
	t.Helper()
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	path := filepath.Join(filepath.Dir(roots.Models), "fixtures", name)
	if strings.HasSuffix(name, ".gguf") {
		path = filepath.Join(roots.Checkpoints, "overgo-hfconvert", name)
	}
	if _, err := os.Stat(path); err != nil {
		testskip.NotApplicable(t, "this machine holds no "+name)
	}
	return path
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
