package server

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestThePageKnowsThePromptEnhancementSchema holds recorded-request/do: the generation page tells a run's request from
// the prompt enhancement it cites by the enhancement's stored schema, which it names as the server stores it
// (promptEnhancementSchema); a schema renamed on one side alone would leave the page reading the enhancement as the
// request again (SQA-470).
func TestThePageKnowsThePromptEnhancementSchema(t *testing.T) {
	t.Parallel()
	script := testcheck.Must(os.ReadFile(filepath.Join(testutil.RepoRoot(t), "internal", "server", "webui", "mod", "generation.js")))(t)
	if want := `const promptEnhancementSchema = "` + promptEnhancementSchema + `";`; !strings.Contains(string(script), want) {
		t.Errorf("generation.js does not name the prompt enhancement's schema as the server stores it: want %s", want)
	}
}
