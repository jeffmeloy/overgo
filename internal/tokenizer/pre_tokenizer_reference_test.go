package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testutil"
)

// preTokenizerPiecesScript writes the fixtures this test reads, from a
// model's own tokenizer file run by the tokenizers library. It is kept where
// Python is kept, on the machine that ran it and not in git: the fixtures are
// the data.
const preTokenizerPiecesScript = "scripts/python_golden_references/pre_tokenizer_pieces.py"

// TestPreTokenizerCutsAsTheModelsOwnTokenizerDoes holds a splitting rule to
// the tokenizer file it was written from. Each fixture of
// fixtures/pre_tokenizer is named for a rule and holds texts with the pieces
// the model's own tokenizer cuts them into before it merges bytes; the rule
// must cut every text into the same pieces. A piece cut otherwise is merged
// otherwise, and a model given other tokens than its own answers whatever it
// is given: no score shows that as a fault.
func TestPreTokenizerCutsAsTheModelsOwnTokenizerDoes(t *testing.T) {
	t.Parallel()
	fixtures, err := filepath.Glob(filepath.Join(testutil.RepoRoot(t), "fixtures", "pre_tokenizer", "*.json"))
	if err != nil || len(fixtures) == 0 {
		t.Fatalf("no rule's pieces are on file (%v); write them with %s", err, preTokenizerPiecesScript)
	}
	for _, fixture := range fixtures {
		raw, err := os.ReadFile(fixture)
		if err != nil {
			t.Fatal(err)
		}
		var reference struct {
			Rule  string `json:"rule"`
			Cases []struct {
				Text   string   `json:"text"`
				Pieces []string `json:"pieces"`
			} `json:"cases"`
		}
		if err := json.Unmarshal(raw, &reference); err != nil || len(reference.Cases) == 0 {
			t.Fatalf("%s holds no text: %v", fixture, err)
		}
		if name := strings.TrimSuffix(filepath.Base(fixture), ".json"); name != reference.Rule || !supportedPreTokenizer(reference.Rule) {
			t.Fatalf("%s holds the pieces of %q, which is not its name or not a rule overgo has", fixture, reference.Rule)
		}
		pieces, differ := 0, 0
		for _, held := range reference.Cases {
			got := preTokenizeFor(reference.Rule, held.Text)
			pieces += len(held.Pieces)
			if !slices.Equal(got, held.Pieces) {
				if differ++; differ <= 3 {
					t.Errorf("%s cuts %q into\n%q\nwhere the model's own tokenizer cuts it into\n%q", reference.Rule, held.Text, got, held.Pieces)
				}
			}
		}
		if differ != 0 {
			t.Errorf("%s cuts %d of %d texts otherwise than the model's own tokenizer", reference.Rule, differ, len(reference.Cases))
			continue
		}
		t.Logf("%s cuts %d texts into the model's own %d pieces", reference.Rule, len(reference.Cases), pieces)
	}
}
