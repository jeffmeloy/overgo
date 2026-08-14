package speechsynth

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"overgo/internal/testevidence"
)

type g8Golden struct {
	VocabSize int `json:"vocab_size"`
	UnkID     int `json:"unk_id"`
	Pieces    []struct {
		Piece string  `json:"piece"`
		Score float64 `json:"score"`
	} `json:"pieces"`
	Cases []struct {
		Raw         string `json:"raw"`
		Prepared    string `json:"prepared"`
		IDsRaw      []int  `json:"ids_raw"`
		IDsPrepared []int  `json:"ids_prepared"`
	} `json:"cases"`
}

// unquoteEscaped decodes the golden dumper's \uXXXX/\UXXXXXXXX escaping.
func unquoteEscaped(t *testing.T, s string) string {
	t.Helper()
	out, err := strconv.Unquote(`"` + s + `"`)
	if err != nil {
		t.Fatalf("unquote %q: %v", s, err)
	}
	return out
}

func fixtureUnigram(t *testing.T, g8 *g8Golden) *Unigram {
	t.Helper()
	pieces := make([]Piece, len(g8.Pieces))
	for i, p := range g8.Pieces {
		pieces[i] = Piece{Piece: unquoteEscaped(t, p.Piece), Score: p.Score}
	}
	tok, err := NewUnigram(pieces, g8.UnkID)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

// TestTokenizerGoldenCases gates the unigram Viterbi against the reference
// sentencepiece encodings (g1 + every g8 raw/prepared case, byte-fallback
// included). Committed goldens: fail-not-skip once the fixture loads.
func TestTokenizerGoldenCases(t *testing.T) {
	g8 := loadFixture[g8Golden](t, "g8_tokenizer.json")
	if len(g8.Pieces) != g8.VocabSize {
		t.Fatalf("fixture pieces %d != vocab_size %d", len(g8.Pieces), g8.VocabSize)
	}
	tok := fixtureUnigram(t, g8)

	g1 := loadFixture[struct {
		Text string `json:"text"`
		IDs  []int  `json:"ids"`
	}](t, "g1_tokenizer.json")
	assertIDs := func(text string, want []int) {
		t.Helper()
		got, err := tok.Encode(text)
		if err != nil {
			t.Fatalf("%q: %v", text, err)
		}
		if len(got) != len(want) {
			t.Fatalf("%q: got %d ids %v, want %d %v", text, len(got), got, len(want), want)
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("%q: id[%d]=%d want %d (got %v want %v)", text, i, got[i], want[i], got, want)
			}
		}
	}
	assertIDs(g1.Text, g1.IDs)
	for _, c := range g8.Cases {
		assertIDs(unquoteEscaped(t, c.Raw), c.IDsRaw)
		assertIDs(unquoteEscaped(t, c.Prepared), c.IDsPrepared)
	}
	t.Logf("tokenizer: g1 + %d g8 cases (raw+prepared) exact", len(g8.Cases))
}

// TestTokenizerModelFileMatchesFixture gates the ModelProto reader: the
// artifact's tokenizer.model must reproduce the pinned piece table (text,
// score, unk id) and the g1 encoding.
func TestTokenizerModelFileMatchesFixture(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	path := filepath.Join(artifactDir(t), "tokenizer.model")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("UNAVAILABLE: tokenizer.model absent at %s; parity NOT verified", path)
	}
	pieces, unkID, err := ReadModelFile(path)
	if err != nil {
		t.Fatal(err)
	}
	g8 := loadFixture[g8Golden](t, "g8_tokenizer.json")
	if len(pieces) != len(g8.Pieces) || unkID != g8.UnkID {
		t.Fatalf("model file pieces=%d unk=%d, fixture pieces=%d unk=%d", len(pieces), unkID, len(g8.Pieces), g8.UnkID)
	}
	for i, p := range g8.Pieces {
		want := unquoteEscaped(t, p.Piece)
		if pieces[i].Piece != want {
			t.Fatalf("piece %d text %q != fixture %q", i, pieces[i].Piece, want)
		}
		if float32(pieces[i].Score) != float32(p.Score) {
			t.Fatalf("piece %d score %g != fixture %g", i, pieces[i].Score, p.Score)
		}
	}
	tok, err := NewUnigram(pieces, unkID)
	if err != nil {
		t.Fatal(err)
	}
	g1 := loadFixture[struct {
		Text string `json:"text"`
		IDs  []int  `json:"ids"`
	}](t, "g1_tokenizer.json")
	got, err := tok.Encode(g1.Text)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(g1.IDs) {
		t.Fatalf("g1 via model file: got %v want %v", got, g1.IDs)
	}
	for i := range got {
		if got[i] != g1.IDs[i] {
			t.Fatalf("g1 via model file: got %v want %v", got, g1.IDs)
		}
	}
}
