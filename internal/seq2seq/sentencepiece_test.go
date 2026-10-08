package seq2seq

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/tokenizer"
)

// modelBytes writes a SentencePiece model file's fields as protobuf: each
// piece with its text and type, the trainer's model type, and, when collapse
// is not the default, whether runs of spaces are collapsed.
func modelBytes(pieces []string, kinds []int, modelType int, collapse *bool) []byte {
	field := func(number, wire int, body ...byte) []byte {
		return append([]byte{byte(number<<wireBits | wire)}, body...)
	}
	sized := func(number int, body []byte) []byte {
		return field(number, wireBytes, append([]byte{byte(len(body))}, body...)...)
	}
	var model []byte
	for index, piece := range pieces {
		entry := append(sized(pieceText, []byte(piece)), field(pieceType, wireVarint, byte(kinds[index]))...)
		model = append(model, sized(modelPieces, entry)...)
	}
	model = append(model, sized(modelTrainer, field(trainerModelType, wireVarint, byte(modelType)))...)
	if collapse != nil {
		value := byte(0)
		if *collapse {
			value = 1
		}
		model = append(model, sized(modelNormalizer, field(normalizerCollapseSpaces, wireVarint, value))...)
	}
	return model
}

// TestPieceMergesSegmentAsByteMergesDo holds the segmentation of a tokenizer
// model that declares byte-pair merges, on a table small enough to follow by
// hand. Two neighbours join in the order their joined piece's score gives,
// the higher first and the leftmost of two alike first, so "abc" is the
// pieces training joined earliest and not the cover a unigram table would
// score best; a user-defined symbol is kept whole and nothing joins across
// it; runs of spaces collapse unless the model says not to; a character no
// piece spells is written as its bytes, and as the unknown piece where a byte
// has none; a control piece's name spelled out in text is not that piece. A
// model that declares no merges is left to be segmented as a unigram table.
func TestPieceMergesSegmentAsByteMergesDo(t *testing.T) {
	t.Parallel()
	const control, unknownKind = 3, 2
	texts := []string{"<unk>", spaceMarker, "a", "b", "c", "ab", "bc", spaceMarker + "a", "abc", "<tools>", "aa", "<0xC3>", "</s>", "<", "/", "s", ">"}
	kinds := []int{unknownKind, pieceNormal, pieceNormal, pieceNormal, pieceNormal, pieceNormal, pieceNormal, pieceNormal, pieceNormal, pieceUserDefined, pieceNormal, pieceByte, control, pieceNormal, pieceNormal, pieceNormal, pieceNormal}
	scores := []float64{0, 0, 0, 0, 0, -1, -0.5, -2, -3, 0, -1, 0, 0, 0, 0, 0, 0}
	pieces := make([]tokenizer.UnigramPiece, len(texts))
	for index, text := range texts {
		pieces[index] = tokenizer.UnigramPiece{Piece: text, Score: scores[index]}
	}
	id := func(text string) int { return slices.Index(texts, text) }
	load := func(modelType int, collapse *bool) (*pieceMerges, bool) {
		t.Helper()
		path := filepath.Join(t.TempDir(), "tokenizer.model")
		if err := os.WriteFile(path, modelBytes(texts, kinds, modelType, collapse), 0o600); err != nil {
			t.Fatal(err)
		}
		table, merges, err := loadPieceMerges(path, pieces, id("<unk>"))
		if err != nil {
			t.Fatal(err)
		}
		return table, merges
	}
	table, merges := load(modelTypeMerges, nil)
	if !merges || !table.collapse {
		t.Fatalf("a model of merges that says nothing of spaces reads merges=%t collapse=%t, want both", merges, table.collapse)
	}
	for text, want := range map[string][]string{
		"abc":       {spaceMarker + "a", "bc"},
		"aaa":       {spaceMarker, "aa", "a"},
		"a<tools>b": {spaceMarker + "a", "<tools>", "b"},
		" a   b ":   {spaceMarker + "a", spaceMarker, "b"},
		"aé":        {spaceMarker + "a", "<0xC3>", "<unk>"},
		"</s>":      {spaceMarker, "<", "/", "s", ">"},
	} {
		ids, err := table.encode(text)
		if err != nil {
			t.Fatal(err)
		}
		expected := make([]int, len(want))
		for index, piece := range want {
			expected[index] = id(piece)
		}
		if !slices.Equal(ids, expected) {
			t.Errorf("%q segments to %v, want %v, the pieces %q", text, ids, expected, want)
		}
	}
	if ids, err := table.encode("   "); err != nil || ids != nil {
		t.Errorf("spaces alone segment to %v (%v), want nothing", ids, err)
	}
	keep := false
	if spaced, _ := load(modelTypeMerges, &keep); spaced.collapse {
		t.Error("a model that says runs of spaces are kept reads as collapsing them")
	} else if ids, _ := spaced.encode("a  b"); !slices.Equal(ids, []int{id(spaceMarker + "a"), id(spaceMarker), id(spaceMarker), id("b")}) {
		t.Errorf("with runs of spaces kept, two spaces segment to %v", ids)
	}
	if _, merges := load(pieceNormal, nil); merges {
		t.Error("a unigram model reads as one of merges")
	}
	if _, _, err := loadPieceMerges(filepath.Join(t.TempDir(), "absent.model"), pieces, 0); err == nil {
		t.Error("an absent model file was read")
	}
}
