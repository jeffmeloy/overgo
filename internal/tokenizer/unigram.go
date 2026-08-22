// Table-based SentencePiece unigram: max-score Viterbi over an explicit
// piece table. One owner for every artifact-file unigram tokenizer (the
// GGUF-vocab UGM path in ugm.go stays separate: it reads GGUF metadata, this
// reads piece tables from ModelProto or HF tokenizer.json artifacts).
// Promoted from speechsynth (ported adaptive sentencepiece behavior).
package tokenizer

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"overgo/internal/jsonfile"
)

// UnigramPiece is one unigram vocabulary entry.
type UnigramPiece struct {
	Piece string
	Score float64
}

// sentencePieceSpaceMarker is SentencePiece's U+2581 word-boundary marker.
const sentencePieceSpaceMarker = "\u2581"

// Unigram owns Viterbi policy over a parsed piece table.
type Unigram struct {
	scores   map[string]float64
	ids      map[string]int
	pieces   []string
	maxPiece int // longest piece length in runes
	unkID    int
}

// NewUnigram builds the table; unkID must index a real piece.
func NewUnigram(pieces []UnigramPiece, unkID int) (*Unigram, error) {
	if len(pieces) == 0 {
		return nil, fmt.Errorf("tokenizer: empty unigram piece table")
	}
	if unkID < 0 || unkID >= len(pieces) {
		return nil, fmt.Errorf("tokenizer: unigram unk id %d outside %d pieces", unkID, len(pieces))
	}
	t := &Unigram{
		scores: make(map[string]float64, len(pieces)),
		ids:    make(map[string]int, len(pieces)),
		pieces: make([]string, len(pieces)),
		unkID:  unkID,
	}
	for i, p := range pieces {
		t.scores[p.Piece] = p.Score
		t.ids[p.Piece] = i
		t.pieces[i] = p.Piece
		if n := len([]rune(p.Piece)); n > t.maxPiece {
			t.maxPiece = n
		}
	}
	return t, nil
}

// Decode applies the SentencePiece decoder: concatenate pieces, restore word
// boundaries, and fuse byte-fallback tokens.
func (t *Unigram) Decode(ids []int) string {
	var decoded []byte
	for _, id := range ids {
		if id < 0 || id >= len(t.pieces) {
			continue
		}
		piece := t.pieces[id]
		if value, ok := unigramByte(piece); ok {
			decoded = append(decoded, value)
			continue
		}
		decoded = append(decoded, strings.ReplaceAll(piece, sentencePieceSpaceMarker, " ")...)
	}
	return strings.TrimPrefix(string(decoded), " ")
}

func unigramByte(piece string) (byte, bool) {
	if len(piece) != 6 || piece[0] != '<' || piece[1] != '0' || piece[2] != 'x' || piece[5] != '>' {
		return 0, false
	}
	hex := func(value byte) (byte, bool) {
		switch {
		case value >= '0' && value <= '9':
			return value - '0', true
		case value >= 'A' && value <= 'F':
			return value - 'A' + 10, true
		case value >= 'a' && value <= 'f':
			return value - 'a' + 10, true
		default:
			return 0, false
		}
	}
	high, highOK := hex(piece[3])
	low, lowOK := hex(piece[4])
	return high<<4 | low, highOK && lowOK
}

// PieceID resolves one literal piece to its table id.
func (t *Unigram) PieceID(piece string) (int, bool) {
	id, ok := t.ids[piece]
	return id, ok
}

// Encode segments text by max-score Viterbi with the dummy prefix and space
// marker. Byte fallback is admitted ONLY for runes absent from the vocab:
// byte pieces carry table score 0.0, so letting them compete for covered
// text would beat every real piece and degrade to per-byte output. UNK is a
// last resort when a byte piece is missing (never fires on a full table).
func (t *Unigram) Encode(text string) ([]int, error) {
	norm := sentencePieceSpaceMarker + strings.ReplaceAll(text, " ", sentencePieceSpaceMarker)
	runes := []rune(norm)
	n := len(runes)
	negInf := math.Inf(-1)
	best := make([]float64, n+1)
	back := make([]int, n+1)       // start index of the piece ending at i
	pieceIDs := make([][]int, n+1) // ids emitted for that piece (>1 = byte fallback)
	for i := 1; i <= n; i++ {
		best[i] = negInf
		back[i] = -1
	}
	for i := 0; i < n; i++ {
		if best[i] == negInf && i > 0 {
			continue
		}
		limit := min(i+t.maxPiece, n)
		for j := i + 1; j <= limit; j++ {
			piece := string(runes[i:j])
			sc, ok := t.scores[piece]
			if !ok {
				continue
			}
			if cand := best[i] + sc; cand > best[j] {
				best[j] = cand
				back[j] = i
				pieceIDs[j] = []int{t.ids[piece]}
			}
		}
		j := i + 1
		r := string(runes[i])
		if _, covered := t.scores[r]; covered {
			continue
		}
		bs := []byte(r)
		idsSeq := make([]int, len(bs))
		score := 0.0
		complete := true
		for bi := range bs {
			bytePiece := fmt.Sprintf("<0x%02X>", bs[bi])
			id, has := t.ids[bytePiece]
			if !has {
				complete = false
				break
			}
			idsSeq[bi] = id
			score += t.scores[bytePiece]
		}
		if complete {
			if cand := best[i] + score; cand > best[j] {
				best[j] = cand
				back[j] = i
				pieceIDs[j] = idsSeq
			}
			continue
		}
		if best[j] == negInf {
			best[j] = best[i]
			back[j] = i
			pieceIDs[j] = []int{t.unkID}
		}
	}
	if best[n] == negInf {
		return nil, fmt.Errorf("tokenizer: unsegmentable text %q", text)
	}
	var revChunks [][]int
	for j := n; j > 0; j = back[j] {
		revChunks = append(revChunks, pieceIDs[j])
	}
	var out []int
	for i := len(revChunks) - 1; i >= 0; i-- {
		out = append(out, revChunks[i]...)
	}
	return out, nil
}

type hfUnigramJSONPiece struct {
	Piece string
	Score float64
}

// HF unigram vocab entries are two-element ["piece", score] arrays.
func (p *hfUnigramJSONPiece) UnmarshalJSON(raw []byte) error {
	var fields []json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return err
	}
	if len(fields) != 2 {
		return fmt.Errorf("tokenizer: unigram piece has %d fields, want 2", len(fields))
	}
	if err := json.Unmarshal(fields[0], &p.Piece); err != nil {
		return err
	}
	return json.Unmarshal(fields[1], &p.Score)
}

// LoadHFUnigramJSON reads an HF-format tokenizer.json whose model is
// Unigram. Returns the table, its artifact unk_id, and whether the artifact
// enables byte fallback (policy owners decide whether that is admissible).
func LoadHFUnigramJSON(path string) (*Unigram, int, bool, error) {
	var artifact struct {
		Model struct {
			Type         string               `json:"type"`
			Vocab        []hfUnigramJSONPiece `json:"vocab"`
			UnkID        int                  `json:"unk_id"`
			ByteFallback bool                 `json:"byte_fallback"`
		} `json:"model"`
	}
	if err := jsonfile.Decode(path, &artifact); err != nil {
		return nil, 0, false, fmt.Errorf("tokenizer: parse unigram tokenizer: %w", err)
	}
	if artifact.Model.Type != "Unigram" {
		return nil, 0, false, fmt.Errorf("tokenizer: model type %q is not Unigram", artifact.Model.Type)
	}
	pieces := make([]UnigramPiece, len(artifact.Model.Vocab))
	for i, piece := range artifact.Model.Vocab {
		pieces[i] = UnigramPiece{Piece: piece.Piece, Score: piece.Score}
	}
	table, err := NewUnigram(pieces, artifact.Model.UnkID)
	if err != nil {
		return nil, 0, false, err
	}
	return table, artifact.Model.UnkID, artifact.Model.ByteFallback, nil
}
