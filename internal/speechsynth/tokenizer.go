// SentencePiece unigram tokenizer for the text conditioner: a ModelProto
// reader for the vendor tokenizer.model plus max-score Viterbi segmentation
// with the dummy-prefix/space-marker normalization SentencePiece applies.
// Ported behavior (adaptive sentencepiece owner), written fresh.
package speechsynth

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strings"
)

// Piece is one unigram vocabulary entry.
type Piece struct {
	Piece string
	Score float64
}

// spaceMarker: SentencePiece's U+2581 word-boundary marker.
const spaceMarker = "▁"

// Unigram owns Viterbi policy over a parsed piece table.
type Unigram struct {
	scores   map[string]float64
	ids      map[string]int
	maxPiece int // longest piece length in runes
	unkID    int
}

// NewUnigram builds the table; unkID must index a real piece.
func NewUnigram(pieces []Piece, unkID int) (*Unigram, error) {
	if len(pieces) == 0 {
		return nil, fmt.Errorf("speechsynth: empty piece table")
	}
	if unkID < 0 || unkID >= len(pieces) {
		return nil, fmt.Errorf("speechsynth: unk id %d outside %d pieces", unkID, len(pieces))
	}
	t := &Unigram{
		scores: make(map[string]float64, len(pieces)),
		ids:    make(map[string]int, len(pieces)),
		unkID:  unkID,
	}
	for i, p := range pieces {
		t.scores[p.Piece] = p.Score
		t.ids[p.Piece] = i
		if n := len([]rune(p.Piece)); n > t.maxPiece {
			t.maxPiece = n
		}
	}
	return t, nil
}

// Encode segments text by max-score Viterbi with the dummy prefix and space
// marker. Byte fallback is admitted ONLY for runes absent from the vocab:
// byte pieces carry table score 0.0, so letting them compete for covered
// text would beat every real piece and degrade to per-byte output. UNK is a
// last resort when a byte piece is missing (never fires on a full table).
func (t *Unigram) Encode(text string) ([]int, error) {
	norm := spaceMarker + strings.ReplaceAll(text, " ", spaceMarker)
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
		return nil, fmt.Errorf("speechsynth: unsegmentable text %q", text)
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

// SentencePiece ModelProto wire constants (only what the reader needs).
const (
	protoWireVarint = 0
	protoWire64Bit  = 1
	protoWireBytes  = 2
	protoWire32Bit  = 5
	fieldPieces     = 1
	pieceFieldText  = 1
	pieceFieldScore = 2
	pieceFieldType  = 3
	pieceTypeUnk    = 2
)

func protoVarint(buf []byte, at int) (uint64, int, error) {
	var value uint64
	for shift := uint(0); ; shift += 7 {
		if at >= len(buf) {
			return 0, 0, fmt.Errorf("varint truncated at %d", at)
		}
		if shift > 63 {
			return 0, 0, fmt.Errorf("varint overflows 64 bits at %d", at)
		}
		octet := buf[at]
		at++
		value |= uint64(octet&0x7f) << shift
		if octet < 0x80 {
			return value, at, nil
		}
	}
}

func protoBytesSpan(buf []byte, at int) (start, end int, err error) {
	size, start, err := protoVarint(buf, at)
	if err != nil {
		return 0, 0, err
	}
	if size > uint64(len(buf)-start) {
		return 0, 0, fmt.Errorf("length-delimited field truncated at %d", at)
	}
	return start, start + int(size), nil
}

func protoSkip(buf []byte, at, wire int) (int, error) {
	switch wire {
	case protoWireVarint:
		_, next, err := protoVarint(buf, at)
		return next, err
	case protoWire64Bit:
		if at+8 > len(buf) {
			return 0, fmt.Errorf("64-bit field truncated at %d", at)
		}
		return at + 8, nil
	case protoWireBytes:
		_, end, err := protoBytesSpan(buf, at)
		return end, err
	case protoWire32Bit:
		if at+4 > len(buf) {
			return 0, fmt.Errorf("32-bit field truncated at %d", at)
		}
		return at + 4, nil
	default:
		return 0, fmt.Errorf("unsupported protobuf wire type %d at %d", wire, at)
	}
}

func parsePieceEntry(body []byte) (Piece, int, error) {
	var out Piece
	kind := 0
	for at := 0; at < len(body); {
		key, next, err := protoVarint(body, at)
		if err != nil {
			return out, 0, err
		}
		at = next
		field, wire := int(key>>3), int(key&7)
		switch {
		case field == pieceFieldText && wire == protoWireBytes:
			start, end, err := protoBytesSpan(body, at)
			if err != nil {
				return out, 0, fmt.Errorf("piece string: %w", err)
			}
			out.Piece, at = string(body[start:end]), end
		case field == pieceFieldScore && wire == protoWire32Bit:
			if at+4 > len(body) {
				return out, 0, fmt.Errorf("piece score truncated at %d", at)
			}
			out.Score = float64(math.Float32frombits(binary.LittleEndian.Uint32(body[at : at+4])))
			at += 4
		case field == pieceFieldType && wire == protoWireVarint:
			value, next, err := protoVarint(body, at)
			if err != nil {
				return out, 0, err
			}
			kind, at = int(value), next
		default:
			at, err = protoSkip(body, at, wire)
			if err != nil {
				return out, 0, err
			}
		}
	}
	if out.Piece == "" {
		return out, 0, fmt.Errorf("sentencepiece entry carries no piece string")
	}
	return out, kind, nil
}

// ReadModelFile parses a vendor tokenizer.model and derives the unknown
// piece id from the entry typed UNKNOWN rather than assuming an index.
func ReadModelFile(path string) ([]Piece, int, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	var pieces []Piece
	unkID := -1
	for at := 0; at < len(raw); {
		key, next, err := protoVarint(raw, at)
		if err != nil {
			return nil, 0, fmt.Errorf("%s: %w", path, err)
		}
		at = next
		field, wire := int(key>>3), int(key&7)
		if field != fieldPieces || wire != protoWireBytes {
			if at, err = protoSkip(raw, at, wire); err != nil {
				return nil, 0, fmt.Errorf("%s: %w", path, err)
			}
			continue
		}
		start, end, err := protoBytesSpan(raw, at)
		if err != nil {
			return nil, 0, fmt.Errorf("%s: piece entry: %w", path, err)
		}
		piece, kind, err := parsePieceEntry(raw[start:end])
		if err != nil {
			return nil, 0, fmt.Errorf("%s piece %d: %w", path, len(pieces), err)
		}
		if kind == pieceTypeUnk && unkID < 0 {
			unkID = len(pieces)
		}
		pieces, at = append(pieces, piece), end
	}
	if unkID < 0 {
		return nil, 0, fmt.Errorf("%s: no piece typed UNKNOWN, so the unk id cannot be derived", path)
	}
	return pieces, unkID, nil
}

// LoadTokenizer reads the artifact's tokenizer.model into a Unigram.
func LoadTokenizer(path string) (*Unigram, error) {
	pieces, unkID, err := ReadModelFile(path)
	if err != nil {
		return nil, err
	}
	return NewUnigram(pieces, unkID)
}
