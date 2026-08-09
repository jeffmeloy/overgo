// SentencePiece ModelProto reader for vendor tokenizer.model artifacts:
// only the piece table (text, score, type) is decoded. Promoted from
// speechsynth alongside the Unigram table it feeds.
package tokenizer

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

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

func parsePieceEntry(body []byte) (UnigramPiece, int, error) {
	var out UnigramPiece
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

// ReadSentencePieceModel parses a vendor tokenizer.model and derives the
// unknown piece id from the entry typed UNKNOWN rather than assuming an index.
func ReadSentencePieceModel(path string) ([]UnigramPiece, int, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	var pieces []UnigramPiece
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
