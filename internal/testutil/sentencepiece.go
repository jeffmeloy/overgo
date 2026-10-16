package testutil

import (
	"encoding/binary"
	"os"
	"testing"
)

// WriteSentencePiece writes a SentencePiece ModelProto holding only the
// piece table: every piece normal, unknown typed UNKNOWN.
func WriteSentencePiece(t testing.TB, path string, pieces []string, unknown int) {
	t.Helper()
	var model []byte
	for index, piece := range pieces {
		entry := binary.AppendUvarint([]byte{pieceText}, uint64(len(piece)))
		entry = append(entry, piece...)
		kind := uint64(normal)
		if index == unknown {
			kind = unknownType
		}
		entry = binary.AppendUvarint(append(entry, pieceType), kind)
		model = binary.AppendUvarint(append(model, pieceEntry), uint64(len(entry)))
		model = append(model, entry...)
	}
	if err := os.WriteFile(path, model, sentencePieceMode); err != nil {
		t.Fatal(err)
	}
}

const (
	// pieceEntry encodes ModelProto.pieces field 1 with length-delimited protobuf wire type.
	pieceEntry = 0x0a
	// pieceText encodes SentencePiece.piece field 1 with length-delimited protobuf wire type.
	pieceText = pieceEntry
	// pieceType encodes SentencePiece.type field 3 with varint protobuf wire type.
	pieceType = 0x18
	// normal and unknownType are SentencePiece ModelProto.Type enum values.
	normal, unknownType = 1, 2
	// sentencePieceMode keeps generated test tokenizers private to their temporary directory.
	sentencePieceMode = 0o600
)
