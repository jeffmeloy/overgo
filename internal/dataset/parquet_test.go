package dataset

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// TestSnappyDecode pins the block format against hand-encoded bytes:
// literals, near and far copies, and the declared-length check.
func TestSnappyDecode(t *testing.T) {
	// "abcdabcdabcdX": literal "abcd" then a copy1 of 8 bytes at
	// distance 4, then a literal "X".
	block := []byte{13}
	block = append(block, byte(3<<2), 'a', 'b', 'c', 'd')
	block = append(block, byte((8-4)<<2|1), 4)
	block = append(block, byte(0<<2), 'X')
	decoded, err := snappyDecode(block, nil)
	if err != nil || string(decoded) != "abcdabcdabcdX" {
		t.Fatalf("decoded = %q, %v", decoded, err)
	}
	if _, err := snappyDecode([]byte{5, byte(0 << 2), 'a'}, nil); err == nil {
		t.Fatal("length mismatch decoded")
	}
	if _, err := snappyDecode([]byte{2, byte((8-4)<<2 | 1), 9}, nil); err == nil {
		t.Fatal("invalid copy distance decoded")
	}
}

// TestRLEHybridDecode pins both run kinds: an RLE run repeating one
// value and a bit-packed group of eight 3-bit values.
func TestRLEHybridDecode(t *testing.T) {
	// RLE run: header 5<<1 (even), value 6 at bit width 3 → one byte.
	rle := []byte{5 << 1, 6}
	values, err := decodeRLEHybrid(rle, 3, 5, nil)
	if err != nil || len(values) != 5 || values[0] != 6 || values[4] != 6 {
		t.Fatalf("rle = %v, %v", values, err)
	}
	// Bit-packed run: header 1<<1|1, 8 values of width 3 in 3 bytes:
	// values 0..7 → 0b10001000 0b11000110 0b11111010.
	packed := []byte{1<<1 | 1, 0x88, 0xC6, 0xFA}
	values, err = decodeRLEHybrid(packed, 3, 8, nil)
	if err != nil || len(values) != 8 {
		t.Fatalf("packed = %v, %v", values, err)
	}
	for index, value := range values {
		if value != uint64(index) {
			t.Fatalf("packed[%d] = %d", index, value)
		}
	}
	// Width zero decodes to zeros without consuming bytes.
	if values, err := decodeRLEHybrid(nil, 0, 3, nil); err != nil || len(values) != 3 || values[0] != 0 {
		t.Fatalf("zero width = %v, %v", values, err)
	}
}

// TestThriftCompactCursor pins the field walk: delta and long-form
// field ids, zigzag integers, binary, nested structs, and skipping.
func TestThriftCompactCursor(t *testing.T) {
	var buffer bytes.Buffer
	// field 1 (delta 1), type i32, zigzag(300)
	buffer.WriteByte(1<<4 | thriftTypeI32)
	var varint [binary.MaxVarintLen64]byte
	buffer.Write(varint[:binary.PutUvarint(varint[:], uint64(300<<1))])
	// field 4 (delta 3), type binary, "seq"
	buffer.WriteByte(3<<4 | thriftTypeBinary)
	buffer.WriteByte(3)
	buffer.WriteString("seq")
	// field 6 (delta 2), nested struct with one bool-true field 1
	buffer.WriteByte(2<<4 | thriftTypeStruct)
	buffer.WriteByte(1<<4 | thriftTypeTrue)
	buffer.WriteByte(thriftTypeStop)
	buffer.WriteByte(thriftTypeStop)

	cursor := &thriftCursor{data: buffer.Bytes()}
	var number int64
	var name string
	skipped := false
	err := cursor.walkStruct(func(fieldID int16, fieldType byte) error {
		switch fieldID {
		case 1:
			value, err := cursor.readI64(fieldType)
			number = value
			return err
		case 4:
			value, err := cursor.readBinary(fieldType)
			name = string(value)
			return err
		default:
			skipped = true
			return cursor.skip(fieldType)
		}
	})
	if err != nil || number != 300 || name != "seq" || !skipped {
		t.Fatalf("walk = number=%d name=%q skipped=%t err=%v", number, name, skipped, err)
	}
}

// TestPlainStringsAndColumnSelection pins the value decoding and the
// named-column preference over synthetic schema metadata.
func TestPlainStringsAndColumnSelection(t *testing.T) {
	var data []byte
	for _, value := range []string{"ACGT", "TTAA"} {
		var length [4]byte
		binary.LittleEndian.PutUint32(length[:], uint32(len(value)))
		data = append(data, length[:]...)
		data = append(data, value...)
	}
	values, err := decodePlainStrings(data, 2, nil)
	if err != nil || len(values) != 2 || values[1] != "TTAA" {
		t.Fatalf("plain = %v, %v", values, err)
	}
	if _, err := decodePlainStrings(data[:5], 2, nil); err == nil {
		t.Fatal("truncated plain decoded")
	}

	metadata := parquetMetadata{schema: []parquetSchemaElement{
		{name: "schema", numChildren: 3},
		{name: "id", kind: parquetTypeByteArray, hasKind: true, repetition: 1},
		{name: "sequence", kind: parquetTypeByteArray, hasKind: true, repetition: 1},
		{name: "count", kind: 2, hasKind: true, repetition: 1},
	}}
	leaf, definition, err := stringColumnLeaf(metadata, "")
	if err != nil || leaf != 1 || definition != 1 {
		t.Fatalf("auto column = (%d, %d, %v), want the sequence leaf", leaf, definition, err)
	}
	if leaf, _, err := stringColumnLeaf(metadata, "id"); err != nil || leaf != 0 {
		t.Fatalf("named column = (%d, %v)", leaf, err)
	}
	if _, _, err := stringColumnLeaf(metadata, "absent"); err == nil {
		t.Fatal("absent column resolved")
	}
	metadata.schema[2].name = "id"
	if _, _, err := stringColumnLeaf(metadata, "id"); err == nil {
		t.Fatal("ambiguous byte-array leaf resolved")
	}
}
