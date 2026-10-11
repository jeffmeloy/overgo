package dataset

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// compactInt is one 32-bit field of a compact-protocol struct: its header,
// the distance from the field before it beside the type, then the value.
func compactInt(distance byte, value int32) []byte {
	const compactI32 = 5
	return binary.AppendUvarint([]byte{distance<<4 | compactI32}, uint64(uint32(value<<1^value>>31)))
}

// pageHeader encodes a page header as a column chunk holds it: the page's
// kind and sizes, then the dictionary or data page's own header at the field
// number that kind has, with its value count and encoding.
func pageHeader(kind int64, size int, values int64) []byte {
	const compactStruct, dataField, dictionaryField, sizesField = 12, 5, 7, 3
	field := byte(dataField)
	if kind == parquetPageDictionary {
		field = dictionaryField
	}
	header := compactInt(1, int32(kind))
	header = append(header, compactInt(1, int32(size))...)
	header = append(header, compactInt(1, int32(size))...)
	header = append(header, (field-sizesField)<<4|compactStruct)
	header = append(header, compactInt(1, int32(values))...)
	header = append(header, compactInt(1, parquetEncodingPlain)...)
	return append(header, 0, 0)
}

// TestPlainIntegerPageAfterADictionaryReadsItsDecimalText holds a 64-bit
// integer column whose data page is written plain, after a dictionary page,
// to the decimal text of its values. Such a page passed the one guard for
// integer columns, which asks only that a dictionary came first, and was then
// read as length-prefixed strings: the value 4 came back as four NUL bytes,
// and other values as a string of a length the page does not hold. Held for
// a required column and for an optional one with a missing value.
func TestPlainIntegerPageAfterADictionaryReadsItsDecimalText(t *testing.T) {
	t.Parallel()
	four := binary.LittleEndian.AppendUint64(nil, 4)
	// One definition level a value, a bit each: present, then missing.
	levels := []byte{2, 0, 0, 0, 3, 1}
	for name, column := range map[string]struct {
		definition int
		data       []byte
		want       []parquetValue
	}{
		"required": {0, four, []parquetValue{{text: "4", valid: true}}},
		"optional": {1, append(levels, four...), []parquetValue{{text: "4", valid: true}, {}}},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			dictionary := binary.LittleEndian.AppendUint64(nil, 7)
			chunk := append(pageHeader(parquetPageDictionary, len(dictionary), 1), dictionary...)
			chunk = append(chunk, pageHeader(parquetPageData, len(column.data), int64(len(column.want)))...)
			chunk = append(chunk, column.data...)
			path := filepath.Join(t.TempDir(), "column.parquet")
			if err := os.WriteFile(path, append([]byte(parquetMagic), chunk...), 0o600); err != nil {
				t.Fatal(err)
			}
			file, err := os.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer file.Close()
			start := int64(len(parquetMagic))
			var read []parquetValue
			err = readColumnChunk(t.Context(), file, parquetColumn{
				kind: parquetTypeInt64, codec: parquetCodecUncompressed, numValues: int64(len(column.want)), compressedSize: int64(len(chunk)),
				dictionaryOffset: start, dataPageOffset: start + int64(len(chunk)-len(column.data)), hasDictionary: true,
			}, column.definition, &parquetBudget{remaining: int64(len(chunk)) << 8}, func(_ uint64, value parquetValue) error {
				read = append(read, value)
				return nil
			})
			if err != nil || !reflect.DeepEqual(read, column.want) {
				t.Fatalf("a plain integer page after a dictionary read as %#v (%v), want %#v", read, err, column.want)
			}
		})
	}
}
