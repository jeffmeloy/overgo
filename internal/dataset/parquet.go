package dataset

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// Parquet reader for the corpus slices evaluation draws on -- the
// narrow subset HuggingFace-written files use: PAR1 framing, thrift
// compact footer metadata, snappy or uncompressed pages, and string
// (byte-array) columns in plain or dictionary encoding. Reading the
// column natively keeps corpus evaluation free of external tooling;
// every construct outside the subset refuses with its name.

const (
	parquetMagic = "PAR1"

	parquetTypeByteArray = 6

	parquetCodecUncompressed = 0
	parquetCodecSnappy       = 1

	parquetPageData       = 0
	parquetPageDictionary = 2
	parquetPageDataV2     = 3

	parquetEncodingPlain           = 0
	parquetEncodingPlainDictionary = 2
	parquetEncodingRLEDictionary   = 8
)

// ReadParquetTextRows streams one byte-array (string) leaf column of a
// parquet file, observing up to limit non-null values in row order.
// An empty column name prefers "sequence", then "text", then the first
// string leaf -- the conventions corpus files actually use.
func ReadParquetTextRows(ctx context.Context, path, column string, limit int, observe func(uint64, string) error) error {
	if ctx == nil || limit <= 0 || observe == nil {
		return errors.New("dataset: incomplete parquet read")
	}
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return err
	}
	metadata, err := readParquetFooter(file, info.Size())
	if err != nil {
		return err
	}
	leaf, maxDefinition, err := stringColumnLeaf(metadata, column)
	if err != nil {
		return err
	}
	emitted := uint64(0)
	for _, group := range metadata.rowGroups {
		if int(emitted) >= limit {
			break
		}
		if leaf >= len(group.columns) {
			return errors.New("dataset: parquet row group misses the string column")
		}
		stop := errors.New("parquet selection complete")
		err := readColumnChunk(ctx, file, group.columns[leaf], maxDefinition, nil, func(_ uint64, value parquetValue) error {
			if !value.valid {
				return nil
			}
			if err := observe(emitted, value.text); err != nil {
				return err
			}
			emitted++
			if int(emitted) >= limit {
				return stop
			}
			return nil
		})
		if err != nil && !errors.Is(err, stop) {
			return err
		}
	}
	if emitted == 0 {
		return errors.New("dataset: parquet column carried no values")
	}
	return nil
}

type parquetMetadata struct {
	schema    []parquetSchemaElement
	rowGroups []parquetRowGroup
}

type parquetSchemaElement struct {
	kind        int64
	repetition  int64
	name        string
	numChildren int64
	hasKind     bool
}

type parquetRowGroup struct {
	columns []parquetColumn
	rows    int64
}

type parquetColumn struct {
	kind             int64
	codec            int64
	numValues        int64
	compressedSize   int64
	dataPageOffset   int64
	dictionaryOffset int64
	hasDictionary    bool
}

func readParquetFooter(file *os.File, size int64) (parquetMetadata, error) {
	var tail [8]byte
	if _, err := file.ReadAt(tail[:], size-8); err != nil {
		return parquetMetadata{}, err
	}
	if string(tail[4:]) != parquetMagic {
		return parquetMetadata{}, errors.New("dataset: not a parquet file")
	}
	length := int64(binary.LittleEndian.Uint32(tail[:4]))
	if length <= 0 || length > size-8 {
		return parquetMetadata{}, errors.New("dataset: parquet footer length is invalid")
	}
	footer := make([]byte, length)
	if _, err := file.ReadAt(footer, size-8-length); err != nil {
		return parquetMetadata{}, err
	}
	return parseFileMetadata(&thriftCursor{data: footer})
}

// parseFileMetadata walks FileMetaData: schema is field 2, row groups
// field 4; everything else skips by type.
func parseFileMetadata(cursor *thriftCursor) (parquetMetadata, error) {
	metadata := parquetMetadata{}
	err := cursor.walkStruct(func(fieldID int16, fieldType byte) error {
		switch fieldID {
		case 2:
			return cursor.walkList(fieldType, func(elementType byte) error {
				element, err := parseSchemaElement(cursor)
				if err != nil {
					return err
				}
				metadata.schema = append(metadata.schema, element)
				return nil
			})
		case 4:
			return cursor.walkList(fieldType, func(elementType byte) error {
				group, err := parseRowGroup(cursor)
				if err != nil {
					return err
				}
				metadata.rowGroups = append(metadata.rowGroups, group)
				return nil
			})
		default:
			return cursor.skip(fieldType)
		}
	})
	if err != nil {
		return parquetMetadata{}, err
	}
	if len(metadata.schema) == 0 || len(metadata.rowGroups) == 0 {
		return parquetMetadata{}, errors.New("dataset: parquet metadata is incomplete")
	}
	return metadata, nil
}

func parseSchemaElement(cursor *thriftCursor) (parquetSchemaElement, error) {
	element := parquetSchemaElement{}
	err := cursor.walkStruct(func(fieldID int16, fieldType byte) error {
		switch fieldID {
		case 1:
			value, err := cursor.readI64(fieldType)
			element.kind, element.hasKind = value, true
			return err
		case 3:
			value, err := cursor.readI64(fieldType)
			element.repetition = value
			return err
		case 4:
			value, err := cursor.readBinary(fieldType)
			element.name = string(value)
			return err
		case 5:
			value, err := cursor.readI64(fieldType)
			element.numChildren = value
			return err
		default:
			return cursor.skip(fieldType)
		}
	})
	return element, err
}

func parseRowGroup(cursor *thriftCursor) (parquetRowGroup, error) {
	group := parquetRowGroup{}
	err := cursor.walkStruct(func(fieldID int16, fieldType byte) error {
		if fieldID == 3 {
			value, err := cursor.readI64(fieldType)
			group.rows = value
			return err
		}
		if fieldID != 1 {
			return cursor.skip(fieldType)
		}
		return cursor.walkList(fieldType, func(elementType byte) error {
			column, err := parseColumnChunk(cursor)
			if err != nil {
				return err
			}
			group.columns = append(group.columns, column)
			return nil
		})
	})
	return group, err
}

func parseColumnChunk(cursor *thriftCursor) (parquetColumn, error) {
	column := parquetColumn{}
	err := cursor.walkStruct(func(fieldID int16, fieldType byte) error {
		if fieldID != 3 {
			return cursor.skip(fieldType)
		}
		return cursor.walkStruct(func(metaID int16, metaType byte) error {
			switch metaID {
			case 1:
				value, err := cursor.readI64(metaType)
				column.kind = value
				return err
			case 4:
				value, err := cursor.readI64(metaType)
				column.codec = value
				return err
			case 5:
				value, err := cursor.readI64(metaType)
				column.numValues = value
				return err
			case 7:
				value, err := cursor.readI64(metaType)
				column.compressedSize = value
				return err
			case 9:
				value, err := cursor.readI64(metaType)
				column.dataPageOffset = value
				return err
			case 11:
				value, err := cursor.readI64(metaType)
				column.dictionaryOffset, column.hasDictionary = value, true
				return err
			default:
				return cursor.skip(metaType)
			}
		})
	})
	return column, err
}

// stringColumnLeaf finds the leaf index and maximum definition level
// (leaf plus optional ancestors) of the named byte-array column; an
// empty name prefers "sequence", then "text", then the first string
// leaf. Repeated fields are outside the subset.
func stringColumnLeaf(metadata parquetMetadata, column string) (int, int, error) {
	type candidate struct {
		leaf     int
		optional int
		name     string
		path     string
	}
	type frame struct {
		remaining int64
		optional  int
		path      string
	}
	var candidates []candidate
	leaf := 0
	stack := []frame{{remaining: metadata.schema[0].numChildren}}
	for _, element := range metadata.schema[1:] {
		if len(stack) == 0 {
			break
		}
		optional := stack[len(stack)-1].optional
		path := element.name
		if parent := stack[len(stack)-1].path; parent != "" {
			path = parent + "." + path
		}
		const repetitionOptional, repetitionRepeated = 1, 2
		if element.repetition == repetitionRepeated {
			return 0, 0, errors.New("dataset: repeated parquet fields are outside the supported subset")
		}
		if element.repetition == repetitionOptional {
			optional++
		}
		stack[len(stack)-1].remaining--
		if element.numChildren > 0 {
			stack = append(stack, frame{remaining: element.numChildren, optional: optional, path: path})
		} else {
			if element.hasKind && element.kind == parquetTypeByteArray {
				candidates = append(candidates, candidate{leaf: leaf, optional: optional, name: element.name, path: path})
			}
			leaf++
			for len(stack) > 0 && stack[len(stack)-1].remaining == 0 {
				stack = stack[:len(stack)-1]
			}
		}
	}
	if len(candidates) == 0 {
		return 0, 0, errors.New("dataset: parquet file carries no string column")
	}
	wanted := []string{column}
	if column == "" {
		wanted = []string{"sequence", "text"}
	}
	for _, name := range wanted {
		var matches int
		var selectedLeaf, selectedOptional int
		for _, found := range candidates {
			if found.path == name || !strings.Contains(name, ".") && found.name == name {
				matches++
				selectedLeaf, selectedOptional = found.leaf, found.optional
			}
		}
		if matches > 1 {
			return 0, 0, fmt.Errorf("dataset: parquet column %q is ambiguous", name)
		}
		if matches == 1 {
			return selectedLeaf, selectedOptional, nil
		}
	}
	if column != "" {
		return 0, 0, fmt.Errorf("dataset: parquet column %q is absent", column)
	}
	return candidates[0].leaf, candidates[0].optional, nil
}

func readColumnChunk(
	ctx context.Context,
	file *os.File,
	column parquetColumn,
	maxDefinition int,
	budget *parquetBudget,
	observe func(uint64, parquetValue) error,
) error {
	if column.kind != parquetTypeByteArray {
		return errors.New("dataset: parquet string column has a different physical type")
	}
	if column.codec != parquetCodecSnappy && column.codec != parquetCodecUncompressed {
		return fmt.Errorf("dataset: parquet codec %d is outside the supported subset", column.codec)
	}
	start := column.dataPageOffset
	if column.hasDictionary && column.dictionaryOffset > 0 && column.dictionaryOffset < start {
		start = column.dictionaryOffset
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	info, err := file.Stat()
	if err != nil {
		return err
	}
	if start < int64(len(parquetMagic)) || column.compressedSize <= 0 || start > info.Size() || column.compressedSize > info.Size()-start || column.numValues < 0 {
		return errors.New("dataset: invalid parquet column bounds")
	}
	if err := budget.take(column.compressedSize, 1); err != nil {
		return err
	}
	chunk := make([]byte, column.compressedSize)
	if _, err := file.ReadAt(chunk, start); err != nil {
		return err
	}
	var dictionary []string
	values := int64(0)
	for offset := int64(0); offset < int64(len(chunk)); {
		if err := ctx.Err(); err != nil {
			return err
		}
		cursor := &thriftCursor{data: chunk[offset:]}
		header, err := parsePageHeader(cursor)
		if err != nil {
			return err
		}
		payloadStart := offset + int64(cursor.position)
		if header.compressedSize < 0 || header.compressedSize > int64(len(chunk))-payloadStart || header.uncompressedSize < 0 || header.numValues < 0 {
			return errors.New("dataset: invalid parquet page bounds")
		}
		payload := chunk[payloadStart : payloadStart+int64(header.compressedSize)]
		offset = payloadStart + int64(header.compressedSize)
		switch header.kind {
		case parquetPageDictionary:
			if dictionary != nil || values != 0 || header.encoding != parquetEncodingPlain {
				return errors.New("dataset: invalid parquet dictionary page")
			}
			plain, err := decompressPage(payload, column.codec, header.uncompressedSize, budget)
			if err != nil {
				return err
			}
			dictionary, err = decodePlainStrings(plain, header.numValues, budget)
			if err != nil {
				return err
			}
		case parquetPageData, parquetPageDataV2:
			if header.numValues > column.numValues-values {
				return errors.New("dataset: parquet page exceeds declared row count")
			}
			pageValues, err := decodeDataPage(payload, column.codec, header, maxDefinition, dictionary, budget)
			if err != nil {
				return err
			}
			for index, value := range pageValues {
				if err := ctx.Err(); err != nil {
					return err
				}
				if err := observe(uint64(values)+uint64(index), value); err != nil {
					return err
				}
			}
			values += int64(header.numValues)
		default:
			return fmt.Errorf("dataset: parquet page type %d is outside the supported subset", header.kind)
		}
	}
	if values != column.numValues {
		return errors.New("dataset: parquet column row count differs from metadata")
	}
	return nil
}

type parquetPageHeader struct {
	kind             int64
	uncompressedSize int64
	compressedSize   int64
	numValues        int64
	encoding         int64
	defLevelBytes    int64
	repLevelBytes    int64
	v2               bool
	v2Compressed     bool
}

func parsePageHeader(cursor *thriftCursor) (parquetPageHeader, error) {
	header := parquetPageHeader{v2Compressed: true}
	err := cursor.walkStruct(func(fieldID int16, fieldType byte) error {
		switch fieldID {
		case 1:
			value, err := cursor.readI64(fieldType)
			header.kind = value
			return err
		case 2:
			value, err := cursor.readI64(fieldType)
			header.uncompressedSize = value
			return err
		case 3:
			value, err := cursor.readI64(fieldType)
			header.compressedSize = value
			return err
		case 5, 7:
			return cursor.walkStruct(func(pageID int16, pageType byte) error {
				switch pageID {
				case 1:
					value, err := cursor.readI64(pageType)
					header.numValues = value
					return err
				case 2:
					value, err := cursor.readI64(pageType)
					header.encoding = value
					return err
				default:
					return cursor.skip(pageType)
				}
			})
		case 8:
			header.v2 = true
			return cursor.walkStruct(func(pageID int16, pageType byte) error {
				switch pageID {
				case 1:
					value, err := cursor.readI64(pageType)
					header.numValues = value
					return err
				case 4:
					value, err := cursor.readI64(pageType)
					header.encoding = value
					return err
				case 5:
					value, err := cursor.readI64(pageType)
					header.defLevelBytes = value
					return err
				case 6:
					value, err := cursor.readI64(pageType)
					header.repLevelBytes = value
					return err
				case 7:
					header.v2Compressed = pageType == thriftTypeTrue
					return nil
				default:
					return cursor.skip(pageType)
				}
			})
		default:
			return cursor.skip(fieldType)
		}
	})
	return header, err
}

func decompressPage(payload []byte, codec, uncompressedSize int64, budget *parquetBudget) ([]byte, error) {
	if uncompressedSize < 0 {
		return nil, errors.New("dataset: negative parquet page size")
	}
	if codec == parquetCodecUncompressed {
		if int64(len(payload)) != uncompressedSize {
			return nil, errors.New("dataset: parquet uncompressed page size differs")
		}
		return payload, nil
	}
	length, consumed := binary.Uvarint(payload)
	if codec != parquetCodecSnappy || consumed <= 0 || length != uint64(uncompressedSize) {
		return nil, errors.New("dataset: parquet compressed page size or codec differs")
	}
	plain, err := snappyDecode(payload, budget)
	if err != nil {
		return nil, err
	}
	if int64(len(plain)) != uncompressedSize {
		return nil, errors.New("dataset: parquet page decompressed to an unexpected size")
	}
	return plain, nil
}

// decodeDataPage preserves physical rows, including absent optional ancestors.
func decodeDataPage(
	payload []byte,
	codec int64,
	header parquetPageHeader,
	maxDefinition int,
	dictionary []string,
	budget *parquetBudget,
) ([]parquetValue, error) {
	var levels, data []byte
	if header.v2 {
		split := header.repLevelBytes + header.defLevelBytes
		if header.repLevelBytes != 0 || header.defLevelBytes < 0 || split > int64(len(payload)) || split > header.uncompressedSize {
			return nil, errors.New("dataset: parquet v2 level bytes exceed the page")
		}
		levels = payload[header.repLevelBytes:split]
		data = payload[split:]
		if header.v2Compressed {
			plain, err := decompressPage(data, codec, header.uncompressedSize-split, budget)
			if err != nil {
				return nil, err
			}
			data = plain
		} else if int64(len(data)) != header.uncompressedSize-split {
			return nil, errors.New("dataset: parquet v2 uncompressed size differs")
		}
	} else {
		plain, err := decompressPage(payload, codec, header.uncompressedSize, budget)
		if err != nil {
			return nil, err
		}
		data = plain
		if maxDefinition > 0 {
			if len(data) < 4 {
				return nil, errors.New("dataset: parquet definition levels are truncated")
			}
			length := int64(binary.LittleEndian.Uint32(data))
			if length+4 > int64(len(data)) {
				return nil, errors.New("dataset: parquet definition levels exceed the page")
			}
			levels = data[4 : 4+length]
			data = data[4+length:]
		}
	}
	if err := budget.take(header.numValues, parquetValueBytes); err != nil {
		return nil, err
	}
	rows := make([]parquetValue, int(header.numValues))
	present := len(rows)
	if maxDefinition > 0 {
		definitions, err := decodeRLEHybrid(levels, bitWidthFor(maxDefinition), len(rows), budget)
		if err != nil {
			return nil, err
		}
		present = 0
		for index, level := range definitions {
			if level > uint64(maxDefinition) {
				return nil, errors.New("dataset: parquet definition level exceeds schema")
			}
			if level == uint64(maxDefinition) {
				rows[index].valid = true
				present++
			}
		}
	} else {
		for index := range rows {
			rows[index].valid = true
		}
	}
	values, err := decodePageStrings(data, header.encoding, present, dictionary, budget)
	if err != nil {
		return nil, err
	}
	next := 0
	for index := range rows {
		if rows[index].valid {
			rows[index].text = values[next]
			next++
		}
	}
	return rows, nil
}

func decodePageStrings(data []byte, encoding int64, present int, dictionary []string, budget *parquetBudget) ([]string, error) {
	switch encoding {
	case parquetEncodingPlain:
		return decodePlainStrings(data, int64(present), budget)
	case parquetEncodingPlainDictionary, parquetEncodingRLEDictionary:
		if dictionary == nil {
			return nil, errors.New("dataset: parquet dictionary page is absent")
		}
		if present == 0 {
			return nil, nil
		}
		if len(data) == 0 {
			return nil, errors.New("dataset: parquet dictionary indices are empty")
		}
		width := int(data[0])
		indices, err := decodeRLEHybrid(data[1:], width, present, budget)
		if err != nil {
			return nil, err
		}
		if err := budget.take(int64(present), parquetStringBytes); err != nil {
			return nil, err
		}
		values := make([]string, len(indices))
		for position, index := range indices {
			if index >= uint64(len(dictionary)) {
				return nil, errors.New("dataset: parquet dictionary index out of range")
			}
			values[position] = dictionary[index]
		}
		return values, nil
	default:
		return nil, fmt.Errorf("dataset: parquet encoding %d is outside the supported subset", encoding)
	}
}

func decodePlainStrings(data []byte, count int64, budget *parquetBudget) ([]string, error) {
	if count < 0 || count > int64(len(data)/binary.Size(uint32(0))) {
		return nil, errors.New("dataset: parquet string count exceeds page")
	}
	if err := budget.take(count, parquetStringBytes); err != nil {
		return nil, err
	}
	values := make([]string, 0, count)
	for offset := 0; int64(len(values)) < count; {
		if offset+4 > len(data) {
			return nil, errors.New("dataset: parquet plain string is truncated")
		}
		length := int(binary.LittleEndian.Uint32(data[offset:]))
		offset += 4
		if offset+length > len(data) {
			return nil, errors.New("dataset: parquet plain string exceeds the page")
		}
		if err := budget.take(int64(length), 1); err != nil {
			return nil, err
		}
		values = append(values, string(data[offset:offset+length]))
		offset += length
	}
	return values, nil
}

func bitWidthFor(maximum int) int {
	width := 0
	for value := maximum; value > 0; value >>= 1 {
		width++
	}
	return width
}

// decodeRLEHybrid decodes parquet's RLE/bit-packed hybrid runs.
func decodeRLEHybrid(data []byte, bitWidth, count int, budget *parquetBudget) ([]uint64, error) {
	if bitWidth < 0 || bitWidth > 64 {
		return nil, errors.New("dataset: parquet bit width exceeds uint64")
	}
	if err := budget.take(int64(count), int64(binary.Size(uint64(0)))); err != nil {
		return nil, err
	}
	if bitWidth == 0 {
		return make([]uint64, count), nil
	}
	values := make([]uint64, 0, count)
	byteWidth := (bitWidth + 7) / 8
	offset := 0
	for len(values) < count {
		runHeader, consumed := binary.Uvarint(data[offset:])
		if consumed <= 0 || runHeader>>1 == 0 || runHeader>>1 > uint64(^uint(0)>>1)/8 {
			return nil, errors.New("dataset: parquet run header is invalid")
		}
		offset += consumed
		if runHeader&1 == 0 {
			runLength := int(runHeader >> 1)
			if offset+byteWidth > len(data) {
				return nil, errors.New("dataset: parquet RLE run exceeds the buffer")
			}
			var value uint64
			for index := range byteWidth {
				value |= uint64(data[offset+index]) << (8 * index)
			}
			offset += byteWidth
			for index := 0; index < runLength && len(values) < count; index++ {
				values = append(values, value)
			}
		} else {
			groups := int(runHeader >> 1)
			if groups > (len(data)-offset)/bitWidth {
				return nil, errors.New("dataset: parquet bit-packed run exceeds the buffer")
			}
			needed := groups * bitWidth
			if offset+needed > len(data) {
				return nil, errors.New("dataset: parquet bit-packed run exceeds the buffer")
			}
			bitOffset := 0
			packed := data[offset : offset+needed]
			for index := 0; index < groups*8 && len(values) < count; index++ {
				var value uint64
				for bit := range bitWidth {
					byteIndex := (bitOffset + bit) / 8
					if packed[byteIndex]&(1<<((bitOffset+bit)%8)) != 0 {
						value |= 1 << bit
					}
				}
				bitOffset += bitWidth
				values = append(values, value)
			}
			offset += needed
		}
	}
	return values, nil
}

// snappyDecode decodes one snappy block (the framing parquet uses).
func snappyDecode(data []byte, budget *parquetBudget) ([]byte, error) {
	length, consumed := binary.Uvarint(data)
	if consumed <= 0 {
		return nil, errors.New("dataset: snappy length is invalid")
	}
	if length > uint64(^uint(0)>>1) {
		return nil, errors.New("dataset: snappy size exceeds addressable memory")
	}
	if err := budget.take(int64(length), 1); err != nil {
		return nil, err
	}
	output := make([]byte, 0, length)
	for offset := consumed; offset < len(data); {
		tag := data[offset]
		switch tag & 3 {
		case 0:
			size := int(tag >> 2)
			offset++
			if size >= 60 {
				extra := size - 59
				if offset+extra > len(data) {
					return nil, errors.New("dataset: snappy literal header is truncated")
				}
				size = 0
				for index := range extra {
					size |= int(data[offset+index]) << (8 * index)
				}
				offset += extra
			}
			size++
			if size < 0 || size > len(data)-offset || size > cap(output)-len(output) {
				return nil, errors.New("dataset: snappy literal exceeds the block")
			}
			output = append(output, data[offset:offset+size]...)
			offset += size
		case 1:
			if offset+2 > len(data) {
				return nil, errors.New("dataset: snappy copy1 is truncated")
			}
			size := int(tag>>2)&0x7 + 4
			distance := int(tag>>5)<<8 | int(data[offset+1])
			offset += 2
			if err := snappyCopy(&output, distance, size); err != nil {
				return nil, err
			}
		case 2:
			if offset+3 > len(data) {
				return nil, errors.New("dataset: snappy copy2 is truncated")
			}
			size := int(tag>>2) + 1
			distance := int(binary.LittleEndian.Uint16(data[offset+1:]))
			offset += 3
			if err := snappyCopy(&output, distance, size); err != nil {
				return nil, err
			}
		default:
			if offset+5 > len(data) {
				return nil, errors.New("dataset: snappy copy4 is truncated")
			}
			size := int(tag>>2) + 1
			distance := int(binary.LittleEndian.Uint32(data[offset+1:]))
			offset += 5
			if err := snappyCopy(&output, distance, size); err != nil {
				return nil, err
			}
		}
	}
	if uint64(len(output)) != length {
		return nil, errors.New("dataset: snappy block decoded to an unexpected size")
	}
	return output, nil
}

func snappyCopy(output *[]byte, distance, size int) error {
	if distance <= 0 || distance > len(*output) || size > cap(*output)-len(*output) {
		return errors.New("dataset: snappy copy distance is invalid")
	}
	for range size {
		*output = append(*output, (*output)[len(*output)-distance])
	}
	return nil
}

// The workspace budget counts requested backing storage, not process RSS or
// allocator overhead. It is cumulative for a row group, so even scratch that
// becomes unreachable remains charged until the next group.
type parquetBudget struct{ remaining int64 }

const (
	parquetWordBytes   = strconv.IntSize / 8
	parquetStringBytes = 2 * parquetWordBytes
	parquetValueBytes  = parquetStringBytes + parquetWordBytes // bool plus alignment
)

type parquetValue struct {
	text  string
	valid bool
}

func (budget *parquetBudget) take(count, width int64) error {
	if count < 0 || width <= 0 || count > int64(^uint(0)>>1)/width {
		return errors.New("dataset: parquet allocation exceeds addressable memory")
	}
	bytes := count * width
	if budget != nil {
		if bytes > budget.remaining {
			return fmt.Errorf("dataset: parquet workspace allocation %d exceeds remaining budget %d", bytes, budget.remaining)
		}
		budget.remaining -= bytes
	}
	return nil
}

// ---- minimal thrift compact protocol ----

const (
	thriftTypeStop   = 0
	thriftTypeTrue   = 1
	thriftTypeFalse  = 2
	thriftTypeByte   = 3
	thriftTypeI16    = 4
	thriftTypeI32    = 5
	thriftTypeI64    = 6
	thriftTypeDouble = 7
	thriftTypeBinary = 8
	thriftTypeList   = 9
	thriftTypeSet    = 10
	thriftTypeMap    = 11
	thriftTypeStruct = 12
)

type thriftCursor struct {
	data     []byte
	position int
}

func (c *thriftCursor) readByte() (byte, error) {
	if c.position >= len(c.data) {
		return 0, io.ErrUnexpectedEOF
	}
	value := c.data[c.position]
	c.position++
	return value, nil
}

func (c *thriftCursor) readUvarint() (uint64, error) {
	value, consumed := binary.Uvarint(c.data[c.position:])
	if consumed <= 0 {
		return 0, errors.New("dataset: thrift varint is invalid")
	}
	c.position += consumed
	return value, nil
}

func (c *thriftCursor) readZigzag() (int64, error) {
	value, err := c.readUvarint()
	if err != nil {
		return 0, err
	}
	return int64(value>>1) ^ -int64(value&1), nil
}

// walkStruct visits every field until the stop byte; the callback must
// consume the field's value (or skip it).
func (c *thriftCursor) walkStruct(visit func(fieldID int16, fieldType byte) error) error {
	lastField := int16(0)
	for {
		header, err := c.readByte()
		if err != nil {
			return err
		}
		if header == thriftTypeStop {
			return nil
		}
		fieldType := header & 0x0F
		delta := int16(header >> 4)
		if delta != 0 {
			lastField += delta
		} else {
			value, err := c.readZigzag()
			if err != nil {
				return err
			}
			lastField = int16(value)
		}
		if err := visit(lastField, fieldType); err != nil {
			return err
		}
	}
}

// walkList visits every element of a list value.
func (c *thriftCursor) walkList(fieldType byte, visit func(elementType byte) error) error {
	if fieldType != thriftTypeList && fieldType != thriftTypeSet {
		return errors.New("dataset: thrift field is not a list")
	}
	header, err := c.readByte()
	if err != nil {
		return err
	}
	size := uint64(header >> 4)
	elementType := header & 0x0F
	if size == 0x0F {
		if size, err = c.readUvarint(); err != nil {
			return err
		}
	}
	for index := uint64(0); index < size; index++ {
		if err := visit(elementType); err != nil {
			return err
		}
	}
	return nil
}

// readI64 reads any integer field as int64.
func (c *thriftCursor) readI64(fieldType byte) (int64, error) {
	switch fieldType {
	case thriftTypeByte:
		value, err := c.readByte()
		return int64(int8(value)), err
	case thriftTypeI16, thriftTypeI32, thriftTypeI64:
		return c.readZigzag()
	case thriftTypeTrue:
		return 1, nil
	case thriftTypeFalse:
		return 0, nil
	default:
		return 0, fmt.Errorf("dataset: thrift field type %d is not an integer", fieldType)
	}
}

func (c *thriftCursor) readBinary(fieldType byte) ([]byte, error) {
	if fieldType != thriftTypeBinary {
		return nil, errors.New("dataset: thrift field is not binary")
	}
	length, err := c.readUvarint()
	if err != nil {
		return nil, err
	}
	if c.position+int(length) > len(c.data) {
		return nil, io.ErrUnexpectedEOF
	}
	value := c.data[c.position : c.position+int(length)]
	c.position += int(length)
	return value, nil
}

// skip consumes one value of the given type without interpreting it.
func (c *thriftCursor) skip(fieldType byte) error {
	switch fieldType {
	case thriftTypeTrue, thriftTypeFalse:
		return nil
	case thriftTypeByte:
		_, err := c.readByte()
		return err
	case thriftTypeI16, thriftTypeI32, thriftTypeI64:
		_, err := c.readZigzag()
		return err
	case thriftTypeDouble:
		if c.position+8 > len(c.data) {
			return io.ErrUnexpectedEOF
		}
		c.position += 8
		return nil
	case thriftTypeBinary:
		_, err := c.readBinary(fieldType)
		return err
	case thriftTypeList, thriftTypeSet:
		return c.walkList(fieldType, func(elementType byte) error { return c.skip(elementType) })
	case thriftTypeMap:
		header, err := c.readByte()
		if err != nil {
			return err
		}
		if header == 0 {
			return nil
		}
		c.position--
		size, err := c.readUvarint()
		if err != nil {
			return err
		}
		kinds, err := c.readByte()
		if err != nil {
			return err
		}
		for range size {
			if err := c.skip(kinds >> 4); err != nil {
				return err
			}
			if err := c.skip(kinds & 0x0F); err != nil {
				return err
			}
		}
		return nil
	case thriftTypeStruct:
		return c.walkStruct(func(_ int16, innerType byte) error { return c.skip(innerType) })
	default:
		return fmt.Errorf("dataset: thrift type %d is outside the supported subset", fieldType)
	}
}
