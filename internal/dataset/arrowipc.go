package dataset

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
)

// Arrow IPC stream reader for the HuggingFace dataset cache -- the
// narrow subset those files actually use: an uncompressed stream of
// one schema and record batches whose columns are int64, float64,
// bool, utf8, or list-of-utf8. lm_eval's benchmark datasets ship in
// exactly this shape, and reading them natively keeps the import path
// free of external tooling. Anything outside the subset refuses with
// the construct's name rather than guessing.

// arrowContinuation frames every IPC message; a zero length after it
// is the end-of-stream marker.
const arrowContinuation = 0xFFFFFFFF

// Flatbuffers union discriminants from Arrow's Message.fbs and
// Schema.fbs -- the wire vocabulary this reader admits.
const (
	arrowHeaderSchema      = 1
	arrowHeaderRecordBatch = 3

	arrowTypeInt     = 2
	arrowTypeFloat   = 3
	arrowTypeUtf8    = 5
	arrowTypeBool    = 6
	arrowTypeList    = 12
	arrowTypeStruct  = 13
	arrowFloatSingle = 1
	arrowFloatDouble = 2
	arrowStructBytes = 16
)

type arrowField struct {
	name     string
	typeKind byte
	intBits  int32
	intSign  bool
	floatPre int16
	child    *arrowField
	children []arrowField
}

// DecodeArrowStream reads one Arrow IPC stream and observes every row
// as ordinal plus field-name-to-JSON-value bindings -- the same shape
// the JSONL benchmark decoder produces, so imports treat both formats
// identically.
func DecodeArrowStream(reader io.Reader, observe func(uint64, map[string]json.RawMessage) error) error {
	if reader == nil || observe == nil {
		return errors.New("dataset: incomplete arrow decode")
	}
	var fields []arrowField
	ordinal := uint64(0)
	for {
		metadata, body, done, err := readArrowMessage(reader)
		if done {
			return nil
		}
		if err != nil {
			return err
		}
		message := flatTable{data: metadata, pos: flatRootTable(metadata)}
		headerType := byte(message.scalarU8(1, 0))
		header, hasHeader := message.tableField(2)
		if !hasHeader {
			return errors.New("dataset: arrow message carries no header")
		}
		switch headerType {
		case arrowHeaderSchema:
			fields, err = readArrowSchema(header)
			if err != nil {
				return err
			}
		case arrowHeaderRecordBatch:
			if fields == nil {
				return errors.New("dataset: arrow record batch before schema")
			}
			ordinal, err = decodeArrowBatch(header, body, fields, ordinal, observe)
			if err != nil {
				return err
			}
		default:
			return fmt.Errorf("dataset: arrow header type %d is outside the supported subset", headerType)
		}
	}
}

func readArrowMessage(reader io.Reader) (metadata, body []byte, done bool, err error) {
	var word [4]byte
	if _, err := io.ReadFull(reader, word[:]); err != nil {
		if errors.Is(err, io.EOF) {
			return nil, nil, true, nil
		}
		return nil, nil, false, err
	}
	if binary.LittleEndian.Uint32(word[:]) != arrowContinuation {
		return nil, nil, false, errors.New("dataset: arrow stream is missing its continuation marker")
	}
	if _, err := io.ReadFull(reader, word[:]); err != nil {
		return nil, nil, false, err
	}
	length := binary.LittleEndian.Uint32(word[:])
	if length == 0 {
		return nil, nil, true, nil
	}
	metadata = make([]byte, length)
	if _, err := io.ReadFull(reader, metadata); err != nil {
		return nil, nil, false, err
	}
	message := flatTable{data: metadata, pos: flatRootTable(metadata)}
	bodyLength := message.scalarI64(3, 0)
	if bodyLength < 0 {
		return nil, nil, false, errors.New("dataset: arrow body length is negative")
	}
	body = make([]byte, bodyLength)
	if _, err := io.ReadFull(reader, body); err != nil {
		return nil, nil, false, err
	}
	return metadata, body, false, nil
}

func readArrowSchema(schema flatTable) ([]arrowField, error) {
	vector, ok := schema.vectorField(1)
	if !ok {
		return nil, errors.New("dataset: arrow schema carries no fields")
	}
	fields := make([]arrowField, vector.length)
	for index := range fields {
		field, err := readArrowField(vector.tableAt(index))
		if err != nil {
			return nil, err
		}
		fields[index] = field
	}
	return fields, nil
}

func readArrowField(table flatTable) (arrowField, error) {
	field := arrowField{typeKind: byte(table.scalarU8(2, 0))}
	field.name, _ = table.stringField(0)
	if _, dictionary := table.tableField(4); dictionary {
		return arrowField{}, fmt.Errorf("dataset: arrow field %q is dictionary-encoded, outside the supported subset", field.name)
	}
	typeTable, _ := table.tableField(3)
	switch field.typeKind {
	case arrowTypeInt:
		field.intBits = int32(typeTable.scalarI32(0, 0))
		field.intSign = typeTable.scalarU8(1, 0) != 0
		if field.intBits != 64 && field.intBits != 32 && field.intBits != 8 {
			return arrowField{}, fmt.Errorf("dataset: arrow field %q int width %d is outside the supported subset", field.name, field.intBits)
		}
	case arrowTypeFloat:
		field.floatPre = int16(typeTable.scalarI16(0, 0))
		if field.floatPre != arrowFloatDouble && field.floatPre != arrowFloatSingle {
			return arrowField{}, fmt.Errorf("dataset: arrow field %q float precision %d is outside the supported subset", field.name, field.floatPre)
		}
	case arrowTypeUtf8, arrowTypeBool:
	case arrowTypeList:
		children, ok := table.vectorField(5)
		if !ok || children.length != 1 {
			return arrowField{}, fmt.Errorf("dataset: arrow list field %q needs exactly one child", field.name)
		}
		child, err := readArrowField(children.tableAt(0))
		if err != nil {
			return arrowField{}, err
		}
		field.child = &child
	case arrowTypeStruct:
		children, ok := table.vectorField(5)
		if !ok || children.length == 0 {
			return arrowField{}, fmt.Errorf("dataset: arrow struct field %q carries no children", field.name)
		}
		field.children = make([]arrowField, children.length)
		for index := range field.children {
			child, err := readArrowField(children.tableAt(index))
			if err != nil {
				return arrowField{}, err
			}
			field.children[index] = child
		}
	default:
		return arrowField{}, fmt.Errorf("dataset: arrow field %q type %d is outside the supported subset", field.name, field.typeKind)
	}
	return field, nil
}

// arrowCursor walks the preorder node and buffer streams one column at
// a time; every column consumes its node, its buffers, then its child.
type arrowCursor struct {
	nodes   flatVector
	buffers flatVector
	node    int
	buffer  int
	body    []byte
}

func (c *arrowCursor) nextNode() (length, nulls int64, err error) {
	if c.node >= c.nodes.length {
		return 0, 0, errors.New("dataset: arrow batch exhausted its field nodes")
	}
	base := c.nodes.structAt(c.node)
	c.node++
	return int64(binary.LittleEndian.Uint64(c.nodes.data[base:])),
		int64(binary.LittleEndian.Uint64(c.nodes.data[base+8:])), nil
}

func (c *arrowCursor) nextBuffer() ([]byte, error) {
	if c.buffer >= c.buffers.length {
		return nil, errors.New("dataset: arrow batch exhausted its buffers")
	}
	base := c.buffers.structAt(c.buffer)
	c.buffer++
	offset := int64(binary.LittleEndian.Uint64(c.buffers.data[base:]))
	length := int64(binary.LittleEndian.Uint64(c.buffers.data[base+8:]))
	if offset < 0 || length < 0 || offset+length > int64(len(c.body)) {
		return nil, errors.New("dataset: arrow buffer exceeds its body")
	}
	return c.body[offset : offset+length], nil
}

func decodeArrowBatch(
	batch flatTable,
	body []byte,
	fields []arrowField,
	ordinal uint64,
	observe func(uint64, map[string]json.RawMessage) error,
) (uint64, error) {
	if _, compressed := batch.tableField(3); compressed {
		return ordinal, errors.New("dataset: compressed arrow batches are outside the supported subset")
	}
	rows := batch.scalarI64(0, 0)
	nodes, _ := batch.vectorField(1)
	buffers, _ := batch.vectorField(2)
	cursor := &arrowCursor{nodes: nodes, buffers: buffers, body: body}
	columns := make([][]json.RawMessage, len(fields))
	for index, field := range fields {
		values, err := decodeArrowColumn(cursor, field)
		if err != nil {
			return ordinal, fmt.Errorf("dataset: arrow column %q: %w", field.name, err)
		}
		if int64(len(values)) != rows {
			return ordinal, fmt.Errorf("dataset: arrow column %q carries %d rows of %d", field.name, len(values), rows)
		}
		columns[index] = values
	}
	for row := int64(0); row < rows; row++ {
		record := make(map[string]json.RawMessage, len(fields))
		for index, field := range fields {
			record[field.name] = columns[index][row]
		}
		if err := observe(ordinal, record); err != nil {
			return ordinal, err
		}
		ordinal++
	}
	return ordinal, nil
}

func decodeArrowColumn(cursor *arrowCursor, field arrowField) ([]json.RawMessage, error) {
	length, nulls, err := cursor.nextNode()
	if err != nil {
		return nil, err
	}
	validity, err := cursor.nextBuffer()
	if err != nil {
		return nil, err
	}
	valid := func(row int64) bool {
		if nulls == 0 {
			return true
		}
		if int64(len(validity))*8 <= row {
			return false
		}
		return validity[row/8]&(1<<(row%8)) != 0
	}
	values := make([]json.RawMessage, length)
	switch field.typeKind {
	case arrowTypeInt:
		data, err := cursor.nextBuffer()
		if err != nil {
			return nil, err
		}
		width := int64(field.intBits / 8)
		for row := int64(0); row < length; row++ {
			if !valid(row) {
				values[row] = json.RawMessage("null")
				continue
			}
			if (row+1)*width > int64(len(data)) {
				return nil, errors.New("int data underflow")
			}
			var value int64
			switch field.intBits {
			case 64:
				value = int64(binary.LittleEndian.Uint64(data[row*8:]))
			case 32:
				value = int64(int32(binary.LittleEndian.Uint32(data[row*4:])))
			default:
				value = int64(int8(data[row]))
			}
			if !field.intSign && value < 0 {
				return nil, errors.New("unsigned value overflows int64")
			}
			values[row] = json.RawMessage(strconv.FormatInt(value, 10))
		}
	case arrowTypeFloat:
		data, err := cursor.nextBuffer()
		if err != nil {
			return nil, err
		}
		for row := int64(0); row < length; row++ {
			if !valid(row) {
				values[row] = json.RawMessage("null")
				continue
			}
			var value float64
			if field.floatPre == arrowFloatDouble {
				if (row+1)*8 > int64(len(data)) {
					return nil, errors.New("float data underflow")
				}
				value = math.Float64frombits(binary.LittleEndian.Uint64(data[row*8:]))
			} else {
				if (row+1)*4 > int64(len(data)) {
					return nil, errors.New("float data underflow")
				}
				value = float64(math.Float32frombits(binary.LittleEndian.Uint32(data[row*4:])))
			}
			encoded, err := json.Marshal(value)
			if err != nil {
				return nil, err
			}
			values[row] = encoded
		}
	case arrowTypeBool:
		data, err := cursor.nextBuffer()
		if err != nil {
			return nil, err
		}
		for row := int64(0); row < length; row++ {
			switch {
			case !valid(row):
				values[row] = json.RawMessage("null")
			case int64(len(data))*8 <= row:
				return nil, errors.New("bool data underflow")
			case data[row/8]&(1<<(row%8)) != 0:
				values[row] = json.RawMessage("true")
			default:
				values[row] = json.RawMessage("false")
			}
		}
	case arrowTypeUtf8:
		offsets, err := cursor.nextBuffer()
		if err != nil {
			return nil, err
		}
		data, err := cursor.nextBuffer()
		if err != nil {
			return nil, err
		}
		if int64(len(offsets)) < 4*(length+1) {
			return nil, errors.New("utf8 offsets underflow")
		}
		for row := int64(0); row < length; row++ {
			if !valid(row) {
				values[row] = json.RawMessage("null")
				continue
			}
			start := int64(int32(binary.LittleEndian.Uint32(offsets[row*4:])))
			end := int64(int32(binary.LittleEndian.Uint32(offsets[(row+1)*4:])))
			if start < 0 || end < start || end > int64(len(data)) {
				return nil, errors.New("utf8 offsets exceed data")
			}
			encoded, err := json.Marshal(string(data[start:end]))
			if err != nil {
				return nil, err
			}
			values[row] = encoded
		}
	case arrowTypeList:
		offsets, err := cursor.nextBuffer()
		if err != nil {
			return nil, err
		}
		if int64(len(offsets)) < 4*(length+1) {
			return nil, errors.New("list offsets underflow")
		}
		children, err := decodeArrowColumn(cursor, *field.child)
		if err != nil {
			return nil, err
		}
		for row := int64(0); row < length; row++ {
			if !valid(row) {
				values[row] = json.RawMessage("null")
				continue
			}
			start := int64(int32(binary.LittleEndian.Uint32(offsets[row*4:])))
			end := int64(int32(binary.LittleEndian.Uint32(offsets[(row+1)*4:])))
			if start < 0 || end < start || end > int64(len(children)) {
				return nil, errors.New("list offsets exceed child rows")
			}
			element := []byte("[")
			for child := start; child < end; child++ {
				if child > start {
					element = append(element, ',')
				}
				element = append(element, children[child]...)
			}
			values[row] = append(element, ']')
		}
	case arrowTypeStruct:
		columns := make([][]json.RawMessage, len(field.children))
		for index, child := range field.children {
			childValues, err := decodeArrowColumn(cursor, child)
			if err != nil {
				return nil, err
			}
			if int64(len(childValues)) != length {
				return nil, errors.New("struct child length differs")
			}
			columns[index] = childValues
		}
		for row := int64(0); row < length; row++ {
			if !valid(row) {
				values[row] = json.RawMessage("null")
				continue
			}
			object := map[string]json.RawMessage{}
			for index, child := range field.children {
				object[child.name] = columns[index][row]
			}
			encoded, err := json.Marshal(object)
			if err != nil {
				return nil, err
			}
			values[row] = encoded
		}
	default:
		return nil, fmt.Errorf("type %d is outside the supported subset", field.typeKind)
	}
	return values, nil
}

// ---- minimal flatbuffers access ----

type flatTable struct {
	data []byte
	pos  int
}

type flatVector struct {
	data   []byte
	pos    int
	length int
	stride int
}

func flatRootTable(data []byte) int {
	return int(binary.LittleEndian.Uint32(data))
}

// fieldPosition resolves a table field id through the vtable; zero
// means absent.
func (t flatTable) fieldPosition(id int) int {
	vtable := t.pos - int(int32(binary.LittleEndian.Uint32(t.data[t.pos:])))
	vtableSize := int(binary.LittleEndian.Uint16(t.data[vtable:]))
	slot := 4 + 2*id
	if slot+2 > vtableSize {
		return 0
	}
	offset := int(binary.LittleEndian.Uint16(t.data[vtable+slot:]))
	if offset == 0 {
		return 0
	}
	return t.pos + offset
}

func (t flatTable) scalarU8(id int, missing byte) byte {
	if position := t.fieldPosition(id); position != 0 {
		return t.data[position]
	}
	return missing
}

func (t flatTable) scalarI16(id int, missing int16) int16 {
	if position := t.fieldPosition(id); position != 0 {
		return int16(binary.LittleEndian.Uint16(t.data[position:]))
	}
	return missing
}

func (t flatTable) scalarI32(id int, missing int32) int32 {
	if position := t.fieldPosition(id); position != 0 {
		return int32(binary.LittleEndian.Uint32(t.data[position:]))
	}
	return missing
}

func (t flatTable) scalarI64(id int, missing int64) int64 {
	if position := t.fieldPosition(id); position != 0 {
		return int64(binary.LittleEndian.Uint64(t.data[position:]))
	}
	return missing
}

func (t flatTable) tableField(id int) (flatTable, bool) {
	position := t.fieldPosition(id)
	if position == 0 {
		return flatTable{}, false
	}
	target := position + int(binary.LittleEndian.Uint32(t.data[position:]))
	return flatTable{data: t.data, pos: target}, true
}

func (t flatTable) stringField(id int) (string, bool) {
	position := t.fieldPosition(id)
	if position == 0 {
		return "", false
	}
	target := position + int(binary.LittleEndian.Uint32(t.data[position:]))
	length := int(binary.LittleEndian.Uint32(t.data[target:]))
	return string(t.data[target+4 : target+4+length]), true
}

// vectorField returns a struct-or-offset vector; stride is 16 for the
// Arrow struct vectors (FieldNode, Buffer) and 4 for offset vectors.
func (t flatTable) vectorField(id int) (flatVector, bool) {
	position := t.fieldPosition(id)
	if position == 0 {
		return flatVector{}, false
	}
	target := position + int(binary.LittleEndian.Uint32(t.data[position:]))
	length := int(binary.LittleEndian.Uint32(t.data[target:]))
	return flatVector{data: t.data, pos: target + 4, length: length}, true
}

// tableAt resolves element index of an offset vector as a table.
func (v flatVector) tableAt(index int) flatTable {
	position := v.pos + 4*index
	target := position + int(binary.LittleEndian.Uint32(v.data[position:]))
	return flatTable{data: v.data, pos: target}
}

// structAt returns the byte position of element index in a 16-byte
// struct vector.
func (v flatVector) structAt(index int) int {
	return v.pos + arrowStructBytes*index
}
