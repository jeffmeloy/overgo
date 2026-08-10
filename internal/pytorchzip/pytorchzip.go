// Package pytorchzip reads PyTorch ZIP checkpoints (.pth/.pt): data.pkl is a
// pickle stream carrying tensor metadata; sibling data/<key> entries hold raw
// little-endian storages (required uncompressed, so bodies stream by offset).
// Retained-capability port of adaptive's reader; neutral home — nothing here
// knows any model family.
package pytorchzip

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path"
	"sort"

	"overgo/internal/tensor/dtype"
)

// TensorMeta: one tensor as described by the checkpoint pickle.
type TensorMeta struct {
	Name          string
	DType         string // torch storage class name, e.g. "BFloat16Storage"
	StorageKey    string
	StorageSize   int64 // elements in the backing storage
	StorageOffset int64 // element offset into the storage
	Shape         []int64
	Stride        []int64
	RequiresGrad  bool
	Numel         int64
}

// ReadTensorMetadata parses data.pkl out of one checkpoint file.
func ReadTensorMetadata(filename string) ([]TensorMeta, error) {
	zr, err := zip.OpenReader(filename)
	if err != nil {
		return nil, err
	}
	defer zr.Close()
	for _, f := range zr.File {
		if path.Base(f.Name) == "data.pkl" {
			b, err := readZipEntryBytes(f, maxPickleBytes)
			if err != nil {
				return nil, err
			}
			return ParseTensorMetadata(b)
		}
	}
	return nil, fmt.Errorf("pytorchzip: data.pkl not found in %s", filename)
}

const (
	maxPickleBytes    = 1 << 20
	scratchChunkBytes = 1 << 20
)

// ParseTensorMetadata decodes a torch state_dict pickle stream.
func ParseTensorMetadata(data []byte) (metas []TensorMeta, err error) {
	// Stack-machine helpers signal truncation/underflow with a TYPED
	// controlled panic (encoding/gob pattern); recover exactly that type into
	// a refusal. Any other panic is a real bug and still crashes.
	defer func() {
		if r := recover(); r != nil {
			pe, ok := r.(pickleParseError)
			if !ok {
				panic(r)
			}
			metas, err = nil, fmt.Errorf("pytorchzip pickle: %s at offset %d", pe.msg, pe.pos)
		}
	}()
	p := &pickleTensorParser{data: data, memo: map[int]pickleValue{}}
	if err := p.parse(); err != nil {
		return nil, err
	}
	if len(p.tensors) == 0 {
		return nil, fmt.Errorf("pytorchzip pickle: no tensors found")
	}
	// Downstream loaders address tensors BY name; an unnamed tensor would be
	// present but unreachable, so it is a malformed stream, not an oddity.
	for _, m := range p.tensors {
		if m.Name == "" {
			return nil, fmt.Errorf("pytorchzip pickle: tensor with storage key %q has no state_dict name", m.StorageKey)
		}
	}
	sort.Slice(p.tensors, func(i, j int) bool { return p.tensors[i].Name < p.tensors[j].Name })
	return p.tensors, nil
}

type pickleValue interface{}

type pickleMark struct{}
type pickleGlobal struct{ module, name string }
type pickleStorageRef struct {
	dtype       string
	key         string
	storageSize int64
}
type pickleTensorRef struct{ index int }
type pickleDict struct{}

type pickleTensorParser struct {
	data    []byte
	pos     int
	stack   []pickleValue
	memo    map[int]pickleValue
	tensors []TensorMeta
	// scalars captures string-keyed scalar dict entries (ints, floats, bools,
	// strings) seen anywhere in the stream — the config a {cfg, model} save
	// pickles beside the weights. nil disables capture (the tensor-only path).
	scalars map[string]any
}

func (p *pickleTensorParser) parse() error {
	for p.pos < len(p.data) {
		op := p.readByte()
		switch op {
		case 0x80: // PROTO
			p.pos++
		case 'c': // GLOBAL
			module, err := p.readLine()
			if err != nil {
				return err
			}
			name, err := p.readLine()
			if err != nil {
				return err
			}
			p.push(pickleGlobal{module: module, name: name})
		case 'q': // BINPUT
			p.memo[int(p.readByte())] = p.peek()
		case 'r': // LONG_BINPUT
			p.memo[int(p.readU32())] = p.peek()
		case 'h': // BINGET
			idx := int(p.readByte())
			v, ok := p.memo[idx]
			if !ok {
				return fmt.Errorf("pytorchzip pickle: missing memo %d", idx)
			}
			p.push(v)
		case 'j': // LONG_BINGET
			idx := int(p.readU32())
			v, ok := p.memo[idx]
			if !ok {
				return fmt.Errorf("pytorchzip pickle: missing memo %d", idx)
			}
			p.push(v)
		case '(': // MARK
			p.push(pickleMark{})
		case ')': // EMPTY_TUPLE
			p.push([]pickleValue{})
		case '}': // EMPTY_DICT
			p.push(pickleDict{})
		case ']': // EMPTY_LIST
			// A config pickled beside the weights (a {cfg, model, step} save) can
			// carry list fields; a list is a slice, like a tuple. The tensor walk
			// ignores the container, but the stream must still parse past it.
			p.push([]pickleValue{})
		case 'a': // APPEND
			val := p.pop()
			lst := p.pop()
			if items, ok := lst.([]pickleValue); ok {
				p.push(append(items, val))
				break
			}
			p.push(lst)
		case 'e': // APPENDS
			items := p.popUntilMark()
			lst := p.pop()
			if existing, ok := lst.([]pickleValue); ok {
				p.push(append(existing, items...))
				break
			}
			p.push(lst)
		case 'G': // BINFLOAT: 8-byte IEEE-754 double, BIG-endian (network order)
			// A {cfg, model, step} save pickles a config alongside the weights;
			// any float field in it emits BINFLOAT, over a number the tensor walk
			// never reads. Without this case the reader refuses the whole file.
			p.push(math.Float64frombits(binary.BigEndian.Uint64(p.read(8))))
		case 'X': // BINUNICODE
			n := int(p.readU32())
			p.push(string(p.read(n)))
		case 'J': // BININT
			p.push(int64(int32(p.readU32())))
		case 'K': // BININT1
			p.push(int64(p.readByte()))
		case 'M': // BININT2
			p.push(int64(binary.LittleEndian.Uint16(p.read(2))))
		case 't': // TUPLE
			p.push(p.popUntilMark())
		case 0x85: // TUPLE1
			a := p.pop()
			p.push([]pickleValue{a})
		case 0x86: // TUPLE2
			b, a := p.pop(), p.pop()
			p.push([]pickleValue{a, b})
		case 0x87: // TUPLE3
			c, b, a := p.pop(), p.pop(), p.pop()
			p.push([]pickleValue{a, b, c})
		case 'Q': // BINPERSID
			v, err := persistentStorageRef(p.pop())
			if err != nil {
				return err
			}
			p.push(v)
		case 0x89: // NEWFALSE
			p.push(false)
		case 0x88: // NEWTRUE
			p.push(true)
		case 'R': // REDUCE
			args := p.pop()
			fn := p.pop()
			v, err := p.reduce(fn, args)
			if err != nil {
				return err
			}
			p.push(v)
		case 's': // SETITEM
			val, key, dict := p.pop(), p.pop(), p.pop()
			p.notePair(key, val)
			p.push(dict)
		case 'u': // SETITEMS
			items := p.popUntilMark()
			dict := p.pop()
			for i := 0; i+1 < len(items); i += 2 {
				p.notePair(items[i], items[i+1])
			}
			p.push(dict)
		case 'b': // BUILD
			_ = p.pop()
		case '.': // STOP
			return nil
		default:
			return fmt.Errorf("pytorchzip pickle: unsupported opcode %#x at %d", op, p.pos-1)
		}
	}
	return io.ErrUnexpectedEOF
}

func (p *pickleTensorParser) reduce(fn, args pickleValue) (pickleValue, error) {
	g, ok := fn.(pickleGlobal)
	if !ok {
		return pickleDict{}, nil
	}
	if g.module == "collections" && g.name == "OrderedDict" {
		return pickleDict{}, nil
	}
	if g.module == "torch._tensor" && g.name == "_rebuild_from_type_v2" {
		wrapper, ok := args.([]pickleValue)
		if !ok || len(wrapper) < 3 {
			return nil, fmt.Errorf("pytorchzip pickle: malformed tensor subclass wrapper")
		}
		return p.reduce(wrapper[0], wrapper[2])
	}
	if g.module != "torch._utils" || g.name != "_rebuild_tensor_v2" {
		return pickleDict{}, nil
	}
	tup, ok := args.([]pickleValue)
	if !ok || len(tup) < 6 {
		return nil, fmt.Errorf("pytorchzip pickle: bad tensor args")
	}
	st, ok := tup[0].(pickleStorageRef)
	if !ok {
		return nil, fmt.Errorf("pytorchzip pickle: tensor missing storage ref")
	}
	offset, ok := asInt64(tup[1])
	if !ok {
		return nil, fmt.Errorf("pytorchzip pickle: tensor missing storage offset")
	}
	shape, ok := asInt64Slice(tup[2])
	if !ok {
		return nil, fmt.Errorf("pytorchzip pickle: tensor missing shape")
	}
	stride, ok := asInt64Slice(tup[3])
	if !ok {
		return nil, fmt.Errorf("pytorchzip pickle: tensor missing stride")
	}
	req, _ := tup[4].(bool)
	numel := int64(1)
	for _, v := range shape {
		numel *= v
	}
	p.tensors = append(p.tensors, TensorMeta{
		DType: st.dtype, StorageKey: st.key, StorageSize: st.storageSize,
		StorageOffset: offset, Shape: shape, Stride: stride, RequiresGrad: req, Numel: numel,
	})
	return pickleTensorRef{index: len(p.tensors) - 1}, nil
}

func (p *pickleTensorParser) notePair(key, val pickleValue) {
	name, ok := key.(string)
	if !ok {
		return
	}
	if ref, ok := val.(pickleTensorRef); ok && ref.index >= 0 && ref.index < len(p.tensors) {
		p.tensors[ref.index].Name = name
		return
	}
	// Config scalars a {cfg, model} save carries beside the weights. The tensor
	// walk ignores them; capture is opt-in so the tensor-only path is unchanged.
	if p.scalars != nil {
		switch v := val.(type) {
		case int64, float64, bool, string:
			p.scalars[name] = v
		}
	}
}

// ReadScalarConfig parses the checkpoint pickle and returns its string-keyed
// scalar entries (ints, floats, bools, strings), flattened by key across any
// nested dicts — the config a {cfg, model, step} save pickles beside the
// weights. Tensor entries are skipped; use ReadTensorMetadata for those.
func ReadScalarConfig(filename string) (map[string]any, error) {
	zr, err := zip.OpenReader(filename)
	if err != nil {
		return nil, err
	}
	defer zr.Close()
	for _, f := range zr.File {
		if path.Base(f.Name) == "data.pkl" {
			b, err := readZipEntryBytes(f, maxPickleBytes)
			if err != nil {
				return nil, err
			}
			return parseScalarConfig(b)
		}
	}
	return nil, fmt.Errorf("pytorchzip: data.pkl not found in %s", filename)
}

func parseScalarConfig(data []byte) (scalars map[string]any, err error) {
	defer func() {
		if r := recover(); r != nil {
			pe, ok := r.(pickleParseError)
			if !ok {
				panic(r)
			}
			scalars, err = nil, fmt.Errorf("pytorchzip pickle: %s at offset %d", pe.msg, pe.pos)
		}
	}()
	p := &pickleTensorParser{data: data, memo: map[int]pickleValue{}, scalars: map[string]any{}}
	if err := p.parse(); err != nil {
		return nil, err
	}
	return p.scalars, nil
}

func persistentStorageRef(v pickleValue) (pickleStorageRef, error) {
	t, ok := v.([]pickleValue)
	if !ok || len(t) < 5 {
		return pickleStorageRef{}, fmt.Errorf("pytorchzip pickle: bad persistent id")
	}
	kind, ok := t[0].(string)
	if !ok || kind != "storage" {
		return pickleStorageRef{}, fmt.Errorf("pytorchzip pickle: unsupported persistent id")
	}
	g, ok := t[1].(pickleGlobal)
	if !ok || g.module != "torch" {
		return pickleStorageRef{}, fmt.Errorf("pytorchzip pickle: unsupported storage class")
	}
	key, ok := t[2].(string)
	if !ok {
		return pickleStorageRef{}, fmt.Errorf("pytorchzip pickle: missing storage key")
	}
	size, ok := asInt64(t[4])
	if !ok {
		return pickleStorageRef{}, fmt.Errorf("pytorchzip pickle: missing storage size")
	}
	return pickleStorageRef{dtype: g.name, key: key, storageSize: size}, nil
}

func asInt64(v pickleValue) (int64, bool) {
	i, ok := v.(int64)
	return i, ok
}

func asInt64Slice(v pickleValue) ([]int64, bool) {
	t, ok := v.([]pickleValue)
	if !ok {
		return nil, false
	}
	out := make([]int64, len(t))
	for i, x := range t {
		v, ok := asInt64(x)
		if !ok {
			return nil, false
		}
		out[i] = v
	}
	return out, true
}

// pickleParseError: controlled non-local exit — truncated input and stack
// underflow are DATA defects and must surface as refusals.
type pickleParseError struct {
	msg string
	pos int
}

func (p *pickleTensorParser) fail(msg string) {
	panic(pickleParseError{msg: msg, pos: p.pos})
}

func (p *pickleTensorParser) popUntilMark() []pickleValue {
	i := len(p.stack) - 1
	for ; i >= 0; i-- {
		if _, ok := p.stack[i].(pickleMark); ok {
			break
		}
	}
	if i < 0 {
		p.fail("no mark on stack")
	}
	items := append([]pickleValue(nil), p.stack[i+1:]...)
	p.stack = p.stack[:i]
	return items
}

func (p *pickleTensorParser) push(v pickleValue) { p.stack = append(p.stack, v) }

func (p *pickleTensorParser) peek() pickleValue {
	if len(p.stack) == 0 {
		p.fail("peek on empty stack")
	}
	return p.stack[len(p.stack)-1]
}

func (p *pickleTensorParser) pop() pickleValue {
	if len(p.stack) == 0 {
		p.fail("pop on empty stack")
	}
	v := p.stack[len(p.stack)-1]
	p.stack = p.stack[:len(p.stack)-1]
	return v
}

func (p *pickleTensorParser) readByte() byte {
	if p.pos >= len(p.data) {
		p.fail("truncated stream")
	}
	b := p.data[p.pos]
	p.pos++
	return b
}

func (p *pickleTensorParser) readU32() uint32 {
	return binary.LittleEndian.Uint32(p.read(4))
}

func (p *pickleTensorParser) read(n int) []byte {
	if n < 0 || p.pos+n > len(p.data) || p.pos+n < p.pos {
		p.fail("truncated stream")
	}
	b := p.data[p.pos : p.pos+n]
	p.pos += n
	return b
}

func (p *pickleTensorParser) readLine() (string, error) {
	start := p.pos
	for p.pos < len(p.data) && p.data[p.pos] != '\n' {
		p.pos++
	}
	if p.pos >= len(p.data) {
		return "", io.ErrUnexpectedEOF
	}
	s := string(p.data[start:p.pos])
	p.pos++
	return s, nil
}

// TensorIdentity is the comparable identity of one validated contiguous
// tensor binding; callers do not infer identity from decoded allocations.
type TensorIdentity struct {
	Name          string
	DType         string
	StorageKey    string
	StorageOffset int64
	Numel         int64
}

type TensorBinding struct {
	Meta     TensorMeta
	Identity TensorIdentity
}

// CompileBindings resolves names against metas and validates each identity.
func CompileBindings(metas []TensorMeta, names []string) ([]TensorBinding, error) {
	byName := make(map[string]TensorMeta, len(metas))
	for _, meta := range metas {
		if meta.Name == "" {
			return nil, fmt.Errorf("pytorchzip bindings: empty tensor name")
		}
		if _, exists := byName[meta.Name]; exists {
			return nil, fmt.Errorf("pytorchzip bindings: duplicate tensor %s", meta.Name)
		}
		byName[meta.Name] = meta
	}
	bindings := make([]TensorBinding, len(names))
	for index, name := range names {
		meta, ok := byName[name]
		if !ok {
			return nil, fmt.Errorf("pytorchzip bindings: missing tensor %s", name)
		}
		identity, err := compileTensorIdentity(meta)
		if err != nil {
			return nil, err
		}
		meta.Shape = append([]int64(nil), meta.Shape...)
		meta.Stride = append([]int64(nil), meta.Stride...)
		bindings[index] = TensorBinding{Meta: meta, Identity: identity}
	}
	return bindings, nil
}

func compileTensorIdentity(meta TensorMeta) (TensorIdentity, error) {
	identity := TensorIdentity{
		Name: meta.Name, DType: meta.DType, StorageKey: meta.StorageKey,
		StorageOffset: meta.StorageOffset, Numel: meta.Numel,
	}
	if meta.Name == "" || meta.DType == "" || meta.StorageKey == "" || meta.StorageOffset < 0 || meta.Numel < 0 || meta.StorageSize < 0 {
		return identity, fmt.Errorf("pytorchzip binding: invalid metadata for %q", meta.Name)
	}
	if meta.Numel > meta.StorageSize || meta.StorageOffset > meta.StorageSize-meta.Numel {
		return identity, fmt.Errorf("pytorchzip binding: %s range exceeds storage", meta.Name)
	}
	elements := int64(1)
	for axis, dim := range meta.Shape {
		if dim <= 0 || elements > math.MaxInt64/dim {
			return identity, fmt.Errorf("pytorchzip binding: %s invalid shape dimension %d at axis %d", meta.Name, dim, axis)
		}
		elements *= dim
	}
	if elements != meta.Numel {
		return identity, fmt.Errorf("pytorchzip binding: %s shape elements=%d numel=%d", meta.Name, elements, meta.Numel)
	}
	if len(meta.Stride) != len(meta.Shape) {
		return identity, fmt.Errorf("pytorchzip binding: %s stride rank=%d shape rank=%d", meta.Name, len(meta.Stride), len(meta.Shape))
	}
	expected := int64(1)
	for dim := len(meta.Shape) - 1; dim >= 0; dim-- {
		if meta.Shape[dim] > 1 && meta.Stride[dim] != expected {
			return identity, fmt.Errorf("pytorchzip binding: %s is not contiguous at dimension %d", meta.Name, dim)
		}
		if meta.Shape[dim] != 0 && expected > math.MaxInt64/meta.Shape[dim] {
			return identity, fmt.Errorf("pytorchzip binding: %s stride extent overflows", meta.Name)
		}
		expected *= meta.Shape[dim]
	}
	return identity, nil
}

// TensorMetaBytes: the storage byte span one tensor occupies.
func TensorMetaBytes(m TensorMeta) (int64, error) {
	width, _, err := storageDecode(m.DType)
	if err != nil {
		return 0, fmt.Errorf("pytorchzip bytes: %w for %s", err, m.Name)
	}
	return m.Numel * width, nil
}

func storageDecode(dtypeName string) (elemBytes int64, convert func(uint16) float32, err error) {
	switch dtypeName {
	case "FloatStorage":
		return 4, nil, nil
	case "HalfStorage":
		return 2, dtype.Float16ToFloat32, nil
	case "BFloat16Storage":
		return 2, dtype.BF16ToFloat32, nil
	default:
		return 0, nil, fmt.Errorf("unsupported dtype %s", dtypeName)
	}
}

// Reader decodes tensor bodies to F32 host storage by direct file offsets.
type Reader struct {
	zr      *zip.ReadCloser
	f       *os.File
	root    string
	entries map[string]*zip.File
	scratch []byte
}

// Open indexes the checkpoint's storage entries for offset reads.
func Open(filename string) (*Reader, error) {
	zr, err := zip.OpenReader(filename)
	if err != nil {
		return nil, err
	}
	root, entries, err := storageEntries(zr.File)
	if err != nil {
		zr.Close()
		return nil, err
	}
	f, err := os.Open(filename)
	if err != nil {
		zr.Close()
		return nil, err
	}
	return &Reader{zr: zr, f: f, root: root, entries: entries}, nil
}

func (r *Reader) Close() error {
	var err error
	if r.f != nil {
		err = r.f.Close()
		r.f = nil
	}
	if r.zr != nil {
		if zerr := r.zr.Close(); err == nil {
			err = zerr
		}
		r.zr = nil
	}
	r.scratch = nil
	return err
}

// ReadBindingValues decodes compiled bindings in order; scratch is reused.
func (r *Reader) ReadBindingValues(bindings []TensorBinding) ([][]float32, error) {
	if r == nil || r.f == nil {
		return nil, fmt.Errorf("pytorchzip: reader is closed")
	}
	out := make([][]float32, len(bindings))
	for index, binding := range bindings {
		values, err := r.ReadBinding(binding)
		if err != nil {
			return nil, err
		}
		out[index] = values
	}
	return out, nil
}

// ReadBinding validates identity then decodes the whole tensor to F32.
func (r *Reader) ReadBinding(binding TensorBinding) ([]float32, error) {
	if r == nil || r.f == nil {
		return nil, fmt.Errorf("pytorchzip: reader is closed")
	}
	identity, err := compileTensorIdentity(binding.Meta)
	if err != nil {
		return nil, err
	}
	if identity != binding.Identity {
		return nil, fmt.Errorf("pytorchzip: binding identity drift for %s", binding.Identity.Name)
	}
	meta := binding.Meta
	elemBytes, convert, err := storageDecode(meta.DType)
	if err != nil {
		return nil, fmt.Errorf("pytorchzip: %w for %s", err, meta.Name)
	}
	if uint64(meta.Numel) > uint64(^uint(0)>>1) || meta.StorageOffset > math.MaxInt64/elemBytes {
		return nil, fmt.Errorf("pytorchzip: %s byte span overflows", meta.Name)
	}
	dataOffset, err := r.storageDataOffset(meta)
	if err != nil {
		return nil, err
	}
	values := make([]float32, int(meta.Numel))
	if err := r.decodeAt(dataOffset+meta.StorageOffset*elemBytes, elemBytes, convert, values); err != nil {
		return nil, err
	}
	return values, nil
}

// ReadTensorRows gathers selected rows of a 2-d row-contiguous tensor.
func (r *Reader) ReadTensorRows(binding TensorBinding, rows []int, rowWidth int) ([]float32, error) {
	if r == nil || r.f == nil {
		return nil, fmt.Errorf("pytorchzip: reader is closed")
	}
	identity, err := compileTensorIdentity(binding.Meta)
	if err != nil {
		return nil, err
	}
	if identity != binding.Identity {
		return nil, fmt.Errorf("pytorchzip: binding identity drift for %s", binding.Identity.Name)
	}
	meta := binding.Meta
	elemBytes, convert, err := storageDecode(meta.DType)
	if err != nil {
		return nil, err
	}
	if len(meta.Shape) != 2 || int(meta.Shape[1]) != rowWidth || rowWidth <= 0 {
		return nil, fmt.Errorf("pytorchzip rows: %s shape=%v rowWidth=%d", meta.Name, meta.Shape, rowWidth)
	}
	if len(meta.Stride) != 2 || meta.Stride[1] != 1 {
		return nil, fmt.Errorf("pytorchzip rows: %s unsupported stride=%v", meta.Name, meta.Stride)
	}
	dataOffset, err := r.storageDataOffset(meta)
	if err != nil {
		return nil, err
	}
	out := make([]float32, len(rows)*rowWidth)
	for i, row := range rows {
		if row < 0 || row >= int(meta.Shape[0]) {
			return nil, fmt.Errorf("pytorchzip rows: row %d outside [0,%d)", row, meta.Shape[0])
		}
		elemOffset := meta.StorageOffset + int64(row)*meta.Stride[0]
		if elemOffset+int64(rowWidth) > meta.StorageSize {
			return nil, fmt.Errorf("pytorchzip rows: %s row %d exceeds storage", meta.Name, row)
		}
		dst := out[i*rowWidth : (i+1)*rowWidth]
		if err := r.decodeAt(dataOffset+elemOffset*elemBytes, elemBytes, convert, dst); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (r *Reader) storageDataOffset(meta TensorMeta) (int64, error) {
	entry, ok := r.entries[r.root+"/data/"+meta.StorageKey]
	if !ok {
		return 0, fmt.Errorf("pytorchzip: storage entry %s not found", r.root+"/data/"+meta.StorageKey)
	}
	return entry.DataOffset()
}

// decodeAt reads len(out) elements at byteOffset in chunks through scratch.
func (r *Reader) decodeAt(byteOffset, elemBytes int64, convert func(uint16) float32, out []float32) error {
	chunkElems := min(max(1, len(out)), max(1, scratchChunkBytes/int(elemBytes)))
	need := chunkElems * int(elemBytes)
	if cap(r.scratch) < need {
		r.scratch = make([]byte, need)
	}
	for start := 0; start < len(out); {
		n := min(len(out)-start, chunkElems)
		buf := r.scratch[:n*int(elemBytes)]
		if _, err := r.f.ReadAt(buf, byteOffset+int64(start)*elemBytes); err != nil {
			return err
		}
		if convert == nil {
			for i := 0; i < n; i++ {
				out[start+i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
			}
		} else {
			for i := 0; i < n; i++ {
				out[start+i] = convert(binary.LittleEndian.Uint16(buf[i*2:]))
			}
		}
		start += n
	}
	return nil
}

func storageEntries(files []*zip.File) (string, map[string]*zip.File, error) {
	var root string
	for _, f := range files {
		if path.Base(f.Name) == "data.pkl" {
			root = path.Dir(f.Name)
			break
		}
	}
	if root == "" {
		return "", nil, fmt.Errorf("pytorchzip: data.pkl not found")
	}
	entries := map[string]*zip.File{}
	for _, f := range files {
		if f.Method != zip.Store && path.Dir(f.Name) == root+"/data" {
			return "", nil, fmt.Errorf("pytorchzip: storage entry %s is compressed", f.Name)
		}
		entries[f.Name] = f
	}
	return root, entries, nil
}

func readZipEntryBytes(f *zip.File, maxBytes int64) ([]byte, error) {
	if int64(f.UncompressedSize64) > maxBytes {
		return nil, fmt.Errorf("pytorchzip: entry %s too large: %d > %d", f.Name, f.UncompressedSize64, maxBytes)
	}
	rc, err := f.Open()
	if err != nil {
		return nil, err
	}
	defer rc.Close()
	out := make([]byte, int(f.UncompressedSize64))
	_, err = io.ReadFull(rc, out)
	return out, err
}
