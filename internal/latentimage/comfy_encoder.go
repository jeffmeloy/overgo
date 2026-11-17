package latentimage

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"strings"
	"sync"

	"overgo/internal/safetensors"
)

// A ComfyUI text encoder file (the MOSS Ultra workflow's Qwen3.5-9B-abliterated_int8_convrot) holds a Qwen3.5 model as
// a Hugging Face checkpoint for generation names it (model.language_model.layers.0..., where MOSS-Image's own text
// encoder, a base model, names language_model.layers.0...), each projection, the token embedding and the head in int8
// ConvRot with an F32 scale a row and a comfy_quant record (convRotView), and a vision tower and the head, which
// conditioning does not read. comfyLanguageView reads such a file as MOSS-Image's encoder holds its own: each
// language-model tensor under its name without the model. prefix, each ConvRot weight as the bfloat16 weight it decodes
// to, decoded as its rows are read, and neither the vision tower nor the head.
const (
	comfyModelPrefix = "model."
	comfyVisual      = "model.visual."
	comfyHead        = "lm_head."
	comfyQuantRecord = ".comfy_quant"
	comfyWeightScale = ".weight_scale"
)

// convRotBlockRows is how many rows a ConvRot reader decodes at once, so a reader reading a weight whole reads its
// values and scales in blocks.
const convRotBlockRows = 256

// comfyLanguageView is the language model of a ComfyUI text encoder source as a Hugging Face checkpoint's tensors, the
// source returned as it is where it holds no ConvRot weight. The view takes the source's files over (Retensored).
func comfyLanguageView(source *safetensors.Source) (*safetensors.Source, error) {
	quantized := false
	for name := range source.Tensors {
		if strings.HasSuffix(name, comfyQuantRecord) {
			quantized = true
			break
		}
	}
	if !quantized {
		return source, nil
	}
	tensors := map[string]safetensors.Tensor{}
	for name, value := range source.Tensors {
		if strings.HasPrefix(name, comfyVisual) || strings.HasPrefix(name, comfyHead) || strings.HasSuffix(name, comfyQuantRecord) || strings.HasSuffix(name, comfyWeightScale) {
			continue
		}
		renamed := strings.TrimPrefix(name, comfyModelPrefix)
		stored, quantized, err := convRotView(source, name)
		if err != nil {
			return nil, err
		}
		if !quantized {
			value.Name = renamed
			tensors[renamed] = value
			continue
		}
		decoded, err := safetensors.NewTensor(renamed, "BF16", stored.shape(), &convRotReader{stored: stored}, 0, int64(stored.rows*stored.columns*2))
		if err != nil {
			return nil, fmt.Errorf("convrot: %s: %w", name, err)
		}
		tensors[renamed] = decoded
	}
	return source.Retensored(tensors), nil
}

// convRotReader reads a ConvRot weight as the bfloat16 rows it decodes to, little-endian, decoding a block of rows at a
// time and keeping the last block, so a weight read whole or row by row decodes each row once.
type convRotReader struct {
	stored convRotStored
	mu     sync.Mutex
	first  uint64
	block  []byte
}

// ReadAt reads the decoded weight's bytes from offset, as many as destination holds or the weight has left.
func (r *convRotReader) ReadAt(destination []byte, offset int64) (int, error) {
	rowBytes := int64(r.stored.columns * 2)
	read := 0
	for read < len(destination) {
		position := offset + int64(read)
		row := uint64(position / rowBytes)
		if row >= r.stored.rows {
			return read, io.EOF
		}
		block, first, err := r.blockOf(row)
		if err != nil {
			return read, err
		}
		read += copy(destination[read:], block[position-int64(first)*rowBytes:])
	}
	return read, nil
}

// blockOf is the decoded block that holds row, and the block's first row.
func (r *convRotReader) blockOf(row uint64) ([]byte, uint64, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.block != nil && row >= r.first && row < r.first+uint64(len(r.block))/(r.stored.columns*2) {
		return r.block, r.first, nil
	}
	first := row - row%convRotBlockRows
	rows := min(convRotBlockRows, r.stored.rows-first)
	columns := r.stored.columns
	quantized := make([]byte, rows*columns)
	if _, err := r.stored.values.ReadAt(quantized, int64((r.stored.first+first)*columns)); err != nil {
		return nil, 0, err
	}
	scales := make([]byte, rows*4)
	if _, err := r.stored.scales.ReadAt(scales, int64((r.stored.first+first)*4)); err != nil {
		return nil, 0, err
	}
	bits := make([]uint16, columns)
	block := make([]byte, rows*columns*2)
	for index := range rows {
		decodeConvRotRow(bits, quantized[index*columns:(index+1)*columns], math.Float32frombits(binary.LittleEndian.Uint32(scales[4*index:])))
		for column, value := range bits {
			binary.LittleEndian.PutUint16(block[(index*columns+uint64(column))*2:], value)
		}
	}
	r.first, r.block = first, block
	return block, first, nil
}
