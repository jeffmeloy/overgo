package tensor

import (
	"errors"
	"fmt"
	"slices"

	"overgo/internal/tensor/dtype"
)

// RepeatHeads: broadcasts single head in [width,1,tokens] tensor across
// requested head count without changing token ordering
func (b *Builder) RepeatHeads(input *Tensor, heads uint32) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || input.Type != dtype.F32 || input.Shape.Rank != 3 ||
		input.Shape.Dims[1] != 1 || heads == 0 {
		b.setError(errors.New("RepeatHeads requires rank-3 F32 [width,1,tokens] input and positive heads"))
		return nil
	}
	shape, err := NewShape(input.Shape.Dims[0], uint64(heads), input.Shape.Dims[2])
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", dtype.F32, shape, OpRepeatHeads, []*Tensor{input}, RepeatHeadsAttributes{Heads: heads})
}

// SwiGLU: computes SiLU(gate) * up
func (b *Builder) SwiGLU(gate, up *Tensor) *Tensor {
	return b.Multiply(b.SiLU(gate), up)
}

// GEGLU: computes GELU(gate) * up
func (b *Builder) GEGLU(gate, up *Tensor) *Tensor {
	return b.Multiply(b.GELU(gate), up)
}

// ReGLU: ReLU(gate) * up
func (b *Builder) ReGLU(gate, up *Tensor) *Tensor {
	return b.Multiply(b.ReLU(gate), up)
}

// MulMat: follows ggml semantics; Left has shape [K,M], right has shape [K,N],
// and result has shape [M,N]
func (b *Builder) MulMat(left, right *Tensor) *Tensor {
	result := b.mulMat(left, right)
	if result == nil || left == nil || left.Name == "" {
		return result
	}
	for _, definition := range b.loras[left.Name] {
		if definition.Scale == 0 || definition.Embedding {
			continue
		}
		a := b.loraInput(definition.AName, definition.AShape, definition.AData)
		c := b.loraInput(definition.BName, definition.BShape, definition.BData)
		delta := b.mulMat(c, b.mulMat(a, right))
		result = b.Add(result, b.Scale(delta, definition.Scale))
	}
	return result
}

func (b *Builder) mulMat(left, right *Tensor) *Tensor {
	if b.err != nil {
		return nil
	}
	if left == nil || right == nil {
		b.setError(errors.New("mul_mat input is nil"))
		return nil
	}
	if left.Shape.Rank != 2 || right.Shape.Rank != 2 {
		b.setError(errors.New("initial mul_mat implementation requires rank-2 inputs"))
		return nil
	}
	if left.Shape.Dims[0] != right.Shape.Dims[0] {
		b.setError(fmt.Errorf(
			"mul_mat inner dimensions differ: %d and %d",
			left.Shape.Dims[0],
			right.Shape.Dims[0],
		))
		return nil
	}
	outputType := left.Type
	if (nativeQuantizedType(left.Type) || left.Type == dtype.BF16) && right.Type == dtype.F32 {
		outputType = dtype.F32
	} else if left.Type != right.Type {
		b.setError(fmt.Errorf("mul_mat types are unsupported: %s and %s", left.Type, right.Type))
		return nil
	}
	shape, err := NewShape(left.Shape.Dims[1], right.Shape.Dims[1])
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", outputType, shape, OpMulMat, []*Tensor{left, right}, nil)
}

// GroupedMulMat: per-group ggml matmul; [K,M,G] x [K,G,N] -> [M,G,N]
func (b *Builder) GroupedMulMat(left, right *Tensor) *Tensor {
	result := b.groupedMulMat(left, right)
	if result == nil || left == nil || left.Name == "" {
		return result
	}
	for _, definition := range b.loras[left.Name] {
		if definition.Scale == 0 || definition.Embedding {
			continue
		}
		a := b.loraInput(definition.AName, definition.AShape, definition.AData)
		c := b.loraInput(definition.BName, definition.BShape, definition.BData)
		delta := b.groupedMulMat(c, b.groupedMulMat(a, right))
		result = b.Add(result, b.Scale(delta, definition.Scale))
	}
	return result
}

func (b *Builder) groupedMulMat(left, right *Tensor) *Tensor {
	if b.err != nil {
		return nil
	}
	if left == nil || right == nil || left.Shape.Rank != 3 || right.Shape.Rank != 3 {
		b.setError(errors.New("grouped_mul_mat requires rank-3 inputs"))
		return nil
	}
	if left.Shape.Dims[0] != right.Shape.Dims[0] || left.Shape.Dims[2] != right.Shape.Dims[1] {
		b.setError(errors.New("grouped_mul_mat inner or group dimensions differ"))
		return nil
	}
	outputType := left.Type
	if nativeQuantizedType(left.Type) && right.Type == dtype.F32 {
		outputType = dtype.F32
	} else if left.Type != right.Type {
		b.setError(fmt.Errorf("grouped_mul_mat types are unsupported: %s and %s", left.Type, right.Type))
		return nil
	}
	shape, err := NewShape(left.Shape.Dims[1], left.Shape.Dims[2], right.Shape.Dims[2])
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", outputType, shape, OpGroupedMulMat, []*Tensor{left, right}, nil)
}

// GetRows gathers vocabulary rows from rank-2 table in ggml layout;
// table shape is [embedding, rows] and result is [embedding, len(rows)]
func (b *Builder) GetRows(table *Tensor, rows []uint32) *Tensor {
	result := b.getRows(table, rows)
	if result == nil || table == nil || table.Name == "" {
		return result
	}
	for _, definition := range b.loras[table.Name] {
		if definition.Scale == 0 || !definition.Embedding {
			continue
		}
		a := b.loraInput(definition.AName, definition.AShape, definition.AData)
		c := b.loraInput(definition.BName, definition.BShape, definition.BData)
		delta := b.mulMat(c, b.getRows(a, rows))
		result = b.Add(result, b.Scale(delta, definition.Scale))
	}
	return result
}

func (b *Builder) getRows(table *Tensor, rows []uint32) *Tensor {
	if b.err != nil {
		return nil
	}
	if table == nil {
		b.setError(errors.New("get_rows input is nil"))
		return nil
	}
	if table.Shape.Rank != 2 {
		b.setError(errors.New("get_rows table must have rank 2"))
		return nil
	}
	if len(rows) == 0 {
		b.setError(errors.New("get_rows row list is empty"))
		return nil
	}
	for index, row := range rows {
		if uint64(row) >= table.Shape.Dims[1] {
			b.setError(fmt.Errorf("get_rows row %d at index %d exceeds table size %d", row, index, table.Shape.Dims[1]))
			return nil
		}
	}
	shape, err := NewShape(table.Shape.Dims[0], uint64(len(rows)))
	if err != nil {
		b.setError(err)
		return nil
	}
	attributes := GetRowsAttributes{Rows: slices.Clone(rows)}
	outputType := table.Type
	if nativeQuantizedType(table.Type) {
		outputType = dtype.F32
	}
	return b.add("", outputType, shape, OpGetRows, []*Tensor{table}, attributes)
}

func nativeQuantizedType(value dtype.Type) bool {
	switch value {
	case dtype.Q4_0, dtype.Q4_1, dtype.Q5_0, dtype.Q5_1,
		dtype.Q8_0, dtype.Q8_1, dtype.Q2K, dtype.Q3K, dtype.Q4K, dtype.Q5K, dtype.Q6K, dtype.Q8K,
		dtype.IQ2XXS, dtype.IQ2XS, dtype.IQ2S, dtype.IQ3XXS, dtype.IQ3S, dtype.IQ1S, dtype.IQ1M,
		dtype.IQ4NL, dtype.IQ4XS, dtype.MXFP4, dtype.NVFP4,
		dtype.Q1_0, dtype.Q2_0, dtype.TQ1_0, dtype.TQ2_0:
		return true
	default:
		return false
	}
}
