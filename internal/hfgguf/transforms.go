package hfgguf

import "fmt"

// TensorTransform describes the same payload operation the converter executes.
// Unknown operations are refused rather than treated as identity transforms.
type TensorTransform struct {
	Operation string `json:"operation"`
	Groups    uint64 `json:"groups,omitzero"`
	Heads     uint64 `json:"heads,omitzero"`
	Width     uint64 `json:"width,omitzero"`
	Rows      uint64 `json:"rows,omitzero"`
	Start     uint64 `json:"start,omitzero"`
	Block     uint64 `json:"block,omitzero"`
}

const (
	transformAddOne      = "add-one-f32"
	transformNegateExp   = "negative-exp-f32"
	transformHeadRows    = "grouped-to-cyclic-head-rows"
	transformHeadColumns = "grouped-to-cyclic-head-columns"
)

func (t TensorTransform) apply(data []byte, elementSize uint64) error {
	switch t.Operation {
	case transformAddOne:
		return addOneF32(data, elementSize)
	case transformNegateExp:
		return negateExpF32(data, elementSize)
	case transformHeadRows, transformHeadColumns:
		if t.Groups == 0 || t.Heads == 0 || t.Heads%t.Groups != 0 || t.Width == 0 {
			return fmt.Errorf("HF/GGUF adapter: invalid head transform %+v", t)
		}
		dims := qwen35LinearDims{keyHeads: t.Groups, valueHeads: t.Heads, valueWidth: t.Width}
		if t.Operation == transformHeadRows {
			return dims.reorderVRows(data, elementSize, t.Rows, t.Start, t.Block)
		}
		return dims.reorderVColumns(data, elementSize)
	default:
		return fmt.Errorf("HF/GGUF adapter: unknown transform %q", t.Operation)
	}
}
