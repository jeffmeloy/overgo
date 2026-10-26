package reference

import (
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// holdsCells reports whether a node of a cell type is one the reference holds as the values it encodes: a QSA state's
// int8 codes, fp16 scales and integer tables fed as inputs (tensor.QSAState), which the QSA step reads and writes as
// those values (qsaAttention).
func holdsCells(node *tensor.Tensor) bool {
	return node.Op == tensor.OpInput && (node.Type == dtype.I8 || node.Type == dtype.F16 || node.Type == dtype.I32)
}
