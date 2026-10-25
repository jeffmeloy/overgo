package hostexpert

import (
	"errors"
	"slices"

	"overgo/internal/tensor/dtype"
)

// Formats is how an expert layer's banks are stored: its gate and up banks'
// format and its down bank's.
type Formats struct {
	Gate, Up, Down dtype.Type
}

// rowFunction computes rows of one format against a token's activations.
type rowFunction func(weights []byte, depth int, activations [][]byte, out [][]float32, first, end int) error

// rowsOf is each format the pool computes and its rows: the 256-value block
// formats against Q8_K activations, the gate and up banks', and the 32-value
// ones against Q8_0, the down bank's.
var rowsOf = map[dtype.Type]rowFunction{
	dtype.IQ4XS: iq4XSRows,
	dtype.IQ3S:  iq3SRows,
	dtype.IQ4NL: iq4NLRows,
	dtype.Q8_0:  q80Rows,
}

// gateFormats and downFormats are the formats of each position.
var (
	gateFormats = []dtype.Type{dtype.IQ4XS, dtype.IQ3S}
	downFormats = []dtype.Type{dtype.IQ4NL, dtype.Q8_0}
)

var errFormats = errors.New("hostexpert: the layer's formats or widths are none the pool computes")

// Q8KBytes is the Q8_K activations of a token of width values, the
// activations an Expert's gate and up rows read.
func Q8KBytes(width int) int { return width / superBlockValues * q8KBlockBytes }

// Supported reports whether the pool computes an expert layer of formats at
// the hidden and intermediate widths: IQ4_XS or IQ3_S gate and up rows of
// whole 256-value blocks and IQ4_NL or Q8_0 down rows of whole 32-value ones,
// every Flash-Next expert layer.
func Supported(formats Formats, hidden, intermediate int) bool {
	return slices.Contains(gateFormats, formats.Gate) && slices.Contains(gateFormats, formats.Up) && slices.Contains(downFormats, formats.Down) &&
		hidden > 0 && hidden%superBlockValues == 0 && intermediate > 0 && intermediate%blockValues == 0
}

// QuantizeQ8K quantizes each row of values, a token's activations, into the
// same row of blocks as Q8_K across the pool's workers.
func (p *Pool) QuantizeQ8K(values [][]float32, blocks [][]byte) error {
	if len(values) != len(blocks) {
		return errShape
	}
	return p.run(len(values), func(row int) error { return quantizeQ8K(values[row], blocks[row]) })
}
