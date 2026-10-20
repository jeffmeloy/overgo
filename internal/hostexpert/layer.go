package hostexpert

// expertWork is one routed expert's work in a layer on the host: its gate
// and up weights, IQ4_XS rows of the hidden width, intermediate rows each;
// its down weights, IQ4_NL rows of the intermediate width, hidden rows each;
// the Q8_K activations of the tokens routed to it; and where each token's
// down product goes, hidden values a token.
type expertWork struct {
	gate, up, down []byte
	activations    [][]byte
	out            [][]float32
}

// layerSpan is how many rows a task of a layer computes: enough that a
// task's cost outweighs taking it, few enough that a layer's tasks outnumber
// the workers.
const layerSpan = 64

// layerScratch holds a layer's intermediate values, each token's gate and up
// products and its activations quantized, grown to the largest layer and kept
// between layers.
type layerScratch struct {
	gates, ups []float32
	quantized  []byte
}

// layer computes each expert's down products of its tokens, Strata's
// native_expert order: the gate and up rows, activation(gate, up) of each,
// those quantized as Q8_0, and the down rows, each of the three spread across
// the pool's workers. A pool computes one layer at a time.
func (p *pool) layer(experts []expertWork, hidden, intermediate int, activation func(gate, up float32) float32,
	scratch *layerScratch) error {
	if intermediate%blockValues != 0 {
		return errDepth(intermediate, blockValues)
	}
	firsts := make([]int, len(experts)+1)
	for index, work := range experts {
		if len(work.out) != len(work.activations) {
			return errShape
		}
		firsts[index+1] = firsts[index] + len(work.activations)
	}
	tokens := firsts[len(experts)]
	quantizedBytes := intermediate / blockValues * q80BlockBytes
	if len(scratch.gates) < tokens*intermediate {
		scratch.gates, scratch.ups = make([]float32, tokens*intermediate), make([]float32, tokens*intermediate)
	}
	if len(scratch.quantized) < tokens*quantizedBytes {
		scratch.quantized = make([]byte, tokens*quantizedBytes)
	}
	gates, ups, quantized := make([][]float32, tokens), make([][]float32, tokens), make([][]byte, tokens)
	for token := range tokens {
		gates[token] = scratch.gates[token*intermediate : (token+1)*intermediate]
		ups[token] = scratch.ups[token*intermediate : (token+1)*intermediate]
		quantized[token] = scratch.quantized[token*quantizedBytes : (token+1)*quantizedBytes]
	}
	spans := (intermediate + layerSpan - 1) / layerSpan
	if err := p.run(len(experts)*spans, func(task int) error {
		expert, first := task/spans, task%spans*layerSpan
		end := min(first+layerSpan, intermediate)
		work := experts[expert]
		gate, up := gates[firsts[expert]:firsts[expert+1]], ups[firsts[expert]:firsts[expert+1]]
		if err := iq4XSRows(work.gate, hidden, work.activations, gate, first, end); err != nil {
			return err
		}
		if err := iq4XSRows(work.up, hidden, work.activations, up, first, end); err != nil {
			return err
		}
		for token := range gate {
			for row := first; row < end; row++ {
				gate[token][row] = activation(gate[token][row], up[token][row])
			}
		}
		return nil
	}); err != nil {
		return err
	}
	if err := p.run(tokens, func(token int) error {
		return quantizeQ80(gates[token], quantized[token])
	}); err != nil {
		return err
	}
	spans = (hidden + layerSpan - 1) / layerSpan
	return p.run(len(experts)*spans, func(task int) error {
		expert, first := task/spans, task%spans*layerSpan
		work := experts[expert]
		return iq4NLRows(work.down, intermediate, quantized[firsts[expert]:firsts[expert+1]], work.out, first, min(first+layerSpan, hidden))
	})
}
