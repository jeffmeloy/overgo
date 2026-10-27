package hostexpert

// Expert is one routed expert's work in a layer on the host: its gate
// and up weights, IQ4_XS rows of the hidden width, intermediate rows each;
// its down weights, IQ4_NL rows of the intermediate width, hidden rows each;
// the Q8_K activations of the tokens routed to it; and where each token's
// down product goes, hidden values a token.
type Expert struct {
	Gate, Up, Down []byte
	Activations    [][]byte
	Out            [][]float32
}

// layerSpan is how many rows a task of a layer computes: enough that a
// task's cost outweighs taking it, few enough that a layer's tasks outnumber
// the workers.
const layerSpan = 64

// callerQuantizedRows and callerActivationRows are the most rows the caller quantizes itself rather than hand them to
// the workers: Flash-Next's intermediate rows of 640 values quantize in about 1.8 us each and its activation rows of
// 2,560 in about 3.5 us, against about 2 us to hand a batch over, so a window's one or two of the first and its single
// token's activations are the caller's (a 4-token window's activations on the caller cost a layer 14 us, on the workers
// about 5): the caller keeps the rows whose quantizing costs less than handing them over.
const (
	callerQuantizedRows  = 2
	callerActivationRows = 1
)

// layerScratch holds a layer's intermediate values, each token's gate and up
// products and its activations quantized, grown to the largest layer and kept
// between layers.
type layerScratch struct {
	gates, ups []float32
	quantized  []byte
	// firsts and the rows are the call's bookkeeping, kept so a decode window's layer allocates nothing.
	firsts           []int
	gateRows, upRows [][]float32
	quantizedRows    [][]byte
}

// Layer computes each expert's down products of its tokens, Strata's
// native_expert order: the gate and up rows, activation(gate, up) of each,
// those quantized as Q8_0, and the down rows, each of the three spread across
// the pool's workers, the banks stored as formats says. A pool computes one
// layer at a time.
func (p *Pool) Layer(experts []Expert, formats Formats, hidden, intermediate int, activation func(gate, up float32) float32) error {
	scratch := &p.scratch
	if !Supported(formats, hidden, intermediate) {
		return errFormats
	}
	gateRows, upRows, downRows := rowsOf[formats.Gate], rowsOf[formats.Up], rowsOf[formats.Down]
	firsts := growTo(&scratch.firsts, len(experts)+1)
	firsts[0] = 0
	for index, work := range experts {
		if len(work.Out) != len(work.Activations) {
			return errShape
		}
		firsts[index+1] = firsts[index] + len(work.Activations)
	}
	tokens := firsts[len(experts)]
	quantizedBytes := intermediate / blockValues * q80BlockBytes
	if len(scratch.gates) < tokens*intermediate {
		scratch.gates, scratch.ups = make([]float32, tokens*intermediate), make([]float32, tokens*intermediate)
	}
	if len(scratch.quantized) < tokens*quantizedBytes {
		scratch.quantized = make([]byte, tokens*quantizedBytes)
	}
	gates, ups, quantized := growTo(&scratch.gateRows, tokens), growTo(&scratch.upRows, tokens), growTo(&scratch.quantizedRows, tokens)
	for token := range tokens {
		gates[token] = scratch.gates[token*intermediate : (token+1)*intermediate]
		ups[token] = scratch.ups[token*intermediate : (token+1)*intermediate]
		quantized[token] = scratch.quantized[token*quantizedBytes : (token+1)*quantizedBytes]
	}
	span := p.taskRows(intermediate, len(experts))
	spans := (intermediate + span - 1) / span
	if err := p.run(len(experts)*spans, func(task int) error {
		expert, first := task/spans, task%spans*span
		end := min(first+span, intermediate)
		work := experts[expert]
		gate, up := gates[firsts[expert]:firsts[expert+1]], ups[firsts[expert]:firsts[expert+1]]
		if err := gateRows(work.Gate, hidden, work.Activations, gate, first, end); err != nil {
			return err
		}
		if err := upRows(work.Up, hidden, work.Activations, up, first, end); err != nil {
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
	// A decode window's few rows are quantized by the caller between the phases, as Strata's host quantizes them
	// between its split phases, where handing them to the workers took longer than the rows; a prompt's many are spread
	// across the workers.
	if tokens <= callerQuantizedRows {
		for token := range tokens {
			if err := quantizeQ80(gates[token], quantized[token]); err != nil {
				return err
			}
		}
	} else if err := p.run(tokens, func(token int) error {
		return quantizeQ80(gates[token], quantized[token])
	}); err != nil {
		return err
	}
	span = p.taskRows(hidden, len(experts))
	spans = (hidden + span - 1) / span
	return p.run(len(experts)*spans, func(task int) error {
		expert, first := task/spans, task%spans*span
		work := experts[expert]
		return downRows(work.Down, intermediate, quantized[firsts[expert]:firsts[expert+1]], work.Out, first, min(first+span, hidden))
	})
}

// taskRows is how many of a phase's rows of each expert a task computes:
// layerSpan, or fewer where its tasks would not reach every worker (or the
// pool's tasks a thread), as one decode token's expert's would not.
func (p *Pool) taskRows(rows, experts int) int {
	tasks := p.workers
	if p.tasksPerThread > 0 {
		tasks = p.tasksPerThread * (p.workers + 1)
	}
	return min(layerSpan, max(1, (rows*experts+tasks-1)/tasks))
}

// growTo answers the first n of the slice at s, grown first where it holds fewer.
func growTo[T any](s *[]T, n int) []T {
	if cap(*s) < n {
		*s = make([]T, n)
	}
	return (*s)[:n]
}
