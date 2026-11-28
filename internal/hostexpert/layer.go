package hostexpert

// Expert is one routed expert's work in a layer on the host: its gate
// and up weights, IQ4_XS rows of the hidden width, intermediate rows each;
// its down weights, IQ4_NL rows of the intermediate width, hidden rows each;
// the Q8_K activations of the tokens routed to it, each split in two
// (quantizeSplitQ8K); and where each token's
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
	// downs are the second halves' down products (quantizeSplitQ8K), which a token's own output sums.
	downs []float32
	// firsts and the rows are the call's bookkeeping, kept so a decode window's layer allocates nothing.
	firsts           []int
	gateRows, upRows [][]float32
	quantizedRows    [][]byte
	partRows         [][]byte
	downRows         [][]float32
}

// Layer computes each expert's down products of its tokens, Strata's native_expert order with each quantized row split
// in two (quantizeSplitQ8K): the gate and up rows against both halves of each token's activations, summed,
// activation(gate, up) of each, those split as Q8_0, and the down rows against both halves, summed, each of the three
// spread across the pool's workers, the banks stored as formats says. A row's product with the two halves stands about
// 250 times nearer its product with the values than with the first half alone, as near as the card's: with the first
// alone, ggml's CPU arithmetic, Flash-Next's host experts stood sixty times the card's distance from the f32 reference
// and flipped an exact projection answer with the card's free memory (flashnext-projection-exact). Each weight row is
// read once for both halves. A pool computes one layer at a time.
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
	rows := splitParts * tokens
	activationBytes, quantizedBytes := Q8KBytes(hidden)/splitParts, intermediate/blockValues*q80BlockBytes
	if len(scratch.gates) < rows*intermediate {
		scratch.gates, scratch.ups = make([]float32, rows*intermediate), make([]float32, rows*intermediate)
	}
	if len(scratch.quantized) < rows*quantizedBytes {
		scratch.quantized = make([]byte, rows*quantizedBytes)
	}
	if len(scratch.downs) < tokens*hidden {
		scratch.downs = make([]float32, tokens*hidden)
	}
	gates, ups, quantized := growTo(&scratch.gateRows, rows), growTo(&scratch.upRows, rows), growTo(&scratch.quantizedRows, rows)
	parts, downs := growTo(&scratch.partRows, rows), growTo(&scratch.downRows, rows)
	for row := range rows {
		gates[row] = scratch.gates[row*intermediate : (row+1)*intermediate]
		ups[row] = scratch.ups[row*intermediate : (row+1)*intermediate]
		quantized[row] = scratch.quantized[row*quantizedBytes : (row+1)*quantizedBytes]
	}
	// A token's rows are its two halves, the values' first: its activations' and, of its down products, its own
	// output's and the scratch the second half's go to.
	for index, work := range experts {
		for token, activations := range work.Activations {
			if len(activations) != splitParts*activationBytes {
				return errShape
			}
			at := firsts[index] + token
			parts[splitParts*at], parts[splitParts*at+1] = activations[:activationBytes], activations[activationBytes:]
			downs[splitParts*at], downs[splitParts*at+1] = work.Out[token], scratch.downs[at*hidden:(at+1)*hidden]
		}
	}
	span := p.taskRows(intermediate, len(experts))
	spans := (intermediate + span - 1) / span
	if err := p.run(len(experts)*spans, func(task int) error {
		expert, first := task/spans, task%spans*span
		end := min(first+span, intermediate)
		work := experts[expert]
		from, to := splitParts*firsts[expert], splitParts*firsts[expert+1]
		gate, up := gates[from:to], ups[from:to]
		if err := gateRows(work.Gate, hidden, parts[from:to], gate, first, end); err != nil {
			return err
		}
		if err := upRows(work.Up, hidden, parts[from:to], up, first, end); err != nil {
			return err
		}
		for token := 0; token < len(gate); token += splitParts {
			for row := first; row < end; row++ {
				gate[token][row] = activation(gate[token][row]+gate[token+1][row], up[token][row]+up[token+1][row])
			}
		}
		return nil
	}); err != nil {
		return err
	}
	// A decode window's few rows are quantized by the caller between the phases, as Strata's host quantizes them
	// between its split phases, where handing them to the workers took longer than the rows; a prompt's many are spread
	// across the workers.
	quantize := func(token int) error {
		at := splitParts * token
		return quantizeSplitQ80(gates[at], scratch.quantized[at*quantizedBytes:(at+splitParts)*quantizedBytes])
	}
	if tokens <= callerQuantizedRows {
		for token := range tokens {
			if err := quantize(token); err != nil {
				return err
			}
		}
	} else if err := p.run(tokens, quantize); err != nil {
		return err
	}
	span = p.taskRows(hidden, len(experts))
	spans = (hidden + span - 1) / span
	return p.run(len(experts)*spans, func(task int) error {
		expert, first := task/spans, task%spans*span
		end := min(first+span, hidden)
		from, to := splitParts*firsts[expert], splitParts*firsts[expert+1]
		out := downs[from:to]
		if err := downRows(experts[expert].Down, intermediate, quantized[from:to], out, first, end); err != nil {
			return err
		}
		for token := 0; token < len(out); token += splitParts {
			for row := first; row < end; row++ {
				out[token][row] += out[token+1][row]
			}
		}
		return nil
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
