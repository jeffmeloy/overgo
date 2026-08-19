// Forward chain for tabular in-context prediction, ported behavior written
// fresh: Fourier cell embedding (train rows carry their label embedding),
// induced-set column towers whose inducing queries see only train rows,
// RoPE row towers over [CLS|columns], a second column/row pass keeping the
// CLS summary, then a support-masked ICL encoder and MLP decoder emitting
// one prediction per row (train and test alike).
package tabularicl

import (
	"fmt"
	"math"

	"overgo/internal/hostmath"
)

// Request: labeled prefix (rows [0,TrainRows)) conditions predictions for
// ALL rows; test rows' Y values are read but never influence any output.
type Request struct {
	Task      string    `json:"task"`
	X         []float32 `json:"x"` // row-major [Rows*Cols]
	Y         []float32 `json:"y"` // len Rows; class ids or scalars
	Rows      int       `json:"rows"`
	Cols      int       `json:"cols"`
	TrainRows int       `json:"train_rows"`
	CatCols   []int     `json:"categorical_columns,omitempty"`
}

// ValidateRequest checks task and input geometry before model loading.
func ValidateRequest(req Request) error {
	if !validTask(req.Task) {
		return fmt.Errorf("tabularicl: unsupported task %q", req.Task)
	}
	if _, err := requestCatMask(req.X, req.Y, req.Rows, req.Cols, req.TrainRows, req.CatCols); err != nil {
		return err
	}
	if req.Task == TaskClassification {
		for row := 0; row < req.TrainRows; row++ {
			class := int(req.Y[row])
			if class < 0 || float32(class) != req.Y[row] {
				return fmt.Errorf("tabularicl: train row %d has invalid class %g", row, req.Y[row])
			}
		}
	}
	return nil
}

// Predict dispatches to the task head; returns [Rows*OutDim] raw outputs.
func (m *Model) Predict(req Request) ([]float32, int, error) {
	if err := ValidateRequest(req); err != nil {
		return nil, 0, err
	}
	head, ok := m.Heads[req.Task]
	if !ok {
		return nil, 0, fmt.Errorf("tabularicl: task %q was not loaded", req.Task)
	}
	out, err := head.Predict(req.X, req.Y, req.Rows, req.Cols, req.TrainRows, req.CatCols)
	if err != nil {
		return nil, 0, err
	}
	return out, head.Dims.OutDim, nil
}

// Predict validates the request and runs the forward chain.
func (h *Head) Predict(x, y []float32, rows, cols, trainRows int, catCols []int) ([]float32, error) {
	catMask, err := requestCatMask(x, y, rows, cols, trainRows, catCols)
	if err != nil {
		return nil, err
	}
	if h.Dims.IsClassifier {
		// MaxClasses is an artifact bound.
		for t := 0; t < trainRows; t++ {
			cls := int(y[t])
			if float32(cls) != y[t] {
				return nil, fmt.Errorf("tabularicl: train row %d has invalid class %g", t, y[t])
			}
			if cls < 0 || cls >= h.Dims.MaxClasses {
				return nil, fmt.Errorf("tabularicl: train row %d class %d outside [0,%d)", t, cls, h.Dims.MaxClasses)
			}
		}
	}
	return h.forward(x, y, rows, cols, trainRows, catMask), nil
}

func requestCatMask(x, y []float32, rows, cols, trainRows int, catCols []int) ([]bool, error) {
	if rows <= 0 || cols <= 0 || len(x) != rows*cols {
		return nil, fmt.Errorf("tabularicl: x len %d, want rows*cols %d*%d", len(x), rows, cols)
	}
	if len(y) != rows {
		return nil, fmt.Errorf("tabularicl: y len %d, want rows %d", len(y), rows)
	}
	if trainRows <= 0 || trainRows >= rows {
		return nil, fmt.Errorf("tabularicl: train_rows %d must be in [1, rows-1] (%d rows)", trainRows, rows)
	}
	for index, value := range x {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("tabularicl: x[%d] is non-finite", index)
		}
	}
	for index, value := range y {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("tabularicl: y[%d] is non-finite", index)
		}
	}
	catMask := make([]bool, cols)
	for _, c := range catCols {
		if c < 0 || c >= cols {
			return nil, fmt.Errorf("tabularicl: categorical column %d out of range (cols %d)", c, cols)
		}
		if catMask[c] {
			return nil, fmt.Errorf("tabularicl: categorical column %d is repeated", c)
		}
		catMask[c] = true
	}
	return catMask, nil
}

// forward: the full input->output chain (mirrors the reference program order).
func (h *Head) forward(x, y []float32, rows, cols, trainRows int, catMask []bool) []float32 {
	input := h.decoderInput(x, y, rows, cols, trainRows, catMask)
	return mlp2(input, rows, h.decL0W, h.decL0B, h.decL1W, h.decL1B, h.Dims.NumCLS*h.Dims.EmbedDim)
}

// decoderInput executes the frozen feature program through the final norm.
func (h *Head) decoderInput(x, y []float32, rows, cols, trainRows int, catMask []bool) []float32 {
	d := h.Dims
	e, cls := d.EmbedDim, d.NumCLS
	cellY := h.cellYEmbed(y, rows)
	cell := h.cellEmbed(x, cellY, rows, cols, trainRows, catMask)
	col1 := colTowers(h.col1, h.col1OutW, h.col1OutB, h.col1LN, cell, rows, cols, trainRows, e, d.ColHeads, d.NumInds)
	withCLS := concatCLS(col1, h.clsTokens, rows, cols, cls, e)
	colsTot := cols + cls
	row1 := rowTowers(h.row1, withCLS, rows, colsTot, colsTot, e, d.RowHeads)
	col2 := colTowers(h.col2, h.col2OutW, h.col2OutB, h.col2LN, row1, rows, colsTot, trainRows, e, d.ColHeads, d.NumInds)
	reps := rowTowers(h.row2, col2, rows, colsTot, cls, e, d.RowHeads)
	yEnc := h.yEncode(y, rows, cls*e)
	return iclContext(h.icl, reps, yEnc, rows, trainRows, cls*e, d.ICLHeads)
}

// supportPrefixMask: only the labeled prefix is visible as attention keys.
func supportPrefixMask(totalKeys, supportKeys int) []bool {
	if supportKeys >= totalKeys {
		return nil
	}
	mask := make([]bool, totalKeys)
	for i := 0; i < supportKeys; i++ {
		mask[i] = true
	}
	return mask
}

// mlp2: linear -> tanh GELU -> linear.
func mlp2(x []float32, rows int, l0w, l0b, l1w, l1b []float32, inDim int) []float32 {
	hid := len(l0b)
	mid := make([]float32, rows*hid)
	hostmath.Linear(mid, x, l0w, rows, inDim, hid)
	addBiasRows(mid, l0b, rows, hid)
	hostmath.GELUTanhInPlace(mid)
	out := make([]float32, rows*len(l1b))
	hostmath.Linear(out, mid, l1w, rows, hid, len(l1b))
	addBiasRows(out, l1b, rows, len(l1b))
	return out
}

// addBiasRows broadcasts one bias row over a [rows, d] matrix.
func addBiasRows(v, bias []float32, rows, d int) {
	for r := 0; r < rows; r++ {
		hostmath.AddBias(v[r*d:(r+1)*d], bias)
	}
}

// cellYEmbed: per-row label embedding added to train-row cells. Classifier:
// clamped lookup; regression: scalar MLP2. Computed for every row; only
// train rows consume it.
func (h *Head) cellYEmbed(y []float32, rows int) []float32 {
	e := h.Dims.EmbedDim
	if h.Dims.IsClassifier {
		out := make([]float32, rows*e)
		for row := 0; row < rows; row++ {
			cls := min(max(int(y[row]), 0), h.Dims.MaxClasses-1)
			copy(out[row*e:(row+1)*e], h.cellYLookup[cls*e:(cls+1)*e])
		}
		return out
	}
	return mlp2(y, rows, h.cellYL0W, h.cellYL0B, h.cellYL1W, h.cellYL1B, 1)
}

// cellEmbed: each cell sums feature-group Fourier embeddings of shifted
// source columns ((c + 2^i - 1) mod cols); categorical sources use the cat
// frequency/projection pair; train rows add their label embedding.
func (h *Head) cellEmbed(x, yEmb []float32, rows, cols, trainRows int, catMask []bool) []float32 {
	d := h.Dims
	e, fgs, nf := d.EmbedDim, d.FeatureGroupSize, d.NumFreq
	out := make([]float32, rows*cols*e)
	feats := make([]float32, 2*nf)
	member := make([]float32, e)
	for t := 0; t < rows; t++ {
		for c := 0; c < cols; c++ {
			cell := out[(t*cols+c)*e : (t*cols+c+1)*e]
			for i := 0; i < fgs; i++ {
				src := (c + (1 << i) - 1) % cols
				g := x[t*cols+src]
				freqs, w, b := h.ff, h.wNum, h.bNum
				if catMask[src] {
					freqs, w, b = h.ffCat, h.wCat, h.bCat
				}
				for f := 0; f < nf; f++ {
					arg := g * freqs[i*nf+f] // fp32 multiply, then fp32-rounded trig
					feats[f] = float32(math.Sin(float64(arg)))
					feats[nf+f] = float32(math.Cos(float64(arg)))
				}
				hostmath.Linear(member, feats, w, 1, 2*nf, e)
				hostmath.AddBias(member, b)
				for j := 0; j < e; j++ {
					cell[j] += member[j]
				}
			}
			if t < trainRows {
				for j := 0; j < e; j++ {
					cell[j] += yEmb[t*e+j]
				}
			}
		}
	}
	return out
}

// mabForward: sandwich-norm attention and SwiGLU residuals over q (queries)
// reading kv (keys/values), both normalized with the shared pre-attn norm.
func mabForward(w *mabWeights, q, kv []float32, tq, tk, d, nhead int, keyMask []bool, rope []float64) []float32 {
	hd := d / nhead
	qn := make([]float32, tq*d)
	hostmath.RMSNormInto(qn, q, w.preAttnLN, tq, d, rmsEps)
	kvn := make([]float32, tk*d)
	hostmath.RMSNormInto(kvn, kv, w.preAttnLN, tk, d, rmsEps)

	qp := make([]float32, tq*d)
	hostmath.Linear(qp, qn, w.qw, tq, d, d)
	addBiasRows(qp, w.qb, tq, d)
	kp := make([]float32, tk*d)
	hostmath.Linear(kp, kvn, w.kw, tk, d, d)
	addBiasRows(kp, w.kb, tk, d)
	vp := make([]float32, tk*d)
	hostmath.Linear(vp, kvn, w.vw, tk, d, d)
	addBiasRows(vp, w.vb, tk, d)
	if rope != nil {
		for t := 0; t < tq; t++ {
			for n := 0; n < nhead; n++ {
				hostmath.ApplyRotaryInterleaved(qp[(t*nhead+n)*hd:(t*nhead+n+1)*hd], rope, t)
			}
		}
		for t := 0; t < tk; t++ {
			for n := 0; n < nhead; n++ {
				hostmath.ApplyRotaryInterleaved(kp[(t*nhead+n)*hd:(t*nhead+n+1)*hd], rope, t)
			}
		}
	}
	// Per-head QK norm; the learned softplus query scale folds into q so the
	// attention core runs with scale 1.
	for t := 0; t < tq; t++ {
		for n := 0; n < nhead; n++ {
			slice := qp[t*d+n*hd : t*d+(n+1)*hd]
			hostmath.RMSNormInto(slice, slice, w.qLN, 1, hd, rmsEps)
			for j := 0; j < hd; j++ {
				slice[j] *= w.scale[j]
			}
		}
	}
	for t := 0; t < tk; t++ {
		for n := 0; n < nhead; n++ {
			slice := kp[t*d+n*hd : t*d+(n+1)*hd]
			hostmath.RMSNormInto(slice, slice, w.kLN, 1, hd, rmsEps)
		}
	}
	core := make([]float32, tq*d)
	hostmath.MaskedBidirectionalAttention(core, qp, kp, vp, tq, tk, nhead, nhead, hd, keyMask)
	a := make([]float32, tq*d)
	hostmath.Linear(a, core, w.ow, tq, d, d)
	addBiasRows(a, w.ob, tq, d)
	hostmath.RMSNormInto(a, a, w.postAttnLN, tq, d, rmsEps)
	x := make([]float32, tq*d)
	for i := range x {
		x[i] = q[i] + a[i]
	}

	ff := len(w.l1b)
	xn := make([]float32, tq*d)
	hostmath.RMSNormInto(xn, x, w.preFFLN, tq, d, rmsEps)
	gate := make([]float32, tq*ff)
	hostmath.Linear(gate, xn, w.gateW, tq, d, ff)
	addBiasRows(gate, w.gateB, tq, ff)
	hostmath.SiLUInPlace(gate)
	lin := make([]float32, tq*ff)
	hostmath.Linear(lin, xn, w.l1w, tq, d, ff)
	addBiasRows(lin, w.l1b, tq, ff)
	for i := range gate {
		gate[i] *= lin[i]
	}
	hostmath.Linear(a, gate, w.l2w, tq, ff, d)
	addBiasRows(a, w.l2b, tq, d)
	hostmath.RMSNormInto(a, a, w.postFFLN, tq, d, rmsEps)
	out := make([]float32, tq*d)
	for i := range out {
		out[i] = x[i] + a[i]
	}
	return out
}

// forward: inducing queries read the masked set; rows read the summary.
func (b *isabBlock) forward(src []float32, t, d, nhead, numInds int, keyMask []bool) []float32 {
	hidden := mabForward(b.mab1, b.ind, src, numInds, t, d, nhead, keyMask, nil)
	return mabForward(b.mab2, src, hidden, t, numInds, d, nhead, nil, nil)
}

// colTowers: per column, the ISAB stack over the row set (inducing queries
// see only train rows), then a linear + norm.
func colTowers(blocks []*isabBlock, outW, outB, lnW, x []float32, rows, cols, trainRows, e, nhead, numInds int) []float32 {
	keyMask := supportPrefixMask(rows, trainRows)
	out := make([]float32, rows*cols*e)
	seq := make([]float32, rows*e)
	for c := 0; c < cols; c++ {
		for t := 0; t < rows; t++ {
			copy(seq[t*e:(t+1)*e], x[(t*cols+c)*e:(t*cols+c+1)*e])
		}
		cur := seq
		for _, b := range blocks {
			cur = b.forward(cur, rows, e, nhead, numInds, keyMask)
		}
		projected := make([]float32, rows*e)
		hostmath.Linear(projected, cur, outW, rows, e, e)
		addBiasRows(projected, outB, rows, e)
		hostmath.RMSNormInto(projected, projected, lnW, rows, e, rmsEps)
		for t := 0; t < rows; t++ {
			copy(out[(t*cols+c)*e:(t*cols+c+1)*e], projected[t*e:(t+1)*e])
		}
	}
	return out
}

// forward: encoder self/full-attention stack over one sequence.
func (enc *encoder) forward(seq []float32, t, d, nhead int, keyMask []bool) []float32 {
	cur := seq
	for _, w := range enc.blocks {
		cur = mabForward(w, cur, cur, t, t, d, nhead, keyMask, enc.rope)
	}
	return cur
}

// rowTowers: per row, bidirectional attention over its tokens; keep the
// first tokens and norm them.
func rowTowers(enc *encoder, x []float32, rows, colsTot, keep, e, nhead int) []float32 {
	out := make([]float32, rows*keep*e)
	for t := 0; t < rows; t++ {
		res := enc.forward(x[t*colsTot*e:(t+1)*colsTot*e], colsTot, e, nhead, nil)
		res = res[:keep*e]
		hostmath.RMSNormInto(res, res, enc.outLN, keep, e, rmsEps)
		copy(out[t*keep*e:(t+1)*keep*e], res)
	}
	return out
}

// concatCLS prefixes every row's token sequence with the CLS tokens.
func concatCLS(x, tokens []float32, rows, cols, cls, e int) []float32 {
	tot := cls + cols
	out := make([]float32, rows*tot*e)
	for t := 0; t < rows; t++ {
		copy(out[t*tot*e:t*tot*e+cls*e], tokens)
		copy(out[t*tot*e+cls*e:(t+1)*tot*e], x[t*cols*e:(t+1)*cols*e])
	}
	return out
}

// yEncode: per-row label encoding at ICL width. Classifier: class column of
// the projection (out-of-range rows keep only the bias — test rows'
// placeholders); regression: scalar MLP2.
func (h *Head) yEncode(y []float32, rows, d int) []float32 {
	if h.Dims.IsClassifier {
		out := make([]float32, rows*d)
		for row := 0; row < rows; row++ {
			cls := int(y[row])
			for column := 0; column < d; column++ {
				value := float64(h.iclYProjB[column])
				if cls >= 0 && cls < h.Dims.MaxClasses {
					value += float64(h.iclYProjW[column*h.Dims.MaxClasses+cls])
				}
				out[row*d+column] = float32(value)
			}
		}
		return out
	}
	return mlp2(y, rows, h.iclYL0W, h.iclYL0B, h.iclYL1W, h.iclYL1B, 1)
}

// iclContext: support-conditioned rows after the final norm.
func iclContext(enc *encoder, reps, yEnc []float32, rows, trainRows, d, nhead int) []float32 {
	for row := 0; row < trainRows && row < rows; row++ {
		for j := 0; j < d; j++ {
			reps[row*d+j] += yEnc[row*d+j]
		}
	}
	keyMask := supportPrefixMask(rows, trainRows)
	out := enc.forward(reps, rows, d, nhead, keyMask)
	hostmath.RMSNormInto(out, out, enc.outLN, rows, d, rmsEps)
	return out
}
