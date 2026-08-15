// Package tabularicl owns tabular in-context-learning prediction (ladder
// rung 2): a set transformer over a labeled row prefix predicting every row.
// A capability port from adaptive_new expressed against overgo's owners —
// behavior and evidence carried, no substrate copied; parity is judged
// against the imported /api/predict golden.
//
// Every dimension is DERIVED from the artifact's own tensor shapes; the
// artifact's config.json restates only derivable facts and is not read.
package tabularicl

import (
	"fmt"
	"io"
	"math"
	"path/filepath"
	"slices"

	"overgo/internal/hostmath"
	"overgo/internal/safetensors"
	"overgo/internal/tensorcatalog"
)

const (
	TaskClassification = "classification"
	TaskRegression     = "regression"
)

var taskNames = [...]string{TaskClassification, TaskRegression}

// Tasks returns the supported head names.
func Tasks() []string { return slices.Clone(taskNames[:]) }

// rmsEps: the one carried execution fact. Not recorded anywhere in the
// artifact (config.json or tensors); vendor source
// models/tabfm-1.0.0-pytorch/tabfm/tabfm/src/pytorch/model.py fixes RMSNorm
// eps, mirrored by the adaptive_new repodb fact model.rms_norm_eps.
const rmsEps = 1e-06

// Dims: head geometry, derived from tensor shapes.
type Dims struct {
	EmbedDim         int // cell_embedder.in_linear.weight rows
	NumFreq          int // cell_embedder.fourier_frequencies cols
	FeatureGroupSize int // cell_embedder.fourier_frequencies rows
	NumCLS           int // cls_tokens rows
	NumInds          int // col_embedder ind_vectors rows
	ColHeads         int // EmbedDim / len(col per_dim_scale)
	RowHeads         int // EmbedDim / len(row per_dim_scale)
	ICLHeads         int // NumCLS*EmbedDim / len(icl per_dim_scale)
	ColBlocks        int // col_embedder blocks enumeration
	RowBlocks        int // row_interactor blocks enumeration
	ICLBlocks        int // icl_predictor blocks enumeration
	MaxClasses       int // y_embedder_lookup rows (classifier only)
	OutDim           int // icl_predictor.decoder.layers.1.weight rows
	IsClassifier     bool
}

// mabWeights: one multihead attention block (sandwich-norm + SwiGLU).
type mabWeights struct {
	qw, qb, kw, kb, vw, vb, ow, ob           []float32
	qLN, kLN, scale                          []float32 // scale: softplus per-dim query scale, precomputed
	preAttnLN, postAttnLN, preFFLN, postFFLN []float32
	l1w, l1b, gateW, gateB, l2w, l2b         []float32
}

// isabBlock: induced-set attention — inducing queries read the masked set,
// then rows read the inducing summary.
type isabBlock struct {
	ind        []float32
	mab1, mab2 *mabWeights
}

// encoder: a stack of self/cross MAB blocks with optional checkpoint RoPE
// frequencies and a trailing norm.
type encoder struct {
	blocks []*mabWeights
	rope   []float64 // interleaved-pair inverse frequencies; nil = no RoPE
	outLN  []float32
}

// Head: one compiled task head.
type Head struct {
	Dims Dims

	// Cell embedder.
	ff, ffCat, wNum, bNum, wCat, bCat []float32
	cellYLookup                       []float32 // classifier: [MaxClasses, EmbedDim]
	cellYL0W, cellYL0B                []float32 // regression MLP2
	cellYL1W, cellYL1B                []float32

	// ICL y encoder.
	iclYProjW, iclYProjB []float32 // classifier: [NumCLS*EmbedDim, MaxClasses]
	iclYL0W, iclYL0B     []float32 // regression MLP2
	iclYL1W, iclYL1B     []float32

	// Column towers.
	col1, col2                     []*isabBlock
	col1OutW, col1OutB, col1LN     []float32
	col2OutW, col2OutB, col2LN     []float32
	clsTokens                      []float32
	row1, row2, icl                *encoder
	decL0W, decL0B, decL1W, decL1B []float32
}

// Model: both task heads, dispatched by task name.
type Model struct {
	Heads map[string]*Head
}

// Load reads both sub-model heads under directory (classification/ and
// regression/, each config.json + model.safetensors).
func Load(directory string) (*Model, error) {
	m := &Model{Heads: make(map[string]*Head, len(taskNames))}
	for _, task := range taskNames {
		head, err := LoadHead(filepath.Join(directory, task))
		if err != nil {
			return nil, fmt.Errorf("tabularicl %s head: %w", task, err)
		}
		m.Heads[task] = head
	}
	return m, nil
}

// LoadTask materializes only the requested task head.
func LoadTask(directory, task string) (*Model, error) {
	if !validTask(task) {
		return nil, fmt.Errorf("tabularicl: unsupported task %q", task)
	}
	head, err := LoadHead(filepath.Join(directory, task))
	if err != nil {
		return nil, fmt.Errorf("tabularicl %s head: %w", task, err)
	}
	return &Model{Heads: map[string]*Head{task: head}}, nil
}

func validTask(task string) bool {
	return slices.Contains(taskNames[:], task)
}

// LoadHead reads one task head and materializes every tensor as f32.
func LoadHead(headDir string) (*Head, error) {
	source, err := safetensors.OpenSource(headDir)
	if err != nil {
		return nil, err
	}
	defer source.Close()
	shapes := make(map[string][]int, len(source.Tensors))
	weights := make(map[string][]float32, len(source.Tensors))
	for name, tensor := range source.Tensors {
		// All-F32 artifact contract (mirrors the reference admission check).
		if tensor.DType != "F32" {
			return nil, fmt.Errorf("tabularicl: tensor %q dtype %s violates the all-F32 contract", name, tensor.DType)
		}
		dims := make([]int, len(tensor.Shape))
		for i, dim := range tensor.Shape {
			if dim == 0 || dim > 1<<31 {
				return nil, fmt.Errorf("tabularicl: tensor %q dimension %d out of range", name, dim)
			}
			dims[i] = int(dim)
		}
		shapes[name] = dims
		reader, err := safetensors.F32Reader(tensor)
		if err != nil {
			return nil, fmt.Errorf("tabularicl: tensor %q: %w", name, err)
		}
		elements := tensor.Elements()
		raw := make([]byte, elements*4)
		if _, err := io.ReadFull(reader, raw); err != nil {
			return nil, fmt.Errorf("tabularicl: tensor %q payload: %w", name, err)
		}
		values := make([]float32, elements)
		for i := range values {
			bits := uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 | uint32(raw[4*i+3])<<24
			values[i] = math.Float32frombits(bits)
		}
		weights[name] = values
	}
	loader := &headLoader{weights: weights, shapes: shapes}
	head, err := loader.compile()
	if err != nil {
		return nil, fmt.Errorf("tabularicl %s: %w", headDir, err)
	}
	return head, nil
}

type headLoader struct {
	weights map[string][]float32
	shapes  map[string][]int
	err     error
}

// tensor fetches by name, recording the first missing tensor.
func (l *headLoader) tensor(name string) []float32 {
	values, ok := l.weights[name]
	if !ok && l.err == nil {
		l.err = fmt.Errorf("missing tensor %q", name)
	}
	return values
}

// deriveDims reads every geometric fact from tensor shapes.
func (l *headLoader) deriveDims() (Dims, error) {
	var d Dims
	inLinear, err := tensorcatalog.Shape(l.shapes, "cell_embedder.in_linear.weight", 2)
	if err != nil {
		return d, err
	}
	if inLinear[1]%2 != 0 {
		return d, fmt.Errorf("in_linear input width %d is not sin+cos pairs", inLinear[1])
	}
	d.EmbedDim, d.NumFreq = inLinear[0], inLinear[1]/2
	fourier, err := tensorcatalog.Shape(l.shapes, "cell_embedder.fourier_frequencies", 2)
	if err != nil {
		return d, err
	}
	if fourier[1] != d.NumFreq {
		return d, fmt.Errorf("fourier freq cols %d != in_linear pairs %d", fourier[1], d.NumFreq)
	}
	d.FeatureGroupSize = fourier[0]
	cls, err := tensorcatalog.Shape(l.shapes, "cls_tokens", 2)
	if err != nil {
		return d, err
	}
	if cls[1] != d.EmbedDim {
		return d, fmt.Errorf("cls_tokens width %d != embed %d", cls[1], d.EmbedDim)
	}
	d.NumCLS = cls[0]
	ind, err := tensorcatalog.Shape(l.shapes, "col_embedder.tf_col.blocks.0.ind_vectors", 2)
	if err != nil {
		return d, err
	}
	if ind[1] != d.EmbedDim {
		return d, fmt.Errorf("ind_vectors width %d != embed %d", ind[1], d.EmbedDim)
	}
	d.NumInds = ind[0]

	// Head counts: per-dim scale length IS the head dim of its tower.
	headsOf := func(name string, width int) (int, error) {
		pds, err := tensorcatalog.Shape(l.shapes, name, 1)
		if err != nil {
			return 0, err
		}
		if pds[0] <= 0 || width%pds[0] != 0 {
			return 0, fmt.Errorf("%s length %d does not divide width %d", name, pds[0], width)
		}
		return width / pds[0], nil
	}
	if d.ColHeads, err = headsOf("col_embedder.tf_col.blocks.0.mab1.attn.per_dim_scale", d.EmbedDim); err != nil {
		return d, err
	}
	if d.RowHeads, err = headsOf("row_interactor.tf_row.blocks.0.attn.per_dim_scale", d.EmbedDim); err != nil {
		return d, err
	}
	if d.ICLHeads, err = headsOf("icl_predictor.tf_icl.blocks.0.attn.per_dim_scale", d.NumCLS*d.EmbedDim); err != nil {
		return d, err
	}
	if d.ColBlocks, err = tensorcatalog.IndexedCount(l.shapes, "col_embedder.tf_col.blocks.", ".ind_vectors"); err != nil {
		return d, err
	}
	if d.RowBlocks, err = tensorcatalog.IndexedCount(l.shapes, "row_interactor.tf_row.blocks.", ".attn.per_dim_scale"); err != nil {
		return d, err
	}
	if d.ICLBlocks, err = tensorcatalog.IndexedCount(l.shapes, "icl_predictor.tf_icl.blocks.", ".attn.per_dim_scale"); err != nil {
		return d, err
	}
	// Second towers must mirror the first (the forward reuses one count).
	if n, err := tensorcatalog.IndexedCount(l.shapes, "col_embedder_2.tf_col.blocks.", ".ind_vectors"); err != nil || n != d.ColBlocks {
		return d, fmt.Errorf("col_embedder_2 blocks %d/%v != col_embedder blocks %d", n, err, d.ColBlocks)
	}
	if n, err := tensorcatalog.IndexedCount(l.shapes, "row_interactor_2.tf_row.blocks.", ".attn.per_dim_scale"); err != nil || n != d.RowBlocks {
		return d, fmt.Errorf("row_interactor_2 blocks %d/%v != row_interactor blocks %d", n, err, d.RowBlocks)
	}
	// Task kind: the classifier ships a class-projection y encoder; the
	// regressor ships a scalar MLP2 y encoder.
	if _, ok := l.shapes["icl_predictor.y_encoder.projection.weight"]; ok {
		d.IsClassifier = true
		lookup, err := tensorcatalog.Shape(l.shapes, "cell_embedder.y_embedder_lookup.weight", 2)
		if err != nil {
			return d, err
		}
		d.MaxClasses = lookup[0]
		projection, err := tensorcatalog.Shape(l.shapes, "icl_predictor.y_encoder.projection.weight", 2)
		if err != nil {
			return d, err
		}
		if projection[1] != d.MaxClasses || projection[0] != d.NumCLS*d.EmbedDim {
			return d, fmt.Errorf("y projection shape %v != [%d %d]", projection, d.NumCLS*d.EmbedDim, d.MaxClasses)
		}
	}
	decoder, err := tensorcatalog.Shape(l.shapes, "icl_predictor.decoder.layers.1.weight", 2)
	if err != nil {
		return d, err
	}
	d.OutDim = decoder[0]
	return d, nil
}

func (l *headLoader) loadMAB(prefix string, headDim int) *mabWeights {
	w := &mabWeights{
		qw: l.tensor(prefix + ".attn.q_proj.weight"), qb: l.tensor(prefix + ".attn.q_proj.bias"),
		kw: l.tensor(prefix + ".attn.k_proj.weight"), kb: l.tensor(prefix + ".attn.k_proj.bias"),
		vw: l.tensor(prefix + ".attn.v_proj.weight"), vb: l.tensor(prefix + ".attn.v_proj.bias"),
		ow: l.tensor(prefix + ".attn.out_proj.weight"), ob: l.tensor(prefix + ".attn.out_proj.bias"),
		qLN:       l.tensor(prefix + ".attn.query_ln.weight"),
		kLN:       l.tensor(prefix + ".attn.key_ln.weight"),
		preAttnLN: l.tensor(prefix + ".pre_attn_ln.weight"), postAttnLN: l.tensor(prefix + ".post_attn_ln.weight"),
		preFFLN: l.tensor(prefix + ".pre_ff_ln.weight"), postFFLN: l.tensor(prefix + ".post_ff_ln.weight"),
		l1w: l.tensor(prefix + ".linear1.weight"), l1b: l.tensor(prefix + ".linear1.bias"),
		gateW: l.tensor(prefix + ".linear1_gate.weight"), gateB: l.tensor(prefix + ".linear1_gate.bias"),
		l2w: l.tensor(prefix + ".linear2.weight"), l2b: l.tensor(prefix + ".linear2.bias"),
	}
	// Learned per-dim query scale: log2(e)/sqrt(headDim) * softplus(raw).
	raw := l.tensor(prefix + ".attn.per_dim_scale")
	if l.err != nil {
		return w
	}
	factor := math.Log2E / math.Sqrt(float64(headDim))
	w.scale = make([]float32, headDim)
	for i := range w.scale {
		w.scale[i] = float32(factor * hostmath.Softplus(float64(raw[i])))
	}
	return w
}

func (l *headLoader) loadISAB(prefix string, headDim int) *isabBlock {
	return &isabBlock{
		ind:  l.tensor(prefix + ".ind_vectors"),
		mab1: l.loadMAB(prefix+".mab1", headDim),
		mab2: l.loadMAB(prefix+".mab2", headDim),
	}
}

func (l *headLoader) loadEncoder(prefix string, blocks, headDim int, ropeName, outLNName string) *encoder {
	enc := &encoder{blocks: make([]*mabWeights, blocks), outLN: l.tensor(outLNName)}
	for i := range enc.blocks {
		enc.blocks[i] = l.loadMAB(fmt.Sprintf("%s.blocks.%d", prefix, i), headDim)
	}
	if ropeName != "" {
		freqs := l.tensor(ropeName)
		if l.err == nil {
			if len(freqs) < headDim/2 {
				l.err = fmt.Errorf("rope freqs %q length %d < head_dim/2 %d", ropeName, len(freqs), headDim/2)
				return enc
			}
			enc.rope = make([]float64, len(freqs))
			for i, f := range freqs {
				enc.rope[i] = float64(f)
			}
		}
	}
	return enc
}

func (l *headLoader) compile() (*Head, error) {
	dims, err := l.deriveDims()
	if err != nil {
		return nil, err
	}
	h := &Head{Dims: dims}
	h.ff = l.tensor("cell_embedder.fourier_frequencies")
	h.ffCat = l.tensor("cell_embedder.fourier_frequencies_cat")
	h.wNum = l.tensor("cell_embedder.in_linear.weight")
	h.bNum = l.tensor("cell_embedder.in_linear.bias")
	h.wCat = l.tensor("cell_embedder.in_linear_cat.weight")
	h.bCat = l.tensor("cell_embedder.in_linear_cat.bias")
	if dims.IsClassifier {
		h.cellYLookup = l.tensor("cell_embedder.y_embedder_lookup.weight")
		h.iclYProjW = l.tensor("icl_predictor.y_encoder.projection.weight")
		h.iclYProjB = l.tensor("icl_predictor.y_encoder.projection.bias")
	} else {
		h.cellYL0W = l.tensor("cell_embedder.y_embedder_lookup.layers.0.weight")
		h.cellYL0B = l.tensor("cell_embedder.y_embedder_lookup.layers.0.bias")
		h.cellYL1W = l.tensor("cell_embedder.y_embedder_lookup.layers.1.weight")
		h.cellYL1B = l.tensor("cell_embedder.y_embedder_lookup.layers.1.bias")
		h.iclYL0W = l.tensor("icl_predictor.y_encoder.layers.0.weight")
		h.iclYL0B = l.tensor("icl_predictor.y_encoder.layers.0.bias")
		h.iclYL1W = l.tensor("icl_predictor.y_encoder.layers.1.weight")
		h.iclYL1B = l.tensor("icl_predictor.y_encoder.layers.1.bias")
	}
	colHeadDim := dims.EmbedDim / dims.ColHeads
	h.col1 = make([]*isabBlock, dims.ColBlocks)
	h.col2 = make([]*isabBlock, dims.ColBlocks)
	for i := range h.col1 {
		h.col1[i] = l.loadISAB(fmt.Sprintf("col_embedder.tf_col.blocks.%d", i), colHeadDim)
		h.col2[i] = l.loadISAB(fmt.Sprintf("col_embedder_2.tf_col.blocks.%d", i), colHeadDim)
		for _, block := range []*isabBlock{h.col1[i], h.col2[i]} {
			if l.err == nil && len(block.ind) != dims.NumInds*dims.EmbedDim {
				l.err = fmt.Errorf("isab block %d ind_vectors len %d != %d", i, len(block.ind), dims.NumInds*dims.EmbedDim)
			}
		}
	}
	h.col1OutW = l.tensor("col_embedder.out_w.weight")
	h.col1OutB = l.tensor("col_embedder.out_w.bias")
	h.col1LN = l.tensor("col_embedder.ln_w.weight")
	h.col2OutW = l.tensor("col_embedder_2.out_w.weight")
	h.col2OutB = l.tensor("col_embedder_2.out_w.bias")
	h.col2LN = l.tensor("col_embedder_2.ln_w.weight")
	h.clsTokens = l.tensor("cls_tokens")
	rowHeadDim := dims.EmbedDim / dims.RowHeads
	h.row1 = l.loadEncoder("row_interactor.tf_row", dims.RowBlocks, rowHeadDim,
		"row_interactor.tf_row.rope.freqs", "row_interactor.out_ln.weight")
	h.row2 = l.loadEncoder("row_interactor_2.tf_row", dims.RowBlocks, rowHeadDim,
		"row_interactor_2.tf_row.rope.freqs", "row_interactor_2.out_ln.weight")
	iclHeadDim := dims.NumCLS * dims.EmbedDim / dims.ICLHeads
	h.icl = l.loadEncoder("icl_predictor.tf_icl", dims.ICLBlocks, iclHeadDim,
		"", "icl_predictor.ln.weight")
	h.decL0W = l.tensor("icl_predictor.decoder.layers.0.weight")
	h.decL0B = l.tensor("icl_predictor.decoder.layers.0.bias")
	h.decL1W = l.tensor("icl_predictor.decoder.layers.1.weight")
	h.decL1B = l.tensor("icl_predictor.decoder.layers.1.bias")
	if l.err != nil {
		return nil, l.err
	}
	return h, nil
}
