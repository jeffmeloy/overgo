package thoughtbank

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// ArchConfig is the fast-weight-bank LM architecture. Shape-determined fields
// are recoverable from the checkpoint tensor inventory alone; the behavioural
// fields (MaxMem, TopKCSA, TopKExperts, NWin, SinkhornIters, MemSeedSlots) are
// not encoded in any tensor shape and come from the pickled ck["cfg"].
type ArchConfig struct {
	// Shape-determined.
	Vocab       int
	DModel      int
	NLayers     int
	NHeads      int
	DHead       int
	NHC         int
	NGroups     int
	NIdxHeads   int
	DLatentQ    int
	CSAm        int
	HCAm        int
	NExperts    int
	NShared     int
	DFF         int
	MemDim      int
	MemReadRank int
	SwiGLU      bool
	Untied      bool // lm_head.weight present separately (not tied to embed)
	// Behavioural (ck["cfg"] only).
	MaxMem        int
	TopKCSA       int
	TopKExperts   int
	NWin          int
	SinkhornIters int
	MemSeedSlots  int
	NormEps       float64
}

type vectorShape [1]int64
type matrixShape [2]int64

func isMatrixShape(shape []int64) bool { return len(shape) == len(matrixShape{}) }

// DefaultNormEps is the reference RMSNorm epsilon. It is a code constant, not a
// checkpoint field: ck["cfg"] carries no norm_eps because the reference never
// varies it (mhc.RMSNorm eps=1e-6).
const DefaultNormEps = 1e-6

// countLayers returns the number of distinct blocks.N indices in the inventory.
func countLayers(shapes map[string][]int64) int {
	seen := map[int]bool{}
	for name := range shapes {
		if !strings.HasPrefix(name, "blocks.") {
			continue
		}
		rest := name[len("blocks."):]
		dot := strings.IndexByte(rest, '.')
		if dot < 0 {
			continue
		}
		if l, err := strconv.Atoi(rest[:dot]); err == nil {
			seen[l] = true
		}
	}
	return len(seen)
}

// countIndexed returns the count of consecutive blocks.0.<prefix>.<i>.<suffix>
// entries starting at i=0 (experts, shared, out_group families).
func countIndexed(shapes map[string][]int64, prefix, suffix string) int {
	n := 0
	for {
		if _, ok := shapes[fmt.Sprintf("blocks.0.%s.%d.%s", prefix, n, suffix)]; !ok {
			return n
		}
		n++
	}
}

// DeriveArchFromShapes recovers the shape-determined architecture from a tensor
// inventory (state_dict name -> shape). swiGLU comes from ck["cfg"]
// (mem_read_swiglu) because it changes fw_A's leading factor and cannot be read
// from the shape alone. A missing or inconsistent tensor is a hard error: a
// wrong shape yields a forward that runs and means nothing.
func DeriveArchFromShapes(shapes map[string][]int64, swiGLU bool) (ArchConfig, error) {
	var c ArchConfig
	c.SwiGLU = swiGLU

	get := func(name string, rank int) ([]int64, error) {
		s, ok := shapes[name]
		if !ok {
			return nil, fmt.Errorf("thoughtbank config: missing tensor %q", name)
		}
		if len(s) != rank {
			return nil, fmt.Errorf("thoughtbank config: %q rank %d, want %d", name, len(s), rank)
		}
		return s, nil
	}
	getVector := func(name string) ([]int64, error) { return get(name, len(vectorShape{})) }
	getMatrix := func(name string) ([]int64, error) { return get(name, len(matrixShape{})) }

	embed, err := getMatrix("embed.weight")
	if err != nil {
		return c, err
	}
	c.Vocab, c.DModel = int(embed[0]), int(embed[1])
	_, c.Untied = shapes["lm_head.weight"]

	c.NLayers = countLayers(shapes)
	if c.NLayers < 1 {
		return c, fmt.Errorf("thoughtbank config: no blocks.N tensors")
	}

	qn, err := getVector("blocks.0._attn.q_norm.weight")
	if err != nil {
		return c, err
	}
	c.DHead = int(qn[0])

	wuq, err := getMatrix("blocks.0._attn.W_uq.weight")
	if err != nil {
		return c, err
	}
	if wuq[0]%int64(c.DHead) != 0 {
		return c, fmt.Errorf("thoughtbank config: W_uq rows %d not a multiple of d_head %d", wuq[0], c.DHead)
	}
	c.NHeads = int(wuq[0]) / c.DHead
	c.DLatentQ = int(wuq[1])

	if wdq, err := getMatrix("blocks.0._attn.W_dq.weight"); err != nil {
		return c, err
	} else if int(wdq[0]) != c.DLatentQ || int(wdq[1]) != c.DModel {
		return c, fmt.Errorf("thoughtbank config: W_dq shape %v disagrees with d_latent_q=%d d_model=%d", wdq, c.DLatentQ, c.DModel)
	}

	// n_hc from the mHC norm gain over the flattened streams [n_hc*d_model].
	mnorm, err := getVector("blocks.0.mhc_attn.norm.weight")
	if err != nil {
		return c, err
	}
	if mnorm[0]%int64(c.DModel) != 0 {
		return c, fmt.Errorf("thoughtbank config: mhc norm %d not a multiple of d_model %d", mnorm[0], c.DModel)
	}
	c.NHC = int(mnorm[0]) / c.DModel

	c.NGroups = countIndexed(shapes, "_attn.out_group", "weight")
	if c.NGroups < 1 {
		return c, fmt.Errorf("thoughtbank config: no _attn.out_group.N")
	}
	c.NExperts = countIndexed(shapes, "_moe.experts", "w12.weight")
	c.NShared = countIndexed(shapes, "_moe.shared", "w12.weight")
	if c.NExperts < 1 || c.NShared < 1 {
		return c, fmt.Errorf("thoughtbank config: experts=%d shared=%d", c.NExperts, c.NShared)
	}

	// d_ff from an expert's second projection [d_model, d_ff].
	w3, err := getMatrix("blocks.0._moe.experts.0.w3.weight")
	if err != nil {
		return c, err
	}
	c.DFF = int(w3[1])
	if int(w3[0]) != c.DModel {
		return c, fmt.Errorf("thoughtbank config: expert w3 rows %d != d_model %d", w3[0], c.DModel)
	}

	// Fast-weight bank: fw_A [na*rank*d_model, mem_dim], fw_B [d_model*rank, mem_dim].
	na := int64(activationProjectionCount(swiGLU))
	fwA, err := getMatrix("blocks.0.fw_A.weight")
	if err != nil {
		return c, err
	}
	c.MemDim = int(fwA[1])
	denom := na * int64(c.DModel)
	if fwA[0]%denom != 0 {
		return c, fmt.Errorf("thoughtbank config: fw_A rows %d not divisible by na*d_model=%d", fwA[0], denom)
	}
	c.MemReadRank = int(fwA[0] / denom)
	if fwB, err := getMatrix("blocks.0.fw_B.weight"); err != nil {
		return c, err
	} else if int(fwB[0]) != c.DModel*c.MemReadRank || int(fwB[1]) != c.MemDim {
		return c, fmt.Errorf("thoughtbank config: fw_B shape %v disagrees with d_model*rank=%d mem_dim=%d", fwB, c.DModel*c.MemReadRank, c.MemDim)
	}
	if nw, err := getVector("blocks.0.norm_fw.weight"); err == nil && int(nw[0]) != c.DModel {
		return c, fmt.Errorf("thoughtbank config: norm_fw %d != d_model %d", nw[0], c.DModel)
	}

	// Attention block sizes: layer 0 is sparse (CSA) carrying pos_a [csa_m, d_head];
	// the first dense (HCA) layer carries pos [hca_m, d_head].
	if pa, ok := shapes["blocks.0._attn.pos_a"]; ok && isMatrixShape(pa) {
		c.CSAm = int(pa[0])
	} else {
		return c, fmt.Errorf("thoughtbank config: blocks.0._attn.pos_a missing (expected a sparse layer 0)")
	}
	for l := 0; l < c.NLayers; l++ {
		if pos, ok := shapes[fmt.Sprintf("blocks.%d._attn.pos", l)]; ok && isMatrixShape(pos) {
			c.HCAm = int(pos[0])
			break
		}
	}
	if c.HCAm == 0 {
		return c, fmt.Errorf("thoughtbank config: no dense _attn.pos tensor for hca_m")
	}
	// n_idx_heads from W_iq [n_idx_heads*d_head, d_latent_q] on the sparse layer.
	if wiq, ok := shapes["blocks.0._attn.W_iq.weight"]; ok && isMatrixShape(wiq) && wiq[0]%int64(c.DHead) == 0 {
		c.NIdxHeads = int(wiq[0]) / c.DHead
	}

	c.NormEps = DefaultNormEps
	return c, nil
}

// ExpectedTensorNames returns the full state_dict tensor set the forward binds
// for config c — the reference for the inventory-completeness cross-check. Even
// layers are sparse (CSA), odd layers dense (HCA), matching the loader.
func ExpectedTensorNames(c ArchConfig) []string {
	names := []string{
		"embed.weight", "A_out_net.weight", "norm_out.weight",
		"thought_stream.write_ctx_q.weight",
		"thought_stream.write_gate.weight", "thought_stream.write_gate.bias",
		"thought_stream.thought_head.weight", "thought_stream.norm_write.weight",
		"thought_stream.write_decision.weight", "thought_stream.write_decision.bias",
	}
	if c.Untied {
		names = append(names, "lm_head.weight")
	}
	for l := 0; l < c.NLayers; l++ {
		p := func(s string) string { return fmt.Sprintf("blocks.%d.%s", l, s) }
		names = append(names,
			p("_attn.W_dq.weight"), p("_attn.W_uq.weight"),
			p("_attn.W_wk.weight"), p("_attn.W_wv.weight"),
			p("_attn.out_proj.weight"), p("_attn.q_norm.weight"),
			p("_attn.kv_norm.weight"), p("_attn.sink_logits"),
		)
		for g := 0; g < c.NGroups; g++ {
			names = append(names, p(fmt.Sprintf("_attn.out_group.%d.weight", g)))
		}
		if l%2 == 0 { // sparse (CSA)
			names = append(names,
				p("_attn.W_kv_a.weight"), p("_attn.W_kv_b.weight"),
				p("_attn.W_z_a.weight"), p("_attn.W_z_b.weight"),
				p("_attn.pos_a"), p("_attn.pos_b"),
				p("_attn.W_iq.weight"), p("_attn.W_w.weight"),
			)
		} else { // dense (HCA)
			names = append(names, p("_attn.W_kv.weight"), p("_attn.W_z.weight"), p("_attn.pos"))
		}
		names = append(names, p("_moe.W_gate.weight"))
		for e := 0; e < c.NExperts; e++ {
			names = append(names, p(fmt.Sprintf("_moe.experts.%d.w12.weight", e)), p(fmt.Sprintf("_moe.experts.%d.w3.weight", e)))
		}
		for s := 0; s < c.NShared; s++ {
			names = append(names, p(fmt.Sprintf("_moe.shared.%d.w12.weight", s)), p(fmt.Sprintf("_moe.shared.%d.w3.weight", s)))
		}
		names = append(names, p("fw_A.weight"), p("fw_B.weight"), p("fw_o.weight"), p("norm_fw.weight"))
		for _, m := range []string{"mhc_attn", "mhc_moe"} {
			names = append(names,
				p(m+".W_pre.weight"), p(m+".W_res.weight"), p(m+".W_post.weight"),
				p(m+".S_pre"), p(m+".S_res"), p(m+".S_post"),
				p(m+".alpha_pre"), p(m+".alpha_res"), p(m+".alpha_post"),
				p(m+".norm.weight"),
			)
		}
		names = append(names, p("norm_attn.weight"), p("norm_moe.weight"), p("A_cross_net.weight"))
	}
	slices.Sort(names)
	return names
}
