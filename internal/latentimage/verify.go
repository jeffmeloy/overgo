package latentimage

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"overgo/internal/safetensors"
)

// Check: one structural-oracle assertion. Want is the config/derived value,
// Got is the value read from a real tensor shape (or a sibling tensor), Source
// names the tensor(s). Equal reports whether the two agree.
type Check struct {
	Name   string
	Want   int
	Got    int
	Source string
}

// Equal reports agreement.
func (c Check) Equal() bool { return c.Want == c.Got }

// String renders one line: NAME want=.. got=.. [tensor].
func (c Check) String() string {
	mark := "OK"
	if !c.Equal() {
		mark = "MISMATCH"
	}
	return fmt.Sprintf("%-8s %-34s want=%-8d got=%-8d  %s", mark, c.Name, c.Want, c.Got, c.Source)
}

// Witness: the full structural oracle -- every derived dim vs the real
// checkpoint, plus derived-from-tensor fields folded back into the Spec.
type Witness struct {
	Checks   []Check
	ModField int // ModFields derived from time_mod_proj.weight (folded into Spec)
}

// Failed reports whether any check disagreed.
func (w Witness) Failed() bool {
	for _, c := range w.Checks {
		if !c.Equal() {
			return true
		}
	}
	return false
}

// VerifyCheckpoint opens the three sub-model safetensors HEADERS (never any
// payload) under dir and asserts every Spec dim against the real tensor shapes,
// deriving each dim FROM a shape and cross-checking it against the config Spec.
// It also derives ModFields from time_mod_proj.weight and folds it into the
// Spec. Returns the Witness (for verbatim reporting) and an error if any check
// disagrees or a required tensor is missing.
func (s *Spec) VerifyCheckpoint(dir string) (*Witness, error) {
	tsrc, err := safetensors.OpenSource(dir + "/transformer")
	if err != nil {
		return nil, fmt.Errorf("open transformer: %w", err)
	}
	defer tsrc.Close()
	vsrc, err := safetensors.OpenSource(dir + "/vae")
	if err != nil {
		return nil, fmt.Errorf("open vae: %w", err)
	}
	defer vsrc.Close()
	esrc, err := safetensors.OpenSource(dir + "/text_encoder")
	if err != nil {
		return nil, fmt.Errorf("open text_encoder: %w", err)
	}
	defer esrc.Close()

	w := &Witness{}
	add := func(name string, want, got int, source string) {
		w.Checks = append(w.Checks, Check{Name: name, Want: want, Got: got, Source: source})
	}
	// dim0/dim1 read a tensor's shape[i]; a missing tensor or short rank is a
	// hard error (the checkpoint does not match the recognized arch).
	dim := func(src *safetensors.Source, name string, axis int) (int, error) {
		t, ok := src.Tensors[name]
		if !ok {
			return 0, fmt.Errorf("missing tensor %q", name)
		}
		if axis >= len(t.Shape) {
			return 0, fmt.Errorf("tensor %q rank %d has no axis %d", name, len(t.Shape), axis)
		}
		return int(t.Shape[axis]), nil
	}
	// bind: derive `got` from a tensor shape, record the check.
	bind := func(name string, want int, src *safetensors.Source, tensor string, axis int) error {
		got, err := dim(src, tensor, axis)
		if err != nil {
			return err
		}
		add(name, want, got, fmt.Sprintf("%s[%d]", tensor, axis))
		return nil
	}

	t, v, e := &s.Transformer, &s.VAE, &s.TextEncoder

	// ---- transformer block counts (config vs count of numbered tensors) ----
	add("t.layers", t.Layers, countBlocks(tsrc, "transformer_blocks."), "count(transformer_blocks.N)")
	add("t.layerwise_text_blocks", t.LayerwiseTextBlocks, countBlocks(tsrc, "text_fusion.layerwise_blocks."), "count(layerwise_blocks.N)")
	add("t.refiner_text_blocks", t.RefinerTextBlocks, countBlocks(tsrc, "text_fusion.refiner_blocks."), "count(refiner_blocks.N)")

	// ---- transformer per-block geometry (image stream, block 0) ----
	for _, b := range []struct {
		name string
		want int
		tens string
		axis int
	}{
		{"t.head_dim", t.HeadDim, "transformer_blocks.0.attn.norm_q.weight", 0},
		{"t.hidden(to_q0)", t.Hidden, "transformer_blocks.0.attn.to_q.weight", 0},
		{"t.hidden(to_q1)", t.Hidden, "transformer_blocks.0.attn.to_q.weight", 1},
		{"t.hidden(img_in0)", t.Hidden, "img_in.weight", 0},
		{"t.kv_dim(to_k0)", t.KVDim, "transformer_blocks.0.attn.to_k.weight", 0},
		{"t.kv_dim(to_v0)", t.KVDim, "transformer_blocks.0.attn.to_v.weight", 0},
		{"t.in_channels(img_in1)", t.InChannels, "img_in.weight", 1},
		{"t.in_channels(final0)", t.InChannels, "final_layer.linear.weight", 0},
		{"t.intermediate(ff.gate0)", t.Intermediate, "transformer_blocks.0.ff.gate.weight", 0},
		{"t.intermediate(ff.down1)", t.Intermediate, "transformer_blocks.0.ff.down.weight", 1},
		{"t.timestep_embed", t.TimestepEmbed, "time_embed.linear_1.weight", 1},
		{"t.text_hidden(txt_in1)", t.TextHidden, "txt_in.linear_1.weight", 1},
		{"t.text_hidden(txt_norm)", t.TextHidden, "txt_in.norm.weight", 0},
		{"t.text_intermediate", t.TextIntermediate, "text_fusion.layerwise_blocks.0.ff.gate.weight", 0},
		{"t.text_hidden(lw.to_q)", t.TextHeads * t.HeadDim, "text_fusion.layerwise_blocks.0.attn.to_q.weight", 0},
		{"t.text_layers(projector)", t.TextLayers, "text_fusion.projector.weight", 1},
	} {
		if err := bind(b.name, b.want, tsrc, b.tens, b.axis); err != nil {
			return nil, err
		}
	}

	// ---- modulation fields: derived from time_mod_proj.weight[0]/Hidden,
	//      cross-checked against transformer_blocks.0.scale_shift_table[0] ----
	modOut, err := dim(tsrc, "time_mod_proj.weight", 0)
	if err != nil {
		return nil, err
	}
	if t.Hidden == 0 || modOut%t.Hidden != 0 {
		return nil, fmt.Errorf("time_mod_proj out %d not a multiple of hidden %d", modOut, t.Hidden)
	}
	w.ModField = modOut / t.Hidden
	t.ModFields = w.ModField
	ssTable, err := dim(tsrc, "transformer_blocks.0.scale_shift_table", 0)
	if err != nil {
		return nil, err
	}
	add("t.mod_fields", w.ModField, ssTable, "time_mod_proj.weight[0]/hidden vs scale_shift_table[0]")

	// ---- VAE conv geometry ----
	for _, b := range []struct {
		name string
		want int
		tens string
		axis int
	}{
		{"v.z_dim(post_quant1)", v.ZDim, "post_quant_conv.weight", 1},
		{"v.z_dim(dec.conv_in1)", v.ZDim, "decoder.conv_in.weight", 1},
		{"v.quant_channels(quant0)", v.QuantChannels, "quant_conv.weight", 0},
		{"v.quant_channels(enc.out0)", v.QuantChannels, "encoder.conv_out.weight", 0},
		{"v.base_dim(enc.in0)", v.BaseDim, "encoder.conv_in.weight", 0},
		{"v.base_dim(dec.out1)", v.BaseDim, "decoder.conv_out.weight", 1},
		{"v.input_channels(enc.in1)", v.InputChannels, "encoder.conv_in.weight", 1},
		{"v.input_channels(dec.out0)", v.InputChannels, "decoder.conv_out.weight", 0},
		{"v.deepest_dim(enc.out1)", v.DeepestDim, "encoder.conv_out.weight", 1},
		{"v.deepest_dim(dec.in0)", v.DeepestDim, "decoder.conv_in.weight", 0},
	} {
		if err := bind(b.name, b.want, vsrc, b.tens, b.axis); err != nil {
			return nil, err
		}
	}

	// ---- text encoder geometry ----
	add("e.hidden_layers", e.HiddenLayers, countBlocks(esrc, "language_model.layers."), "count(language_model.layers.N)")
	for _, b := range []struct {
		name string
		want int
		tens string
		axis int
	}{
		{"e.hidden(embed1)", e.Hidden, "language_model.embed_tokens.weight", 1},
		{"e.vocab(embed0)", e.VocabSize, "language_model.embed_tokens.weight", 0},
		{"e.q_dim(q_proj0)", e.Heads * e.HeadDim, "language_model.layers.0.self_attn.q_proj.weight", 0},
		{"e.kv_dim(k_proj0)", e.KVHeads * e.HeadDim, "language_model.layers.0.self_attn.k_proj.weight", 0},
		{"e.kv_dim(v_proj0)", e.KVHeads * e.HeadDim, "language_model.layers.0.self_attn.v_proj.weight", 0},
		{"e.head_dim(q_norm)", e.HeadDim, "language_model.layers.0.self_attn.q_norm.weight", 0},
	} {
		if err := bind(b.name, b.want, esrc, b.tens, b.axis); err != nil {
			return nil, err
		}
	}
	// cross-model fusion boundary: text encoder hidden == transformer text_hidden
	add("fusion.hidden", t.TextHidden, e.Hidden, "transformer.text_hidden_dim vs text_encoder.embed_tokens[1]")

	if w.Failed() {
		return w, fmt.Errorf("latentimage: %d structural check(s) disagreed with checkpoint", failCount(w))
	}
	return w, nil
}

// countBlocks returns 1 + the max numeric index N appearing immediately after
// prefix across the source's tensor names (0 if none) -- the block count.
func countBlocks(src *safetensors.Source, prefix string) int {
	maxIdx := -1
	for _, name := range src.Names() {
		if !strings.HasPrefix(name, prefix) {
			continue
		}
		rest := name[len(prefix):]
		dot := strings.IndexByte(rest, '.')
		if dot <= 0 {
			continue
		}
		n, err := strconv.Atoi(rest[:dot])
		if err != nil {
			continue
		}
		if n > maxIdx {
			maxIdx = n
		}
	}
	return maxIdx + 1
}

func failCount(w *Witness) int {
	n := 0
	for _, c := range w.Checks {
		if !c.Equal() {
			n++
		}
	}
	return n
}

// SortedChecks returns the checks in stable name order for reporting.
func (w *Witness) SortedChecks() []Check {
	out := append([]Check(nil), w.Checks...)
	sort.SliceStable(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}
