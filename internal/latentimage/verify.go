package latentimage

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"overgo/internal/checked"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
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
func (c Check) Equal() bool { return checked.Equal(c.Want, c.Got) }

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

func (w Witness) Failed() bool { return checked.Nonzero(failedCheckCount(w.Checks)) }

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
	// bind: derive `got` from a tensor shape, record the check.
	bind := func(name string, want int, src *safetensors.Source, tensor string, axis int) error {
		got, err := safetensors.Dimension(src, tensor, axis)
		if err != nil {
			return err
		}
		add(name, want, got, fmt.Sprintf("%s[%d]", tensor, axis))
		return nil
	}

	t, v, e := &s.Transformer, &s.VAE, &s.TextEncoder
	product := func(values ...int) int {
		result, ok := checked.ProductInt(values...)
		if !ok {
			return checked.UnknownCount()
		}
		return result
	}

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
		{"t.head_dim", t.HeadDim, "transformer_blocks.0.attn.norm_q.weight", tensor.FirstOffset},
		{"t.hidden(to_q0)", t.Hidden, "transformer_blocks.0.attn.to_q.weight", tensor.FirstOffset},
		{"t.hidden(to_q1)", t.Hidden, "transformer_blocks.0.attn.to_q.weight", tensor.SingletonExtent},
		{"t.hidden(img_in0)", t.Hidden, "img_in.weight", tensor.FirstOffset},
		{"t.kv_dim(to_k0)", t.KVDim, "transformer_blocks.0.attn.to_k.weight", tensor.FirstOffset},
		{"t.kv_dim(to_v0)", t.KVDim, "transformer_blocks.0.attn.to_v.weight", tensor.FirstOffset},
		{"t.in_channels(img_in1)", t.InChannels, "img_in.weight", tensor.SingletonExtent},
		{"t.in_channels(final0)", t.InChannels, "final_layer.linear.weight", tensor.FirstOffset},
		{"t.intermediate(ff.gate0)", t.Intermediate, "transformer_blocks.0.ff.gate.weight", tensor.FirstOffset},
		{"t.intermediate(ff.down1)", t.Intermediate, "transformer_blocks.0.ff.down.weight", tensor.SingletonExtent},
		{"t.timestep_embed", t.TimestepEmbed, "time_embed.linear_1.weight", tensor.SingletonExtent},
		{"t.text_hidden(txt_in1)", t.TextHidden, "txt_in.linear_1.weight", tensor.SingletonExtent},
		{"t.text_hidden(txt_norm)", t.TextHidden, "txt_in.norm.weight", tensor.FirstOffset},
		{"t.text_intermediate", t.TextIntermediate, "text_fusion.layerwise_blocks.0.ff.gate.weight", tensor.FirstOffset},
		{"t.text_hidden(lw.to_q)", product(t.TextHeads, t.HeadDim), "text_fusion.layerwise_blocks.0.attn.to_q.weight", tensor.FirstOffset},
		{"t.text_layers(projector)", t.TextLayers, "text_fusion.projector.weight", tensor.SingletonExtent},
	} {
		if err := bind(b.name, b.want, tsrc, b.tens, b.axis); err != nil {
			return nil, err
		}
	}

	// ---- modulation fields: derived from time_mod_proj.weight[0]/Hidden,
	//      cross-checked against transformer_blocks.0.scale_shift_table[0] ----
	modOut, err := safetensors.Dimension(tsrc, "time_mod_proj.weight", tensor.FirstOffset)
	if err != nil {
		return nil, err
	}
	modFields, ok := checked.DivExactInt(modOut, t.Hidden)
	if !ok {
		return nil, fmt.Errorf("time_mod_proj out %d not a multiple of hidden %d", modOut, t.Hidden)
	}
	w.ModField = modFields
	t.ModFields = w.ModField
	ssTable, err := safetensors.Dimension(tsrc, "transformer_blocks.0.scale_shift_table", tensor.FirstOffset)
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
		{"v.z_dim(post_quant1)", v.ZDim, "post_quant_conv.weight", tensor.SingletonExtent},
		{"v.z_dim(dec.conv_in1)", v.ZDim, "decoder.conv_in.weight", tensor.SingletonExtent},
		{"v.quant_channels(quant0)", v.QuantChannels, "quant_conv.weight", tensor.FirstOffset},
		{"v.quant_channels(enc.out0)", v.QuantChannels, "encoder.conv_out.weight", tensor.FirstOffset},
		{"v.base_dim(enc.in0)", v.BaseDim, "encoder.conv_in.weight", tensor.FirstOffset},
		{"v.base_dim(dec.out1)", v.BaseDim, "decoder.conv_out.weight", tensor.SingletonExtent},
		{"v.input_channels(enc.in1)", v.InputChannels, "encoder.conv_in.weight", tensor.SingletonExtent},
		{"v.input_channels(dec.out0)", v.InputChannels, "decoder.conv_out.weight", tensor.FirstOffset},
		{"v.deepest_dim(enc.out1)", v.DeepestDim, "encoder.conv_out.weight", tensor.SingletonExtent},
		{"v.deepest_dim(dec.in0)", v.DeepestDim, "decoder.conv_in.weight", tensor.FirstOffset},
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
		{"e.hidden(embed1)", e.Hidden, "language_model.embed_tokens.weight", tensor.SingletonExtent},
		{"e.vocab(embed0)", e.VocabSize, "language_model.embed_tokens.weight", tensor.FirstOffset},
		{"e.q_dim(q_proj0)", product(e.Heads, e.HeadDim), "language_model.layers.0.self_attn.q_proj.weight", tensor.FirstOffset},
		{"e.kv_dim(k_proj0)", product(e.KVHeads, e.HeadDim), "language_model.layers.0.self_attn.k_proj.weight", tensor.FirstOffset},
		{"e.kv_dim(v_proj0)", product(e.KVHeads, e.HeadDim), "language_model.layers.0.self_attn.v_proj.weight", tensor.FirstOffset},
		{"e.head_dim(q_norm)", e.HeadDim, "language_model.layers.0.self_attn.q_norm.weight", tensor.FirstOffset},
	} {
		if err := bind(b.name, b.want, esrc, b.tens, b.axis); err != nil {
			return nil, err
		}
	}
	// cross-model fusion boundary: text encoder hidden == transformer text_hidden
	add("fusion.hidden", t.TextHidden, e.Hidden, "transformer.text_hidden_dim vs text_encoder.embed_tokens[1]")

	if failures := failedCheckCount(w.Checks); checked.Nonzero(failures) {
		return w, fmt.Errorf("latentimage: %d structural check(s) disagreed with checkpoint", failures)
	}
	return w, nil
}

// countBlocks returns 1 + the max numeric index N appearing immediately after
// prefix across the source's tensor names (0 if none) -- the block count.
func countBlocks(src *safetensors.Source, prefix string) int {
	maxIdx := checked.UnknownCount()
	for _, name := range src.Names() {
		if !strings.HasPrefix(name, prefix) {
			continue
		}
		rest := name[len(prefix):]
		dot := strings.IndexByte(rest, '.')
		if !checked.PositiveInts(dot) {
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
	return maxIdx + tensor.SingletonExtent
}

func failedCheckCount(checks []Check) int {
	count := tensor.FirstOffset
	for _, c := range checks {
		if !c.Equal() {
			count++
		}
	}
	return count
}

// SortedChecks returns the checks in stable name order for reporting.
func (w *Witness) SortedChecks() []Check {
	out := append([]Check(nil), w.Checks...)
	sort.SliceStable(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}
