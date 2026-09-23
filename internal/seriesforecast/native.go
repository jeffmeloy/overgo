// The native forecaster: a checkpoint that ships its own configuration
// (input and output patch lengths, quantiles, a stacked mixing transformer)
// rather than a transformers one. Ported from the model repository's torch
// reference for one target series: linear detrending of the context, running
// RevIN statistics, an input embedding that carries each patch's rolled
// future window, sequence attention followed by variate attention (which,
// over a single variate, is its value path), a quantile head over a whole
// output patch, iterative refinement of the statistics across the forecast
// patches, and linear stitching of overlapping patch forecasts.

package seriesforecast

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/extent"
	"overgo/internal/hostmath"
	"overgo/internal/jsonfile"
	"overgo/internal/tensorcatalog"
)

const (
	// nativeRopeTimescale is the reference's rotary maximum timescale; the
	// native configuration does not state one.
	nativeRopeTimescale = 10000
	// nativeMaxContext is the reference forecaster's longest context; a
	// longer series keeps its most recent values.
	nativeMaxContext = 15360
)

// nativeConfig is the checkpoint's own configuration file.
type nativeConfig struct {
	InputPatchLen  int       `json:"input_patch_len"`
	OutputPatchLen int       `json:"output_patch_len"`
	Quantiles      []float64 `json:"quantiles"`
	InputTransform string    `json:"input_transform"`
	Residual       struct {
		Activation   string `json:"activation"`
		IdentitySkip bool   `json:"identity_skip"`
		Prenorm      string `json:"prenorm"`
		UseBias      bool   `json:"use_bias"`
	} `json:"residual_block_config"`
	Transformer struct {
		Layer struct {
			AttentionNorm   string `json:"attention_norm"`
			FeedforwardNorm string `json:"feedforward_norm"`
			QKNorm          string `json:"qk_norm"`
			VNorm           string `json:"v_norm"`
			Activation      string `json:"ff_activation"`
			Causal          bool   `json:"causal_attention"`
			RopeSequence    bool   `json:"use_rope_seq"`
			RopeVariate     bool   `json:"use_rope_var"`
			UseBias         bool   `json:"use_bias"`
			EfficientScale  bool   `json:"use_memory_efficient_attention"`
		} `json:"transformer"`
	} `json:"transformer_config"`
	VariateAttention  bool    `json:"use_variate_attention"`
	Stitching         bool    `json:"use_stitching"`
	Detrending        bool    `json:"use_linear_detrending"`
	DetrendThreshold  float64 `json:"linear_detrending_threshold"`
	IterativeRevin    bool    `json:"use_iterative_cpm_revin"`
	FrozenRunningStat bool    `json:"use_frozen_running_stats"`
	ValueClip         float64 `json:"value_clip"`
}

// native holds what the native forward needs beyond the shared geometry.
type native struct {
	outputPatch, rolls, median int
	detrendThreshold           float64
	valueClip                  float64
}

// isNative reports whether a checkpoint directory's configuration is the
// native one, which names its transformer stack instead of a model type.
func isNative(path string) (bool, error) {
	var probe struct {
		Transformer *struct{} `json:"transformer_config"`
	}
	if err := jsonfile.Decode(path, &probe); err != nil {
		return false, fmt.Errorf("seriesforecast: parse config.json: %w", err)
	}
	return probe.Transformer != nil, nil
}

// loadNativeConfig refuses a configuration asking for a variant this port
// does not run.
func loadNativeConfig(path string) (nativeConfig, error) {
	var config nativeConfig
	if err := jsonfile.Decode(path, &config); err != nil {
		return nativeConfig{}, fmt.Errorf("seriesforecast: parse config.json: %w", err)
	}
	layer := config.Transformer.Layer
	if config.InputPatchLen <= 0 || config.OutputPatchLen <= config.InputPatchLen || config.OutputPatchLen%config.InputPatchLen != 0 ||
		len(config.Quantiles) == 0 || config.InputTransform != "identity" || config.ValueClip <= 0 ||
		config.Residual.Activation != "relu" || config.Residual.IdentitySkip || config.Residual.Prenorm != "none" || config.Residual.UseBias ||
		layer.AttentionNorm != "rms" || layer.FeedforwardNorm != "rms" || layer.QKNorm != "rms" || layer.VNorm != "none" ||
		layer.Activation != "relu" || !layer.Causal || !layer.RopeSequence || layer.RopeVariate || layer.UseBias || !layer.EfficientScale ||
		!config.VariateAttention || !config.Stitching || !config.Detrending || !config.IterativeRevin || config.FrozenRunningStat {
		return nativeConfig{}, errors.New("seriesforecast: the native configuration asks for a variant this port does not run")
	}
	return config, nil
}

// loadNative binds a native checkpoint onto the canonical layer layout the
// shared decoder reads, keeping the variate branch's value path.
func loadNative(configPath string, weights map[string][]float32, shapes map[string][]int) (*Model, error) {
	config, err := loadNativeConfig(configPath)
	if err != nil {
		return nil, err
	}
	canonical, canonicalShapes, err := canonicalizeNative(weights, shapes)
	if err != nil {
		return nil, err
	}
	d := Dims{PatchLen: config.InputPatchLen, Horizon: config.OutputPatchLen, Quantiles: extent.SingletonExtent + len(config.Quantiles),
		RopeTheta: nativeRopeTimescale, RMSEps: float64(math.Nextafter32(extent.SingletonExtent, extent.PairedExtent) - extent.SingletonExtent)}
	outputLayer, err := tensorcatalog.Shape(canonicalShapes, "tokenizer.output_layer.weight", extent.PairedExtent)
	if err != nil {
		return nil, err
	}
	d.Hidden = outputLayer[0]
	if d.Layers, err = tensorcatalog.IndexedCount(canonicalShapes, "stacked_xf.", ".attn.qkv_proj.weight"); err != nil {
		return nil, err
	}
	queryNorm, err := tensorcatalog.Shape(canonicalShapes, "stacked_xf.0.attn.query_ln.scale", extent.SingletonExtent)
	if err != nil {
		return nil, err
	}
	d.HeadDim = queryNorm[0]
	if !checked.PositiveInts(d.HeadDim) || checked.Nonzero(d.Hidden%d.HeadDim) {
		return nil, fmt.Errorf("seriesforecast: hidden %d not divisible by head dim %d", d.Hidden, d.HeadDim)
	}
	d.Heads = d.Hidden / d.HeadDim
	head, err := tensorcatalog.Shape(canonicalShapes, "output_head.weight", extent.PairedExtent)
	if err != nil {
		return nil, err
	}
	embed, err := tensorcatalog.Shape(canonicalShapes, "tokenizer.hidden_layer.weight", extent.PairedExtent)
	if err != nil {
		return nil, err
	}
	if head[0] != d.Horizon*len(config.Quantiles) || head[1] != d.Hidden || embed[1] != extent.PairedExtent*(d.PatchLen+d.Horizon) {
		return nil, errors.New("seriesforecast: native head or embedding width disagrees with the configuration")
	}
	return &Model{Dims: d, Weights: canonical, Shapes: canonicalShapes, Levels: config.Quantiles, native: &native{
		outputPatch: config.OutputPatchLen, rolls: config.OutputPatchLen / config.InputPatchLen, median: len(config.Quantiles) / extent.PairedExtent,
		detrendThreshold: config.DetrendThreshold, valueClip: config.ValueClip,
	}}, nil
}

// canonicalizeNative renames the native tensors onto the canonical layout.
// The variate branch keeps its norms and value and output projections; its
// query and key path is dropped, since over one variate the attention weight
// is one and the branch is its value path.
func canonicalizeNative(weights map[string][]float32, shapes map[string][]int) (map[string][]float32, map[string][]int, error) {
	outWeights := map[string][]float32{}
	outShapes := map[string][]int{}
	move := func(source, target string) error {
		values, found := weights[source]
		if !found {
			return fmt.Errorf("seriesforecast: native tensor %s is absent", source)
		}
		outWeights[target], outShapes[target] = values, shapes[source]
		return nil
	}
	for _, name := range []string{"hidden_layer", "output_layer", "residual_layer"} {
		if err := move("pre_transformer_resblock."+name+".weight", "tokenizer."+name+".weight"); err != nil {
			return nil, nil, err
		}
	}
	for _, name := range []string{"output_head.weight", "output_head.bias"} {
		if err := move(name, name); err != nil {
			return nil, nil, err
		}
	}
	renames := [][2]string{
		{"pre_seq_attn_ln.weight", "pre_attn_ln.scale"}, {"post_seq_attn_ln.weight", "post_attn_ln.scale"},
		{"pre_ff_ln.weight", "pre_ff_ln.scale"}, {"post_ff_ln.weight", "post_ff_ln.scale"},
		{"ff0.weight", "ff0.weight"}, {"ff1.weight", "ff1.weight"},
		{"seq_attn.query_ln.weight", "attn.query_ln.scale"}, {"seq_attn.key_ln.weight", "attn.key_ln.scale"},
		{"seq_attn.per_dim_scale.per_dim_scale", "attn.per_dim_scale.per_dim_scale"},
		{"seq_attn.out_proj.weight", "attn.out.weight"},
		{"pre_var_attn_ln.weight", "pre_var_ln.scale"}, {"post_var_attn_ln.weight", "post_var_ln.scale"},
		{"var_attn.value_proj.weight", "var_attn.value.weight"}, {"var_attn.out_proj.weight", "var_attn.out.weight"},
	}
	for layer := 0; ; layer++ {
		source := fmt.Sprintf("transformer_stack.layers.%d.", layer)
		if _, found := weights[source+"pre_seq_attn_ln.weight"]; !found {
			if layer == 0 {
				return nil, nil, errors.New("seriesforecast: native layout has no transformer_stack.layers.0")
			}
			return outWeights, outShapes, nil
		}
		target := fmt.Sprintf("stacked_xf.%d.", layer)
		for _, rename := range renames {
			if err := move(source+rename[0], target+rename[1]); err != nil {
				return nil, nil, err
			}
		}
		var qkv []float32
		for _, projection := range []string{"query_proj", "key_proj", "value_proj"} {
			qkv = append(qkv, weights[source+"seq_attn."+projection+".weight"]...)
		}
		hidden := shapes[source+"seq_attn.query_proj.weight"]
		if len(hidden) != extent.PairedExtent || len(qkv) != extent.TripleExtent*hidden[0]*hidden[1] {
			return nil, nil, fmt.Errorf("seriesforecast: native layer %d sequence projections are incomplete", layer)
		}
		outWeights[target+"attn.qkv_proj.weight"], outShapes[target+"attn.qkv_proj.weight"] = qkv, []int{extent.TripleExtent * hidden[0], hidden[1]}
	}
}

// forecastNative runs one series through the native forward and returns
// horizon rows of the median point forecast followed by the sorted
// quantiles.
func (m *Model) forecastNative(series []float32, horizon int) ([]float32, error) {
	d, n := m.Dims, m.native
	if !checked.PositiveInts(horizon) {
		return nil, errors.New("seriesforecast: horizon must be positive")
	}
	if len(series) > nativeMaxContext {
		series = series[len(series)-nativeMaxContext:]
	}
	context, contextMasks, err := padToPatches(series, nil, d.PatchLen)
	if err != nil {
		return nil, err
	}
	trend := detrend(context, contextMasks, n.detrendThreshold)
	// Masked context values stay zero.
	values := make([]float32, len(context))
	for index, value := range context {
		if contextMasks[index] != 0 {
			continue
		}
		values[index] = value
		if trend.apply {
			values[index] -= float32(trend.at(index - len(values) + extent.SingletonExtent))
		}
	}
	// The horizon is a whole number of output patches, forecast from
	// overlapping patches that are stitched back together.
	target := n.outputPatch * ((horizon + n.outputPatch - extent.SingletonExtent) / n.outputPatch)
	overlap := n.outputPatch - d.PatchLen
	forecastPatches := max((target-overlap+d.PatchLen-extent.SingletonExtent)/d.PatchLen, extent.SingletonExtent)
	contextPatches := len(values) / d.PatchLen
	patches := contextPatches + forecastPatches + n.rolls - extent.SingletonExtent
	values = append(values, make([]float32, (patches-contextPatches)*d.PatchLen)...)
	masks := append(contextMasks, make([]float32, (patches-contextPatches)*d.PatchLen)...)
	for index := len(context); index < len(masks); index++ {
		masks[index] = extent.SingletonExtent
	}
	counts, mu, sigma := make([]float64, patches), make([]float64, patches), make([]float64, patches)
	var runningN, runningMu, runningSigma float64
	for patch := range patches {
		window := values[patch*d.PatchLen : (patch+1)*d.PatchLen]
		runningN, runningMu, runningSigma = mergePatch(runningN, runningMu, runningSigma, window, masks[patch*d.PatchLen:(patch+1)*d.PatchLen])
		counts[patch], mu[patch], sigma[patch] = runningN, runningMu, runningSigma
	}

	hidden := make([]float32, patches*d.Hidden)
	embedding := make([]float32, extent.PairedExtent*(d.PatchLen+n.outputPatch))
	for patch := range patches {
		clear(embedding)
		divisor := revinDivisor(sigma[patch])
		for j := range d.PatchLen {
			at := patch*d.PatchLen + j
			if masks[at] == 0 {
				embedding[j] = float32((float64(values[at]) - mu[patch]) / divisor)
			}
			embedding[d.PatchLen+n.outputPatch+j] = masks[at]
		}
		// The rolled future window of a target series is always masked.
		for j := range n.outputPatch {
			embedding[extent.PairedExtent*d.PatchLen+n.outputPatch+j] = extent.SingletonExtent
		}
		m.nativeEmbed(hidden[patch*d.Hidden:(patch+1)*d.Hidden], embedding)
	}
	invFreq := hostmath.RopeInvFreq(d.RopeTheta, d.HeadDim)
	for index := range d.Layers {
		l, err := m.layerWeights(index)
		if err != nil {
			return nil, err
		}
		m.layerForward(hidden, l, invFreq, patches)
	}
	width := n.outputPatch * len(m.Levels)
	logits := make([]float32, patches*width)
	hostmath.Linear(logits, hidden, m.Weights["output_head.weight"], patches, d.Hidden, width)
	for patch := range patches {
		hostmath.AddBias(logits[patch*width:(patch+1)*width], m.Weights["output_head.bias"])
	}
	m.refineForecastStatistics(logits, counts, mu, sigma, contextPatches)
	rows := n.outputPatch
	forecasts := make([][]float64, forecastPatches)
	for index := range forecasts {
		patch := contextPatches - extent.SingletonExtent + index
		forecasts[index] = make([]float64, width)
		for at := range width {
			value := float64(logits[patch*width+at])*sigma[patch] + mu[patch]
			forecasts[index][at] = max(-n.valueClip, min(n.valueClip, value))
		}
	}
	stitched := stitch(forecasts, rows, d.PatchLen, len(m.Levels))
	out := make([]float32, horizon*d.Quantiles)
	for step := range horizon {
		row := stitched[step*len(m.Levels) : (step+1)*len(m.Levels)]
		if trend.apply {
			for column := range row {
				row[column] += trend.at(step + extent.SingletonExtent)
			}
		}
		slices.Sort(row)
		out[step*d.Quantiles] = float32(row[n.median])
		for column, value := range row {
			out[step*d.Quantiles+extent.SingletonExtent+column] = float32(value)
		}
	}
	return out, nil
}

// nativeEmbed is the pre-transformer residual block: ReLU hidden layer plus
// a linear skip, without biases.
func (m *Model) nativeEmbed(dst, x []float32) {
	hiddenShape := m.Shapes["tokenizer.hidden_layer.weight"]
	inner := make([]float32, hiddenShape[0])
	hostmath.Linear(inner, x, m.Weights["tokenizer.hidden_layer.weight"], extent.SingletonExtent, len(x), len(inner))
	reluInPlace(inner)
	hostmath.Linear(dst, inner, m.Weights["tokenizer.output_layer.weight"], extent.SingletonExtent, len(inner), len(dst))
	skip := make([]float32, len(dst))
	hostmath.Linear(skip, x, m.Weights["tokenizer.residual_layer.weight"], extent.SingletonExtent, len(x), len(dst))
	for index := range dst {
		dst[index] += skip[index]
	}
}

// reluInPlace clamps negative activations to zero.
func reluInPlace(values []float32) {
	for i, value := range values {
		values[i] = max(value, 0)
	}
}

// refineForecastStatistics replaces the running statistics of each forecast
// patch with ones that also count the median forecast of the forecast
// patches before it, in place.
func (m *Model) refineForecastStatistics(logits []float32, counts, mu, sigma []float64, contextPatches int) {
	d, n := m.Dims, m.native
	quantiles := len(m.Levels)
	width := n.outputPatch * quantiles
	var carryN, carryMu, carrySigma float64
	anchor := make([]float32, n.outputPatch)
	unmasked := make([]float32, d.PatchLen)
	offset := extent.FirstOffset
	for patch := range counts {
		forecast := patch >= contextPatches
		if forecast {
			predicted := anchor[offset*d.PatchLen : (offset+extent.SingletonExtent)*d.PatchLen]
			counts[patch], mu[patch], sigma[patch] = mergePatch(carryN, carryMu, carrySigma, predicted, unmasked)
			offset = (offset + extent.SingletonExtent) % n.rolls
		} else {
			offset = extent.FirstOffset
		}
		carryN, carryMu, carrySigma = counts[patch], mu[patch], sigma[patch]
		if offset == extent.FirstOffset {
			for at := range n.outputPatch {
				value := float64(logits[patch*width+at*quantiles+n.median])*sigma[patch] + mu[patch]
				anchor[at] = float32(max(-n.valueClip, min(n.valueClip, value)))
			}
		}
	}
}

// stitch joins overlapping patch forecasts: each holds patchLen new steps
// followed by the steps the next one also forecasts, which are blended
// linearly from the earlier forecast to the later one.
func stitch(forecasts [][]float64, rows, patchLen, columns int) []float64 {
	overlap := rows - patchLen
	if len(forecasts) == extent.SingletonExtent {
		return forecasts[0]
	}
	out := slices.Clone(forecasts[0][:patchLen*columns])
	for index, next := range forecasts[extent.SingletonExtent:] {
		previous := forecasts[index]
		for step := range overlap {
			weight := extent.SingletonExtent - float64(step)/float64(overlap-extent.SingletonExtent)
			for column := range columns {
				out = append(out, weight*previous[(patchLen+step)*columns+column]+(extent.SingletonExtent-weight)*next[step*columns+column])
			}
		}
		out = append(out, next[overlap*columns:patchLen*columns]...)
	}
	return append(out, forecasts[len(forecasts)-1][patchLen*columns:]...)
}

// trend is a least-squares line over the context against time normalized by
// the context length, kept only when removing it leaves a spread below the
// threshold fraction of the original.
type trend struct {
	slope, intercept, length float64
	apply                    bool
}

// at is the line at a step relative to the last context value (0 there,
// negative before it, positive in the forecast).
func (t trend) at(step int) float64 { return t.slope*float64(step)/t.length + t.intercept }

func detrend(values, masks []float32, threshold float64) trend {
	t := trend{length: float64(len(values))}
	var n, sumT, sumT2, sumY, sumTY, sumY2 float64
	for index, value := range values {
		if masks[index] != 0 {
			continue
		}
		time, y := float64(index-len(values)+1)/t.length, float64(value)
		n, sumT, sumT2, sumY, sumTY, sumY2 = n+1, sumT+time, sumT2+time*time, sumY+y, sumTY+time*y, sumY2+y*y
	}
	count := max(n, extent.SingletonExtent)
	if determinant := n*sumT2 - sumT*sumT; determinant != 0 {
		t.slope = (n*sumTY - sumT*sumY) / determinant
		t.intercept = (sumY - t.slope*sumT) / count
	} else {
		t.intercept = sumY / count
	}
	var sumD, sumD2 float64
	for index, value := range values {
		if masks[index] == 0 {
			residual := float64(value) - t.at(index-len(values)+1)
			sumD, sumD2 = sumD+residual, sumD2+residual*residual
		}
	}
	spread := func(sum, squares float64) float64 {
		mean := sum / count
		return math.Sqrt(max(squares/count-mean*mean, 0))
	}
	t.apply = spread(sumD, sumD2) < threshold*spread(sumY, sumY2)
	return t
}
