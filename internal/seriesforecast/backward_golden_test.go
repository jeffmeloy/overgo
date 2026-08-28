package seriesforecast

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"overgo/internal/hostmath"
	"overgo/internal/testutil"
)

func readGrad(t *testing.T, name string, out any) {
	t.Helper()
	path := testutil.FixturePath(t, name)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("UNAVAILABLE: %s absent; backward parity NOT verified", name)
	}
	if err := json.Unmarshal(raw, out); err != nil {
		t.Fatal(err)
	}
}

func f32(v []float64) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(x)
	}
	return out
}

// Wired-component tolerance: a few composed f64-host ops against the
// reference's f64 autograd through f32 storage.
const wiredTol = 1e-3

func closeTo(t *testing.T, name string, got []float32, want []float64, tol float64) {
	t.Helper()
	testutil.RequireElementsWithin(t, name, got, want, tol)
}

func TestResidualBlockBackwardMatchesGolden(t *testing.T) {
	var g struct {
		InDim, HDim, OutDim int       `json:"-"`
		In                  int       `json:"in_dim"`
		H                   int       `json:"hdim"`
		Out                 int       `json:"out_dim"`
		HiddenWeight        []float64 `json:"hidden_weight"`
		HiddenBias          []float64 `json:"hidden_bias"`
		OutputWeight        []float64 `json:"output_weight"`
		OutputBias          []float64 `json:"output_bias"`
		ResidualWeight      []float64 `json:"residual_weight"`
		ResidualBias        []float64 `json:"residual_bias"`
		X                   []float64 `json:"x"`
		Dout                []float64 `json:"dout"`
		GradX               []float64 `json:"grad_x"`
		GradHiddenWeight    []float64 `json:"grad_hidden_weight"`
		GradHiddenBias      []float64 `json:"grad_hidden_bias"`
		GradOutputWeight    []float64 `json:"grad_output_weight"`
		GradOutputBias      []float64 `json:"grad_output_bias"`
		GradResidualWeight  []float64 `json:"grad_residual_weight"`
		GradResidualBias    []float64 `json:"grad_residual_bias"`
	}
	readGrad(t, "timesfm_resblock_grad_golden.json", &g)
	model := &Model{
		Shapes: map[string][]int{
			"blk.hidden_layer.weight":   {g.H, g.In},
			"blk.output_layer.weight":   {g.Out, g.H},
			"blk.residual_layer.weight": {g.Out, g.In},
		},
		Weights: map[string][]float32{
			"blk.hidden_layer.weight": f32(g.HiddenWeight), "blk.hidden_layer.bias": f32(g.HiddenBias),
			"blk.output_layer.weight": f32(g.OutputWeight), "blk.output_layer.bias": f32(g.OutputBias),
			"blk.residual_layer.weight": f32(g.ResidualWeight), "blk.residual_layer.bias": f32(g.ResidualBias),
		},
	}
	grads := Grads{}
	dx, err := model.residualBlockBackward("blk", f32(g.X), f32(g.Dout), grads)
	if err != nil {
		t.Fatal(err)
	}
	closeTo(t, "grad_x", dx, g.GradX, wiredTol)
	closeTo(t, "grad_hidden_weight", grads["blk.hidden_layer.weight"], g.GradHiddenWeight, wiredTol)
	closeTo(t, "grad_hidden_bias", grads["blk.hidden_layer.bias"], g.GradHiddenBias, wiredTol)
	closeTo(t, "grad_output_weight", grads["blk.output_layer.weight"], g.GradOutputWeight, wiredTol)
	closeTo(t, "grad_output_bias", grads["blk.output_layer.bias"], g.GradOutputBias, wiredTol)
	closeTo(t, "grad_residual_weight", grads["blk.residual_layer.weight"], g.GradResidualWeight, wiredTol)
	closeTo(t, "grad_residual_bias", grads["blk.residual_layer.bias"], g.GradResidualBias, wiredTol)
}

func TestRevinBackwardMatchesGolden(t *testing.T) {
	var g struct {
		N         int       `json:"n"`
		X         []float64 `json:"x"`
		Mu        float64   `json:"mu"`
		Sigma     float64   `json:"sigma"`
		Dnormed   []float64 `json:"dnormed"`
		GradX     []float64 `json:"grad_x"`
		GradMu    float64   `json:"grad_mu"`
		GradSigma float64   `json:"grad_sigma"`
	}
	readGrad(t, "timesfm_revin_grad_golden.json", &g)
	dx, dMu, dSigma := revinBackward(f32(g.Dnormed), f32(g.X), g.Mu, g.Sigma)
	closeTo(t, "grad_x", dx, g.GradX, wiredTol)
	if math.Abs(dMu-g.GradMu) > wiredTol || math.Abs(dSigma-g.GradSigma) > wiredTol {
		t.Fatalf("grad_mu/sigma %g/%g, want %g/%g", dMu, dSigma, g.GradMu, g.GradSigma)
	}
}

func TestPatchStatsBackwardMatchesGolden(t *testing.T) {
	var g struct {
		PatchLen   int       `json:"patch_len"`
		NP         int       `json:"np"`
		Series     []float64 `json:"series"`
		Mask       []float64 `json:"mask"`
		PatchMu    []float64 `json:"patch_mu"`
		PatchSigma []float64 `json:"patch_sigma"`
		DMu        []float64 `json:"d_mu"`
		DSigma     []float64 `json:"d_sigma"`
		GradSeries []float64 `json:"grad_series"`
	}
	readGrad(t, "timesfm_patchstats_grad_golden.json", &g)
	series, masks := f32(g.Series), f32(g.Mask)
	mu := make([]float64, g.NP)
	sigma := make([]float64, g.NP)
	patchStats(series, masks, g.PatchLen, mu, sigma)
	closeTo(t, "patch_mu (forward)", f32(mu), g.PatchMu, wiredTol)
	closeTo(t, "patch_sigma (forward)", f32(sigma), g.PatchSigma, wiredTol)
	dSeries := make([]float32, len(series))
	patchStatsBackward(dSeries, series, masks, g.PatchLen, mu, sigma, g.DMu, g.DSigma)
	closeTo(t, "grad_series", dSeries, g.GradSeries, wiredTol)
}

func TestPatchEmbedBackwardMatchesGolden(t *testing.T) {
	var g struct {
		PatchLen         int       `json:"patch_len"`
		NP               int       `json:"np"`
		InDim            int       `json:"in_dim"`
		HDim             int       `json:"hdim"`
		OutDim           int       `json:"out_dim"`
		HiddenWeight     []float64 `json:"hidden_weight"`
		HiddenBias       []float64 `json:"hidden_bias"`
		OutputWeight     []float64 `json:"output_weight"`
		OutputBias       []float64 `json:"output_bias"`
		ResidualWeight   []float64 `json:"residual_weight"`
		ResidualBias     []float64 `json:"residual_bias"`
		Series           []float64 `json:"series"`
		Mask             []float64 `json:"mask"`
		Dout             []float64 `json:"dout"`
		GradSeries       []float64 `json:"grad_series"`
		GradHiddenWeight []float64 `json:"grad_hidden_weight"`
		GradOutputWeight []float64 `json:"grad_output_weight"`
	}
	readGrad(t, "timesfm_patchembed_grad_golden.json", &g)
	model := &Model{
		Dims: Dims{PatchLen: g.PatchLen, Hidden: g.OutDim},
		Shapes: map[string][]int{
			"tokenizer.hidden_layer.weight":   {g.HDim, g.InDim},
			"tokenizer.output_layer.weight":   {g.OutDim, g.HDim},
			"tokenizer.residual_layer.weight": {g.OutDim, g.InDim},
		},
		Weights: map[string][]float32{
			"tokenizer.hidden_layer.weight": f32(g.HiddenWeight), "tokenizer.hidden_layer.bias": f32(g.HiddenBias),
			"tokenizer.output_layer.weight": f32(g.OutputWeight), "tokenizer.output_layer.bias": f32(g.OutputBias),
			"tokenizer.residual_layer.weight": f32(g.ResidualWeight), "tokenizer.residual_layer.bias": f32(g.ResidualBias),
		},
	}
	series, masks := f32(g.Series), f32(g.Mask)
	mu := make([]float64, g.NP)
	sigma := make([]float64, g.NP)
	patchStats(series, masks, g.PatchLen, mu, sigma)
	grads := Grads{}
	dSeries, err := model.patchEmbedBackward(series, masks, mu, sigma, f32(g.Dout), grads)
	if err != nil {
		t.Fatal(err)
	}
	closeTo(t, "grad_series", dSeries, g.GradSeries, wiredTol)
	closeTo(t, "grad_hidden_weight", grads["tokenizer.hidden_layer.weight"], g.GradHiddenWeight, wiredTol)
	closeTo(t, "grad_output_weight", grads["tokenizer.output_layer.weight"], g.GradOutputWeight, wiredTol)
}

// syntheticLayerModel builds a one-layer model from fixture tensors under
// the canonical stacked_xf names.
func syntheticLayerModel(hid, nh, hd int, w map[string][]float32) *Model {
	return &Model{
		Dims:    Dims{Hidden: hid, Layers: 1, Heads: nh, HeadDim: hd, RopeTheta: 10000, RMSEps: 1e-06},
		Weights: w,
		Shapes:  map[string][]int{},
	}
}

func TestAttentionLayerBackwardMatchesGolden(t *testing.T) {
	var g struct {
		N, Hid, NH, HD            int
		QkvW, QLN, KLN, Pds, OutW []float64
		X, Dout                   []float64
		GradX, GradQkvW, GradOutW []float64
		GradQLN, GradKLN, GradPds []float64
	}
	raw := map[string]json.RawMessage{}
	readGrad(t, "timesfm_attnlayer_grad_golden.json", &raw)
	for key, dst := range map[string]any{
		"n": &g.N, "hid": &g.Hid, "nh": &g.NH, "hd": &g.HD,
		"qkvW": &g.QkvW, "qLN": &g.QLN, "kLN": &g.KLN, "pds": &g.Pds, "outW": &g.OutW,
		"x": &g.X, "dout": &g.Dout, "grad_x": &g.GradX, "grad_qkvW": &g.GradQkvW,
		"grad_outW": &g.GradOutW, "grad_qLN": &g.GradQLN, "grad_kLN": &g.GradKLN, "grad_pds": &g.GradPds,
	} {
		if err := json.Unmarshal(raw[key], dst); err != nil {
			t.Fatalf("%s: %v", key, err)
		}
	}
	model := syntheticLayerModel(g.Hid, g.NH, g.HD, map[string][]float32{
		"stacked_xf.0.attn.qkv_proj.weight":             f32(g.QkvW),
		"stacked_xf.0.attn.out.weight":                  f32(g.OutW),
		"stacked_xf.0.attn.query_ln.scale":              f32(g.QLN),
		"stacked_xf.0.attn.key_ln.scale":                f32(g.KLN),
		"stacked_xf.0.attn.per_dim_scale.per_dim_scale": f32(g.Pds),
	})
	l, err := model.layerWeights(0)
	if err == nil {
		t.Fatal("layerWeights should fail without norms; use direct struct")
	}
	matrix := g.Hid * g.Hid
	qkv := model.Weights["stacked_xf.0.attn.qkv_proj.weight"]
	l = layer{
		q: qkv[:matrix], k: qkv[matrix : 2*matrix], v: qkv[2*matrix:],
		o:           model.Weights["stacked_xf.0.attn.out.weight"],
		queryLN:     model.Weights["stacked_xf.0.attn.query_ln.scale"],
		keyLN:       model.Weights["stacked_xf.0.attn.key_ln.scale"],
		perDimScale: model.Weights["stacked_xf.0.attn.per_dim_scale.per_dim_scale"],
	}
	grads := Grads{}
	invFreq := hostmath.RopeInvFreq(model.Dims.RopeTheta, g.HD)
	dx := model.attnSubBackward(0, l, f32(g.X), f32(g.Dout), invFreq, g.N, grads)
	closeTo(t, "grad_x", dx, g.GradX, wiredTol)
	closeTo(t, "grad_qkvW", grads["stacked_xf.0.attn.qkv_proj.weight"], g.GradQkvW, wiredTol)
	closeTo(t, "grad_outW", grads["stacked_xf.0.attn.out.weight"], g.GradOutW, wiredTol)
	closeTo(t, "grad_qLN", grads["stacked_xf.0.attn.query_ln.scale"], g.GradQLN, wiredTol)
	closeTo(t, "grad_kLN", grads["stacked_xf.0.attn.key_ln.scale"], g.GradKLN, wiredTol)
	closeTo(t, "grad_pds", grads["stacked_xf.0.attn.per_dim_scale.per_dim_scale"], g.GradPds, wiredTol)
}

func TestDecoderLayerBackwardMatchesGolden(t *testing.T) {
	raw := map[string]json.RawMessage{}
	readGrad(t, "timesfm_decoderlayer_grad_golden.json", &raw)
	var n, hid, nh, hd int
	get := func(key string, dst any) {
		if err := json.Unmarshal(raw[key], dst); err != nil {
			t.Fatalf("%s: %v", key, err)
		}
	}
	get("n", &n)
	get("hid", &hid)
	get("nh", &nh)
	get("hd", &hd)
	vec := func(key string) []float32 {
		var v []float64
		get(key, &v)
		return f32(v)
	}
	want := func(key string) []float64 {
		var v []float64
		get(key, &v)
		return v
	}
	model := syntheticLayerModel(hid, nh, hd, map[string][]float32{
		"stacked_xf.0.pre_attn_ln.scale":                vec("preA"),
		"stacked_xf.0.post_attn_ln.scale":               vec("postA"),
		"stacked_xf.0.pre_ff_ln.scale":                  vec("preF"),
		"stacked_xf.0.post_ff_ln.scale":                 vec("postF"),
		"stacked_xf.0.ff0.weight":                       vec("ff0"),
		"stacked_xf.0.ff1.weight":                       vec("ff1"),
		"stacked_xf.0.attn.qkv_proj.weight":             vec("qkvW"),
		"stacked_xf.0.attn.out.weight":                  vec("outW"),
		"stacked_xf.0.attn.query_ln.scale":              vec("qLN"),
		"stacked_xf.0.attn.key_ln.scale":                vec("kLN"),
		"stacked_xf.0.attn.per_dim_scale.per_dim_scale": vec("pds"),
	})
	grads := Grads{}
	invFreq := hostmath.RopeInvFreq(model.Dims.RopeTheta, hd)
	dx, err := model.layerBackward(0, vec("x"), vec("dout"), invFreq, n, grads)
	if err != nil {
		t.Fatal(err)
	}
	closeTo(t, "grad_x", dx, want("grad_x"), wiredTol)
	for name, key := range map[string]string{
		"stacked_xf.0.pre_attn_ln.scale":                "grad_preA",
		"stacked_xf.0.post_attn_ln.scale":               "grad_postA",
		"stacked_xf.0.pre_ff_ln.scale":                  "grad_preF",
		"stacked_xf.0.post_ff_ln.scale":                 "grad_postF",
		"stacked_xf.0.ff0.weight":                       "grad_ff0",
		"stacked_xf.0.ff1.weight":                       "grad_ff1",
		"stacked_xf.0.attn.qkv_proj.weight":             "grad_qkvW",
		"stacked_xf.0.attn.out.weight":                  "grad_outW",
		"stacked_xf.0.attn.query_ln.scale":              "grad_qLN",
		"stacked_xf.0.attn.key_ln.scale":                "grad_kLN",
		"stacked_xf.0.attn.per_dim_scale.per_dim_scale": "grad_pds",
	} {
		closeTo(t, key, grads[name], want(key), wiredTol)
	}
}
