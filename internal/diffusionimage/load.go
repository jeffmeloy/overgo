// Package diffusionimage owns the u-vit flow-matching image-generation
// capability (ladder rung 9): an asymmetric residual U-shaped transformer
// (conv ResBlocks at the outer levels, tensor-product-attention transformer
// blocks in the middle), trained as an OT linear-flow velocity field and
// sampled by Euler integration. Reference: adaptive_new
// go/extmodel/asymmetric_image_transformer.go (+_backward, _training);
// vendor source ships in-artifact (train.py, models/uvit.py).
//
// Every geometric dimension is DERIVED from the artifact's flat tensor
// lengths and module index walk; config.json contributes only the family
// tag. Three facts are vendor-source constants the shapes cannot carry:
// GroupNorm groups (nn.GroupNorm(32, ch)), norm eps (torch default 1e-5),
// and RoPE theta (Rotary base=10000).
package diffusionimage

import (
	"fmt"
	"io"
	"math"
	"path/filepath"
	"strings"

	"overgo/internal/jsonfile"
	"overgo/internal/safetensors"
)

const (
	// vendorGroups: models/uvit.py ResBlock nn.GroupNorm(32, channels).
	vendorGroups = 32
	// vendorNormEps: torch GroupNorm/LayerNorm default (vendor passes none).
	vendorNormEps = 1e-5
	// vendorRopeTheta: models/uvit.py Rotary(dim, base=10000).
	vendorRopeTheta = 10000
	// compilePrefix: torch.compile checkpoint key prefix, stripped on load.
	compilePrefix = "_orig_mod."
	// modelType: config.json family tag.
	modelType = "uvit"
)

// Config: derived architecture facts (see package doc for provenance).
type Config struct {
	InChannels   int
	BaseChannels int
	PatchSize    int
	NumLevels    int
	MidBlocks    int
	Groups       int
	Heads        int
	QRank        int
	KVRank       int
	NormEps      float64
	RopeTheta    float64
}

// convBinding: one Conv2d/ConvTranspose2d parameter pair.
type convBinding struct {
	name         string
	weight, bias []float32
}

// block: one level operator (ResBlock or transformer block).
type block interface {
	forward(*Model, []float32, int, int, int, int) []float32
	backward(*Model, []float32, []float32, int, int, int, int, Grads) []float32
}

// resBlock: GroupNorm-SiLU-Conv3x3 twice with a learned residual scale.
type resBlock struct {
	name                                           string
	norm1Weight, norm1Bias, conv1Weight, conv1Bias []float32
	norm2Weight, norm2Bias, conv2Weight, conv2Bias []float32
	residualScale                                  float32
}

// tpaWeights: tensor-product attention factors plus xATGLU output proj.
type tpaWeights struct {
	WAq, WAk, WAv []float32
	WBq, WBk, WBv []float32
	Woproj        []float32
	Boproj        []float32
	AlphaO        float64
}

// attnBlock: pre-norm TPA + xATGLU MLP with learned residual scales.
// Token layout is the vendor's raw reshape: the NCHW slab reinterpreted as
// [b*h*w, c] rows (golden-locked; see tokensToImageNCHW).
type attnBlock struct {
	name                                           string
	norm1Weight, norm1Bias, norm2Weight, norm2Bias []float32
	attention                                      tpaWeights
	mlpProjection, mlpOutput                       []float32
	mlpProjected, mlpHidden                        int
	mlpAlpha, attentionScale, mlpScale             float32
}

// level: block stack plus the 1x1 transition conv toward the next level.
type level struct {
	name       string
	blocks     []block
	transition convBinding
}

// Model: derived config plus compiled operator bindings. raw retains the
// checkpoint tensor map the bindings alias — the training update surface.
type Model struct {
	Cfg            Config
	patch, final   convBinding
	encoders       []level
	middle         []*attnBlock
	decoders       []level
	middleResidual float32
	raw            map[string][]float32
}

// Load reads config.json (family tag) and the artifact's single top-level
// safetensors file, strips the torch.compile prefix, derives the
// architecture from flat tensor lengths, and binds weights.
func Load(directory string) (*Model, error) {
	var cfg struct {
		ModelType string `json:"model_type"`
	}
	if err := jsonfile.Decode(filepath.Join(directory, "config.json"), &cfg); err != nil {
		return nil, fmt.Errorf("diffusionimage: parse config.json: %w", err)
	}
	if cfg.ModelType != modelType {
		return nil, fmt.Errorf("diffusionimage: model_type %q, want %q", cfg.ModelType, modelType)
	}
	tensors, err := loadTensors(directory)
	if err != nil {
		return nil, err
	}
	return Compile(tensors)
}

func loadTensors(directory string) (map[string][]float32, error) {
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		return nil, err
	}
	defer source.Close()
	tensors := make(map[string][]float32, len(source.Tensors))
	for name, tensor := range source.Tensors {
		reader, err := safetensors.F32Reader(tensor)
		if err != nil {
			return nil, fmt.Errorf("diffusionimage: tensor %q: %w", name, err)
		}
		elements := tensor.Elements()
		raw := make([]byte, elements*4)
		if _, err := io.ReadFull(reader, raw); err != nil {
			return nil, fmt.Errorf("diffusionimage: tensor %q payload: %w", name, err)
		}
		values := make([]float32, elements)
		for i := range values {
			bits := uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 | uint32(raw[4*i+3])<<24
			values[i] = math.Float32frombits(bits)
		}
		tensors[strings.TrimPrefix(name, compilePrefix)] = values
	}
	return tensors, nil
}

// deriveConfig reconstructs the architecture from flat tensor lengths.
func deriveConfig(tensors map[string][]float32) (Config, error) {
	need := func(name string) ([]float32, error) {
		v, ok := tensors[name]
		if !ok {
			return nil, fmt.Errorf("diffusionimage: checkpoint missing %s", name)
		}
		return v, nil
	}
	patchBias, err := need("patch_embed.bias")
	if err != nil {
		return Config{}, err
	}
	finalBias, err := need("final_proj.bias")
	if err != nil {
		return Config{}, err
	}
	patchWeight, err := need("patch_embed.weight")
	if err != nil {
		return Config{}, err
	}
	c := Config{
		InChannels: len(finalBias), BaseChannels: len(patchBias),
		Groups: vendorGroups, NormEps: vendorNormEps, RopeTheta: vendorRopeTheta,
	}
	if c.InChannels <= 0 || c.BaseChannels <= 0 {
		return Config{}, fmt.Errorf("diffusionimage: empty projection biases")
	}
	// patch_embed.weight is [base, in, k, k] flat.
	kk := len(patchWeight) / (c.BaseChannels * c.InChannels)
	c.PatchSize = int(math.Round(math.Sqrt(float64(kk))))
	if c.PatchSize*c.PatchSize != kk || c.PatchSize <= 0 {
		return Config{}, fmt.Errorf("diffusionimage: patch_embed.weight len %d not square kernel over %dx%d", len(patchWeight), c.BaseChannels, c.InChannels)
	}
	if fw, err := need("final_proj.weight"); err != nil {
		return Config{}, err
	} else if len(fw) != len(patchWeight) {
		// ConvTranspose2d [base, in, k, k]: same element count as patch_embed.
		return Config{}, fmt.Errorf("diffusionimage: final_proj.weight len %d != patch_embed %d", len(fw), len(patchWeight))
	}
	// Levels: encoder modules alternate blocks (even index) and 1x1
	// transitions (odd index); count the level modules.
	for module := 0; ; module += 2 {
		if _, ok := tensors[fmt.Sprintf("encoders.%d.blocks.0.norm1.weight", module)]; !ok {
			break
		}
		c.NumLevels++
	}
	if c.NumLevels == 0 {
		return Config{}, fmt.Errorf("diffusionimage: no encoder levels")
	}
	// Middle blocks: contiguous middle.N transformer stack.
	for ; ; c.MidBlocks++ {
		if _, ok := tensors[fmt.Sprintf("middle.%d.norm1.weight", c.MidBlocks)]; !ok {
			break
		}
	}
	if c.MidBlocks == 0 {
		return Config{}, fmt.Errorf("diffusionimage: no middle blocks")
	}
	// Middle width doubles per level below the top.
	mid, err := need("middle.0.norm1.weight")
	if err != nil {
		return Config{}, err
	}
	cMid := len(mid)
	if want := c.BaseChannels << uint(c.NumLevels-1); cMid != want {
		return Config{}, fmt.Errorf("diffusionimage: middle width %d != base %d << %d", cMid, c.BaseChannels, c.NumLevels-1)
	}
	// TPA ranks and heads from factor lengths: len(WAq)=heads*qRank*cMid,
	// len(WBq)=qRank*headDim*cMid, headDim=cMid/heads =>
	// heads^2 = len(WAq)*cMid/len(WBq).
	waq, err := need("middle.0.attn.c_qkv.W_A_q.weight")
	if err != nil {
		return Config{}, err
	}
	wbq, err := need("middle.0.attn.c_qkv.W_B_q.weight")
	if err != nil {
		return Config{}, err
	}
	wak, err := need("middle.0.attn.c_qkv.W_A_k.weight")
	if err != nil {
		return Config{}, err
	}
	headsSq := len(waq) * cMid / len(wbq)
	c.Heads = int(math.Round(math.Sqrt(float64(headsSq))))
	if c.Heads <= 0 || c.Heads*c.Heads != headsSq || cMid%c.Heads != 0 {
		return Config{}, fmt.Errorf("diffusionimage: heads^2=%d not a square dividing width %d", headsSq, cMid)
	}
	c.QRank = len(waq) / (c.Heads * cMid)
	c.KVRank = len(wak) / (c.Heads * cMid)
	if c.QRank*c.Heads*cMid != len(waq) || c.KVRank*c.Heads*cMid != len(wak) || c.QRank <= 0 || c.KVRank <= 0 {
		return Config{}, fmt.Errorf("diffusionimage: TPA factor lengths q=%d k=%d indivisible at heads=%d width=%d", len(waq), len(wak), c.Heads, cMid)
	}
	if _, ok := tensors["learned_middle_residual_scale"]; !ok {
		return Config{}, fmt.Errorf("diffusionimage: checkpoint missing learned_middle_residual_scale")
	}
	return c, nil
}

// Compile derives the config from the tensors and binds every weight.
func Compile(tensors map[string][]float32) (*Model, error) {
	cfg, err := deriveConfig(tensors)
	if err != nil {
		return nil, err
	}
	bind := func(name string) ([]float32, error) {
		if v, ok := tensors[name]; ok {
			return v, nil
		}
		return nil, fmt.Errorf("diffusionimage: checkpoint missing %s", name)
	}
	scalar := func(name string) (float32, error) {
		v, err := bind(name)
		if err != nil || len(v) == 0 {
			return 0, fmt.Errorf("diffusionimage: scalar %s missing or empty", name)
		}
		return v[0], nil
	}
	bindMany := func(prefix string, suffixes ...string) ([][]float32, error) {
		bound := make([][]float32, len(suffixes))
		for i, suffix := range suffixes {
			v, err := bind(prefix + suffix)
			if err != nil {
				return nil, err
			}
			bound[i] = v
		}
		return bound, nil
	}
	compileAttn := func(prefix string) (*attnBlock, error) {
		v, err := bindMany(prefix,
			".norm1.weight", ".norm1.bias", ".norm2.weight", ".norm2.bias",
			".attn.c_qkv.W_A_q.weight", ".attn.c_qkv.W_A_k.weight", ".attn.c_qkv.W_A_v.weight",
			".attn.c_qkv.W_B_q.weight", ".attn.c_qkv.W_B_k.weight", ".attn.c_qkv.W_B_v.weight",
			".attn.o_proj.proj.weight", ".attn.o_proj.proj.bias", ".attn.o_proj.alpha",
			".mlp.0.proj.weight", ".mlp.0.alpha", ".mlp.1.weight",
			".learned_residual_scale_attn", ".learned_residual_scale_mlp")
		if err != nil {
			return nil, err
		}
		for _, scalarIndex := range []int{12, 14, 16, 17} {
			if len(v[scalarIndex]) == 0 {
				return nil, fmt.Errorf("diffusionimage: %s scalar tensor %d empty", prefix, scalarIndex)
			}
		}
		hidden := len(v[0])
		if hidden == 0 || len(v[13])%hidden != 0 || len(v[15])%hidden != 0 {
			return nil, fmt.Errorf("diffusionimage: %s MLP tensor lengths incompatible with width %d", prefix, hidden)
		}
		projected, gated := len(v[13])/hidden, len(v[15])/hidden
		if projected != 2*gated {
			return nil, fmt.Errorf("diffusionimage: %s MLP projection %d != 2*gated %d", prefix, projected, gated)
		}
		return &attnBlock{
			name:        prefix,
			norm1Weight: v[0], norm1Bias: v[1], norm2Weight: v[2], norm2Bias: v[3],
			attention: tpaWeights{
				WAq: v[4], WAk: v[5], WAv: v[6], WBq: v[7], WBk: v[8], WBv: v[9],
				Woproj: v[10], Boproj: v[11], AlphaO: float64(v[12][0]),
			},
			mlpProjection: v[13], mlpAlpha: v[14][0], mlpOutput: v[15],
			mlpProjected: projected, mlpHidden: gated,
			attentionScale: v[16][0], mlpScale: v[17][0],
		}, nil
	}
	compileRes := func(prefix string) (*resBlock, error) {
		v, err := bindMany(prefix,
			".norm1.weight", ".norm1.bias", ".conv1.weight", ".conv1.bias",
			".norm2.weight", ".norm2.bias", ".conv2.weight", ".conv2.bias",
			".learned_residual_scale")
		if err != nil {
			return nil, err
		}
		if len(v[8]) == 0 {
			return nil, fmt.Errorf("diffusionimage: %s residual scale empty", prefix)
		}
		return &resBlock{
			name:        prefix,
			norm1Weight: v[0], norm1Bias: v[1], conv1Weight: v[2], conv1Bias: v[3],
			norm2Weight: v[4], norm2Bias: v[5], conv2Weight: v[6], conv2Bias: v[7],
			residualScale: v[8][0],
		}, nil
	}
	compileLevels := func(family string) ([]level, error) {
		levels := make([]level, cfg.NumLevels)
		module := 0
		for l := range levels {
			levels[l].name = fmt.Sprintf("%s.%d", family, module)
			// Block type and count discovered per index (thresholds trained
			// into the checkpoint; conv1 presence marks a ResBlock).
			for b := 0; ; b++ {
				prefix := fmt.Sprintf("%s.%d.blocks.%d", family, module, b)
				var compiled block
				var err error
				if _, ok := tensors[prefix+".conv1.weight"]; ok {
					compiled, err = compileRes(prefix)
				} else if _, ok := tensors[prefix+".norm1.weight"]; ok {
					compiled, err = compileAttn(prefix)
				} else {
					break
				}
				if err != nil {
					return nil, err
				}
				levels[l].blocks = append(levels[l].blocks, compiled)
			}
			if len(levels[l].blocks) == 0 {
				return nil, fmt.Errorf("diffusionimage: %s.%d has no blocks", family, module)
			}
			module++
			if l < len(levels)-1 {
				name := fmt.Sprintf("%s.%d", family, module)
				weight, err := bind(name + ".weight")
				if err != nil {
					return nil, err
				}
				bias, _ := bind(name + ".bias")
				levels[l].transition = convBinding{name: name, weight: weight, bias: bias}
				module++
			}
		}
		return levels, nil
	}

	m := &Model{Cfg: cfg, raw: tensors}
	var err2 error
	if m.patch.weight, err2 = bind("patch_embed.weight"); err2 != nil {
		return nil, err2
	}
	m.patch.name = "patch_embed"
	m.patch.bias, _ = bind("patch_embed.bias")
	if m.final.weight, err2 = bind("final_proj.weight"); err2 != nil {
		return nil, err2
	}
	m.final.name = "final_proj"
	m.final.bias, _ = bind("final_proj.bias")
	if m.middleResidual, err2 = scalar("learned_middle_residual_scale"); err2 != nil {
		return nil, err2
	}
	if m.encoders, err2 = compileLevels("encoders"); err2 != nil {
		return nil, err2
	}
	if m.decoders, err2 = compileLevels("decoders"); err2 != nil {
		return nil, err2
	}
	m.middle = make([]*attnBlock, cfg.MidBlocks)
	for i := range m.middle {
		if m.middle[i], err2 = compileAttn(fmt.Sprintf("middle.%d", i)); err2 != nil {
			return nil, err2
		}
	}
	return m, nil
}
