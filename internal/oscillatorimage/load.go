// Package oscillatorimage owns the conditional-oscillator image-generation
// capability (ladder rung 8): coupled Kuramoto phase dynamics driven by a
// class-conditional term, a sin/cos phase readout, and a resize-conv decoder.
// Reference: adaptive_new go/extmodel/conditional_oscillator_image.go (the
// artifact is adaptive-TRAINED; provenance family carried by the artifact).
//
// Every geometric dimension is DERIVED from the artifact's flat tensor
// lengths; config.json contributes only execution facts the shapes cannot
// carry (integration horizon, plane split, readout conventions) and restated
// dims that are cross-checked against the derivation.
package oscillatorimage

import (
	"fmt"
	"path/filepath"
	"strings"

	"overgo/internal/jsonfile"
	"overgo/internal/safetensors"
)

const (
	convKernel = 3
	convTaps   = convKernel * convKernel
	upsample   = 2
	// decoderNegativeSlope: not serialized in the artifact config; the
	// reference records it as a repodb reference fact (conditional-oscillator
	// decoder.negative_slope = 0.2, source torch leaky_relu in the artifact
	// generator scripts).
	decoderNegativeSlope = 0.2
)

// Config: artifact config.json. num_steps/dt and the k_* scales are Euler
// execution facts; in_h/in_w split the readout plane (only the product is
// shape-constrained); relativization/encoding/tanh_out are readout/decoder
// conventions. The reference reads all of them from config, none derived.
type Config struct {
	ModelType      string  `json:"model_type"`
	N              int     `json:"n"`
	NCond          int     `json:"n_cond"`
	NClasses       int     `json:"n_classes"`
	InChannels     int     `json:"in_channels"`
	InH            int     `json:"in_h"`
	InW            int     `json:"in_w"`
	OutChannels    int     `json:"out_channels"`
	NumSteps       int     `json:"num_steps"`
	Dt             float64 `json:"dt"`
	KScale         float64 `json:"k_scale"`
	KCondScale     float64 `json:"k_cond_scale"`
	KDriveScale    float64 `json:"k_drive_scale"`
	Relativization string  `json:"relativization"`
	Encoding       string  `json:"encoding"`
	TanhOut        bool    `json:"tanh_out"`
	BlockChannels  []int   `json:"block_channels"`
}

// DecoderBlock: one upsample-conv-conv stage.
type DecoderBlock struct {
	W1, B1, W2, B2 []float32
	Cout           int
}

// Model: validated config, derived namespace, and bound tensors.
type Model struct {
	Cfg       Config
	Namespace string
	Slope     float64

	Omega, OmegaCond []float32
	K, KCond         []float32
	Drive            []float32
	Blocks           []DecoderBlock
	ToOutW, ToOutB   []float32
}

// OutH and OutW: decoded image dims (each block doubles H and W).
func (c Config) OutH() int { return c.InH << uint(len(c.BlockChannels)) }
func (c Config) OutW() int { return c.InW << uint(len(c.BlockChannels)) }

func (c Config) readoutWidth() int {
	if c.Encoding == "sin_cos" {
		return 2 * c.N
	}
	return c.N
}

// Load reads config.json and the safetensors weights, derives dims from flat
// tensor lengths, and cross-checks the config's restated dims.
func Load(directory string) (*Model, error) {
	var cfg Config
	if err := jsonfile.Decode(filepath.Join(directory, "config.json"), &cfg); err != nil {
		return nil, fmt.Errorf("oscillatorimage: parse config.json: %w", err)
	}
	tensors, err := loadTensors(directory)
	if err != nil {
		return nil, err
	}
	return bind(cfg, tensors)
}

func loadTensors(directory string) (map[string][]float32, error) {
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		return nil, err
	}
	defer source.Close()
	catalog, err := source.MaterializeF32(safetensors.F32Selection{})
	if err != nil {
		return nil, fmt.Errorf("oscillatorimage: materialize: %w", err)
	}
	return catalog.Values, nil
}

// namespaceOf: the artifact family namespace, derived from the unique
// ".to_out.weight" tensor (never asserted in code).
func namespaceOf(tensors map[string][]float32) (string, error) {
	const anchor = ".to_out.weight"
	namespace := ""
	for name := range tensors {
		if !strings.HasSuffix(name, anchor) {
			continue
		}
		if namespace != "" {
			return "", fmt.Errorf("oscillatorimage: multiple output tensors (%q, %q)", namespace, name)
		}
		namespace = strings.TrimSuffix(name, anchor)
	}
	if namespace == "" {
		return "", fmt.Errorf("oscillatorimage: no *.to_out.weight tensor")
	}
	return namespace, nil
}

func bind(cfg Config, tensors map[string][]float32) (*Model, error) {
	namespace, err := namespaceOf(tensors)
	if err != nil {
		return nil, err
	}
	get := func(suffix string) ([]float32, error) {
		values, ok := tensors[namespace+"."+suffix]
		if !ok {
			return nil, fmt.Errorf("oscillatorimage: missing tensor %s.%s", namespace, suffix)
		}
		return values, nil
	}
	m := &Model{Cfg: cfg, Namespace: namespace, Slope: decoderNegativeSlope}
	for _, binding := range []struct {
		dst    *[]float32
		suffix string
	}{
		{&m.Omega, "omega"}, {&m.OmegaCond, "omega_cond"},
		{&m.K, "k"}, {&m.KCond, "k_cond"}, {&m.Drive, "k_drive"},
		{&m.ToOutW, "to_out.weight"}, {&m.ToOutB, "to_out.bias"},
	} {
		if *binding.dst, err = get(binding.suffix); err != nil {
			return nil, err
		}
	}
	// Derive dims from flat lengths.
	n, nc := len(m.Omega), len(m.OmegaCond)
	if n <= 0 || nc <= 0 {
		return nil, fmt.Errorf("oscillatorimage: empty frequency tensors")
	}
	if len(m.K) != n*n || len(m.KCond) != nc*nc {
		return nil, fmt.Errorf("oscillatorimage: coupling %d/%d != %d^2/%d^2", len(m.K), len(m.KCond), n, nc)
	}
	if len(m.Drive)%(n*nc) != 0 {
		return nil, fmt.Errorf("oscillatorimage: drive %d not divisible by n*nc=%d", len(m.Drive), n*nc)
	}
	nClasses := len(m.Drive) / (n * nc)
	// Decoder blocks: contiguous indices; cout from bias length, cin chained.
	var blocks []DecoderBlock
	cin := 0
	for index := 0; ; index++ {
		b1, ok := tensors[fmt.Sprintf("%s.blocks.%d.b1", namespace, index)]
		if !ok {
			break
		}
		part := func(part string) ([]float32, error) {
			return get(fmt.Sprintf("blocks.%d.%s", index, part))
		}
		w1, err := part("w1")
		if err != nil {
			return nil, err
		}
		w2, err := part("w2")
		if err != nil {
			return nil, err
		}
		b2, err := part("b2")
		if err != nil {
			return nil, err
		}
		cout := len(b1)
		if cout <= 0 || len(b2) != cout || len(w2) != cout*cout*convTaps || len(w1)%(cout*convTaps) != 0 {
			return nil, fmt.Errorf("oscillatorimage: block %d tensor lengths inconsistent", index)
		}
		blockCin := len(w1) / (cout * convTaps)
		if index == 0 {
			cin = blockCin
		} else if blockCin != blocks[index-1].Cout {
			return nil, fmt.Errorf("oscillatorimage: block %d cin %d != previous cout %d", index, blockCin, blocks[index-1].Cout)
		}
		blocks = append(blocks, DecoderBlock{W1: w1, B1: b1, W2: w2, B2: b2, Cout: cout})
	}
	if len(blocks) == 0 {
		return nil, fmt.Errorf("oscillatorimage: no decoder blocks")
	}
	m.Blocks = blocks
	lastCout := blocks[len(blocks)-1].Cout
	outChannels := len(m.ToOutB)
	if outChannels <= 0 || len(m.ToOutW) != outChannels*lastCout*convTaps {
		return nil, fmt.Errorf("oscillatorimage: output conv %d != %d*%d*%d", len(m.ToOutW), outChannels, lastCout, convTaps)
	}
	// Cross-check config restatements against the derivation (shapes win).
	derived := []struct {
		name         string
		fromShapes   int
		configClaims int
	}{
		{"n", n, cfg.N}, {"n_cond", nc, cfg.NCond}, {"n_classes", nClasses, cfg.NClasses},
		{"in_channels", cin, cfg.InChannels}, {"out_channels", outChannels, cfg.OutChannels},
		{"block_count", len(blocks), len(cfg.BlockChannels)},
	}
	for _, check := range derived {
		if check.fromShapes != check.configClaims {
			return nil, fmt.Errorf("oscillatorimage: %s derived %d but config claims %d", check.name, check.fromShapes, check.configClaims)
		}
	}
	for index, block := range blocks {
		if cfg.BlockChannels[index] != block.Cout {
			return nil, fmt.Errorf("oscillatorimage: block %d cout derived %d but config claims %d", index, block.Cout, cfg.BlockChannels[index])
		}
	}
	return m, validate(cfg)
}

func validate(cfg Config) error {
	if cfg.NumSteps <= 0 || cfg.Dt == 0 {
		return fmt.Errorf("oscillatorimage: integration facts num_steps=%d dt=%g", cfg.NumSteps, cfg.Dt)
	}
	if cfg.InH <= 0 || cfg.InW <= 0 {
		return fmt.Errorf("oscillatorimage: plane split %dx%d", cfg.InH, cfg.InW)
	}
	if cfg.readoutWidth() != cfg.InChannels*cfg.InH*cfg.InW {
		return fmt.Errorf("oscillatorimage: readout width %d != decoder input %d*%d*%d",
			cfg.readoutWidth(), cfg.InChannels, cfg.InH, cfg.InW)
	}
	switch cfg.Relativization {
	case "ref_oscillator", "mean_relative", "":
	default:
		return fmt.Errorf("oscillatorimage: unknown relativization %q", cfg.Relativization)
	}
	switch cfg.Encoding {
	case "sin", "sin_cos", "":
	default:
		return fmt.Errorf("oscillatorimage: unknown encoding %q", cfg.Encoding)
	}
	return nil
}
