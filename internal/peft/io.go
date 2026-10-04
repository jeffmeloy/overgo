package peft

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/safetensors"
)

// The names PEFT gives an adapter's tensors: the wrapped model's module path
// under a fixed prefix, then which factor of the pair the tensor is.
const (
	weightPrefix = "base_model.model."
	downSuffix   = ".lora_A.weight"
	upSuffix     = ".lora_B.weight"
	// peftType and taskType are the adapter kind and task PEFT records.
	peftType = "LORA"
	taskType = "CAUSAL_LM"
	// factorRank is the rank of a factor tensor: each is a matrix. A module's
	// pair is two such tensors, a down and an up.
	factorRank     = 2
	factorsPerPair = 2
	// regexMeta are the characters that would make a pattern key a regular
	// expression rather than a module name; "." is a path separator here.
	regexMeta = `\^$*+?()[]{}|`
)

// document is adapter_config.json as PEFT writes it. A reader takes the fields
// it needs and the ones that change the arithmetic, and ignores the rest.
type document struct {
	AlphaPattern          map[string]float64 `json:"alpha_pattern"`
	BaseModel             string             `json:"base_model_name_or_path"`
	Bias                  string             `json:"bias"`
	FanInFanOut           bool               `json:"fan_in_fan_out"`
	InferenceMode         bool               `json:"inference_mode"`
	InitLoRAWeights       bool               `json:"init_lora_weights"`
	LayerReplication      json.RawMessage    `json:"layer_replication"`
	LoRABias              bool               `json:"lora_bias"`
	Alpha                 float64            `json:"lora_alpha"`
	Dropout               float64            `json:"lora_dropout"`
	ModulesToSave         json.RawMessage    `json:"modules_to_save"`
	PeftType              string             `json:"peft_type"`
	Rank                  int                `json:"r"`
	RankPattern           map[string]int     `json:"rank_pattern"`
	TargetModules         json.RawMessage    `json:"target_modules"`
	TaskType              string             `json:"task_type"`
	TrainableTokenIndices json.RawMessage    `json:"trainable_token_indices"`
	UseDoRA               bool               `json:"use_dora"`
	UseRSLoRA             bool               `json:"use_rslora"`
}

// Save writes the adapter to directory as PEFT does: the tensors, then the
// config that marks the directory complete. Tensors are stored F32.
func Save(directory string, adapter Adapter) error {
	if err := adapter.Validate(); err != nil {
		return err
	}
	if err := os.MkdirAll(directory, clioptions.OutputDirectoryMode); err != nil {
		return err
	}
	tensors := make(map[string][]float32, factorsPerPair*len(adapter.Modules))
	shapes := make(map[string][]int, factorsPerPair*len(adapter.Modules))
	for name, module := range adapter.Modules {
		down, up := weightPrefix+name+downSuffix, weightPrefix+name+upSuffix
		tensors[down], shapes[down] = module.Down, []int{module.Rank, module.In}
		tensors[up], shapes[up] = module.Up, []int{module.Out, module.Rank}
	}
	if err := safetensors.Save(filepath.Join(directory, weightsFile), tensors, shapes, map[string]string{"format": "pt"}); err != nil {
		return err
	}
	return jsonfile.Write(filepath.Join(directory, configFile), adapter.document(), clioptions.OutputFileMode)
}

// document is the config PEFT would write for the adapter: the defaults, with a
// pattern entry for each module whose rank or alpha differs from them.
func (adapter Adapter) document() document {
	config := adapter.Config
	rankPattern, alphaPattern := map[string]int{}, map[string]float64{}
	targets := slices.Clone(config.TargetModules)
	for name, module := range adapter.Modules {
		if module.Rank != config.Rank {
			rankPattern[name] = module.Rank
		}
		if module.Alpha != config.Alpha {
			alphaPattern[name] = module.Alpha
		}
		if len(config.TargetModules) == 0 {
			targets = append(targets, name[strings.LastIndex(name, ".")+1:])
		}
	}
	slices.Sort(targets)
	targets = slices.Compact(targets)
	encoded, _ := json.Marshal(targets) // a string slice always encodes
	return document{
		AlphaPattern: alphaPattern, BaseModel: config.BaseModel, Bias: "none", InferenceMode: true, InitLoRAWeights: true,
		Alpha: config.Alpha, Dropout: config.Dropout, PeftType: peftType, Rank: config.Rank, RankPattern: rankPattern,
		TargetModules: encoded, TaskType: taskType, UseRSLoRA: config.RSLoRA,
	}
}

// Load reads the adapter in directory. It refuses an adapter whose arithmetic
// the factors alone do not carry: DoRA, bias or modules saved whole, an
// embedding adapter, layer replication, a transposed fan-in layout, or tensors
// that are not a LoRA factor pair.
func Load(directory string) (Adapter, error) {
	var doc document
	if err := jsonfile.Decode(filepath.Join(directory, configFile), &doc); err != nil {
		return Adapter{}, fmt.Errorf("peft: read %s: %w", configFile, err)
	}
	if err := doc.supported(); err != nil {
		return Adapter{}, err
	}
	var targets []string
	if err := json.Unmarshal(doc.TargetModules, &targets); err != nil && present(doc.TargetModules) {
		// A string target is a regular expression over module names; the tensors name the adapted ones.
		targets = nil
	}
	adapter := Adapter{
		Config:  Config{BaseModel: doc.BaseModel, Rank: doc.Rank, Alpha: doc.Alpha, Dropout: doc.Dropout, TargetModules: targets, RSLoRA: doc.UseRSLoRA},
		Modules: map[string]Module{},
	}
	source, err := safetensors.OpenFile(filepath.Join(directory, weightsFile))
	if err != nil {
		return Adapter{}, fmt.Errorf("peft: open %s: %w", weightsFile, err)
	}
	defer source.Close()
	type factors struct{ down, up *safetensors.Tensor }
	pairs := map[string]*factors{}
	for _, name := range source.Names() {
		tensor := source.Tensors[name]
		rest, prefixed := strings.CutPrefix(name, weightPrefix)
		var module string
		var isDown bool
		switch {
		case prefixed && strings.HasSuffix(rest, downSuffix):
			module, isDown = strings.TrimSuffix(rest, downSuffix), true
		case prefixed && strings.HasSuffix(rest, upSuffix):
			module = strings.TrimSuffix(rest, upSuffix)
		default:
			return Adapter{}, fmt.Errorf("peft: tensor %q is not a LoRA linear factor: embedding, DoRA, bias and saved-module tensors are not supported", name)
		}
		if pairs[module] == nil {
			pairs[module] = &factors{}
		}
		if isDown {
			pairs[module].down = &tensor
		} else {
			pairs[module].up = &tensor
		}
	}
	for name, pair := range pairs {
		if pair.down == nil || pair.up == nil {
			return Adapter{}, fmt.Errorf("peft: module %q has one factor of its pair", name)
		}
		downShape, downErr := safetensors.HostShape(*pair.down, factorRank)
		upShape, upErr := safetensors.HostShape(*pair.up, factorRank)
		if downErr != nil || upErr != nil || downShape[0] != upShape[1] {
			return Adapter{}, fmt.Errorf("peft: module %q has factors of shapes %v and %v, want rank-by-inputs and outputs-by-rank", name, pair.down.Shape, pair.up.Shape)
		}
		rank := downShape[0]
		if want := patterned(doc.RankPattern, name, doc.Rank); rank != want {
			return Adapter{}, fmt.Errorf("peft: module %q has rank %d, its config says %d", name, rank, want)
		}
		down, err := safetensors.ReadF32(*pair.down)
		if err != nil {
			return Adapter{}, fmt.Errorf("peft: module %q: %w", name, err)
		}
		up, err := safetensors.ReadF32(*pair.up)
		if err != nil {
			return Adapter{}, fmt.Errorf("peft: module %q: %w", name, err)
		}
		adapter.Modules[name] = Module{Rank: rank, In: downShape[1], Out: upShape[0], Alpha: patterned(doc.AlphaPattern, name, doc.Alpha), Down: down, Up: up}
	}
	return adapter, adapter.Validate()
}

// present reports whether an optional config field holds a value.
func present(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	return len(trimmed) != 0 && !bytes.Equal(trimmed, []byte("null")) && !bytes.Equal(trimmed, []byte("[]")) && !bytes.Equal(trimmed, []byte("{}"))
}

// supported refuses a config whose adapter the factors alone would not carry.
func (doc document) supported() error {
	switch {
	case doc.PeftType != peftType:
		return fmt.Errorf("peft: adapter type %q is not %s", doc.PeftType, peftType)
	case doc.Rank <= 0:
		return fmt.Errorf("peft: adapter rank %d is not positive", doc.Rank)
	case doc.UseDoRA:
		return fmt.Errorf("peft: a DoRA adapter carries magnitude vectors this reader does not apply")
	case doc.Bias != "" && doc.Bias != "none":
		return fmt.Errorf("peft: adapter bias mode %q trains base biases this reader does not apply", doc.Bias)
	case doc.LoRABias:
		return fmt.Errorf("peft: adapter lora_bias carries bias vectors this reader does not apply")
	case doc.FanInFanOut:
		return fmt.Errorf("peft: fan_in_fan_out adapters hold transposed layers this reader does not apply")
	case present(doc.ModulesToSave):
		return fmt.Errorf("peft: adapter modules_to_save holds whole modules this reader does not apply")
	case present(doc.LayerReplication):
		return fmt.Errorf("peft: adapter layer_replication repeats layers this reader does not apply")
	case present(doc.TrainableTokenIndices):
		return fmt.Errorf("peft: adapter trainable_token_indices trains embedding rows this reader does not apply")
	}
	for name := range doc.RankPattern {
		if strings.ContainsAny(name, regexMeta) {
			return fmt.Errorf("peft: rank_pattern key %q is a regular expression, not a module name", name)
		}
	}
	for name := range doc.AlphaPattern {
		if strings.ContainsAny(name, regexMeta) {
			return fmt.Errorf("peft: alpha_pattern key %q is a regular expression, not a module name", name)
		}
	}
	return nil
}

// patterned is the value a pattern map gives the module name, else fallback.
// An exact key wins; otherwise the longest key that names the module's trailing
// path, which is how PEFT matches a key against a module.
func patterned[T any](patterns map[string]T, name string, fallback T) T {
	if value, found := patterns[name]; found {
		return value
	}
	longest := ""
	for key := range patterns {
		if strings.HasSuffix(name, "."+key) && len(key) > len(longest) {
			longest = key
		}
	}
	if longest != "" {
		return patterns[longest]
	}
	return fallback
}
