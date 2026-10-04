package peft

import (
	"encoding/binary"
	"encoding/json"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/safetensors"
	"overgo/internal/tensor/dtype"
)

const (
	queryModule = "model.layers.0.self_attn.q_proj"
	valueModule = "model.layers.0.self_attn.v_proj"
)

// testAdapter has a query module at the config's rank and alpha and a value
// module of its own rank and alpha, so a round trip needs both patterns.
func testAdapter() Adapter {
	return Adapter{
		Config: Config{BaseModel: "base/model", Rank: 2, Alpha: 4, Dropout: 0.05},
		Modules: map[string]Module{
			queryModule: {Rank: 2, In: 4, Out: 3, Alpha: 4, Down: []float32{1, 2, 3, 4, -1, 0, 1, 2}, Up: []float32{0.5, 1, -1, 0, 2, 1}},
			valueModule: {Rank: 1, In: 4, Out: 2, Alpha: 3, Down: []float32{1, -1, 2, 0.5}, Up: []float32{2, -1}},
		},
	}
}

// product is Up·Down times scale, by the definition and in float64.
func product(module Module, scale float64) []float64 {
	result := make([]float64, module.Out*module.In)
	for row := range module.Out {
		for column := range module.In {
			for rank := range module.Rank {
				result[row*module.In+column] += float64(module.Up[row*module.Rank+rank]) * float64(module.Down[rank*module.In+column])
			}
			result[row*module.In+column] *= scale
		}
	}
	return result
}

func near(t *testing.T, got []float32, want []float64, tolerance float64, what string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s has %d values, want %d", what, len(got), len(want))
	}
	for index := range got {
		if math.Abs(float64(got[index])-want[index]) > tolerance {
			t.Fatalf("%s[%d] = %v, want %v", what, index, got[index], want[index])
		}
	}
}

// TestDeltaIsTheScaledProduct: a module's update is alpha over its rank times
// Up·Down, and alpha over the root of its rank when the adapter is rank-stabilized.
func TestDeltaIsTheScaledProduct(t *testing.T) {
	adapter := testAdapter()
	module := adapter.Modules[queryModule]
	delta, err := adapter.Delta(queryModule)
	if err != nil {
		t.Fatal(err)
	}
	near(t, delta, product(module, 4.0/2), 1e-6, "delta")
	adapter.Config.RSLoRA = true
	delta, err = adapter.Delta(queryModule)
	if err != nil {
		t.Fatal(err)
	}
	near(t, delta, product(module, 4/math.Sqrt(2)), 1e-6, "rank-stabilized delta")
	if _, err := adapter.Delta("model.layers.9.self_attn.q_proj"); err == nil {
		t.Fatal("a module the adapter lacks gave an update")
	}
}

// TestValidateRefusesWhatCannotBeApplied: factors that do not fill their
// shapes, scales that are not positive, values that are not finite, a tensor
// name given for a module name, and an empty adapter.
func TestValidateRefusesWhatCannotBeApplied(t *testing.T) {
	for name, change := range map[string]func(*Adapter){
		"short factor": func(a *Adapter) { m := a.Modules[queryModule]; m.Down = m.Down[1:]; a.Modules[queryModule] = m },
		"zero alpha":   func(a *Adapter) { m := a.Modules[queryModule]; m.Alpha = 0; a.Modules[queryModule] = m },
		"not finite": func(a *Adapter) {
			m := a.Modules[queryModule]
			m.Up[0] = float32(math.NaN())
			a.Modules[queryModule] = m
		},
		"tensor name":    func(a *Adapter) { a.Modules[queryModule+".weight"] = a.Modules[queryModule] },
		"no modules":     func(a *Adapter) { a.Modules = nil },
		"no rank":        func(a *Adapter) { a.Config.Rank = 0 },
		"full dropout":   func(a *Adapter) { a.Config.Dropout = 1 },
		"negative alpha": func(a *Adapter) { a.Config.Alpha = -1 },
	} {
		adapter := testAdapter()
		change(&adapter)
		if adapter.Validate() == nil {
			t.Errorf("%s: an adapter that cannot be applied validated", name)
		}
	}
	if err := testAdapter().Validate(); err != nil {
		t.Fatalf("the test adapter does not validate: %v", err)
	}
}

// TestSaveWritesWhatPEFTWrites: the directory holds the config PEFT reads and
// tensors under its names and shapes, with a pattern entry for the module whose
// rank and alpha differ from the defaults; Load returns the adapter saved.
func TestSaveWritesWhatPEFTWrites(t *testing.T) {
	directory := filepath.Join(t.TempDir(), "adapter")
	adapter := testAdapter()
	if err := Save(directory, adapter); err != nil {
		t.Fatal(err)
	}
	var config map[string]any
	data, err := os.ReadFile(filepath.Join(directory, configFile))
	if err != nil || json.Unmarshal(data, &config) != nil {
		t.Fatalf("config unreadable: %v", err)
	}
	if config["peft_type"] != "LORA" || config["task_type"] != "CAUSAL_LM" || config["r"] != 2.0 || config["lora_alpha"] != 4.0 ||
		config["bias"] != "none" || config["base_model_name_or_path"] != "base/model" || config["use_dora"] != false {
		t.Fatalf("config = %v, want the fields PEFT's LoraConfig carries", config)
	}
	if want := map[string]any{valueModule: 1.0}; !mapsEqual(config["rank_pattern"], want) {
		t.Fatalf("rank_pattern = %v, want %v", config["rank_pattern"], want)
	}
	if want := map[string]any{valueModule: 3.0}; !mapsEqual(config["alpha_pattern"], want) {
		t.Fatalf("alpha_pattern = %v, want %v", config["alpha_pattern"], want)
	}
	if got, _ := json.Marshal(config["target_modules"]); string(got) != `["q_proj","v_proj"]` {
		t.Fatalf("target_modules = %s, want the adapted layer names, sorted", got)
	}
	source, err := safetensors.OpenFile(filepath.Join(directory, weightsFile))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	wantNames := []string{
		"base_model.model." + queryModule + ".lora_A.weight", "base_model.model." + queryModule + ".lora_B.weight",
		"base_model.model." + valueModule + ".lora_A.weight", "base_model.model." + valueModule + ".lora_B.weight",
	}
	if !slices.Equal(source.Names(), wantNames) {
		t.Fatalf("tensors = %v, want %v", source.Names(), wantNames)
	}
	if shape := source.Tensors[wantNames[0]].Shape; !slices.Equal(shape, []uint64{2, 4}) {
		t.Fatalf("lora_A shape = %v, want rank by inputs", shape)
	}
	if shape := source.Tensors[wantNames[1]].Shape; !slices.Equal(shape, []uint64{3, 2}) {
		t.Fatalf("lora_B shape = %v, want outputs by rank", shape)
	}
	loaded, err := Load(directory)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Config.Rank != 2 || loaded.Config.Alpha != 4 || loaded.Config.Dropout != 0.05 || loaded.Config.BaseModel != "base/model" ||
		!slices.Equal(loaded.Config.TargetModules, []string{"q_proj", "v_proj"}) || len(loaded.Modules) != 2 {
		t.Fatalf("loaded config = %+v with %d modules", loaded.Config, len(loaded.Modules))
	}
	for name, module := range adapter.Modules {
		got := loaded.Modules[name]
		if got.Rank != module.Rank || got.In != module.In || got.Out != module.Out || got.Alpha != module.Alpha ||
			!slices.Equal(got.Down, module.Down) || !slices.Equal(got.Up, module.Up) {
			t.Fatalf("module %s = %+v after a round trip, want %+v", name, got, module)
		}
	}
}

func mapsEqual(got any, want map[string]any) bool {
	entries, ok := got.(map[string]any)
	return ok && maps.Equal(entries, want)
}

// rawTensor is a tensor written by hand, as a PEFT adapter from a half-precision
// training run arrives.
type rawTensor struct {
	dtype string
	shape []int
	data  []byte
}

func f32Bytes(values ...float32) []byte {
	data := make([]byte, 4*len(values))
	for index, value := range values {
		binary.LittleEndian.PutUint32(data[4*index:], math.Float32bits(value))
	}
	return data
}

func bf16Bytes(values ...float32) []byte {
	data := make([]byte, 2*len(values))
	for index, value := range values {
		binary.LittleEndian.PutUint16(data[2*index:], dtype.Float32ToBF16(value))
	}
	return data
}

func writeRaw(t *testing.T, path string, tensors map[string]rawTensor) {
	t.Helper()
	header := map[string]any{"__metadata__": map[string]string{"format": "pt"}}
	var body []byte
	for _, name := range slices.Sorted(maps.Keys(tensors)) {
		raw := tensors[name]
		header[name] = map[string]any{"dtype": raw.dtype, "shape": raw.shape, "data_offsets": []int{len(body), len(body) + len(raw.data)}}
		body = append(body, raw.data...)
	}
	encoded, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	prefix := make([]byte, 8)
	binary.LittleEndian.PutUint64(prefix, uint64(len(encoded)))
	if err := os.WriteFile(path, append(append(prefix, encoded...), body...), 0o600); err != nil {
		t.Fatal(err)
	}
}

func writeConfig(t *testing.T, directory string, fields map[string]any) {
	t.Helper()
	config := map[string]any{"peft_type": "LORA", "r": 2, "lora_alpha": 4, "target_modules": []string{"q_proj"}, "bias": "none"}
	maps.Copy(config, fields)
	data, err := json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, configFile), data, 0o600); err != nil {
		t.Fatal(err)
	}
}

// TestLoadReadsHalfPrecisionFactorsAndSuffixPatterns: an adapter a half-precision
// run saved, with a config that names its modules by trailing path and holds
// fields newer PEFT versions add, loads with each module's own rank and alpha.
func TestLoadReadsHalfPrecisionFactorsAndSuffixPatterns(t *testing.T) {
	directory := t.TempDir()
	prefix := "base_model.model."
	writeRaw(t, filepath.Join(directory, weightsFile), map[string]rawTensor{
		prefix + queryModule + ".lora_A.weight": {"BF16", []int{2, 3}, bf16Bytes(1, 2, 3, 4, 5, 6)},
		prefix + queryModule + ".lora_B.weight": {"BF16", []int{2, 2}, bf16Bytes(1, 0, 0, 1)},
		prefix + valueModule + ".lora_A.weight": {"BF16", []int{1, 3}, bf16Bytes(1, 1, 1)},
		prefix + valueModule + ".lora_B.weight": {"BF16", []int{2, 1}, bf16Bytes(1, 2)},
	})
	writeConfig(t, directory, map[string]any{
		"rank_pattern": map[string]int{"v_proj": 1}, "alpha_pattern": map[string]float64{"self_attn.v_proj": 1},
		"eva_config": nil, "exclude_modules": nil, "target_modules": "q_proj|v_proj",
	})
	adapter, err := Load(directory)
	if err != nil {
		t.Fatal(err)
	}
	query, value := adapter.Modules[queryModule], adapter.Modules[valueModule]
	if query.Rank != 2 || query.Alpha != 4 || value.Rank != 1 || value.Alpha != 1 || !slices.Equal(query.Down, []float32{1, 2, 3, 4, 5, 6}) {
		t.Fatalf("modules = %+v and %+v, want ranks 2 and 1, alphas 4 and 1, the BF16 values exact", query, value)
	}
	if len(adapter.Config.TargetModules) != 0 {
		t.Fatalf("a target pattern read as the list %v", adapter.Config.TargetModules)
	}
}

// TestLoadRefusesWhatItCannotApply: each adapter whose arithmetic the factors
// alone do not carry, and each malformed one, is refused with its reason.
func TestLoadRefusesWhatItCannotApply(t *testing.T) {
	prefix := "base_model.model." + queryModule
	good := func() map[string]rawTensor {
		return map[string]rawTensor{
			prefix + ".lora_A.weight": {"F32", []int{2, 3}, f32Bytes(1, 2, 3, 4, 5, 6)},
			prefix + ".lora_B.weight": {"F32", []int{2, 2}, f32Bytes(1, 0, 0, 1)},
		}
	}
	for _, scenario := range []struct {
		name    string
		fields  map[string]any
		tensors func() map[string]rawTensor
		want    string
	}{
		{"not lora", map[string]any{"peft_type": "PREFIX_TUNING"}, good, "is not LORA"},
		{"dora", map[string]any{"use_dora": true}, good, "DoRA"},
		{"bias trained", map[string]any{"bias": "lora_only"}, good, "bias mode"},
		{"lora bias", map[string]any{"lora_bias": true}, good, "lora_bias"},
		{"transposed layers", map[string]any{"fan_in_fan_out": true}, good, "fan_in_fan_out"},
		{"saved modules", map[string]any{"modules_to_save": []string{"lm_head"}}, good, "modules_to_save"},
		{"layer replication", map[string]any{"layer_replication": [][]int{{0, 2}}}, good, "layer_replication"},
		{"token rows", map[string]any{"trainable_token_indices": []int{1}}, good, "trainable_token_indices"},
		{"regular expression key", map[string]any{"rank_pattern": map[string]int{`.*\.q_proj`: 2}}, good, "regular expression"},
		{"rank disagrees with config", map[string]any{"r": 4}, good, "rank 2, its config says 4"},
		{"embedding adapter", nil, func() map[string]rawTensor {
			tensors := good()
			tensors["base_model.model.model.embed_tokens.lora_embedding_A"] = rawTensor{"F32", []int{2, 3}, f32Bytes(1, 2, 3, 4, 5, 6)}
			return tensors
		}, "not a LoRA linear factor"},
		{"one factor", nil, func() map[string]rawTensor {
			tensors := good()
			delete(tensors, prefix+".lora_B.weight")
			return tensors
		}, "one factor"},
		{"ranks of the pair differ", nil, func() map[string]rawTensor {
			tensors := good()
			tensors[prefix+".lora_B.weight"] = rawTensor{"F32", []int{2, 3}, f32Bytes(1, 2, 3, 4, 5, 6)}
			return tensors
		}, "rank-by-inputs and outputs-by-rank"},
	} {
		directory := t.TempDir()
		writeRaw(t, filepath.Join(directory, weightsFile), scenario.tensors())
		writeConfig(t, directory, scenario.fields)
		if _, err := Load(directory); err == nil || !strings.Contains(err.Error(), scenario.want) {
			t.Errorf("%s: err = %v, want it to say %q", scenario.name, err, scenario.want)
		}
	}
	if _, err := Load(t.TempDir()); err == nil {
		t.Error("a directory with no adapter loaded")
	}
}

// writeBase writes a one-file checkpoint of a query and a value weight, a norm,
// and a weight the adapter leaves alone, in the given storage.
func writeBase(t *testing.T, directory, storage string) map[string][]float32 {
	t.Helper()
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	values := map[string][]float32{
		queryModule + ".weight":                  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		valueModule + ".weight":                  {-1, -2, -3, -4, -5, -6, -7, -8},
		"model.layers.0.input_layernorm.weight":  {1, 1, 1, 1},
		"model.layers.0.self_attn.k_proj.weight": {0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2},
	}
	shapes := map[string][]int{
		queryModule + ".weight": {3, 4}, valueModule + ".weight": {2, 4},
		"model.layers.0.input_layernorm.weight": {4}, "model.layers.0.self_attn.k_proj.weight": {2, 4},
	}
	if storage == "F32" {
		if err := safetensors.Save(filepath.Join(directory, "model.safetensors"), values, shapes, nil); err != nil {
			t.Fatal(err)
		}
		return values
	}
	raw := map[string]rawTensor{}
	for name, floats := range values {
		raw[name] = rawTensor{storage, shapes[name], bf16Bytes(floats...)}
	}
	writeRaw(t, filepath.Join(directory, "model.safetensors"), raw)
	return values
}

// TestMergeAddsTheUpdateAndKeepsTheCheckpoint: the adapted weights gain exactly
// the update the adapter defines, every other tensor is copied byte for byte,
// and the merged checkpoint has the base's tensors, shapes and storage.
func TestMergeAddsTheUpdateAndKeepsTheCheckpoint(t *testing.T) {
	for _, storage := range []string{"F32", "BF16"} {
		root := t.TempDir()
		base, merged := filepath.Join(root, "base"), filepath.Join(root, "merged")
		values := writeBase(t, base, storage)
		adapter := testAdapter()
		report, err := Merge(adapter, base, merged)
		if err != nil || report.Tensors != 4 || report.Merged != 2 {
			t.Fatalf("%s: merge = %+v, %v, want four tensors with two merged", storage, report, err)
		}
		source, err := safetensors.OpenSource(merged)
		if err != nil {
			t.Fatal(err)
		}
		original, err := safetensors.OpenSource(base)
		if err != nil {
			t.Fatal(err)
		}
		for name, tensor := range original.Tensors {
			got, found := source.Tensors[name]
			if !found || got.DType != tensor.DType || !slices.Equal(got.Shape, tensor.Shape) {
				t.Fatalf("%s: tensor %s = %+v after the merge, want %s %v", storage, name, got, tensor.DType, tensor.Shape)
			}
		}
		for module, factors := range adapter.Modules {
			delta := product(factors, factors.Alpha/float64(factors.Rank)) // alpha over rank, as PEFT defines it
			want := make([]float64, len(delta))
			for index := range want {
				want[index] = float64(values[module+".weight"][index]) + delta[index]
				if storage == "BF16" {
					want[index] = float64(dtype.RoundBF16(float32(want[index])))
				}
			}
			got, err := safetensors.ReadF32(source.Tensors[module+".weight"])
			if err != nil {
				t.Fatal(err)
			}
			near(t, got, want, 1e-5, storage+" merged "+module)
		}
		for _, name := range []string{"model.layers.0.input_layernorm.weight", "model.layers.0.self_attn.k_proj.weight"} {
			before, after := make([]byte, original.Tensors[name].Size()), make([]byte, source.Tensors[name].Size())
			if _, err := original.Tensors[name].ReadAt(before, 0); err != nil {
				t.Fatal(err)
			}
			if _, err := source.Tensors[name].ReadAt(after, 0); err != nil || !slices.Equal(before, after) {
				t.Fatalf("%s: tensor %s changed though the adapter does not touch it (%v)", storage, name, err)
			}
		}
		_ = source.Close()
		_ = original.Close()
	}
}

// TestMergeKeepsTheShardLayout: a sharded base merges into the same shards.
func TestMergeKeepsTheShardLayout(t *testing.T) {
	root := t.TempDir()
	base, merged := filepath.Join(root, "base"), filepath.Join(root, "merged")
	if err := os.MkdirAll(base, 0o755); err != nil {
		t.Fatal(err)
	}
	first, second := "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
	writeRaw(t, filepath.Join(base, first), map[string]rawTensor{queryModule + ".weight": {"F32", []int{3, 4}, f32Bytes(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)}})
	writeRaw(t, filepath.Join(base, second), map[string]rawTensor{
		valueModule + ".weight": {"F32", []int{2, 4}, f32Bytes(-1, -2, -3, -4, -5, -6, -7, -8)}, "lm_head.weight": {"F32", []int{2}, f32Bytes(1, 1)},
	})
	index, err := json.Marshal(map[string]any{"metadata": map[string]any{"total_size": 4 * 22}, "weight_map": map[string]string{
		queryModule + ".weight": first, valueModule + ".weight": second, "lm_head.weight": second,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(base, "model.safetensors.index.json"), index, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Merge(testAdapter(), base, merged); err != nil {
		t.Fatal(err)
	}
	source, err := safetensors.OpenSource(merged)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	if !slices.Equal(source.Shards(), []string{first, second}) || source.Tensors[valueModule+".weight"].Shard != second || !source.Indexed() {
		t.Fatalf("merged shards = %v (indexed %t), want the base's two", source.Shards(), source.Indexed())
	}
}

// TestMergeRefusesWhatItCannotMerge: a module the base lacks, a shape that
// differs, and half-precision storage the merge cannot round back to.
func TestMergeRefusesWhatItCannotMerge(t *testing.T) {
	root := t.TempDir()
	base := filepath.Join(root, "base")
	writeBase(t, base, "F32")
	missing := testAdapter()
	missing.Modules["model.layers.7.self_attn.q_proj"] = missing.Modules[queryModule]
	wrong := testAdapter()
	module := wrong.Modules[queryModule]
	module.Out, module.Up = 4, make([]float32, 4*module.Rank)
	wrong.Modules[queryModule] = module
	half := filepath.Join(root, "half")
	if err := os.MkdirAll(half, 0o755); err != nil {
		t.Fatal(err)
	}
	writeRaw(t, filepath.Join(half, "model.safetensors"), map[string]rawTensor{
		queryModule + ".weight": {"F16", []int{3, 4}, make([]byte, 24)}, valueModule + ".weight": {"F32", []int{2, 4}, make([]byte, 32)},
	})
	for name, scenario := range map[string]struct {
		adapter Adapter
		base    string
		want    string
	}{
		"missing":   {missing, base, "no tensor"},
		"shape":     {wrong, base, "has shape"},
		"half":      {testAdapter(), half, "only F32 and BF16"},
		"no module": {Adapter{Config: testAdapter().Config}, base, "no modules"},
	} {
		if _, err := Merge(scenario.adapter, scenario.base, filepath.Join(root, "out-"+name)); err == nil || !strings.Contains(err.Error(), scenario.want) {
			t.Errorf("%s: err = %v, want it to say %q", name, err, scenario.want)
		}
		if _, err := os.Stat(filepath.Join(root, "out-"+name)); err == nil {
			t.Errorf("%s: a refused merge published a checkpoint", name)
		}
	}
}
