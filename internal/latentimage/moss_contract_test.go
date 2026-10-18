package latentimage

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

type mossTolerance struct {
	Absolute float64 `json:"atol"`
	Relative float64 `json:"rtol"`
}
type mossTensorContract struct {
	DType      string   `json:"dtype"`
	Shape      []uint64 `json:"shape"`
	Comparison string   `json:"comparison"`
}
type mossReferenceCase struct {
	Name           string                        `json:"name"`
	Artifact       artifact.ID                   `json:"artifact"`
	Producer       artifact.ID                   `json:"producer"`
	Backend        string                        `json:"backend_policy"`
	Environment    map[string]string             `json:"environment"`
	MetadataSHA256 string                        `json:"metadata_sha256"`
	Inputs         map[string]artifact.ID        `json:"inputs"`
	Repeatability  string                        `json:"repeatability"`
	Tensors        map[string]mossTensorContract `json:"tensors"`
}
type mossOracleContract struct {
	Version    int         `json:"version"`
	Corpus     artifact.ID `json:"corpus"`
	Source     artifact.ID `json:"source"`
	Checkpoint artifact.ID `json:"checkpoint"`
	HostAudit  struct {
		CPU               string `json:"cpu"`
		OS                string `json:"os"`
		PhysicalCores     int    `json:"physical_cores"`
		LogicalProcessors int    `json:"logical_processors"`
		Source            string `json:"source"`
		Scope             string `json:"scope"`
	} `json:"host_audit"`
	Cases      []mossReferenceCase `json:"cases"`
	Comparison struct {
		SourceURL             string                   `json:"source_url"`
		SourceSHA256          string                   `json:"source_sha256"`
		SourceVersion         string                   `json:"source_version"`
		Rule                  string                   `json:"rule"`
		Tolerances            map[string]mossTolerance `json:"dtype_tolerances"`
		FiniteOnly            bool                     `json:"finite_only"`
		ShapeAndDType         string                   `json:"shape_and_dtype"`
		ReferenceAndBackend   string                   `json:"reference_and_backend"`
		RepeatabilityAbsolute float64                  `json:"repeatability_atol"`
		RepeatabilityRelative float64                  `json:"repeatability_rtol"`
		Scope                 string                   `json:"scope"`
	} `json:"comparison"`
}

func mossContract(t *testing.T) mossOracleContract {
	t.Helper()
	raw := mossFixtureBytes(t, "oracle-contract.json", "0c2767456a75075671de88538d8c1ce0f673ef41c2bae897c4e59449fad0ca86")
	var contract mossOracleContract
	if err := strictjson.DecodeBytes(raw, &contract); err != nil {
		t.Fatal(err)
	}
	if err := contract.validate(); err != nil {
		t.Fatal(err)
	}
	return contract
}

func (c mossOracleContract) validate() error {
	p := c.Comparison
	if c.Version != 1 || c.Corpus.String() != "dataset:sha256:fb09c610a25c80eb6c2a017a6da1c178bfe095e7fd48e84f1b74c63a58b7e964" || c.Source.DigestHex() != mossReferenceSource || c.Checkpoint.DigestHex() != "0d9b600f6232957565c26e86c9f9b0d19411b87fe97d665bbe7b93bd7770a9db" {
		return errors.New("oracle identity differs")
	}
	if p.SourceVersion != "torch 2.12.0+cu126" || p.SourceSHA256 != "679d499d12941216627732cc0d7c220a7be09766aa4b97518df9b0754069fc4c" || p.SourceURL != "https://docs.pytorch.org/docs/2.12/testing.html#torch.testing.assert_close" {
		return errors.New("comparison authority differs")
	}
	// These values are the pinned upstream testing defaults, not fitted native errors.
	if len(p.Tolerances) != 2 || p.Tolerances["F32"] != (mossTolerance{1e-5, 1.3e-6}) || p.Tolerances["BF16"] != (mossTolerance{1e-5, 0.016}) {
		return errors.New("comparison tolerances differ from the frozen upstream policy")
	}
	if !p.FiniteOnly || p.ShapeAndDType != "exact" || p.ReferenceAndBackend != "exact" || p.RepeatabilityAbsolute != 0 || p.RepeatabilityRelative != 0 || p.Rule != "abs(actual - expected) <= atol + rtol * abs(expected)" {
		return errors.New("comparison requirements differ")
	}
	if c.HostAudit.CPU == "" || c.HostAudit.Source == "" || c.HostAudit.Scope == "" {
		return errors.New("oracle host audit is absent")
	}
	counts := map[string]int{"primitives": 96, "tokenizer": 12, "conditioning": 14, "image-cpu": 37, "image-cuda-auto": 37, "image-cuda-math": 37, "full-english": 220, "full-chinese-rectangle": 220, "full-layout": 220}
	seen := map[string]bool{}
	for _, reference := range c.Cases {
		if seen[reference.Name] || counts[reference.Name] != len(reference.Tensors) || len(reference.Tensors) == 0 {
			return fmt.Errorf("missing or duplicate capture coverage %q", reference.Name)
		}
		seen[reference.Name] = true
		hash, err := hex.DecodeString(reference.MetadataSHA256)
		if err != nil || len(hash) != sha256.Size || reference.Artifact.Kind() != artifact.KindTensorSet || reference.Producer.Kind() != artifact.KindFile || reference.Repeatability == "" {
			return errors.New("reference provenance is incomplete")
		}
		if reference.Environment["package_torch"] != "2.12.0+cu126" || reference.Environment["package_transformers"] != "5.14.1" || reference.Environment["package_safetensors"] != "0.8.0" {
			return errors.New("reference dependencies differ")
		}
		if !slices.Contains([]string{"cpu-tokenizer", "cpu-sdpa-auto", "cuda-sdpa-auto", "cuda-sdpa-math"}, reference.Backend) {
			return errors.New("undeclared reference backend")
		}
		if strings.HasPrefix(reference.Backend, "cuda-") && (reference.Environment["cuda"] != "12.6" || reference.Environment["device"] != "NVIDIA GeForce RTX 4090 D") {
			return errors.New("CUDA reference environment differs")
		}
		for name, tensor := range reference.Tensors {
			if name == "" || len(tensor.Shape) == 0 || slices.Contains(tensor.Shape, 0) || !slices.Contains([]string{"F32", "BF16", "BOOL", "I64"}, tensor.DType) {
				return errors.New("invalid tensor geometry or dtype")
			}
			if tensor.Comparison != "exact" && tensor.Comparison != "upstream-dtype" {
				return errors.New("missing tensor comparison policy")
			}
			if (tensor.DType == "I64" || tensor.DType == "BOOL" || strings.HasSuffix(name, ".sigmas") || strings.HasSuffix(name, ".timesteps") || name == "initial_latents" || strings.HasPrefix(name, "input.")) && tensor.Comparison != "exact" {
				return errors.New("discrete or fixed input comparison was relaxed")
			}
		}
	}
	if len(seen) != len(counts) {
		return errors.New("missing reference case")
	}
	return nil
}

// compare checks floating-point captures against one explicitly selected upstream case.
// Integer captures require byte equality in the tokenizer and mask consumers.
func (c mossOracleContract) compare(caseName string, reference artifact.ID, backend, stage, dtype string, shape []uint64, actual, expected []float32) error {
	for _, entry := range c.Cases {
		if entry.Name != caseName {
			continue
		}
		spec, found := entry.Tensors[stage]
		if !found || entry.Artifact != reference || entry.Backend != backend || spec.DType != dtype || !slices.Equal(spec.Shape, shape) {
			return errors.New("capture differs from selected reference contract")
		}
		if dtype != "F32" && dtype != "BF16" {
			return errors.New("floating-point comparator refuses integer captures")
		}
		count := uint64(1)
		for _, dimension := range shape {
			if dimension == 0 || count > uint64(len(actual))/dimension {
				return errors.New("capture payload does not cover shape")
			}
			count *= dimension
		}
		if count != uint64(len(actual)) || len(actual) != len(expected) {
			return errors.New("capture payload does not cover shape")
		}
		tolerance := c.Comparison.Tolerances[dtype]
		for i, got := range actual {
			want := expected[i]
			if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) || math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) {
				return fmt.Errorf("nonfinite capture at %d", i)
			}
			if spec.Comparison == "exact" {
				if math.Float32bits(got) != math.Float32bits(want) {
					return fmt.Errorf("exact capture differs at %d", i)
				}
				continue
			}
			limit := tolerance.Absolute + tolerance.Relative*math.Abs(float64(want))
			if math.Abs(float64(got)-float64(want)) > limit {
				return fmt.Errorf("capture error at %d exceeds %g", i, limit)
			}
		}
		return nil
	}
	return errors.New("unknown reference case")
}

// TestMOSSGoldenContract binds captured upstream data to acceptance before native comparison.
// It checks reference coverage and comparison behavior without claiming native model parity.
func TestMOSSGoldenContract(t *testing.T) {
	t.Parallel()
	contract := mossContract(t)
	cases := map[string]mossReferenceCase{}
	for _, entry := range contract.Cases {
		cases[entry.Name] = entry
	}
	for _, name := range []string{"primitives", "tokenizer"} {
		source := mossGolden(t, name)
		entry := cases[name]
		for _, meta := range source.Metadata {
			data, err := json.Marshal(meta)
			if err != nil {
				t.Fatal(err)
			}
			hash := sha256.Sum256(data)
			if hex.EncodeToString(hash[:]) != entry.MetadataSHA256 {
				t.Fatal("portable reference metadata changed")
			}
		}
		for tensorName, value := range source.Tensors {
			spec := entry.Tensors[tensorName]
			if spec.DType != value.DType || !slices.Equal(spec.Shape, value.Shape) {
				t.Fatalf("portable contract differs for %s", tensorName)
			}
		}
	}
	for _, name := range []string{"full-english", "full-chinese-rectangle", "full-layout"} {
		entry := cases[name]
		for step := range 50 {
			for _, prefix := range []string{"latents.", "velocity."} {
				if _, ok := entry.Tensors[fmt.Sprint(prefix, step)]; !ok {
					t.Fatalf("missing %s %s%d", name, prefix, step)
				}
			}
		}
		for prediction := range 100 {
			if _, ok := entry.Tensors[fmt.Sprint("prediction.", prediction)]; !ok {
				t.Fatalf("missing conditional or negative prediction in %s", name)
			}
		}
		for _, stage := range []string{"initial_latents", "conditioning.0.ids", "conditioning.0.mask", "conditioning.0.hidden", "conditioning.1.ids", "conditioning.1.mask", "conditioning.1.hidden", "rope", "timestep_embedding.input", "timestep_embedding", "modulation", "text_refinement", "block.0", "block.21", "block.41", "vae.unpacked", "vae.decoded", "pixels", "schedule.sigmas", "schedule.timesteps"} {
			if _, ok := entry.Tensors[stage]; !ok {
				t.Fatalf("missing %s stage %s", name, stage)
			}
		}
		height := uint64(1024)
		if name == "full-chinese-rectangle" {
			height = 768
		}
		if !slices.Equal(entry.Tensors["pixels"].Shape, []uint64{1, 3, height, 1024}) {
			t.Fatalf("full canvas geometry differs for %s", name)
		}
	}
	if !strings.Contains(cases["image-cuda-auto"].Environment["attention_dispatch"], "efficient_attention") || !strings.Contains(cases["image-cuda-math"].Environment["attention_dispatch"], "attention_math") {
		t.Fatal("paired CUDA attention evidence differs")
	}
	for _, dtype := range []string{"F32", "BF16"} {
		t.Run(dtype, func(t *testing.T) {
			// A controlled capture exercises the same frozen comparison used by numerical consumers.
			entry := mossReferenceCase{Name: "controlled", Artifact: cases["primitives"].Artifact, Backend: "cpu-sdpa-auto", Tensors: map[string]mossTensorContract{"value": {DType: dtype, Shape: []uint64{1}, Comparison: "upstream-dtype"}}}
			candidate := contract
			candidate.Cases = []mossReferenceCase{entry}
			compare := func(got, want float32) error {
				return candidate.compare(entry.Name, entry.Artifact, entry.Backend, "value", dtype, []uint64{1}, []float32{got}, []float32{want})
			}
			tolerance := contract.Comparison.Tolerances[dtype]
			limit := tolerance.Absolute + tolerance.Relative*100
			if err := compare(float32(100+limit/2), 100); err != nil {
				t.Fatal(err)
			}
			if err := compare(math.Nextafter32(float32(100+limit), float32(math.Inf(1))), 100); err == nil {
				t.Fatal("out-of-policy value passed")
			}
			for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
				if compare(value, value) == nil {
					t.Fatal("nonfinite values passed")
				}
			}
			if candidate.compare(entry.Name, entry.Artifact, "cuda-sdpa-auto", "value", dtype, []uint64{1}, []float32{1}, []float32{1}) == nil {
				t.Fatal("wrong reference backend passed")
			}
			if candidate.compare(entry.Name, cases["tokenizer"].Artifact, entry.Backend, "value", dtype, []uint64{1}, []float32{1}, []float32{1}) == nil {
				t.Fatal("wrong reference identity passed")
			}
			if candidate.compare(entry.Name, entry.Artifact, entry.Backend, "value", dtype, []uint64{1, 1}, []float32{1}, []float32{1}) == nil {
				t.Fatal("wrong shape passed")
			}
			if candidate.compare(entry.Name, entry.Artifact, entry.Backend, "value", dtype, []uint64{1}, nil, nil) == nil {
				t.Fatal("missing payload passed")
			}
			entry.Tensors["value"] = mossTensorContract{DType: dtype, Shape: []uint64{1}, Comparison: "exact"}
			if compare(1, 1) != nil || compare(math.Nextafter32(1, 2), 1) == nil || compare(float32(math.Copysign(0, -1)), 0) == nil {
				t.Fatal("exact comparison lost bit identity")
			}
		})
	}
	for _, change := range []func(*mossOracleContract){
		func(c *mossOracleContract) { c.Cases = c.Cases[1:] },
		func(c *mossOracleContract) {
			c.Cases[0].Tensors["transformer.mask"] = mossTensorContract{DType: "BOOL", Shape: []uint64{2, 4}, Comparison: "upstream-dtype"}
		},
		func(c *mossOracleContract) { c.Comparison.Tolerances["BF16"] = mossTolerance{1, 1} },
		func(c *mossOracleContract) { c.Comparison.RepeatabilityAbsolute = 1 },
		func(c *mossOracleContract) { c.Comparison.FiniteOnly = false },
	} {
		raw, err := json.Marshal(contract)
		if err != nil {
			t.Fatal(err)
		}
		var changed mossOracleContract
		if err := json.Unmarshal(raw, &changed); err != nil {
			t.Fatal(err)
		}
		change(&changed)
		if changed.validate() == nil {
			t.Fatal("incomplete or relaxed contract passed")
		}
	}
}
