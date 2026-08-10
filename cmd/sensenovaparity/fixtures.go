package main

// Device-neutral SenseNova oracle fixture schemas + sample decoders. Mirrors
// the adaptive_new CUDA harness structs (sensenova_edit_cuda_windows_test.go
// and the flowterminal generation oracle) so this CPU ladder reads the SAME
// JSON the device tests read. Every hidden/KV/boundary tensor is a SPARSE
// probe: strided int64 indices + f32 values (edit: base64; generation: plain
// arrays) plus a non_finite count and, for the generation oracle, full-tensor
// mean/std/rms moments.

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// intContract: an integer input vector recorded as element count + sha256 of
// its little-endian int64 encoding (ids/time/height/width).
type intContract struct {
	Elements    int    `json:"elements"`
	SHA256LEI64 string `json:"sha256_le_i64"`
}

// sampledTensor: a tensor recorded as a sparse probe. Edit oracle uses the
// base64 forms; the generation oracle uses plain JSON arrays plus moments.
type sampledTensor struct {
	Shape              []int     `json:"shape"`
	Elements           int       `json:"elements"`
	IndicesLEI64Base64 string    `json:"indices_le_i64_base64"`
	ValuesLEF32Base64  string    `json:"values_le_f32_base64"`
	Indices            []int64   `json:"indices"`
	Values             []float64 `json:"values"`
	Mean               float64   `json:"mean"`
	Std                float64   `json:"std"`
	RMS                float64   `json:"rms"`
	NonFinite          int       `json:"non_finite"`
}

// samples: (index, value) probe pairs, whichever encoding the fixture carries.
func (s sampledTensor) samples() (indices []int, values []float32, err error) {
	if s.IndicesLEI64Base64 != "" {
		ib, e := base64.StdEncoding.DecodeString(s.IndicesLEI64Base64)
		if e != nil || len(ib)%8 != 0 {
			return nil, nil, fmt.Errorf("bad base64 int64 indices (%d bytes): %v", len(ib), e)
		}
		vb, e := base64.StdEncoding.DecodeString(s.ValuesLEF32Base64)
		if e != nil || len(vb)%4 != 0 {
			return nil, nil, fmt.Errorf("bad base64 f32 values (%d bytes): %v", len(vb), e)
		}
		n := len(ib) / 8
		if n != len(vb)/4 {
			return nil, nil, fmt.Errorf("index/value sample count mismatch %d/%d", n, len(vb)/4)
		}
		indices = make([]int, n)
		values = make([]float32, n)
		for i := 0; i < n; i++ {
			indices[i] = int(int64(binary.LittleEndian.Uint64(ib[i*8:])))
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(vb[i*4:]))
		}
		return indices, values, nil
	}
	if len(s.Indices) != len(s.Values) {
		return nil, nil, fmt.Errorf("plain index/value sample count mismatch %d/%d", len(s.Indices), len(s.Values))
	}
	indices = make([]int, len(s.Indices))
	values = make([]float32, len(s.Values))
	for i := range s.Indices {
		indices[i] = int(s.Indices[i])
		values[i] = float32(s.Values[i])
	}
	return indices, values, nil
}

// sampleCount: number of probe coordinates.
func (s sampledTensor) sampleCount() int {
	if s.IndicesLEI64Base64 != "" {
		if b, err := base64.StdEncoding.DecodeString(s.IndicesLEI64Base64); err == nil {
			return len(b) / 8
		}
		return 0
	}
	return len(s.Indices)
}

// ---- edit oracle (adaptive_gpt.sensenova_edit_oracle/v1) -------------------

type editOracleInput struct {
	IDs    intContract `json:"ids"`
	Time   intContract `json:"time"`
	Height intContract `json:"height"`
	Width  intContract `json:"width"`
}

type editOracleKV struct {
	Keys   sampledTensor `json:"keys"`
	Values sampledTensor `json:"values"`
}

type editOracleStep struct {
	Timestep            float64       `json:"timestep"`
	NextTimestep        float64       `json:"next_timestep"`
	Z                   sampledTensor `json:"z"`
	ConditionalBoundary sampledTensor `json:"conditional_boundary"`
	SourceOnlyBoundary  sampledTensor `json:"source_only_boundary"`
	GuidedVelocity      sampledTensor `json:"guided_velocity"`
	NextZ               sampledTensor `json:"next_z"`
}

type editOracle struct {
	Schema  string `json:"schema"`
	Request struct {
		Prompt        string  `json:"prompt"`
		Width         int     `json:"width"`
		Height        int     `json:"height"`
		Steps         int     `json:"steps"`
		Seed          int64   `json:"seed"`
		CFGScale      float64 `json:"cfg_scale"`
		ImgCFGScale   float64 `json:"img_cfg_scale"`
		TimestepShift float64 `json:"timestep_shift"`
	} `json:"request"`
	SourceContract struct {
		GridHW     [][]int       `json:"grid_hw"`
		TokenCount int           `json:"token_count"`
		Pixels     sampledTensor `json:"pixels"`
		Embedding  sampledTensor `json:"embedding"`
	} `json:"source_contract"`
	TextInputs struct {
		Conditional editOracleInput `json:"conditional"`
		SourceOnly  editOracleInput `json:"source_only"`
	} `json:"text_inputs"`
	PrefixLayers map[string]struct {
		Conditional sampledTensor `json:"conditional"`
		SourceOnly  sampledTensor `json:"source_only"`
	} `json:"prefix_layers"`
	GenerationLayers map[string]struct {
		Conditional sampledTensor `json:"conditional"`
		SourceOnly  sampledTensor `json:"source_only"`
	} `json:"generation_layers"`
	PrefixKV map[string]struct {
		Conditional editOracleKV `json:"conditional"`
		SourceOnly  editOracleKV `json:"source_only"`
	} `json:"prefix_kv"`
	Steps []editOracleStep `json:"steps"`
}

// ---- generation oracle (adaptive_gpt.sensenova_generation_oracle/v1) -------

type genOracleStep struct {
	Timestep              float64       `json:"timestep"`
	NextTimestep          float64       `json:"next_timestep"`
	Z                     sampledTensor `json:"z"`
	ConditionalBoundary   sampledTensor `json:"conditional_boundary"`
	UnconditionalBoundary sampledTensor `json:"unconditional_boundary"`
	GuidedVelocity        sampledTensor `json:"guided_velocity"`
	NextZ                 sampledTensor `json:"next_z"`
}

type genOracle struct {
	Schema  string `json:"schema"`
	Request struct {
		Prompt        string  `json:"prompt"`
		Width         int     `json:"width"`
		Height        int     `json:"height"`
		Steps         int     `json:"steps"`
		Seed          int64   `json:"seed"`
		CFGScale      float64 `json:"cfg_scale"`
		TimestepShift float64 `json:"timestep_shift"`
	} `json:"request"`
	TextInputs struct {
		ConditionalIDs   []int `json:"conditional_ids"`
		UnconditionalIDs []int `json:"unconditional_ids"`
	} `json:"text_inputs"`
	Steps []genOracleStep `json:"steps"`
}

// finiteProbes: decodes every probe value and confirms it is finite, cross-
// checking the recorded non_finite count. Returns the sample count.
func (s sampledTensor) finiteProbes() (int, error) {
	_, values, err := s.samples()
	if err != nil {
		return 0, err
	}
	nonFinite := 0
	for _, v := range values {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			nonFinite++
		}
	}
	if nonFinite != s.NonFinite {
		return len(values), fmt.Errorf("decoded non_finite=%d != recorded %d", nonFinite, s.NonFinite)
	}
	return len(values), nil
}

func loadJSON[T any](path string, out *T) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(raw, out)
}
