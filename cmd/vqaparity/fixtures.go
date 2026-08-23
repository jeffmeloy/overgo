package main

import (
	"fmt"
	"math"
	"path/filepath"

	"overgo/internal/checked"
	"overgo/internal/fixtureasset"
	"overgo/internal/jsonfile"
)

// Golden fixture schemas (adaptive_new fixtures/rxbrain_vqa_*.json).

type goldenTensor struct {
	Shape      []int     `json:"shape"`
	Sha256     string    `json:"sha256"`
	ProbeIndex []int     `json:"probe_index"`
	ProbeValue []float64 `json:"probe_value"`
	L2         float64   `json:"l2"`
	Mean       float64   `json:"mean"`
}

type processorGolden struct {
	Request struct {
		Images   []string `json:"images"`
		Question string   `json:"question"`
	} `json:"request"`
	InputIDs           []int        `json:"input_ids"`
	ImageGridTHW       []int        `json:"image_grid_thw"`
	ImageMaskPositions []int        `json:"image_mask_positions"`
	NumImageMaskTokens int          `json:"num_image_mask_tokens"`
	PixelValues        goldenTensor `json:"pixel_values"`
	PixelValuesAsset   string       `json:"pixel_values_asset"`
}

type visionGolden struct {
	Block0QKVAsset string                  `json:"block0_qkv_asset"`
	PreBlock0Asset string                  `json:"pre_block0_asset"`
	BlockLastAsset string                  `json:"block_last_asset"`
	ImageGridTHW   []int                   `json:"image_grid_thw"`
	PixelValues    goldenTensor            `json:"pixel_values"`
	VisionTensors  map[string]goldenTensor `json:"vision_tensors"`
}

type prefillGolden struct {
	InputIDs       []int `json:"input_ids"`
	PromptLen      int   `json:"prompt_len"`
	ImageGridTHW   []int `json:"image_grid_thw"`
	PrefillTensors struct {
		LanguageInputsEmbeds    goldenTensor `json:"language_inputs_embeds"`
		ModalityMask            []int        `json:"modality_mask"`
		InputImageMaskPositions []int        `json:"input_image_mask_positions"`
	} `json:"prefill_tensors"`
}

type motGolden struct {
	InputIDs         []int                   `json:"input_ids"`
	PromptLen        int                     `json:"prompt_len"`
	ImageGridTHW     []int                   `json:"image_grid_thw"`
	LayerIndex       int                     `json:"layer_index"`
	LayerOutputAsset string                  `json:"layer_output_asset"`
	MoTTensors       map[string]goldenTensor `json:"mot_tensors"`
}

type logitsGolden struct {
	Shape      []int     `json:"shape"`
	ProbeIndex []int     `json:"probe_index"`
	ProbeValue []float64 `json:"probe_value"`
	TopIndex   []int     `json:"top_index"`
	TopValue   []float64 `json:"top_value"`
}

type terminalGolden struct {
	InputIDs        []int                   `json:"input_ids"`
	PromptLen       int                     `json:"prompt_len"`
	ImageGridTHW    []int                   `json:"image_grid_thw"`
	TerminalTensors map[string]goldenTensor `json:"terminal_tensors"`
	LastLogits      logitsGolden            `json:"last_logits"`
}

type decodeStepRecord struct {
	Step        int          `json:"step"`
	Position    int          `json:"position"`
	TokenIn     int          `json:"token_in"`
	NextToken   int          `json:"next_token"`
	FinalHidden goldenTensor `json:"final_hidden"`
	Logits      logitsGolden `json:"logits"`
}

type decodeStepsGolden struct {
	InputIDs        []int              `json:"input_ids"`
	PromptLen       int                `json:"prompt_len"`
	ImageGridTHW    []int              `json:"image_grid_thw"`
	FirstToken      int                `json:"first_token"`
	DecodeStepCount int                `json:"decode_step_count"`
	GeneratedTokens []int              `json:"generated_tokens"`
	DecodeSteps     []decodeStepRecord `json:"decode_steps"`
}

func loadGoldenJSON[T any](fixturesDir, name string) (T, error) {
	var out T
	if err := jsonfile.Decode(filepath.Join(fixturesDir, name), &out); err != nil {
		return out, fmt.Errorf("%s: %w", name, err)
	}
	return out, nil
}

// loadTensorAsset: raw little-endian f32 asset checked against the golden
// sha256 and shape product.
func loadTensorAsset(fixturesDir, asset string, tensor goldenTensor) ([]float32, error) {
	elements, ok := checked.ProductInt(tensor.Shape...)
	if !ok || elements <= 0 {
		return nil, fmt.Errorf("%s has invalid shape %v", asset, tensor.Shape)
	}
	return fixtureasset.LoadF32(fixturesDir, asset, tensor.Sha256, elements)
}

// probeCheck: worst-probe rule — fail when worst |got-want| exceeds
// atol + rtol*|want-at-worst| (the reference comparator).
func probeCheck(name string, values []float32, tensor goldenTensor, atol, rtol float64) (string, error) {
	if len(tensor.ProbeIndex) == 0 || len(tensor.ProbeIndex) != len(tensor.ProbeValue) {
		return "", fmt.Errorf("%s: golden probes %d/%d", name, len(tensor.ProbeIndex), len(tensor.ProbeValue))
	}
	got := make([]float32, len(tensor.ProbeIndex))
	for i, index := range tensor.ProbeIndex {
		if index < 0 || index >= len(values) {
			return "", fmt.Errorf("%s: probe index %d outside [0,%d)", name, index, len(values))
		}
		got[i] = values[index]
	}
	return probeValuesCheck(name, got, tensor.ProbeValue, atol, rtol)
}

func probeValuesCheck(name string, got []float32, want []float64, atol, rtol float64) (string, error) {
	if len(got) != len(want) {
		return "", fmt.Errorf("%s: got %d probes, want %d", name, len(got), len(want))
	}
	var worst, worstGot, worstWant float64
	for i := range got {
		g, w := float64(got[i]), want[i]
		if d := math.Abs(g - w); d > worst {
			worst, worstGot, worstWant = d, g, w
		}
	}
	limit := atol + rtol*math.Abs(worstWant)
	if worst > limit {
		return "", fmt.Errorf("%s: worst probe got %.6f vs golden %.6f (|d|=%.4e > %.4e)", name, worstGot, worstWant, worst, limit)
	}
	return fmt.Sprintf("%s worst|d|=%.3e", name, worst), nil
}

func requireIntSliceEqual(name string, got, want []int) error {
	if len(got) != len(want) {
		return fmt.Errorf("%s len %d != %d", name, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			return fmt.Errorf("%s[%d] = %d, want %d", name, i, got[i], want[i])
		}
	}
	return nil
}
