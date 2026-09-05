// Package vqaserve serves visual question answering through the RxBrain
// branch-routed program on the CUDA device: the processor prepares an
// image and a question into prompt state, the device runs the vision
// tower, the merger, the chained prefill and the decode, and the
// checkpoint's tokenizer decodes the answer. The parity harness drives the
// same pipeline against its goldens; the media capability catalog drives it
// from a page. Neither owns a copy.
package vqaserve

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/executor"
	"overgo/internal/hfbpe"
	"overgo/internal/patchtower"
	"overgo/internal/routedlm"
)

// Binding is RxBrain's `_v` branch naming contract.
var Binding = routedlm.RxBrainBinding()

// PromptRoles are the role literals of this checkpoint's prompt specials;
// sourced from its tokenizer_config.json added_tokens_decoder role
// assignments (the reference OvergoDB profile facts, derivation
// tokenizer-special-token-role/v1).
var PromptRoles = routedlm.PromptRoleLiterals{
	BOS:        "<｜hy_begin▁of▁sentence｜>",
	EOS:        "<｜hy_end▁of▁sentence｜>",
	User:       "<｜hy_User｜>",
	Assistant:  "<｜hy_Assistant｜>",
	Think:      "<think>",
	ThinkEnd:   "</think>",
	Answer:     "<answer>",
	AnswerEnd:  "</answer>",
	NoThink:    "/no_think",
	ImageStart: "<｜hy_place▁holder▁no▁666｜>",
	ImageEnd:   "<｜hy_place▁holder▁no▁667｜>",
	Image:      "<｜hy_place▁holder▁no▁669｜>",
	Video:      "<｜hy_place▁holder▁no▁670｜>",
	NewLine:    "<｜hy_place▁holder▁no▁671｜>",
	FlowLatent: "<｜hy_place▁holder▁no▁672｜>",
}

// Prepared is the processor's output for one image and question: the
// preprocessed pixels, the rendered prompt ids with the image-token
// positions, the checkpoint's stop tokens, and the image grid.
type Prepared struct {
	Pixels              []float32
	InputIDs, Positions []int
	EOSIDs              []int
	GridT, GridH, GridW int
}

// DecodeChain turns token ids into text through the checkpoint tokenizer.
func DecodeChain(modelDir string, ids []int) (string, error) {
	tok, err := hfbpe.Load(modelDir)
	if err != nil {
		return "", err
	}
	return tok.Decode(ids), nil
}

// readEOSTokenIDs reads the scalar or list generation stop tokens.
func readEOSTokenIDs(modelDir string) ([]int, error) {
	raw, err := os.ReadFile(filepath.Join(modelDir, "generation_config.json"))
	if err != nil {
		return nil, err
	}
	var parsed struct {
		EOS json.RawMessage `json:"eos_token_id"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("generation_config: %w", err)
	}
	if len(parsed.EOS) == 0 {
		return nil, fmt.Errorf("generation_config: no eos_token_id")
	}
	var list []int
	if err := json.Unmarshal(parsed.EOS, &list); err == nil {
		return list, nil
	}
	var scalar int
	if err := json.Unmarshal(parsed.EOS, &scalar); err != nil {
		return nil, fmt.Errorf("generation_config: eos_token_id shape: %w", err)
	}
	return []int{scalar}, nil
}

// Prepare runs the processor over the image bytes and the question: the
// image preprocesses into pixels and a grid, the prompt renders with the
// image tokens the grid needs, and the stop tokens come from the
// checkpoint's generation config.
func Prepare(modelDir string, image []byte, question string) (Prepared, error) {
	pre, err := patchtower.LoadPreprocessConfig(modelDir)
	if err != nil {
		return Prepared{}, err
	}
	tok, err := hfbpe.Load(modelDir)
	if err != nil {
		return Prepared{}, err
	}
	specials, err := routedlm.LoadPromptSpecials(modelDir, PromptRoles)
	if err != nil {
		return Prepared{}, err
	}
	rgb, height, width, err := patchtower.DecodeImageBytesRGB(image)
	if err != nil {
		return Prepared{}, err
	}
	pixels, gridT, gridH, gridW, err := patchtower.PreprocessImage(pre, rgb, height, width)
	if err != nil {
		return Prepared{}, err
	}
	inputIDs, err := routedlm.RenderVisionQAPrompt(tok, specials, question, gridH, gridW, pre.MergeSize)
	if err != nil {
		return Prepared{}, err
	}
	eosIDs, err := readEOSTokenIDs(modelDir)
	if err != nil {
		return Prepared{}, err
	}
	// The checkpoint's answer-end special closes the answer the way its EOS
	// does, so the decoded text is the answer alone.
	return Prepared{
		Pixels: pixels, InputIDs: inputIDs, Positions: routedlm.ImageMaskPositions(inputIDs, specials),
		EOSIDs: append(eosIDs, specials.AnswerEnd), GridT: gridT, GridH: gridH, GridW: gridW,
	}, nil
}

// Answer runs the prepared prompt through the device pipeline with a decode
// budget and decodes the generated chain: one device worker and executor
// per answer, released on return. log receives the stage boundaries.
func Answer(ctx context.Context, modelDir string, prepared Prepared, maxSteps int, log func(string)) (Result, string, error) {
	pc, err := NewPrefillContext(modelDir, prepared.InputIDs, prepared.Positions, prepared.GridT, prepared.GridH, prepared.GridW)
	if err != nil {
		return Result{}, "", err
	}
	defer pc.Src.Close()
	worker, err := device.New(device.DefaultOrdinal())
	if err != nil {
		return Result{}, "", fmt.Errorf("vqa serve worker: %w", err)
	}
	defer worker.Close()
	exe, err := executor.NewWithWorker(worker)
	if err != nil {
		return Result{}, "", fmt.Errorf("vqa serve executor: %w", err)
	}
	defer exe.Close()
	result, err := RunPipeline(ctx, worker, exe, pc, prepared.Pixels, Options{MaxSteps: maxSteps, EOSIDs: prepared.EOSIDs, Log: log})
	if err != nil {
		return Result{}, "", err
	}
	text, err := DecodeChain(modelDir, result.Generated)
	return result, text, err
}
