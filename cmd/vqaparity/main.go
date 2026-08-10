// vqaparity walks the RxBrain vision-QA fixture parity ladder IN ORDER on
// the canonical bridgev2 case against overgo's neutral host packages
// (patchtower + routedlm + hfbpe), asserting each stage at the reference
// tolerances and stopping at the first failing stage.
package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"overgo/internal/hfbpe"
	"overgo/internal/patchtower"
	"overgo/internal/routedlm"
	"overgo/internal/safetensors"
)

// binding: RxBrain's `_v` branch naming contract.
var binding = routedlm.RxBrainBinding()

// Role literals of this checkpoint's prompt specials; sourced from its
// tokenizer_config.json added_tokens_decoder role assignments (the reference
// repodb profile facts, derivation tokenizer-special-token-role/v1).
var promptRoles = routedlm.PromptRoleLiterals{
	BOS:        "<\uFF5Chy_begin\u2581of\u2581sentence\uFF5C>",
	EOS:        "<\uFF5Chy_end\u2581of\u2581sentence\uFF5C>",
	User:       "<\uFF5Chy_User\uFF5C>",
	Assistant:  "<\uFF5Chy_Assistant\uFF5C>",
	Think:      "<think>",
	ThinkEnd:   "</think>",
	Answer:     "<answer>",
	AnswerEnd:  "</answer>",
	NoThink:    "/no_think",
	ImageStart: "<\uFF5Chy_place\u2581holder\u2581no\u2581666\uFF5C>",
	ImageEnd:   "<\uFF5Chy_place\u2581holder\u2581no\u2581667\uFF5C>",
	Image:      "<\uFF5Chy_place\u2581holder\u2581no\u2581669\uFF5C>",
	Video:      "<\uFF5Chy_place\u2581holder\u2581no\u2581670\uFF5C>",
	NewLine:    "<\uFF5Chy_place\u2581holder\u2581no\u2581671\uFF5C>",
	FlowLatent: "<\uFF5Chy_place\u2581holder\u2581no\u2581672\uFF5C>",
}

type ladder struct {
	modelDir    string
	fixturesDir string
	logPath     string
	failed      bool
	stagesRun   int
	stagesPass  int
}

func (l *ladder) log(line string) {
	stamp := time.Now().UTC().Format("2006-01-02T15:04:05Z")
	full := stamp + " " + line + "\n"
	fmt.Print(full)
	f, err := os.OpenFile(l.logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err == nil {
		_, _ = f.WriteString(full)
		_ = f.Close()
	}
}

// stage: runs fn unless a prior stage failed; logs PASS/FAIL + wall.
func (l *ladder) stage(name string, fn func() (string, error)) {
	if l.failed {
		return
	}
	l.stagesRun++
	start := time.Now()
	detail, err := fn()
	wall := time.Since(start).Round(time.Millisecond)
	if err != nil {
		l.failed = true
		l.log(fmt.Sprintf("STAGE %-28s FAIL wall=%-9s %v", name, wall, err))
		return
	}
	l.stagesPass++
	l.log(fmt.Sprintf("STAGE %-28s PASS wall=%-9s %s", name, wall, detail))
}

func main() {
	modelDir := flag.String("model", `C:\Users\jeffm\adaptive_new\models\Hy-Embodied-RxBrain-1.0`, "checkpoint dir")
	fixturesDir := flag.String("fixtures", `C:\Users\jeffm\adaptive_new\fixtures`, "fixture dir")
	logPath := flag.String("log", `build\rxbrain_port_log.txt`, "liveness log path")
	fromStage := flag.String("from", "", "skip stages before this name")
	toStage := flag.String("to", "", "stop after this stage name")
	flag.Parse()
	l := &ladder{modelDir: *modelDir, fixturesDir: *fixturesDir, logPath: *logPath}
	if err := run(l, *fromStage, *toStage); err != nil {
		l.log("LADDER ERROR " + err.Error())
		os.Exit(1)
	}
	if l.failed {
		l.log(fmt.Sprintf("LADDER STOPPED after %d/%d stages passed", l.stagesPass, l.stagesRun))
		os.Exit(1)
	}
	l.log(fmt.Sprintf("LADDER COMPLETE %d/%d stages passed", l.stagesPass, l.stagesRun))
}

func run(l *ladder, fromStage, toStage string) error {
	skipUntil := fromStage
	stopAfter := toStage
	gate := func(name string, fn func() (string, error)) {
		if skipUntil != "" {
			if name == skipUntil {
				skipUntil = ""
			} else {
				return
			}
		}
		if stopAfter == "done" {
			return
		}
		l.stage(name, fn)
		if name == stopAfter {
			stopAfter = "done"
		}
	}

	pg, err := loadGoldenJSON[processorGolden](l.fixturesDir, "rxbrain_vqa_processor_golden.json")
	if err != nil {
		return err
	}
	vg, err := loadGoldenJSON[visionGolden](l.fixturesDir, "rxbrain_vqa_vision_golden.json")
	if err != nil {
		return err
	}
	fg, err := loadGoldenJSON[prefillGolden](l.fixturesDir, "rxbrain_vqa_prefill_golden.json")
	if err != nil {
		return err
	}
	if len(vg.ImageGridTHW) != 3 {
		return fmt.Errorf("vision golden grid %v", vg.ImageGridTHW)
	}
	gridT, gridH, gridW := vg.ImageGridTHW[0], vg.ImageGridTHW[1], vg.ImageGridTHW[2]

	spec, err := patchtower.LoadSpec(l.modelDir)
	if err != nil {
		return err
	}
	cfg, err := routedlm.LoadConfig(l.modelDir, binding)
	if err != nil {
		return err
	}
	src, err := safetensors.OpenSource(l.modelDir)
	if err != nil {
		return err
	}
	defer src.Close()

	// ---- processor ----------------------------------------------------
	gate("processor", func() (string, error) {
		pre, err := patchtower.LoadPreprocessConfig(l.modelDir)
		if err != nil {
			return "", err
		}
		tok, err := hfbpe.Load(l.modelDir)
		if err != nil {
			return "", err
		}
		specials, err := routedlm.LoadPromptSpecials(l.modelDir, promptRoles)
		if err != nil {
			return "", err
		}
		imagePath := ""
		for _, candidate := range []string{
			firstOrEmpty(pg.Request.Images),
			filepath.Join(l.modelDir, "Hy-Embodied-RxBrain-1.0", "demo_cases", "bridgev2_move_toy", "input", "obs_1.jpg"),
		} {
			if candidate == "" {
				continue
			}
			if _, err := os.Stat(candidate); err == nil {
				imagePath = candidate
				break
			}
		}
		if imagePath == "" {
			return "", fmt.Errorf("demo image not found")
		}
		raw, err := os.ReadFile(imagePath)
		if err != nil {
			return "", err
		}
		rgb, h, w, err := patchtower.DecodeImageBytesRGB(raw)
		if err != nil {
			return "", err
		}
		pv, gt, gh, gw, err := patchtower.PreprocessImage(pre, rgb, h, w)
		if err != nil {
			return "", err
		}
		if gt != pg.ImageGridTHW[0] || gh != pg.ImageGridTHW[1] || gw != pg.ImageGridTHW[2] {
			return "", fmt.Errorf("grid [%d,%d,%d] != golden %v", gt, gh, gw, pg.ImageGridTHW)
		}
		ids, err := routedlm.RenderVisionQAPrompt(tok, specials, pg.Request.Question, gh, gw, pre.MergeSize)
		if err != nil {
			return "", err
		}
		if err := requireIntSliceEqual("input ids", ids, pg.InputIDs); err != nil {
			return "", err
		}
		positions := routedlm.ImageMaskPositions(ids, specials)
		if err := requireIntSliceEqual("image mask positions", positions, pg.ImageMaskPositions); err != nil {
			return "", err
		}
		if len(positions) != pg.NumImageMaskTokens {
			return "", fmt.Errorf("image mask token count %d != golden %d", len(positions), pg.NumImageMaskTokens)
		}
		refPV, err := loadTensorAsset(l.fixturesDir, pg.PixelValuesAsset, pg.PixelValues)
		if err != nil {
			return "", err
		}
		if len(pv) != len(refPV) {
			return "", fmt.Errorf("pixel_values len %d != golden %d", len(pv), len(refPV))
		}
		var maxAbs, sumSq float64
		for i := range pv {
			d := float64(pv[i]) - float64(refPV[i])
			if d < 0 {
				d = -d
			}
			if d > maxAbs {
				maxAbs = d
			}
			sumSq += d * d
		}
		rms := math.Sqrt(sumSq / float64(len(pv)))
		if rms > 0.012 {
			return "", fmt.Errorf("pixel rms %.4g > 0.012", rms)
		}
		if maxAbs > 0.25 {
			return "", fmt.Errorf("pixel max|diff| %.4g > 0.25", maxAbs)
		}
		return fmt.Sprintf("ids=%d grid=[%d,%d,%d] pixel rms=%.4g max=%.4g", len(ids), gt, gh, gw, rms, maxAbs), nil
	})

	// ---- vision tower -------------------------------------------------
	pixelValues, err := loadTensorAsset(l.fixturesDir, pg.PixelValuesAsset, pg.PixelValues)
	if err != nil {
		return err
	}
	var preBlock0 []float32
	gate("pre_block0", func() (string, error) {
		patchWeights, err := patchtower.LoadPatchEmbedWeights(src, spec)
		if err != nil {
			return "", err
		}
		pos, err := patchtower.LoadPosEmbedTable(src, spec)
		if err != nil {
			return "", err
		}
		preBlock0, err = patchtower.PreBlock0Values(pixelValues, gridT, gridH, gridW, spec, patchWeights, pos)
		if err != nil {
			return "", err
		}
		return probeCheck("pre_block0", preBlock0, vg.VisionTensors["pre_block0"], 0.02, 0.02)
	})

	preAsset, err := loadTensorAsset(l.fixturesDir, vg.PreBlock0Asset, vg.VisionTensors["pre_block0"])
	if err != nil {
		return err
	}
	nPatch := gridT * gridH * gridW
	var block0 patchtower.BlockStageOutputs
	gate("block0_qkv", func() (string, error) {
		w, err := patchtower.LoadBlockWeights(src, spec, 0)
		if err != nil {
			return "", err
		}
		block0, err = patchtower.BlockForwardStages(preAsset, nPatch, spec, w)
		if err != nil {
			return "", err
		}
		if detail, err := probeCheck("block0_norm1", block0.Norm1, vg.VisionTensors["block0_norm1"], 0.02, 0.02); err != nil {
			return detail, err
		}
		return probeCheck("block0_qkv", block0.QKV, vg.VisionTensors["block0_qkv"], 0.08, 0.04)
	})
	gate("block0_attn", func() (string, error) {
		return probeCheck("block0_attn", block0.AttnProjected, vg.VisionTensors["block0_attn"], 0.12, 0.06)
	})
	gate("block0", func() (string, error) {
		return probeCheck("block0", block0.Output, vg.VisionTensors["block0"], 0.08, 0.04)
	})
	gate("block_last", func() (string, error) {
		hidden := preAsset
		for layer := 0; layer < spec.Depth; layer++ {
			w, err := patchtower.LoadBlockWeights(src, spec, layer)
			if err != nil {
				return "", err
			}
			stages, err := patchtower.BlockForwardStages(hidden, nPatch, spec, w)
			if err != nil {
				return "", err
			}
			hidden = stages.Output
		}
		return probeCheck("block_last", hidden, vg.VisionTensors["block_last"], 0.12, 0.06)
	})

	// ---- merger + prefill (from the golden block_last, the reference
	// ladder discipline: each stage restarts from the previous fixture) ---
	blockLast, err := loadTensorAsset(l.fixturesDir, vg.BlockLastAsset, vg.VisionTensors["block_last"])
	if err != nil {
		return err
	}
	merger, err := patchtower.LoadMergerWeights(src, spec)
	if err != nil {
		return err
	}
	imageRows, err := patchtower.ValidateMergerInputs(blockLast, gridT, gridH, gridW, spec)
	if err != nil {
		return err
	}
	gate("merger", func() (string, error) {
		values := make([]float32, imageRows*spec.OutHidden)
		scratch := patchtower.NewMergerScratch(spec)
		for row := 0; row < imageRows; row++ {
			patchtower.MergerRowInto(values[row*spec.OutHidden:(row+1)*spec.OutHidden], blockLast, row, gridH, gridW, spec, merger, &scratch)
		}
		return probeCheck("merger", values, vg.VisionTensors["merger"], 0.08, 0.04)
	})

	var prefill []float32
	buildPrefill := func() ([]float32, error) {
		scratch := patchtower.NewMergerScratch(spec)
		return routedlm.PrefillValues(src, cfg, binding, fg.InputIDs, fg.PrefillTensors.InputImageMaskPositions, imageRows, func(dst []float32, ordinal int) error {
			patchtower.MergerRowInto(dst, blockLast, ordinal, gridH, gridW, spec, merger, &scratch)
			return nil
		})
	}
	gate("prefill", func() (string, error) {
		var err error
		prefill, err = buildPrefill()
		if err != nil {
			return "", err
		}
		return probeCheck("prefill_embeds", prefill, fg.PrefillTensors.LanguageInputsEmbeds, 0.10, 0.05)
	})
	if prefill == nil {
		if prefill, err = buildPrefill(); err != nil {
			return err
		}
	}
	mask := fg.PrefillTensors.ModalityMask
	promptLen := fg.PromptLen

	// ---- MoT layer 0 sub-stages --------------------------------------
	var layer0 *routedlm.PromptLayerState
	gate("mot_layer0_sub_stages", func() (string, error) {
		w, err := routedlm.LoadLayerWeights(src, cfg, binding, 0)
		if err != nil {
			return "", err
		}
		layer0, err = routedlm.PromptLayerForward(prefill, mask, cfg, w)
		if err != nil {
			return "", err
		}
		mg, err := loadGoldenJSON[motGolden](l.fixturesDir, "rxbrain_vqa_mot_golden.json")
		if err != nil {
			return "", err
		}
		if _, err := probeCheck("layer0_pre_attn_norm", layer0.NormRows, mg.MoTTensors["layer0_pre_attn_norm"], 0.02, 0.02); err != nil {
			return "", err
		}
		qg, err := loadGoldenJSON[motGolden](l.fixturesDir, "rxbrain_vqa_mot_qkv_golden.json")
		if err != nil {
			return "", err
		}
		for _, item := range []struct {
			name   string
			values []float32
		}{
			{"layer0_q_proj", layer0.QProj},
			{"layer0_k_proj", layer0.KProj},
			{"layer0_v_proj", layer0.VProj},
		} {
			if _, err := probeCheck(item.name, item.values, qg.MoTTensors[item.name], 0.02, 0.02); err != nil {
				return "", err
			}
		}
		ng, err := loadGoldenJSON[motGolden](l.fixturesDir, "rxbrain_vqa_mot_qk_norm_golden.json")
		if err != nil {
			return "", err
		}
		hd := cfg.HeadDim
		relayout := func(flat []float32, heads int) []float32 {
			out := make([]float32, heads*promptLen*hd)
			width := heads * hd
			for token := 0; token < promptLen; token++ {
				for head := 0; head < heads; head++ {
					copy(out[(head*promptLen+token)*hd:(head*promptLen+token+1)*hd], flat[token*width+head*hd:token*width+(head+1)*hd])
				}
			}
			return out
		}
		if _, err := probeCheck("layer0_q_rope_norm", relayout(layer0.QHeads, cfg.NumAttentionHeads), ng.MoTTensors["layer0_q_rope_norm"], 0.08, 0.04); err != nil {
			return "", err
		}
		if _, err := probeCheck("layer0_k_rope_norm", relayout(layer0.KHeads, cfg.NumKeyValueHeads), ng.MoTTensors["layer0_k_rope_norm"], 0.08, 0.04); err != nil {
			return "", err
		}
		ag, err := loadGoldenJSON[motGolden](l.fixturesDir, "rxbrain_vqa_mot_attn_context_golden.json")
		if err != nil {
			return "", err
		}
		if _, err := probeCheck("layer0_attn_context", layer0.Context, ag.MoTTensors["layer0_attn_context"], 0.12, 0.06); err != nil {
			return "", err
		}
		og, err := loadGoldenJSON[motGolden](l.fixturesDir, "rxbrain_vqa_mot_attn_projected_golden.json")
		if err != nil {
			return "", err
		}
		if _, err := probeCheck("layer0_attn_projected", layer0.AttnProjected, og.MoTTensors["layer0_attn_projected"], 0.12, 0.06); err != nil {
			return "", err
		}
		return "norm+qkv+rope_norm+context+projected", nil
	})
	gate("mot_layer0_output", func() (string, error) {
		lg, err := loadGoldenJSON[motGolden](l.fixturesDir, "rxbrain_vqa_mot_layer0_output_golden.json")
		if err != nil {
			return "", err
		}
		return probeCheck("layer0_output", layer0.Output, lg.MoTTensors["layer0_output"], 0.18, 0.08)
	})
	layer0 = nil

	// ---- MoT layers 1..31, each from the previous golden boundary -----
	for layer := 1; layer < cfg.NumHiddenLayers; layer++ {
		layer := layer
		gate("mot_layer"+strconv.Itoa(layer)+"_output", func() (string, error) {
			prev, err := loadGoldenJSON[motGolden](l.fixturesDir, "rxbrain_vqa_mot_layer"+strconv.Itoa(layer-1)+"_output_golden.json")
			if err != nil {
				return "", err
			}
			prevKey := "layer" + strconv.Itoa(layer-1) + "_output"
			prevHidden, err := loadTensorAsset(l.fixturesDir, prev.LayerOutputAsset, prev.MoTTensors[prevKey])
			if err != nil {
				return "", err
			}
			golden, err := loadGoldenJSON[motGolden](l.fixturesDir, "rxbrain_vqa_mot_layer"+strconv.Itoa(layer)+"_output_golden.json")
			if err != nil {
				return "", err
			}
			w, err := routedlm.LoadLayerWeights(src, cfg, binding, layer)
			if err != nil {
				return "", err
			}
			state, err := routedlm.PromptLayerForward(prevHidden, mask, cfg, w)
			if err != nil {
				return "", err
			}
			key := "layer" + strconv.Itoa(layer) + "_output"
			return probeCheck(key, state.Output, golden.MoTTensors[key], 0.24, 0.10)
		})
	}

	// ---- terminal -----------------------------------------------------
	terminal, err := routedlm.LoadTerminalWeights(src, cfg, binding)
	if err != nil {
		return err
	}
	tg, err := loadGoldenJSON[terminalGolden](l.fixturesDir, "rxbrain_vqa_terminal_golden.json")
	if err != nil {
		return err
	}
	gate("terminal", func() (string, error) {
		lg, err := loadGoldenJSON[motGolden](l.fixturesDir, "rxbrain_vqa_mot_layer31_output_golden.json")
		if err != nil {
			return "", err
		}
		layer31, err := loadTensorAsset(l.fixturesDir, lg.LayerOutputAsset, lg.MoTTensors["layer31_output"])
		if err != nil {
			return "", err
		}
		finalHidden := tg.TerminalTensors["final_hidden"]
		logitIDs := append([]int{}, tg.LastLogits.ProbeIndex...)
		logitIDs = append(logitIDs, tg.LastLogits.TopIndex...)
		gotHidden, gotLogits, err := routedlm.TerminalProbeValues(layer31, cfg, terminal, 0, finalHidden.ProbeIndex, logitIDs)
		if err != nil {
			return "", err
		}
		if _, err := probeValuesCheck("final_hidden", gotHidden, finalHidden.ProbeValue, 0.24, 0.10); err != nil {
			return "", err
		}
		probeLogits := gotLogits[:len(tg.LastLogits.ProbeIndex)]
		topLogits := gotLogits[len(tg.LastLogits.ProbeIndex):]
		if _, err := probeValuesCheck("last_logits sparse", probeLogits, tg.LastLogits.ProbeValue, 0.75, 0.12); err != nil {
			return "", err
		}
		if _, err := probeValuesCheck("last_logits top", topLogits, tg.LastLogits.TopValue, 0.75, 0.12); err != nil {
			return "", err
		}
		if len(tg.LastLogits.TopIndex) == 0 {
			return "", fmt.Errorf("terminal golden missing top logits")
		}
		gotTopID, gotTopLogit, err := routedlm.TerminalTopToken(layer31, cfg, terminal, 0)
		if err != nil {
			return "", err
		}
		if gotTopID != tg.LastLogits.TopIndex[0] {
			return "", fmt.Errorf("terminal top token id=%d, want %d", gotTopID, tg.LastLogits.TopIndex[0])
		}
		if _, err := probeValuesCheck("last_logits argmax", []float32{gotTopLogit}, []float64{tg.LastLogits.TopValue[0]}, 0.75, 0.12); err != nil {
			return "", err
		}
		return fmt.Sprintf("top id=%d logit=%.4f", gotTopID, gotTopLogit), nil
	})

	// ---- decode: 12 steps from prompt boundaries ----------------------
	dg, err := loadGoldenJSON[decodeStepsGolden](l.fixturesDir, "rxbrain_vqa_decode_steps_golden.json")
	if err != nil {
		return err
	}
	var generated []int
	gate("decode_steps", func() (string, error) {
		if dg.FirstToken != tg.LastLogits.TopIndex[0] {
			return "", fmt.Errorf("decode first_token=%d, terminal top token=%d", dg.FirstToken, tg.LastLogits.TopIndex[0])
		}
		if len(dg.DecodeSteps) != dg.DecodeStepCount || len(dg.GeneratedTokens) != len(dg.DecodeSteps)+1 || dg.GeneratedTokens[0] != dg.FirstToken {
			return "", fmt.Errorf("decode golden shape steps=%d count=%d chain=%d", len(dg.DecodeSteps), dg.DecodeStepCount, len(dg.GeneratedTokens))
		}
		// Per-layer resident caches from the golden layer boundaries
		// (layer0 from prefill embeds, layer N from layer N-1's asset).
		resident := make([]*routedlm.ResidentKV, cfg.NumHiddenLayers)
		weights := make([]routedlm.LayerWeights, cfg.NumHiddenLayers)
		for layer := 0; layer < cfg.NumHiddenLayers; layer++ {
			w, err := routedlm.LoadLayerWeights(src, cfg, binding, layer)
			if err != nil {
				return "", err
			}
			weights[layer] = w
			hidden := prefill
			if layer > 0 {
				prev, err := loadGoldenJSON[motGolden](l.fixturesDir, "rxbrain_vqa_mot_layer"+strconv.Itoa(layer-1)+"_output_golden.json")
				if err != nil {
					return "", err
				}
				prevKey := "layer" + strconv.Itoa(layer-1) + "_output"
				hidden, err = loadTensorAsset(l.fixturesDir, prev.LayerOutputAsset, prev.MoTTensors[prevKey])
				if err != nil {
					return "", err
				}
			}
			resident[layer], err = routedlm.PromptResidentKV(hidden, mask, cfg, w)
			if err != nil {
				return "", err
			}
		}
		segments := routedlm.VisualSegments(mask)
		decodeMask := make([]int, promptLen+len(dg.DecodeSteps)+1)
		copy(decodeMask, mask)
		tokenID := dg.FirstToken
		generated = append(generated[:0], tokenID)
		for stepIndex, step := range dg.DecodeSteps {
			if step.Step != stepIndex || step.Position != promptLen+stepIndex || step.TokenIn != tokenID {
				return "", fmt.Errorf("decode step %d golden pos=%d token_in=%d, have token=%d", stepIndex, step.Position, step.TokenIn, tokenID)
			}
			embedding, err := routedlm.EmbeddingRows(src, cfg, binding, []int{tokenID})
			if err != nil {
				return "", err
			}
			tokenPos := resident[0].Tokens
			row, err := routedlm.DecodeLayerStack(embedding, decodeMask, segments, resident, tokenPos, cfg, weights)
			if err != nil {
				return "", fmt.Errorf("decode step %d: %w", stepIndex, err)
			}
			logitIDs := append([]int{}, step.Logits.ProbeIndex...)
			logitIDs = append(logitIDs, step.Logits.TopIndex...)
			gotHidden, gotLogits, err := routedlm.TerminalProbeValues(row, cfg, terminal, 0, step.FinalHidden.ProbeIndex, logitIDs)
			if err != nil {
				return "", err
			}
			if _, err := probeValuesCheck(fmt.Sprintf("step%d final_hidden", stepIndex), gotHidden, step.FinalHidden.ProbeValue, 0.45, 0.16); err != nil {
				return "", err
			}
			probeLogits := gotLogits[:len(step.Logits.ProbeIndex)]
			topLogits := gotLogits[len(step.Logits.ProbeIndex):]
			if _, err := probeValuesCheck(fmt.Sprintf("step%d logits sparse", stepIndex), probeLogits, step.Logits.ProbeValue, 1.75, 0.2); err != nil {
				return "", err
			}
			if _, err := probeValuesCheck(fmt.Sprintf("step%d logits top", stepIndex), topLogits, step.Logits.TopValue, 1.75, 0.2); err != nil {
				return "", err
			}
			nextID, _, err := routedlm.TerminalTopToken(row, cfg, terminal, 0)
			if err != nil {
				return "", err
			}
			if nextID != step.NextToken || dg.GeneratedTokens[stepIndex+1] != nextID {
				return "", fmt.Errorf("decode step %d next token id=%d, want step=%d chain=%d", stepIndex, nextID, step.NextToken, dg.GeneratedTokens[stepIndex+1])
			}
			generated = append(generated, nextID)
			tokenID = nextID
		}
		return fmt.Sprintf("12 steps, chain %v", generated), nil
	})

	gate("decode_text", func() (string, error) {
		tok, err := hfbpe.Load(l.modelDir)
		if err != nil {
			return "", err
		}
		if len(generated) == 0 {
			generated = append([]int(nil), dg.GeneratedTokens...)
		}
		if err := requireIntSliceEqual("generated chain", generated, dg.GeneratedTokens); err != nil {
			return "", err
		}
		text := tok.Decode(generated)
		const want = "The stovetop holds a metal pot on the left burner"
		if text != want {
			return "", fmt.Errorf("decoded text %q != %q", text, want)
		}
		return fmt.Sprintf("text=%q", text), nil
	})
	return nil
}

func firstOrEmpty(values []string) string {
	if len(values) == 0 {
		return ""
	}
	return values[0]
}
