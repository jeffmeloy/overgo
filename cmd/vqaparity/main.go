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
	"slices"
	"strconv"

	"overgo/internal/hfbpe"
	"overgo/internal/parity"
	"overgo/internal/patchtower"
	"overgo/internal/routedlm"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
)

// binding: RxBrain's `_v` branch naming contract.
var binding = routedlm.RxBrainBinding()

// Role literals of this checkpoint's prompt specials; sourced from its
// tokenizer_config.json added_tokens_decoder role assignments (the reference
// OvergoDB profile facts, derivation tokenizer-special-token-role/v1).
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

type campaignContext struct {
	*parity.Campaign
	modelDir    string
	fixturesDir string
}

func main() {
	modelDir := flag.String("model", `C:\Users\jeffm\adaptive_new\models\Hy-Embodied-RxBrain-1.0`, "checkpoint dir")
	fixturesDir := flag.String("fixtures", `C:\Users\jeffm\adaptive_new\fixtures`, "fixture dir")
	logPath := flag.String("log", `build\rxbrain_port_log.txt`, "liveness log path")
	fromStage := flag.String("from", "", "skip stages before this name")
	toStage := flag.String("to", "", "stop after this stage name")
	deviceMode := flag.Bool("device", false, "run the CUDA device terminal parity + measurement")
	deviceDecodeMode := flag.Bool("device-decode", false, "run the CUDA device 32-layer decode parity + measurement")
	deviceVisionMode := flag.Bool("device-vision", false, "run the CUDA device vision-blocks parity + measurement")
	deviceMergerMode := flag.Bool("device-merger", false, "run the CUDA device merger parity + measurement")
	devicePrefillMode := flag.Bool("device-prefill", false, "run the CUDA device branch-routed prefill parity + measurement")
	deviceFullMode := flag.Bool("device-full", false, "run the CUDA device full pipeline (merger->prefill->decode) + measurement")
	recipeServeMode := flag.Bool("recipe-serve", false, "serve the canonical case THROUGH the activated VQA recipe (gate on active recipe, real processor, decode-until-EOS)")
	recipeVerifyMode := flag.Bool("recipe-verify-activate", false, "verify the canonical real-device VQA case, activate its recipe, then replay through the active recipe")
	repoFlag := flag.String("repo", "", "OvergoDB store for recipe modes; empty resolves via the data-root contract")
	imageFlag := flag.String("image", `C:\Users\jeffm\adaptive_new\models\Hy-Embodied-RxBrain-1.0\Hy-Embodied-RxBrain-1.0\demo_cases\bridgev2_move_toy\input\obs_1.jpg`, "serve image path")
	questionFlag := flag.String("question", "What objects are on the stovetop, and where is the green toy?", "serve question")
	flag.Parse()
	l := &campaignContext{Campaign: parity.NewCampaign(*logPath), modelDir: *modelDir, fixturesDir: *fixturesDir}
	if *recipeVerifyMode {
		if err := verifyActivateVQA(l, *repoFlag, *imageFlag, *questionFlag); err != nil {
			l.Log("RECIPE verify ERROR " + err.Error())
			os.Exit(1)
		}
		return
	}
	if *recipeServeMode {
		if err := runRecipeServe(l, *repoFlag, *imageFlag, *questionFlag); err != nil {
			l.Log("RECIPE serve ERROR " + err.Error())
			os.Exit(1)
		}
		return
	}
	if *deviceFullMode {
		if err := runDeviceFull(l); err != nil {
			l.Log("DEVICE FULL ERROR " + err.Error())
			os.Exit(1)
		}
		return
	}
	if *devicePrefillMode {
		if err := runDevicePrefill(l); err != nil {
			l.Log("DEVICE PREFILL ERROR " + err.Error())
			os.Exit(1)
		}
		return
	}
	if *deviceMergerMode {
		if err := runDeviceMerger(l); err != nil {
			l.Log("DEVICE MERGER ERROR " + err.Error())
			os.Exit(1)
		}
		return
	}
	if *deviceVisionMode {
		if err := runDeviceVision(l); err != nil {
			l.Log("DEVICE VISION ERROR " + err.Error())
			os.Exit(1)
		}
		return
	}
	if *deviceDecodeMode {
		if err := runDeviceDecode(l); err != nil {
			l.Log("DEVICE DECODE ERROR " + err.Error())
			os.Exit(1)
		}
		return
	}
	if *deviceMode {
		if err := runDevice(l); err != nil {
			l.Log("DEVICE ERROR " + err.Error())
			os.Exit(1)
		}
		return
	}
	if err := run(l, *fromStage, *toStage); err != nil {
		l.Log("LADDER ERROR " + err.Error())
		os.Exit(1)
	}
	if l.Failed() {
		l.Log(fmt.Sprintf("LADDER STOPPED after %d/%d stages passed", l.Count(parity.Pass), l.Total()))
		os.Exit(1)
	}
	l.Log(fmt.Sprintf("LADDER COMPLETE %d/%d stages passed", l.Count(parity.Pass), l.Total()))
}

func run(l *campaignContext, fromStage, toStage string) error {
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
		l.Run(name, func() (parity.Verdict, float64, string, error) {
			detail, err := fn()
			return parity.Pass, math.NaN(), detail, err
		})
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
	gridT, gridH, gridW := vg.ImageGridTHW[tensor.FirstOffset], vg.ImageGridTHW[tensor.SingletonExtent], vg.ImageGridTHW[tensor.PairedExtent]

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
		if gt != pg.ImageGridTHW[tensor.FirstOffset] || gh != pg.ImageGridTHW[tensor.SingletonExtent] || gw != pg.ImageGridTHW[tensor.PairedExtent] {
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
		if limit := acceptance(acceptPixelRMS).Absolute; rms > limit {
			return "", fmt.Errorf("pixel rms %.4g > %.4g", rms, limit)
		}
		if limit := acceptance(acceptPixelMaxAbs).Absolute; maxAbs > limit {
			return "", fmt.Errorf("pixel max|diff| %.4g > %.4g", maxAbs, limit)
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
		return probeCheck("pre_block0", preBlock0, vg.VisionTensors["pre_block0"], acceptNorm)
	})

	preAsset, err := loadTensorAsset(l.fixturesDir, vg.PreBlock0Asset, vg.VisionTensors["pre_block0"])
	if err != nil {
		return err
	}
	nPatch := gridT * gridH * gridW
	var block0 patchtower.BlockStageOutputs
	gate("block0_qkv", func() (string, error) {
		w, err := patchtower.LoadBlockWeights(src, spec, tensor.FirstOffset)
		if err != nil {
			return "", err
		}
		block0, err = patchtower.BlockForwardStages(preAsset, nPatch, spec, w)
		if err != nil {
			return "", err
		}
		if detail, err := probeCheck("block0_norm1", block0.Norm1, vg.VisionTensors["block0_norm1"], acceptNorm); err != nil {
			return detail, err
		}
		return probeCheck("block0_qkv", block0.QKV, vg.VisionTensors["block0_qkv"], acceptProjection)
	})
	gate("block0_attn", func() (string, error) {
		return probeCheck("block0_attn", block0.AttnProjected, vg.VisionTensors["block0_attn"], acceptAttention)
	})
	gate("block0", func() (string, error) {
		return probeCheck("block0", block0.Output, vg.VisionTensors["block0"], acceptProjection)
	})
	gate("block_last", func() (string, error) {
		hidden := preAsset
		for layer := tensor.FirstOffset; layer < spec.Depth; layer++ {
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
		return probeCheck("block_last", hidden, vg.VisionTensors["block_last"], acceptAttention)
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
		for row := tensor.FirstOffset; row < imageRows; row++ {
			patchtower.MergerRowInto(values[row*spec.OutHidden:(row+tensor.SingletonExtent)*spec.OutHidden], blockLast, row, gridH, gridW, spec, merger, &scratch)
		}
		return probeCheck("merger", values, vg.VisionTensors["merger"], acceptProjection)
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
		return probeCheck("prefill_embeds", prefill, fg.PrefillTensors.LanguageInputsEmbeds, acceptPrefill)
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
		w, err := routedlm.LoadLayerWeights(src, cfg, binding, tensor.FirstOffset)
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
		if _, err := probeCheck("layer0_pre_attn_norm", layer0.NormRows, mg.MoTTensors["layer0_pre_attn_norm"], acceptNorm); err != nil {
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
			if _, err := probeCheck(item.name, item.values, qg.MoTTensors[item.name], acceptNorm); err != nil {
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
			for token := range promptLen {
				for head := range heads {
					copy(out[(head*promptLen+token)*hd:(head*promptLen+token+tensor.SingletonExtent)*hd], flat[token*width+head*hd:token*width+(head+tensor.SingletonExtent)*hd])
				}
			}
			return out
		}
		if _, err := probeCheck("layer0_q_rope_norm", relayout(layer0.QHeads, cfg.NumAttentionHeads), ng.MoTTensors["layer0_q_rope_norm"], acceptProjection); err != nil {
			return "", err
		}
		if _, err := probeCheck("layer0_k_rope_norm", relayout(layer0.KHeads, cfg.NumKeyValueHeads), ng.MoTTensors["layer0_k_rope_norm"], acceptProjection); err != nil {
			return "", err
		}
		ag, err := loadGoldenJSON[motGolden](l.fixturesDir, "rxbrain_vqa_mot_attn_context_golden.json")
		if err != nil {
			return "", err
		}
		if _, err := probeCheck("layer0_attn_context", layer0.Context, ag.MoTTensors["layer0_attn_context"], acceptAttention); err != nil {
			return "", err
		}
		og, err := loadGoldenJSON[motGolden](l.fixturesDir, "rxbrain_vqa_mot_attn_projected_golden.json")
		if err != nil {
			return "", err
		}
		if _, err := probeCheck("layer0_attn_projected", layer0.AttnProjected, og.MoTTensors["layer0_attn_projected"], acceptAttention); err != nil {
			return "", err
		}
		return "norm+qkv+rope_norm+context+projected", nil
	})
	gate("mot_layer0_output", func() (string, error) {
		lg, err := loadGoldenJSON[motGolden](l.fixturesDir, "rxbrain_vqa_mot_layer0_output_golden.json")
		if err != nil {
			return "", err
		}
		return probeCheck("layer0_output", layer0.Output, lg.MoTTensors["layer0_output"], acceptFirstLayer)
	})
	layer0 = nil

	// ---- MoT layers 1..31, each from the previous golden boundary -----
	for layer := tensor.SingletonExtent; layer < cfg.NumHiddenLayers; layer++ {
		gate("mot_layer"+strconv.Itoa(layer)+"_output", func() (string, error) {
			prev, err := loadGoldenJSON[motGolden](l.fixturesDir, "rxbrain_vqa_mot_layer"+strconv.Itoa(layer-tensor.SingletonExtent)+"_output_golden.json")
			if err != nil {
				return "", err
			}
			prevKey := "layer" + strconv.Itoa(layer-tensor.SingletonExtent) + "_output"
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
			return probeCheck(key, state.Output, golden.MoTTensors[key], acceptLayer)
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
		logitIDs := slices.Clone(tg.LastLogits.ProbeIndex)
		logitIDs = append(logitIDs, tg.LastLogits.TopIndex...)
		gotHidden, gotLogits, err := routedlm.TerminalProbeValues(layer31, cfg, terminal, tensor.FirstOffset, finalHidden.ProbeIndex, logitIDs)
		if err != nil {
			return "", err
		}
		if _, err := probeValuesCheck("final_hidden", gotHidden, finalHidden.ProbeValue, acceptLayer); err != nil {
			return "", err
		}
		probeLogits := gotLogits[:len(tg.LastLogits.ProbeIndex)]
		topLogits := gotLogits[len(tg.LastLogits.ProbeIndex):]
		if _, err := probeValuesCheck("last_logits sparse", probeLogits, tg.LastLogits.ProbeValue, acceptTerminal); err != nil {
			return "", err
		}
		if _, err := probeValuesCheck("last_logits top", topLogits, tg.LastLogits.TopValue, acceptTerminal); err != nil {
			return "", err
		}
		if len(tg.LastLogits.TopIndex) == 0 {
			return "", fmt.Errorf("terminal golden missing top logits")
		}
		gotTopID, gotTopLogit, err := routedlm.TerminalTopToken(layer31, cfg, terminal, tensor.FirstOffset)
		if err != nil {
			return "", err
		}
		if gotTopID != tg.LastLogits.TopIndex[0] {
			return "", fmt.Errorf("terminal top token id=%d, want %d", gotTopID, tg.LastLogits.TopIndex[0])
		}
		if _, err := probeValuesCheck("last_logits argmax", []float32{gotTopLogit}, []float64{tg.LastLogits.TopValue[0]}, acceptTerminal); err != nil {
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
		if len(dg.DecodeSteps) != dg.DecodeStepCount || len(dg.GeneratedTokens) != len(dg.DecodeSteps)+tensor.SingletonExtent || dg.GeneratedTokens[tensor.FirstOffset] != dg.FirstToken {
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
				prev, err := loadGoldenJSON[motGolden](l.fixturesDir, "rxbrain_vqa_mot_layer"+strconv.Itoa(layer-tensor.SingletonExtent)+"_output_golden.json")
				if err != nil {
					return "", err
				}
				prevKey := "layer" + strconv.Itoa(layer-tensor.SingletonExtent) + "_output"
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
		decodeMask := make([]int, promptLen+len(dg.DecodeSteps)+tensor.SingletonExtent)
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
			logitIDs := slices.Clone(step.Logits.ProbeIndex)
			logitIDs = append(logitIDs, step.Logits.TopIndex...)
			gotHidden, gotLogits, err := routedlm.TerminalProbeValues(row, cfg, terminal, tensor.FirstOffset, step.FinalHidden.ProbeIndex, logitIDs)
			if err != nil {
				return "", err
			}
			if _, err := probeValuesCheck(fmt.Sprintf("step%d final_hidden", stepIndex), gotHidden, step.FinalHidden.ProbeValue, acceptDecodeHidden); err != nil {
				return "", err
			}
			probeLogits := gotLogits[:len(step.Logits.ProbeIndex)]
			topLogits := gotLogits[len(step.Logits.ProbeIndex):]
			if _, err := probeValuesCheck(fmt.Sprintf("step%d logits sparse", stepIndex), probeLogits, step.Logits.ProbeValue, acceptDecodeLogits); err != nil {
				return "", err
			}
			if _, err := probeValuesCheck(fmt.Sprintf("step%d logits top", stepIndex), topLogits, step.Logits.TopValue, acceptDecodeLogits); err != nil {
				return "", err
			}
			nextID, _, err := routedlm.TerminalTopToken(row, cfg, terminal, tensor.FirstOffset)
			if err != nil {
				return "", err
			}
			chainIndex := stepIndex + tensor.SingletonExtent
			if nextID != step.NextToken || dg.GeneratedTokens[chainIndex] != nextID {
				return "", fmt.Errorf("decode step %d next token id=%d, want step=%d chain=%d", stepIndex, nextID, step.NextToken, dg.GeneratedTokens[chainIndex])
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
