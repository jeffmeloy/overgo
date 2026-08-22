// Command mot-train-probe executes full modality-transformer training as a
// director-supervised model-session: the real vision/text forward supplies
// the hidden rows (the same golden-verified prefill splice the VQA parity
// ladder runs), the trainable surface is the vision-branch fork of EVERY
// layer as f32 masters from the BF16 serving storage, the objective is the
// pipeline's own next-token cross-entropy at the prompt's text positions,
// and the run publishes one typed session observation. This is the
// full-pipeline run the organ cells named as their promotion gate.
package main

import (
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/fixtureasset"
	"overgo/internal/jsonfile"
	"overgo/internal/patchtower"
	"overgo/internal/repodb"
	"overgo/internal/routedlm"
	"overgo/internal/runrecord"
	"overgo/internal/safetensors"
	"overgo/internal/trainingworkflow"
)

func main() {
	clioptions.MainNamed("mot-train-probe", run)
}

// bindingByName: the declaration lookup — model facts live in the named
// binding, generic code selects it.
func bindingByName(name string) (routedlm.BranchBinding, error) {
	switch name {
	case "rxbrain":
		return routedlm.RxBrainBinding(), nil
	case "sensenova":
		return routedlm.SenseNovaBinding(), nil
	}
	return routedlm.BranchBinding{}, fmt.Errorf("mot train probe: unknown binding %q", name)
}

type prefillGolden struct {
	InputIDs       []int `json:"input_ids"`
	PromptLen      int   `json:"prompt_len"`
	PrefillTensors struct {
		ModalityMask            []int `json:"modality_mask"`
		InputImageMaskPositions []int `json:"input_image_mask_positions"`
	} `json:"prefill_tensors"`
}

type goldenTensor struct {
	Sha256 string `json:"sha256"`
	Shape  []int  `json:"shape"`
}

type visionGolden struct {
	ImageGridTHW   []int                   `json:"image_grid_thw"`
	BlockLastAsset string                  `json:"block_last_asset"`
	VisionTensors  map[string]goldenTensor `json:"vision_tensors"`
}

func loadTensorAsset(fixturesDir, asset string, golden goldenTensor) ([]float32, error) {
	elements := 1
	for _, dim := range golden.Shape {
		elements *= dim
	}
	return fixtureasset.LoadF32(fixturesDir, asset, golden.Sha256, elements)
}

func run() error {
	modelDir := flag.String("model", `C:\Users\jeffm\adaptive_new\models\Hy-Embodied-RxBrain-1.0`, "branch-routed checkpoint dir")
	fixturesDir := flag.String("fixtures", `C:\Users\jeffm\adaptive_new\fixtures`, "golden fixture dir (real prompt + vision rows)")
	bindingName := flag.String("binding", "rxbrain", "branch binding name")
	steps := flag.Int("steps", 3, "observed training steps")
	maxWall := flag.Duration("max-wall", 25*time.Minute, "abort when the first step projects past this bound")
	storePath := flag.String("store", "repodb-store", "RepoDB for the session observation and recipe authority")
	stimulus := flag.String("stimulus", "docs/verification/rxbrain-mot-training-stimulus.txt", "committed stimulus declaration grounding the recipe dataset")
	flag.Parse()
	ctx := context.Background()

	binding, err := bindingByName(*bindingName)
	if err != nil {
		return err
	}
	store, err := repodb.Open(*storePath)
	if err != nil {
		return err
	}
	defer store.Close()

	// Recipe authority + model identity, then session admission BEFORE any
	// weight touches memory.
	shards, err := filepath.Glob(filepath.Join(*modelDir, "model-*-of-*.safetensors"))
	if err != nil || len(shards) == 0 {
		return fmt.Errorf("mot train probe: no shards under %s", *modelDir)
	}
	modelFile, err := os.Open(shards[0])
	if err != nil {
		return err
	}
	modelID, _, err := artifact.Identify(artifact.KindModel, modelFile)
	modelFile.Close()
	if err != nil {
		return err
	}
	recipeID, err := trainingworkflow.BootstrapTokenRecipe(ctx, store, shards[0], *stimulus)
	if err != nil {
		return err
	}
	observer, err := trainingworkflow.NewObserver(store, false)
	if err != nil {
		return err
	}
	if err := observer.Admit(ctx, modelID, recipeID); err != nil {
		return err
	}
	fmt.Printf("session admitted: model=%s recipe=%s\n", modelID, recipeID)

	// Real prompt: golden-verified token ids, modality mask, and vision rows
	// through the real merger — the exact splice the serving prefill runs.
	loadStarted := time.Now()
	cfg, err := routedlm.LoadConfig(*modelDir, binding)
	if err != nil {
		return err
	}
	src, err := safetensors.OpenSource(*modelDir)
	if err != nil {
		return err
	}
	defer src.Close()
	var vg visionGolden
	if err := jsonfile.Decode(filepath.Join(*fixturesDir, "rxbrain_vqa_vision_golden.json"), &vg); err != nil {
		return err
	}
	var fg prefillGolden
	if err := jsonfile.Decode(filepath.Join(*fixturesDir, "rxbrain_vqa_prefill_golden.json"), &fg); err != nil {
		return err
	}
	if len(vg.ImageGridTHW) != 3 {
		return fmt.Errorf("mot train probe: vision golden grid %v", vg.ImageGridTHW)
	}
	spec, err := patchtower.LoadSpec(*modelDir)
	if err != nil {
		return err
	}
	blockLast, err := loadTensorAsset(*fixturesDir, vg.BlockLastAsset, vg.VisionTensors["block_last"])
	if err != nil {
		return err
	}
	merger, err := patchtower.LoadMergerWeights(src, spec)
	if err != nil {
		return err
	}
	gridT, gridH, gridW := vg.ImageGridTHW[0], vg.ImageGridTHW[1], vg.ImageGridTHW[2]
	imageRows, err := patchtower.ValidateMergerInputs(blockLast, gridT, gridH, gridW, spec)
	if err != nil {
		return err
	}
	scratch := patchtower.NewMergerScratch(spec)
	hidden, err := routedlm.PrefillValues(src, cfg, binding, fg.InputIDs, fg.PrefillTensors.InputImageMaskPositions, imageRows, func(dst []float32, ordinal int) error {
		patchtower.MergerRowInto(dst, blockLast, ordinal, gridH, gridW, spec, merger, &scratch)
		return nil
	})
	if err != nil {
		return err
	}
	mask := fg.PrefillTensors.ModalityMask

	// Supervised positions: every prompt row whose next token is a text
	// token — the pipeline's own next-token contract on the real mixed
	// sequence.
	var targets []routedlm.MoTTarget
	for pos := 0; pos+1 < fg.PromptLen; pos++ {
		if mask[pos+1] == 0 {
			targets = append(targets, routedlm.MoTTarget{Position: pos, Token: fg.InputIDs[pos+1]})
		}
	}
	if len(targets) == 0 {
		return fmt.Errorf("mot train probe: no supervised text positions")
	}

	// Every layer's dual-branch weights + the terminal (frozen).
	layers := make([]routedlm.LayerWeights, cfg.NumHiddenLayers)
	for layer := range layers {
		if layers[layer], err = routedlm.LoadLayerWeights(src, cfg, binding, layer); err != nil {
			return err
		}
	}
	terminal, err := routedlm.LoadTerminalWeights(src, cfg, binding)
	if err != nil {
		return err
	}
	headRaw := make([]byte, cfg.VocabSize*cfg.HiddenSize*2)
	if terminal.Head.DType != "BF16" {
		return fmt.Errorf("mot train probe: head dtype %s, want BF16", terminal.Head.DType)
	}
	if _, err := terminal.Head.ReadAt(headRaw, 0); err != nil {
		return err
	}
	head := make([]uint16, cfg.VocabSize*cfg.HiddenSize)
	for i := range head {
		head[i] = binary.LittleEndian.Uint16(headRaw[i*2:])
	}
	headRaw = nil
	trainer, err := routedlm.NewModalityTransformerTrainer(cfg, layers, terminal.FinalNorm[0], head)
	if err != nil {
		return err
	}
	defer trainer.Close()
	observer.Phase(runrecord.PhaseLoad, time.Since(loadStarted))
	fmt.Printf("full modality-transformer: layers=%d trainable_parameters=%d prompt_tokens=%d image_rows=%d supervised_positions=%d\n",
		cfg.NumHiddenLayers, trainer.ParameterCount(), fg.PromptLen, imageRows, len(targets))
	fmt.Printf("derived_lr=%.6g derived_momentum=%.10g load_wall=%s\n",
		trainer.Config().BaseLearningRate, trainer.Config().Momentum, time.Since(loadStarted).Round(time.Millisecond))

	initial, err := trainer.Loss(hidden, mask, targets)
	if err != nil {
		return err
	}
	fmt.Printf("initial loss %.6f (next-token cross-entropy over %d real positions)\n", initial, len(targets))

	trainStarted := time.Now()
	runErr := func() error {
		for step := 0; step < *steps; step++ {
			stepStarted := time.Now()
			result, err := trainer.Step(hidden, mask, targets)
			if err != nil {
				return err
			}
			observer.SampleStep()
			wall := time.Since(stepStarted)
			fmt.Printf("step %d/%d loss %.6f grad_l2 %.6f lr %.6g wall %s\n",
				result.Step, *steps, result.Loss, result.GradientL2, result.LearningRate, wall.Round(time.Millisecond))
			if result.GradientL2 <= 0 {
				return fmt.Errorf("mot train probe: step %d gradient norm %g", result.Step, result.GradientL2)
			}
			if step == 0 && *maxWall > 0 {
				projected := wall * time.Duration(*steps+1)
				if projected > *maxWall {
					return fmt.Errorf("mot train probe: first step %s projects %d steps to %s, over %s", wall.Round(time.Second), *steps, projected.Round(time.Minute), *maxWall)
				}
			}
		}
		final, err := trainer.Loss(hidden, mask, targets)
		if err != nil {
			return err
		}
		fmt.Printf("after-evaluation loss %.6f (initial %.6f)\n", final, initial)
		if !(final < initial) {
			return fmt.Errorf("mot train probe: no descent: %.6f -> %.6f", initial, final)
		}
		return nil
	}()
	observer.Phase(runrecord.PhaseForwardBackward, time.Since(trainStarted))
	streamed := uint64(*steps) * uint64(fg.PromptLen)
	observation, observeErr := observer.Finish(ctx, modelID, recipeID, runErr, streamed)
	if runErr != nil {
		return runErr
	}
	if observeErr != nil {
		return observeErr
	}
	fmt.Printf("session observation: %s\n", observation)
	return nil
}
