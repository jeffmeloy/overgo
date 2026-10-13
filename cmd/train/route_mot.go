package main

import (
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"path/filepath"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/fixtureasset"
	"overgo/internal/jsonfile"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/patchtower"
	"overgo/internal/routedlm"
	"overgo/internal/runrecord"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/trainingprogram"
)

// The image grid's axes, in the order the vision golden records them.
const (
	gridTime = iota
	gridHeight
	gridWidth
	gridAxes
)

// motBinding: the declaration lookup — model facts live in the named
// binding, generic code selects it.
func motBinding(name string) (routedlm.BranchBinding, error) {
	switch name {
	case "rxbrain":
		return routedlm.RxBrainBinding(), nil
	case "sensenova":
		return routedlm.SenseNovaBinding(), nil
	}
	return routedlm.BranchBinding{}, fmt.Errorf("train -route mot: unknown binding %q", name)
}

type motPrefillGolden struct {
	InputIDs       []int `json:"input_ids"`
	PromptLen      int   `json:"prompt_len"`
	PrefillTensors struct {
		ModalityMask            []int `json:"modality_mask"`
		InputImageMaskPositions []int `json:"input_image_mask_positions"`
	} `json:"prefill_tensors"`
}

type motGoldenTensor struct {
	Sha256 string `json:"sha256"`
	Shape  []int  `json:"shape"`
}

type motVisionGolden struct {
	ImageGridTHW   []int                      `json:"image_grid_thw"`
	BlockLastAsset string                     `json:"block_last_asset"`
	VisionTensors  map[string]motGoldenTensor `json:"vision_tensors"`
}

func loadMotTensorAsset(fixturesDir, asset string, golden motGoldenTensor) ([]float32, error) {
	elements := 1
	for _, dim := range golden.Shape {
		elements *= dim
	}
	return fixtureasset.LoadF32(fixturesDir, asset, golden.Sha256, elements)
}

// trainMot is the mot route: full modality-transformer training as a
// director-supervised model-session. The real vision/text forward supplies
// the hidden rows (the same golden-verified prefill splice the VQA parity
// ladder runs), the trainable surface is the vision-branch fork of every
// layer as f32 masters from the BF16 serving storage, the objective is the
// pipeline's own next-token cross-entropy at the prompt's text positions,
// and the run publishes one typed session observation and the trained fork
// as a checkpoint: the full-pipeline run the organ cells named as their
// promotion gate.
func trainMot(ctx context.Context, flags *flag.FlagSet, args []string) (err error) {
	modelDir := flags.String("model", "", "branch-routed checkpoint dir")
	fixturesDir := flags.String("fixtures", "", "golden fixture dir")
	bindingName := flags.String("binding", "", "branch binding name")
	steps := clioptions.IntOverride(flags, "steps", "required observed training steps")
	maxWall := clioptions.DurationOverride(flags, "max-wall", "optional projected-wall bound")
	storePath := flags.String("store", "overgodb-store", "OvergoDB for the session observation, recipe authority and checkpoint record")
	stimulus := flags.String("stimulus", "docs/verification/rxbrain-mot-training-stimulus.txt", "committed stimulus declaration grounding the recipe dataset")
	out := flags.String("out", "", "directory the trained vision fork's checkpoint publishes to")
	proof := proofFlag(flags)
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *modelDir == "" || *fixturesDir == "" || *bindingName == "" || *out == "" || *steps <= 0 {
		return fmt.Errorf("train -route mot: model, fixtures, binding, out, and positive steps required")
	}

	binding, err := motBinding(*bindingName)
	if err != nil {
		return err
	}
	store, err := overgodb.Open(*storePath)
	if err != nil {
		return err
	}
	defer store.Close()

	shards, err := filepath.Glob(filepath.Join(*modelDir, "model-*-of-*.safetensors"))
	if err != nil || len(shards) == 0 {
		return fmt.Errorf("train -route mot: no shards under %s", *modelDir)
	}
	session, err := openSession(ctx, store, shards[0], *stimulus)
	if err != nil {
		return err
	}
	session.proof = *proof
	defer session.settle(ctx, &err)

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
	var vg motVisionGolden
	if err := jsonfile.Decode(filepath.Join(*fixturesDir, "rxbrain_vqa_vision_golden.json"), &vg); err != nil {
		return err
	}
	var fg motPrefillGolden
	if err := jsonfile.Decode(filepath.Join(*fixturesDir, "rxbrain_vqa_prefill_golden.json"), &fg); err != nil {
		return err
	}
	if len(vg.ImageGridTHW) != gridAxes {
		return fmt.Errorf("train -route mot: vision golden grid %v", vg.ImageGridTHW)
	}
	spec, err := patchtower.LoadSpec(*modelDir)
	if err != nil {
		return err
	}
	blockLast, err := loadMotTensorAsset(*fixturesDir, vg.BlockLastAsset, vg.VisionTensors["block_last"])
	if err != nil {
		return err
	}
	merger, err := patchtower.LoadMergerWeights(src, spec)
	if err != nil {
		return err
	}
	gridT, gridH, gridW := vg.ImageGridTHW[gridTime], vg.ImageGridTHW[gridHeight], vg.ImageGridTHW[gridWidth]
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
		return fmt.Errorf("train -route mot: no supervised text positions")
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
	scalarBytes := binary.Size(uint16(0))
	headRaw := make([]byte, cfg.VocabSize*cfg.HiddenSize*int(scalarBytes))
	if terminal.Head.DType != "BF16" {
		return fmt.Errorf("train -route mot: head dtype %s, want BF16", terminal.Head.DType)
	}
	if _, err := terminal.Head.ReadAt(headRaw, int64(tensor.FirstOffset)); err != nil {
		return err
	}
	head := make([]uint16, cfg.VocabSize*cfg.HiddenSize)
	for i := range head {
		head[i] = binary.LittleEndian.Uint16(headRaw[i*int(scalarBytes):])
	}
	headRaw = nil
	trainer, err := routedlm.NewModalityTransformerTrainer(cfg, layers, terminal.FinalNorm[0], head, session.authority.Optimizer)
	if err != nil {
		return err
	}
	defer trainer.Close()
	plan, err := trainingprogram.CompileProbeSessionPlan(trainingprogram.ProbeSpec{
		Objective: session.authority.Objective, Updates: *steps, MaximumSequence: fg.PromptLen,
		MaxProjectedWall: *maxWall, Parameters: trainer.ParameterCount(), Optimizer: session.authority.Optimizer,
	})
	if err != nil {
		return err
	}
	session.observer.Phase(runrecord.PhaseLoad, time.Since(loadStarted))
	fmt.Printf("full modality-transformer: layers=%d trainable_parameters=%d prompt_tokens=%d image_rows=%d supervised_positions=%d\n",
		cfg.NumHiddenLayers, trainer.ParameterCount(), fg.PromptLen, imageRows, len(targets))
	fmt.Printf("derived_lr=%.6g derived_momentum=%.10g load_wall=%s\n",
		trainer.Config().BaseLearningRate, trainer.Config().Momentum, time.Since(loadStarted).Round(time.Millisecond))
	return boundedRun{name: "train -route mot", plan: plan, session: session,
		step: func(int) (optimizer.ObservedStepResult, error) { return trainer.Step(hidden, mask, targets) },
		loss: func() (float64, error) { return trainer.Loss(hidden, mask, targets) },
		// Every update trains on the one declared prompt.
		publish: func(ctx context.Context) error {
			return session.publishPlanned(ctx, *out, trainer, plan.Updates())
		},
	}.train(ctx)
}
