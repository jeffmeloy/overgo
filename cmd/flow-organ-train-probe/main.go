// flow-organ-train-probe: bounded observed training smoke for a routed-LM
// flow-head organ (the SenseNova-family fm_head MLP). The probe is the
// production driver of the flow-head trainer and the reproducible generator
// of its verification evidence: it compiles the flow plan from the real
// artifact's own declarations, packs the fm_head organ as f32 masters
// decoded from the BF16 serving storage, trains on a deterministic committed
// stimulus with fully DERIVED hyperparameters (base LR n_params^-1/2,
// momentum from the CLT effective-samples rule), observes every step with a
// first-step projection guard, and proves descent by re-evaluating the same
// pair's loss after the observed steps. Full-pipeline MoT training remains
// the promotion gate beyond the organ; this claim covers the organ's
// training machinery at real-artifact-smoke tier.
package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"overgo/internal/dataroot"
	"overgo/internal/latentimage"
	"overgo/internal/pytorchzip"
	"overgo/internal/routedlm"
	"overgo/internal/safetensors"
)

func main() {
	model := flag.String("model", "", "model directory (safetensors + config.json)")
	organ := flag.String("organ", "flow-head", "trainable organ: flow-head (SenseNova fm_head), latent-bridge (RxBrain llm2vae/vae2llm), denoiser-final (Krea-2 MMDiT output head), or video-head (Wan2.1 head)")
	steps := flag.Int("steps", 3, "observed Muon steps")
	rows := flag.Int("rows", 8, "stimulus rows")
	timestep := flag.Float64("timestep", 0.25, "flow timestep in [0,1)")
	maxWall := flag.Duration("max-wall", 25*time.Minute, "abort when the first measured step projects the run past this bound")
	flag.Parse()
	var err error
	switch *organ {
	case "flow-head":
		err = run(*model, *steps, *rows, *timestep, *maxWall)
	case "latent-bridge":
		err = runLatentBridge(*model, *steps, *rows, *maxWall)
	case "denoiser-final":
		err = runDenoiserFinal(*model, latentimage.KreaFinalLayerBinding(), *steps, *rows, *maxWall)
	case "video-head":
		err = runDenoiserFinal(*model, latentimage.WanHeadBinding(), *steps, *rows, *maxWall)
	case "edit-head":
		err = runEditHead(*model, *steps, *rows, *maxWall)
	default:
		err = fmt.Errorf("flow-organ-train-probe: unknown organ %q (flow-head, latent-bridge)", *organ)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// runDenoiserFinal: the Krea-2-family MMDiT output head trains the velocity
// MSE objective on a deterministic committed stimulus.
func runDenoiserFinal(modelDir string, binding latentimage.FinalLayerBinding, steps, rows int, maxWall time.Duration) error {
	if modelDir == "" || steps <= 0 || rows <= 0 {
		return fmt.Errorf("flow-organ-train-probe: -model is required; -steps and -rows must be positive")
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return err
	}
	loadStart := time.Now()
	src, err := safetensors.OpenSource(roots.ResolveModelPath(modelDir))
	if err != nil {
		return err
	}
	defer src.Close()
	weights, err := latentimage.LoadFinalLayerWeights(src, binding)
	if err != nil {
		return err
	}
	trainer, err := latentimage.NewFinalLayerTrainer(weights, 1e-6)
	if err != nil {
		return err
	}
	defer trainer.Close()
	fmt.Printf("trainable parameters=%d hidden=%d out=%d derived_lr=%.4g derived_momentum=%.4f rows=%d load_wall=%s\n",
		trainer.ParameterCount(), weights.Hidden, weights.Out,
		trainer.Config().BaseLearningRate, trainer.Config().Momentum, rows, time.Since(loadStart).Round(time.Millisecond))

	return trainHeadStimulus(trainer, weights.Hidden, weights.Out, steps, rows, maxWall)
}

// trainHeadStimulus: the shared bounded observed loop for final-layer heads.
func trainHeadStimulus(trainer *latentimage.FinalLayerTrainer, hiddenDim, outDim, steps, rows int, maxWall time.Duration) error {
	xF32, zF32, targetF32 := stimulus(rows, hiddenDim, outDim)
	hidden := make([]float64, len(xF32))
	for i, v := range xF32 {
		hidden[i] = float64(v)
	}
	temb := make([]float64, hiddenDim)
	for i := range temb {
		temb[i] = float64(zF32[i%len(zF32)]) * 0.5
	}
	target := make([]float64, len(targetF32))
	for i, v := range targetF32 {
		target[i] = float64(v)
	}

	start := time.Now()
	var first float64
	for step := 0; step < steps; step++ {
		stepStart := time.Now()
		result, err := trainer.Step(hidden, temb, target)
		if err != nil {
			return err
		}
		if math.IsNaN(result.Loss) || math.IsInf(result.Loss, 0) {
			return fmt.Errorf("flow-organ-train-probe: step %d loss is non-finite", step+1)
		}
		fmt.Printf("step %d/%d: loss=%.6f grad_l2=%.4g lr=%.4g wall=%s\n",
			step+1, steps, result.Loss, result.GradientL2, result.LearningRate, time.Since(stepStart).Round(time.Millisecond))
		if step == 0 {
			first = result.Loss
			if result.GradientL2 <= 0 {
				return fmt.Errorf("flow-organ-train-probe: first step carried no gradient")
			}
			if projected := time.Duration(steps+1) * time.Since(start); projected > maxWall {
				return fmt.Errorf("flow-organ-train-probe: first step projects the run to %s, past the %s bound", projected.Round(time.Second), maxWall)
			}
		}
	}
	after, err := trainer.Loss(hidden, temb, target)
	if err != nil {
		return err
	}
	if math.IsNaN(after) || math.IsInf(after, 0) || !(after < first) {
		return fmt.Errorf("flow-organ-train-probe: loss did not descend: before=%.6f after=%.6f", first, after)
	}
	fmt.Printf("descent: steps=%d loss %.6f -> %.6f total_wall=%s\n",
		steps, first, after, time.Since(start).Round(time.Millisecond))
	return nil
}

// runEditHead: the LiveEdit (Wan-family) output head trains from the .pt
// checkpoint through the same LayerNorm head trainer as Wan.
func runEditHead(modelPath string, steps, rows int, maxWall time.Duration) error {
	if modelPath == "" || steps <= 0 || rows <= 0 {
		return fmt.Errorf("flow-organ-train-probe: -model is required; -steps and -rows must be positive")
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return err
	}
	loadStart := time.Now()
	path := roots.ResolveModelPath(modelPath)
	catalog, err := pytorchzip.ReadCatalog(path)
	if err != nil {
		return err
	}
	normalize := func(name string) string {
		for _, prefix := range []string{"_fsdp_wrapped_module.", "module.", "model."} {
			name = strings.TrimPrefix(name, prefix)
		}
		return name
	}
	wanted := map[string]int{"head.modulation": 0, "head.head.weight": 1, "head.head.bias": 2}
	metas := make([]pytorchzip.TensorMeta, 3)
	found := 0
	for _, meta := range catalog.Tensors {
		if slot, ok := wanted[normalize(meta.Name)]; ok {
			metas[slot] = meta
			found++
		}
	}
	if found != 3 {
		return fmt.Errorf("flow-organ-train-probe: checkpoint has %d of 3 head tensors", found)
	}
	names := []string{metas[0].Name, metas[1].Name, metas[2].Name}
	bindings, err := pytorchzip.CompileBindings(catalog.Tensors, names)
	if err != nil {
		return err
	}
	reader, err := pytorchzip.Open(path)
	if err != nil {
		return err
	}
	defer reader.Close()
	values, err := reader.ReadBindingValues(bindings)
	if err != nil {
		return err
	}
	linearShape := metas[1].Shape
	if len(linearShape) != 2 {
		return fmt.Errorf("flow-organ-train-probe: head weight shape %v", linearShape)
	}
	weights := latentimage.FinalLayerWeights{
		Hidden: int(linearShape[1]), Out: int(linearShape[0]),
		Table: values[0], Linear: values[1], Bias: values[2],
	}
	trainer, err := latentimage.NewFinalLayerTrainer(weights, 1e-6)
	if err != nil {
		return err
	}
	defer trainer.Close()
	fmt.Printf("trainable parameters=%d hidden=%d out=%d derived_lr=%.4g derived_momentum=%.4f rows=%d load_wall=%s\n",
		trainer.ParameterCount(), weights.Hidden, weights.Out,
		trainer.Config().BaseLearningRate, trainer.Config().Momentum, rows, time.Since(loadStart).Round(time.Millisecond))
	return trainHeadStimulus(trainer, weights.Hidden, weights.Out, steps, rows, maxWall)
}

// runLatentBridge: the RxBrain-family projection pair trains its bridge
// reconstruction objective on a deterministic committed stimulus.
func runLatentBridge(modelDir string, steps, rows int, maxWall time.Duration) error {
	if modelDir == "" || steps <= 0 || rows <= 0 {
		return fmt.Errorf("flow-organ-train-probe: -model is required; -steps and -rows must be positive")
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return err
	}
	loadStart := time.Now()
	src, err := safetensors.OpenSource(roots.ResolveModelPath(modelDir))
	if err != nil {
		return err
	}
	defer src.Close()
	weights, err := routedlm.LoadLatentBridgeWeights(src, routedlm.RxBrainLatentBridgeBinding())
	if err != nil {
		return err
	}
	trainer, err := routedlm.NewLatentBridgeTrainer(weights)
	if err != nil {
		return err
	}
	defer trainer.Close()
	fmt.Printf("trainable parameters=%d hidden=%d latent=%d derived_lr=%.4g derived_momentum=%.4f rows=%d load_wall=%s\n",
		trainer.ParameterCount(), weights.Hidden, weights.Latent,
		trainer.Config().BaseLearningRate, trainer.Config().Momentum, rows, time.Since(loadStart).Round(time.Millisecond))

	hidden, _, _ := stimulus(rows, weights.Hidden, 1)
	start := time.Now()
	var first float64
	for step := 0; step < steps; step++ {
		stepStart := time.Now()
		result, err := trainer.Step(hidden)
		if err != nil {
			return err
		}
		if math.IsNaN(result.Loss) || math.IsInf(result.Loss, 0) {
			return fmt.Errorf("flow-organ-train-probe: step %d loss is non-finite", step+1)
		}
		fmt.Printf("step %d/%d: loss=%.6f grad_l2=%.4g lr=%.4g wall=%s\n",
			step+1, steps, result.Loss, result.GradientL2, result.LearningRate, time.Since(stepStart).Round(time.Millisecond))
		if step == 0 {
			first = result.Loss
			if result.GradientL2 <= 0 {
				return fmt.Errorf("flow-organ-train-probe: first step carried no gradient")
			}
			if projected := time.Duration(steps+1) * time.Since(start); projected > maxWall {
				return fmt.Errorf("flow-organ-train-probe: first step projects the run to %s, past the %s bound", projected.Round(time.Second), maxWall)
			}
		}
	}
	after, err := trainer.Loss(hidden)
	if err != nil {
		return err
	}
	if math.IsNaN(after) || math.IsInf(after, 0) || !(after < first) {
		return fmt.Errorf("flow-organ-train-probe: loss did not descend: before=%.6f after=%.6f", first, after)
	}
	fmt.Printf("descent: steps=%d loss %.6f -> %.6f total_wall=%s\n",
		steps, first, after, time.Since(start).Round(time.Millisecond))
	return nil
}

// stimulus: the committed deterministic probe pair — xorshift32 seeded rows,
// documented in the evidence as probe stimulus rather than pipeline data.
func stimulus(rows, hidden, flowDim int) (x, z, target []float32) {
	seed := uint32(88675123)
	next := func() float32 {
		seed ^= seed << 13
		seed ^= seed >> 17
		seed ^= seed << 5
		return float32(seed%2000)/1000 - 1
	}
	x = make([]float32, rows*hidden)
	for i := range x {
		x[i] = next() * 0.5
	}
	z = make([]float32, rows*flowDim)
	target = make([]float32, rows*flowDim)
	for i := range z {
		z[i] = next() * 0.2
		target[i] = next()
	}
	return x, z, target
}

func run(modelDir string, steps, rows int, timestep float64, maxWall time.Duration) error {
	if modelDir == "" || steps <= 0 || rows <= 0 {
		return fmt.Errorf("flow-organ-train-probe: -model is required; -steps and -rows must be positive")
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return err
	}
	directory := roots.ResolveModelPath(modelDir)
	loadStart := time.Now()
	binding := routedlm.SenseNovaBinding()
	flowBinding := routedlm.SenseNovaFlowBinding()
	cfg, err := routedlm.LoadConfig(directory, binding)
	if err != nil {
		return err
	}
	flowCfg, err := routedlm.LoadFlowConfig(directory)
	if err != nil {
		return err
	}
	src, err := safetensors.OpenSource(directory)
	if err != nil {
		return err
	}
	defer src.Close()
	plan, err := routedlm.CompileFlowPlan(src, cfg, flowCfg, flowBinding)
	if err != nil {
		return err
	}
	terminal, err := routedlm.LoadFlowTerminalWeights(src, plan, flowBinding)
	if err != nil {
		return err
	}
	trainer, err := routedlm.NewFlowHeadTrainer(plan, terminal.Head)
	if err != nil {
		return err
	}
	defer trainer.Close()
	fmt.Printf("trainable parameters=%d hidden=%d flow_dim=%d derived_lr=%.4g derived_momentum=%.4f rows=%d load_wall=%s\n",
		trainer.ParameterCount(), plan.Hidden, plan.FlowDim,
		trainer.Config().BaseLearningRate, trainer.Config().Momentum, rows, time.Since(loadStart).Round(time.Millisecond))

	x, z, target := stimulus(rows, plan.Hidden, plan.FlowDim)
	evaluate := func() (float64, error) {
		v, err := trainer.Velocity(x, z, timestep)
		if err != nil {
			return 0, err
		}
		var total float64
		invN := 1 / float64(len(target))
		for i := range v {
			d := float64(v[i]) - float64(target[i])
			total += d * d * invN
		}
		return total, nil
	}

	start := time.Now()
	var first float64
	for step := 0; step < steps; step++ {
		stepStart := time.Now()
		result, err := trainer.Step(x, z, target, timestep)
		if err != nil {
			return err
		}
		if math.IsNaN(result.Loss) || math.IsInf(result.Loss, 0) {
			return fmt.Errorf("flow-organ-train-probe: step %d loss is non-finite", step+1)
		}
		fmt.Printf("step %d/%d: loss=%.6f grad_l2=%.4g lr=%.4g wall=%s\n",
			step+1, steps, result.Loss, result.GradientL2, result.LearningRate, time.Since(stepStart).Round(time.Millisecond))
		if step == 0 {
			first = result.Loss
			if result.GradientL2 <= 0 {
				return fmt.Errorf("flow-organ-train-probe: first step carried no gradient")
			}
			if projected := time.Duration(steps+1) * time.Since(start); projected > maxWall {
				return fmt.Errorf("flow-organ-train-probe: first step projects the run to %s, past the %s bound", projected.Round(time.Second), maxWall)
			}
		}
	}
	after, err := evaluate()
	if err != nil {
		return err
	}
	if math.IsNaN(after) || math.IsInf(after, 0) || !(after < first) {
		return fmt.Errorf("flow-organ-train-probe: loss did not descend: before=%.6f after=%.6f", first, after)
	}
	fmt.Printf("descent: steps=%d loss %.6f -> %.6f total_wall=%s\n",
		steps, first, after, time.Since(start).Round(time.Millisecond))
	return nil
}
