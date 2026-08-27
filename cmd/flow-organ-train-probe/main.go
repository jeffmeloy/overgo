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

	"encoding/binary"
	"overgo/internal/checked"
	"overgo/internal/clioptions"
	"overgo/internal/dataroot"

	"overgo/internal/gguf"
	"overgo/internal/hostmath"
	"overgo/internal/latentimage"
	"overgo/internal/optimizer"
	"overgo/internal/pytorchzip"
	"overgo/internal/quant"
	"overgo/internal/routedlm"
	"overgo/internal/safetensors"
	"overgo/internal/tensor/dtype"
	"overgo/internal/trainingprogram"
)

func main() {
	model := flag.String("model", "", "model directory (safetensors + config.json)")
	organ := flag.String("organ", "flow-head", "trainable organ: flow-head (SenseNova fm_head), latent-bridge (RxBrain llm2vae/vae2llm), denoiser-final (Krea-2 MMDiT output head), or video-head (Wan2.1 head)")
	steps := clioptions.IntOverride(flag.CommandLine, "steps", "required observed Muon steps")
	rows := clioptions.IntOverride(flag.CommandLine, "rows", "required stimulus rows")
	timestep := flag.Float64("timestep", 0.25, "flow timestep in [0,1)")
	tensorName := flag.String("tensor", "blk.0.ffn_gate.weight", "ternary-master: the quantized matrix to train")
	maxWall := clioptions.DurationOverride(flag.CommandLine, "max-wall", "optional projected-wall bound")
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
	case "ternary-master":
		err = runTernaryMaster(*model, *tensorName, *steps, *rows, *maxWall)
	default:
		err = fmt.Errorf("flow-organ-train-probe: unknown organ %q (flow-head, latent-bridge)", *organ)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func probeSession(objective trainingprogram.ObjectiveKind, steps, parameters int, maxWall time.Duration) (trainingprogram.TrainingSessionPlan, error) {
	return trainingprogram.CompileProbeSessionPlan(trainingprogram.ProbeSpec{
		Objective: objective, Updates: steps, MaxProjectedWall: maxWall,
		Parameters: parameters, Optimizer: trainingprogram.BuiltinOptimizerPolicy(),
	})
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
	trainer, err := latentimage.NewFinalLayerTrainer(weights, 1e-6, trainingprogram.BuiltinOptimizerPolicy())
	if err != nil {
		return err
	}
	defer trainer.Close()
	session, err := probeSession(trainingprogram.ObjectiveFlowMatching, steps, trainer.ParameterCount(), maxWall)
	if err != nil {
		return err
	}
	fmt.Printf("trainable parameters=%d hidden=%d out=%d derived_lr=%.4g derived_momentum=%.4f rows=%d load_wall=%s\n",
		trainer.ParameterCount(), weights.Hidden, weights.Out,
		trainer.Config().BaseLearningRate, trainer.Config().Momentum, rows, time.Since(loadStart).Round(time.Millisecond))

	return trainHeadStimulus(trainer, weights.Hidden, weights.Out, session, rows)
}

// trainHeadStimulus: the shared bounded observed loop for final-layer heads.
func trainHeadStimulus(trainer *latentimage.FinalLayerTrainer, hiddenDim, outDim int, session trainingprogram.TrainingSessionPlan, rows int) error {
	xF32, zF32, targetF32, err := stimulus(session, rows, hiddenDim, outDim)
	if err != nil {
		return err
	}
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
	for step := 0; step < session.Updates(); step++ {
		stepStart := time.Now()
		result, err := trainer.Step(hidden, temb, target)
		if err != nil {
			return err
		}
		if !checked.Finite64(result.Loss) {
			return fmt.Errorf("flow-organ-train-probe: step %d loss is non-finite", step+1)
		}
		fmt.Printf("step %d/%d: loss=%.6f grad_l2=%.4g lr=%.4g wall=%s\n",
			step+1, session.Updates(), result.Loss, result.GradientL2, result.LearningRate, time.Since(stepStart).Round(time.Millisecond))
		if step == 0 {
			first = result.Loss
			if result.GradientL2 <= 0 {
				return fmt.Errorf("flow-organ-train-probe: first step carried no gradient")
			}
			if err := session.AdmitStepWall(time.Since(start)); err != nil {
				return fmt.Errorf("flow-organ-train-probe: %w", err)
			}
		}
	}
	after, err := trainer.Loss(hidden, temb, target)
	if err != nil {
		return err
	}
	if !checked.Finite64(after) || !(after < first) {
		return fmt.Errorf("flow-organ-train-probe: loss did not descend: before=%.6f after=%.6f", first, after)
	}
	fmt.Printf("descent: steps=%d loss %.6f -> %.6f total_wall=%s\n",
		session.Updates(), first, after, time.Since(start).Round(time.Millisecond))
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
	metas := make([]pytorchzip.TensorMeta, len(wanted))
	found := 0
	for _, meta := range catalog.Tensors {
		if slot, ok := wanted[normalize(meta.Name)]; ok {
			metas[slot] = meta
			found++
		}
	}
	if found != len(wanted) {
		return fmt.Errorf("flow-organ-train-probe: checkpoint has %d of %d head tensors", found, len(wanted))
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
	trainer, err := latentimage.NewFinalLayerTrainer(weights, 1e-6, trainingprogram.BuiltinOptimizerPolicy())
	if err != nil {
		return err
	}
	defer trainer.Close()
	session, err := probeSession(trainingprogram.ObjectiveFlowMatching, steps, trainer.ParameterCount(), maxWall)
	if err != nil {
		return err
	}
	fmt.Printf("trainable parameters=%d hidden=%d out=%d derived_lr=%.4g derived_momentum=%.4f rows=%d load_wall=%s\n",
		trainer.ParameterCount(), weights.Hidden, weights.Out,
		trainer.Config().BaseLearningRate, trainer.Config().Momentum, rows, time.Since(loadStart).Round(time.Millisecond))
	return trainHeadStimulus(trainer, weights.Hidden, weights.Out, session, rows)
}

// runTernaryMaster: the master-weight lane for ternary serving artifacts.
// One quantized matrix dequantizes into f32 masters, the masters train a
// linear MSE objective on a deterministic committed stimulus, and the
// trained masters REQUANTIZE back to the serving dtype -- descent is
// required through the ternary round trip, proving the lane moves the
// serving artifact's behavior rather than a float shadow.
func runTernaryMaster(modelPath, tensorName string, steps, rows int, maxWall time.Duration) error {
	if modelPath == "" || tensorName == "" || steps <= 0 || rows <= 0 {
		return fmt.Errorf("flow-organ-train-probe: -model and -tensor are required; -steps and -rows must be positive")
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return err
	}
	loadStart := time.Now()
	file, err := gguf.Open(roots.ResolveModelPath(modelPath))
	if err != nil {
		return err
	}
	defer file.Close()
	info, ok := file.Tensor(tensorName)
	if !ok {
		return fmt.Errorf("flow-organ-train-probe: tensor %q not in artifact", tensorName)
	}
	if info.Dimensions != 2 {
		return fmt.Errorf("flow-organ-train-probe: tensor %q has %d dimensions, need a matrix", tensorName, info.Dimensions)
	}
	in, out := int(info.Shape[0]), int(info.Shape[1])
	elements := uint64(in) * uint64(out)
	raw := make([]byte, info.Size)
	if err := file.ReadTensorData(info, raw); err != nil {
		return err
	}
	// Scaled fp8 (the gemma fp8-native layout): [rows*in e4m3 | rows F32
	// scales] decoded row-wise as e4m3 * scale; requantization recomputes
	// per-row scales as max|row|/448 and re-encodes round-to-nearest-even.
	scaledFP8 := uint64(len(raw)) == elements+uint64(out)*4
	var masters []float32
	var requantize func([]float32) ([]byte, error)
	if scaledFP8 {
		payload := raw[:elements]
		scaleBytes := raw[elements:]
		scales := make([]float32, out)
		for r := 0; r < out; r++ {
			scales[r] = math.Float32frombits(binary.LittleEndian.Uint32(scaleBytes[r*4:]))
		}
		masters = make([]float32, elements)
		for r := 0; r < out; r++ {
			for i := 0; i < in; i++ {
				masters[r*in+i] = dtype.F8E4M3ToFloat32(payload[r*in+i]) * scales[r]
			}
		}
		requantize = func(values []float32) ([]byte, error) {
			packed := make([]byte, int(elements)+out*4)
			for r := 0; r < out; r++ {
				var peak float64
				for i := 0; i < in; i++ {
					if a := math.Abs(float64(values[r*in+i])); a > peak {
						peak = a
					}
				}
				scale := float32(peak / 448)
				if scale == 0 {
					scale = 1
				}
				for i := 0; i < in; i++ {
					packed[r*in+i] = dtype.Float32ToF8E4M3(values[r*in+i] / scale)
				}
				binary.LittleEndian.PutUint32(packed[int(elements)+r*4:], math.Float32bits(scale))
			}
			return packed, nil
		}
	} else {
		masters, err = quant.Dequantize(info.Type, raw, elements)
		if err != nil {
			return err
		}
		servingType := info.Type
		requantize = func(values []float32) ([]byte, error) {
			return quant.Quantize(servingType, values)
		}
		if _, err := requantize(masters); err != nil {
			return fmt.Errorf("flow-organ-train-probe: %q is not requantizable to %s: %w", tensorName, servingType, err)
		}
	}

	plan, err := optimizer.CompilePlan(len(masters), []optimizer.GroupSpec{{
		Name: tensorName, Start: 0, End: len(masters), Rows: out, Cols: in,
	}})
	if err != nil {
		return err
	}
	gradients := make([]float32, len(masters))
	session, err := probeSession(trainingprogram.ObjectiveFlowMatching, steps, len(masters), maxWall)
	if err != nil {
		return err
	}
	config := session.Optimizer()
	stepper, err := optimizer.NewStepper(masters, gradients, plan, config)
	if err != nil {
		return err
	}
	defer stepper.Close()
	fmt.Printf("trainable parameters=%d matrix=%dx%d dtype=%v derived_lr=%.4g derived_momentum=%.4f rows=%d load_wall=%s"+"\n",
		len(masters), out, in, info.Type, config.BaseLearningRate, config.Momentum, rows, time.Since(loadStart).Round(time.Millisecond))

	xF32, _, targetF32, err := stimulus(session, rows, in, out)
	if err != nil {
		return err
	}
	target := targetF32
	evaluate := func(weights []float32) float64 {
		y := make([]float32, rows*out)
		hostmath.Linear(y, xF32, weights, rows, in, out)
		invN := 1 / float64(len(target))
		var loss float64
		for i := range y {
			d := float64(y[i]) - float64(target[i])
			loss += d * d * invN
		}
		return loss
	}
	servingLoss := func() (float64, error) {
		packed, err := requantize(masters)
		if err != nil {
			return 0, err
		}
		var roundTrip []float32
		if scaledFP8 {
			roundTrip = make([]float32, elements)
			scaleBytes := packed[elements:]
			for r := 0; r < out; r++ {
				scale := math.Float32frombits(binary.LittleEndian.Uint32(scaleBytes[r*4:]))
				for i := 0; i < in; i++ {
					roundTrip[r*in+i] = dtype.F8E4M3ToFloat32(packed[r*in+i]) * scale
				}
			}
		} else {
			roundTrip, err = quant.Dequantize(info.Type, packed, elements)
			if err != nil {
				return 0, err
			}
		}
		return evaluate(roundTrip), nil
	}

	servingBefore, err := servingLoss()
	if err != nil {
		return err
	}
	start := time.Now()
	var first float64
	for step := 0; step < session.Updates(); step++ {
		stepStart := time.Now()
		loss := evaluate(masters)
		if !checked.Finite64(loss) {
			return fmt.Errorf("flow-organ-train-probe: step %d loss is non-finite", step+1)
		}
		y := make([]float32, rows*out)
		hostmath.Linear(y, xF32, masters, rows, in, out)
		dY := make([]float32, len(y))
		invN := 1 / float64(len(target))
		for i := range y {
			dY[i] = float32(2 * (float64(y[i]) - float64(target[i])) * invN)
		}
		clear(gradients)
		dX := make([]float32, rows*in)
		hostmath.LinearBackward(dX, gradients, nil, xF32, masters, dY, rows, in, out, false)
		var gradientSquared float64
		for _, g := range gradients {
			gradientSquared += float64(g) * float64(g)
		}
		if err := stepper.Step(); err != nil {
			return err
		}
		fmt.Printf("step %d/%d: master_loss=%.6f grad_l2=%.4g lr=%.4g wall=%s"+"\n",
			step+1, session.Updates(), loss, math.Sqrt(gradientSquared), config.BaseLearningRate, time.Since(stepStart).Round(time.Millisecond))
		if step == 0 {
			first = loss
			if gradientSquared == 0 {
				return fmt.Errorf("flow-organ-train-probe: first step carried no gradient")
			}
			if err := session.AdmitStepWall(time.Since(start)); err != nil {
				return fmt.Errorf("flow-organ-train-probe: %w", err)
			}
		}
	}
	masterAfter := evaluate(masters)
	servingAfter, err := servingLoss()
	if err != nil {
		return err
	}
	if !(masterAfter < first) {
		return fmt.Errorf("flow-organ-train-probe: master loss did not descend: %.6f -> %.6f", first, masterAfter)
	}
	if !checked.Finite64(servingAfter) || !(servingAfter < servingBefore) {
		return fmt.Errorf("flow-organ-train-probe: serving round-trip loss did not descend: %.6f -> %.6f", servingBefore, servingAfter)
	}
	fmt.Printf("descent: steps=%d master %.6f -> %.6f; serving round-trip %.6f -> %.6f total_wall=%s"+"\n",
		session.Updates(), first, masterAfter, servingBefore, servingAfter, time.Since(start).Round(time.Millisecond))
	return nil
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
	trainer, err := routedlm.NewLatentBridgeTrainer(weights, trainingprogram.BuiltinOptimizerPolicy())
	if err != nil {
		return err
	}
	defer trainer.Close()
	session, err := probeSession(trainingprogram.ObjectiveLatentL2, steps, trainer.ParameterCount(), maxWall)
	if err != nil {
		return err
	}
	fmt.Printf("trainable parameters=%d hidden=%d latent=%d derived_lr=%.4g derived_momentum=%.4f rows=%d load_wall=%s\n",
		trainer.ParameterCount(), weights.Hidden, weights.Latent,
		trainer.Config().BaseLearningRate, trainer.Config().Momentum, rows, time.Since(loadStart).Round(time.Millisecond))

	hidden, _, _, err := stimulus(session, rows, weights.Hidden, 1)
	if err != nil {
		return err
	}
	start := time.Now()
	var first float64
	for step := 0; step < session.Updates(); step++ {
		stepStart := time.Now()
		result, err := trainer.Step(hidden)
		if err != nil {
			return err
		}
		if !checked.Finite64(result.Loss) {
			return fmt.Errorf("flow-organ-train-probe: step %d loss is non-finite", step+1)
		}
		fmt.Printf("step %d/%d: loss=%.6f grad_l2=%.4g lr=%.4g wall=%s\n",
			step+1, session.Updates(), result.Loss, result.GradientL2, result.LearningRate, time.Since(stepStart).Round(time.Millisecond))
		if step == 0 {
			first = result.Loss
			if result.GradientL2 <= 0 {
				return fmt.Errorf("flow-organ-train-probe: first step carried no gradient")
			}
			if err := session.AdmitStepWall(time.Since(start)); err != nil {
				return fmt.Errorf("flow-organ-train-probe: %w", err)
			}
		}
	}
	after, err := trainer.Loss(hidden)
	if err != nil {
		return err
	}
	if !checked.Finite64(after) || !(after < first) {
		return fmt.Errorf("flow-organ-train-probe: loss did not descend: before=%.6f after=%.6f", first, after)
	}
	fmt.Printf("descent: steps=%d loss %.6f -> %.6f total_wall=%s\n",
		session.Updates(), first, after, time.Since(start).Round(time.Millisecond))
	return nil
}

// stimulus: the committed deterministic probe pair — xorshift32 seeded rows,
// documented in the evidence as probe stimulus rather than pipeline data.
func stimulus(session trainingprogram.TrainingSessionPlan, rows, hidden, flowDim int) (x, z, target []float32, err error) {
	derived, err := session.Seed("stimulus")
	if err != nil {
		return nil, nil, nil, err
	}
	seed := uint32(derived)
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
	return x, z, target, nil
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
	flowProfile, err := routedlm.InspectFlowProfile(directory)
	if err != nil {
		return err
	}
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
	plan, err := routedlm.CompileFlowPlan(src, cfg, flowCfg, flowProfile)
	if err != nil {
		return err
	}
	terminal, err := routedlm.LoadFlowTerminalWeights(src, plan, flowProfile)
	if err != nil {
		return err
	}
	trainer, err := routedlm.NewFlowHeadTrainer(plan, terminal.Head, trainingprogram.BuiltinOptimizerPolicy())
	if err != nil {
		return err
	}
	defer trainer.Close()
	session, err := probeSession(trainingprogram.ObjectiveFlowMatching, steps, trainer.ParameterCount(), maxWall)
	if err != nil {
		return err
	}
	fmt.Printf("trainable parameters=%d hidden=%d flow_dim=%d derived_lr=%.4g derived_momentum=%.4f rows=%d load_wall=%s\n",
		trainer.ParameterCount(), plan.Hidden, plan.FlowDim,
		trainer.Config().BaseLearningRate, trainer.Config().Momentum, rows, time.Since(loadStart).Round(time.Millisecond))

	x, z, target, err := stimulus(session, rows, plan.Hidden, plan.FlowDim)
	if err != nil {
		return err
	}
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
	for step := 0; step < session.Updates(); step++ {
		stepStart := time.Now()
		result, err := trainer.Step(x, z, target, timestep)
		if err != nil {
			return err
		}
		if !checked.Finite64(result.Loss) {
			return fmt.Errorf("flow-organ-train-probe: step %d loss is non-finite", step+1)
		}
		fmt.Printf("step %d/%d: loss=%.6f grad_l2=%.4g lr=%.4g wall=%s\n",
			step+1, session.Updates(), result.Loss, result.GradientL2, result.LearningRate, time.Since(stepStart).Round(time.Millisecond))
		if step == 0 {
			first = result.Loss
			if result.GradientL2 <= 0 {
				return fmt.Errorf("flow-organ-train-probe: first step carried no gradient")
			}
			if err := session.AdmitStepWall(time.Since(start)); err != nil {
				return fmt.Errorf("flow-organ-train-probe: %w", err)
			}
		}
	}
	after, err := evaluate()
	if err != nil {
		return err
	}
	if !checked.Finite64(after) || !(after < first) {
		return fmt.Errorf("flow-organ-train-probe: loss did not descend: before=%.6f after=%.6f", first, after)
	}
	fmt.Printf("descent: steps=%d loss %.6f -> %.6f total_wall=%s\n",
		session.Updates(), first, after, time.Since(start).Round(time.Millisecond))
	return nil
}
