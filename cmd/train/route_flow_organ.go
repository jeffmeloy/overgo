package main

import (
	"cmp"
	"context"
	"flag"
	"fmt"
	"maps"
	"math"
	"os"
	"slices"
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
	"overgo/internal/overgodb"
	"overgo/internal/pytorchzip"
	"overgo/internal/quant"
	"overgo/internal/routedlm"
	"overgo/internal/safetensors"
	"overgo/internal/tensor/dtype"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// flowOrganRoute names the flow-organ route in its bounded run's refusals.
const flowOrganRoute = "train -route flow-organ"

// trainFlowOrgan is the flow-organ route: bounded observed training of one
// trainable organ of a real artifact, chosen by -organ, under a supervised
// session that publishes the trained organ as a checkpoint.
func trainFlowOrgan(flags *flag.FlagSet, args []string) error {
	model := flags.String("model", "", "model directory (safetensors + config.json)")
	organ := flags.String("organ", "flow-head", "trainable organ: flow-head (SenseNova fm_head), latent-bridge (RxBrain llm2vae/vae2llm), denoiser-final (Krea-2 MMDiT output head), video-head (Wan2.1 head), edit-head, or ternary-master")
	steps := clioptions.IntOverride(flags, "steps", "required observed Muon steps")
	rows := clioptions.IntOverride(flags, "rows", "required stimulus rows")
	timestep := flags.Float64("timestep", 0.25, "flow timestep in [0,1)")
	tensorName := flags.String("tensor", "blk.0.ffn_gate.weight", "ternary-master: the quantized matrix to train")
	maxWall := clioptions.DurationOverride(flags, "max-wall", "optional projected-wall bound")
	storePath := flags.String("store", "overgodb-store", "OvergoDB for the session observation, recipe authority and checkpoint record")
	stimulusPath := flags.String("stimulus", "", "committed stimulus declaration grounding the recipe dataset")
	out := flags.String("out", "", "directory the trained organ's checkpoint publishes to")
	if err := flags.Parse(args); err != nil {
		return err
	}
	organs := map[string]func(organRun) error{
		"flow-head":      func(run organRun) error { return trainFlowHead(run, *timestep) },
		"latent-bridge":  runLatentBridge,
		"denoiser-final": func(run organRun) error { return runDenoiserFinal(run, latentimage.KreaFinalLayerBinding()) },
		"video-head":     func(run organRun) error { return runDenoiserFinal(run, latentimage.WanHeadBinding()) },
		"edit-head":      runEditHead,
		"ternary-master": func(run organRun) error { return runTernaryMaster(run, *tensorName) },
	}
	train, known := organs[*organ]
	if !known {
		return fmt.Errorf("%s: unknown organ %q (%s)", flags.Name(), *organ, strings.Join(slices.Sorted(maps.Keys(organs)), ", "))
	}
	if *model == "" || *stimulusPath == "" || *out == "" || *steps <= 0 || *rows <= 0 {
		return fmt.Errorf("%s: model, stimulus, and out are required; steps and rows must be positive", flags.Name())
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return err
	}
	path := roots.ResolveModelPath(*model)
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	// A directory identifies by its weights file; the edit head's .pt and
	// the ternary master's GGUF are their own identity.
	weights := path
	if info.IsDir() {
		if weights, err = trainingworkflow.IdentifyModelWeights(path); err != nil {
			return err
		}
	}
	ctx := context.Background()
	store, err := overgodb.Open(*storePath)
	if err != nil {
		return err
	}
	defer store.Close()
	session, err := openSession(ctx, store, weights, *stimulusPath)
	if err != nil {
		return err
	}
	return train(organRun{session: session, model: path, out: *out, steps: *steps, rows: *rows, maxWall: *maxWall})
}

// organRun is what every organ's bounded run shares: its supervised
// session, the resolved model path, the run's bounds, and where the trained
// organ's checkpoint publishes.
type organRun struct {
	session     routeSession
	model, out  string
	steps, rows int
	maxWall     time.Duration
}

// train takes the organ through the shared bounded run under plan and
// publishes what trainer trained; every update trains on the one declared
// stimulus.
func (run organRun) train(plan trainingprogram.TrainingSessionPlan, trainer planned, step func(int) (optimizer.ObservedStepResult, error), loss func() (float64, error)) error {
	return boundedRun{name: flowOrganRoute, plan: plan, session: run.session, step: step, loss: loss,
		publish: func(ctx context.Context) error {
			return run.session.publishPlanned(ctx, run.out, trainer, plan.Updates())
		},
	}.train(context.Background())
}

// probeSession compiles a route's bounded run under the builtin optimizer
// policy; a zero maxWall leaves the projected wall unbounded.
func probeSession(objective trainingprogram.ObjectiveKind, steps, parameters int, maxWall time.Duration) (trainingprogram.TrainingSessionPlan, error) {
	return trainingprogram.CompileProbeSessionPlan(trainingprogram.ProbeSpec{
		Objective: objective, Updates: steps, MaxProjectedWall: maxWall,
		Parameters: parameters, Optimizer: trainingprogram.BuiltinOptimizerPolicy(),
	})
}

// runDenoiserFinal: the Krea-2-family MMDiT output head trains the velocity
// MSE objective on a deterministic committed stimulus.
func runDenoiserFinal(run organRun, binding latentimage.FinalLayerBinding) error {
	loadStart := time.Now()
	src, err := safetensors.OpenSource(run.model)
	if err != nil {
		return err
	}
	defer src.Close()
	weights, err := latentimage.LoadFinalLayerWeights(src, binding)
	if err != nil {
		return err
	}
	return trainHeadStimulus(run, weights, loadStart)
}

// trainHeadStimulus: the shared bounded observed run for final-layer heads.
func trainHeadStimulus(run organRun, weights latentimage.FinalLayerWeights, loadStart time.Time) error {
	trainer, err := latentimage.NewFinalLayerTrainer(weights, 1e-6, trainingprogram.BuiltinOptimizerPolicy())
	if err != nil {
		return err
	}
	defer trainer.Close()
	session, err := probeSession(trainingprogram.ObjectiveFlowMatching, run.steps, trainer.ParameterCount(), run.maxWall)
	if err != nil {
		return err
	}
	fmt.Printf("trainable parameters=%d hidden=%d out=%d derived_lr=%.4g derived_momentum=%.4f rows=%d load_wall=%s\n",
		trainer.ParameterCount(), weights.Hidden, weights.Out,
		trainer.Config().BaseLearningRate, trainer.Config().Momentum, run.rows, time.Since(loadStart).Round(time.Millisecond))
	xF32, zF32, targetF32, err := stimulus(session, run.rows, weights.Hidden, weights.Out)
	if err != nil {
		return err
	}
	hidden := make([]float64, len(xF32))
	for i, v := range xF32 {
		hidden[i] = float64(v)
	}
	temb := make([]float64, weights.Hidden)
	for i := range temb {
		temb[i] = float64(zF32[i%len(zF32)]) * 0.5
	}
	target := make([]float64, len(targetF32))
	for i, v := range targetF32 {
		target[i] = float64(v)
	}
	return run.train(session, trainer,
		func(int) (optimizer.ObservedStepResult, error) { return trainer.Step(hidden, temb, target) },
		func() (float64, error) { return trainer.Loss(hidden, temb, target) })
}

// runEditHead: the LiveEdit (Wan-family) output head trains from the .pt
// checkpoint through the same LayerNorm head trainer as Wan.
func runEditHead(run organRun) error {
	loadStart := time.Now()
	path := run.model
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
		return fmt.Errorf("train -route flow-organ:checkpoint has %d of %d head tensors", found, len(wanted))
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
		return fmt.Errorf("train -route flow-organ:head weight shape %v", linearShape)
	}
	weights := latentimage.FinalLayerWeights{
		Hidden: int(linearShape[1]), Out: int(linearShape[0]),
		Table: values[0], Linear: values[1], Bias: values[2],
	}
	return trainHeadStimulus(run, weights, loadStart)
}

// runTernaryMaster: the master-weight lane for ternary serving artifacts.
// One quantized matrix dequantizes into f32 masters, the masters train a
// linear MSE objective on a deterministic committed stimulus, the trained
// masters publish as the checkpoint, and they REQUANTIZE back to the
// serving dtype -- descent is required through the ternary round trip,
// proving the lane moves the serving artifact's behavior rather than a
// float shadow.
func runTernaryMaster(run organRun, tensorName string) error {
	rows := run.rows
	loadStart := time.Now()
	file, err := gguf.Open(run.model)
	if err != nil {
		return err
	}
	defer file.Close()
	info, ok := file.Tensor(tensorName)
	if !ok {
		return fmt.Errorf("train -route flow-organ:tensor %q not in artifact", tensorName)
	}
	if info.Dimensions != 2 {
		return fmt.Errorf("train -route flow-organ:tensor %q has %d dimensions, need a matrix", tensorName, info.Dimensions)
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
		for r := range out {
			scales[r] = math.Float32frombits(binary.LittleEndian.Uint32(scaleBytes[r*4:]))
		}
		masters = make([]float32, elements)
		for r := range out {
			for i := range in {
				masters[r*in+i] = dtype.F8E4M3ToFloat32(payload[r*in+i]) * scales[r]
			}
		}
		requantize = func(values []float32) ([]byte, error) {
			packed := make([]byte, int(elements)+out*4)
			for r := range out {
				var peak float64
				for i := range in {
					if a := math.Abs(float64(values[r*in+i])); a > peak {
						peak = a
					}
				}
				scale := float32(peak / 448)
				scale = cmp.Or(scale, 1)
				for i := range in {
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
			return fmt.Errorf("train -route flow-organ:%q is not requantizable to %s: %w", tensorName, servingType, err)
		}
	}

	plan, err := optimizer.CompilePlan(len(masters), []optimizer.GroupSpec{{
		Name: tensorName, Start: 0, End: len(masters), Rows: out, Cols: in,
	}})
	if err != nil {
		return err
	}
	gradients := make([]float32, len(masters))
	session, err := probeSession(trainingprogram.ObjectiveFlowMatching, run.steps, len(masters), run.maxWall)
	if err != nil {
		return err
	}
	config := session.Optimizer()
	stepper, err := optimizer.NewStepper(masters, gradients, plan, config)
	if err != nil {
		return err
	}
	defer stepper.Close()
	trained := ternaryMasters{masters: masters, plan: plan, stepper: stepper}
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
			for r := range out {
				scale := math.Float32frombits(binary.LittleEndian.Uint32(scaleBytes[r*4:]))
				for i := range in {
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
	var updates int
	if err := run.train(session, trained,
		func(int) (optimizer.ObservedStepResult, error) {
			loss := evaluate(masters)
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
			return optimizer.Advance(stepper, config, &updates, loss, math.Sqrt(gradientSquared))
		},
		func() (float64, error) { return evaluate(masters), nil }); err != nil {
		return err
	}
	servingAfter, err := servingLoss()
	if err != nil {
		return err
	}
	if !checked.Finite64(servingAfter) || !(servingAfter < servingBefore) {
		return fmt.Errorf("train -route flow-organ:serving round-trip loss did not descend: %.6f -> %.6f", servingBefore, servingAfter)
	}
	fmt.Printf("serving round-trip descent: %.6f -> %.6f\n", servingBefore, servingAfter)
	return nil
}

// ternaryMasters is the f32 master matrix a ternary-master run trains over
// its one-group plan, the weights its checkpoint publishes.
type ternaryMasters struct {
	masters []float32
	plan    optimizer.Plan
	stepper optimizer.Stepper
}

// OptimizerPlan is the one-group Muon plan over the masters.
func (m ternaryMasters) OptimizerPlan() optimizer.Plan { return m.plan }

// Snapshot copies the optimizer state the next step continues from.
func (m ternaryMasters) Snapshot() optimizer.State { return m.stepper.Snapshot() }

// SaveWeights writes the trained masters at path, the weights Snapshot pairs
// with.
func (m ternaryMasters) SaveWeights(path string) error {
	return optimizer.SavePlanWeights(path, m.plan, m.masters)
}

// runLatentBridge: the RxBrain-family projection pair trains its bridge
// reconstruction objective on a deterministic committed stimulus.
func runLatentBridge(run organRun) error {
	loadStart := time.Now()
	src, err := safetensors.OpenSource(run.model)
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
	session, err := probeSession(trainingprogram.ObjectiveLatentL2, run.steps, trainer.ParameterCount(), run.maxWall)
	if err != nil {
		return err
	}
	fmt.Printf("trainable parameters=%d hidden=%d latent=%d derived_lr=%.4g derived_momentum=%.4f rows=%d load_wall=%s\n",
		trainer.ParameterCount(), weights.Hidden, weights.Latent,
		trainer.Config().BaseLearningRate, trainer.Config().Momentum, run.rows, time.Since(loadStart).Round(time.Millisecond))

	hidden, _, _, err := stimulus(session, run.rows, weights.Hidden, 1)
	if err != nil {
		return err
	}
	return run.train(session, trainer,
		func(int) (optimizer.ObservedStepResult, error) { return trainer.Step(hidden) },
		func() (float64, error) { return trainer.Loss(hidden) })
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

// trainFlowHead is bounded observed training of a routed-LM flow-head organ
// (the SenseNova-family fm_head MLP), the production driver of the flow-head
// trainer and the reproducible generator of its verification evidence: it
// compiles the flow plan from the real artifact's own declarations, packs
// the fm_head organ as f32 masters decoded from the BF16 serving storage,
// trains on a deterministic committed stimulus with derived hyperparameters
// (base LR n_params^-1/2, momentum from the CLT effective-samples rule),
// observes every step with a first-step projection guard, proves descent by
// re-evaluating the same pair's loss after the observed steps, and publishes
// the trained organ. Full-pipeline MoT training remains the promotion gate
// beyond the organ.
func trainFlowHead(run organRun, timestep float64) error {
	directory := run.model
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
	session, err := probeSession(trainingprogram.ObjectiveFlowMatching, run.steps, trainer.ParameterCount(), run.maxWall)
	if err != nil {
		return err
	}
	fmt.Printf("trainable parameters=%d hidden=%d flow_dim=%d derived_lr=%.4g derived_momentum=%.4f rows=%d load_wall=%s\n",
		trainer.ParameterCount(), plan.Hidden, plan.FlowDim,
		trainer.Config().BaseLearningRate, trainer.Config().Momentum, run.rows, time.Since(loadStart).Round(time.Millisecond))

	x, z, target, err := stimulus(session, run.rows, plan.Hidden, plan.FlowDim)
	if err != nil {
		return err
	}
	return run.train(session, trainer,
		func(int) (optimizer.ObservedStepResult, error) { return trainer.Step(x, z, target, timestep) },
		func() (float64, error) {
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
		})
}
