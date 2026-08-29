package diffusionimage

import (
	"fmt"
	"strings"

	"overgo/internal/optimizer"
	"overgo/internal/trainingprogram"
)

// OTLinearFlowPathInto: linear optimal-transport flow state and velocity
// target (Lipman 2022 as executed by the artifact's train.py):
// x_t = (1-(1-sigma_min)t)*x0 + t*x1, target = x1 - (1-sigma_min)*x0.
func OTLinearFlowPathInto(xT, target, x1, x0, t []float32, batch, elemsPerSample int, sigmaMin float64) error {
	if batch <= 0 || elemsPerSample <= 0 {
		return fmt.Errorf("ot flow: shape batch=%d elems=%d", batch, elemsPerSample)
	}
	n := batch * elemsPerSample
	if len(xT) != n || len(target) != n || len(x1) != n || len(x0) != n {
		return fmt.Errorf("ot flow: lens xT=%d target=%d x1=%d x0=%d want=%d", len(xT), len(target), len(x1), len(x0), n)
	}
	if len(t) != batch {
		return fmt.Errorf("ot flow: len t=%d want=%d", len(t), batch)
	}
	oneMinusSigmaMin := 1 - sigmaMin
	for bi := range batch {
		tv := float64(t[bi])
		sigmaT := 1 - oneMinusSigmaMin*tv
		base := bi * elemsPerSample
		for i := range elemsPerSample {
			j := base + i
			x0v, x1v := float64(x0[j]), float64(x1[j])
			xT[j] = float32(sigmaT*x0v + tv*x1v)
			target[j] = float32(x1v - oneMinusSigmaMin*x0v)
		}
	}
	return nil
}

// ScaledMSELossGradInto: loss = scale*mean((pred-target)^2), dst = dLoss/dpred.
func ScaledMSELossGradInto(dst, pred, target []float32, scale float64) (float64, error) {
	if len(pred) == 0 {
		return 0, fmt.Errorf("scaled mse: empty prediction")
	}
	if len(dst) != len(pred) || len(target) != len(pred) {
		return 0, fmt.Errorf("scaled mse: len dst=%d pred=%d target=%d", len(dst), len(pred), len(target))
	}
	invN := 1 / float64(len(pred))
	gradScale := 2 * scale * invN
	var loss float64
	for i, p := range pred {
		diff := float64(p) - float64(target[i])
		loss += diff * diff
		dst[i] = float32(gradScale * diff)
	}
	return scale * loss * invN, nil
}

// Trainer: compiled Muon state over one image model.
type Trainer struct {
	model     *Model
	pack      *optimizer.TensorPack
	update    optimizer.Stepper
	program   trainingprogram.TrainingProgram
	execution trainingprogram.Execution[trainingState]
}

type trainingState struct {
	x, target     []float32
	batch         int
	height, width int
	prediction    []float32
	trace         forwardTrace
	gradients     Grads
	loss          float64
}

func NewTrainer(model *Model, config optimizer.Config) (*Trainer, error) {
	if model == nil {
		return nil, fmt.Errorf("diffusionimage train: model is required")
	}
	pack, err := optimizer.NewTensorPack(model.parameters(), model.muonGeometry())
	if err != nil {
		return nil, err
	}
	update, err := pack.NewStepper(config)
	if err != nil {
		return nil, err
	}
	plan := pack.Plan()
	program, err := trainingprogram.CompileObjectiveProgram(trainingprogram.ObjectiveFlowMatching, nil, plan)
	if err != nil {
		_ = update.Close()
		return nil, err
	}
	trainer := &Trainer{model: model, pack: pack, update: update, program: program}
	execution, err := trainingprogram.BindObjective(program, trainer.forward, trainer.backward, trainer.optimize)
	if err != nil {
		_ = update.Close()
		return nil, err
	}
	trainer.execution = execution
	return trainer, nil
}

func (trainer *Trainer) Program() trainingprogram.TrainingProgram { return trainer.program }

func (trainer *Trainer) Close() error {
	if trainer == nil || trainer.update == nil {
		return nil
	}
	return trainer.update.Close()
}

// Step: forward, OT loss/backward, Muon update.
func (trainer *Trainer) Step(x, target []float32, b, height, width int) (float64, error) {
	if trainer == nil || trainer.model == nil || trainer.pack == nil || trainer.update == nil {
		return 0, fmt.Errorf("diffusionimage train: trainer is unavailable")
	}
	state := trainingState{x: x, target: target, batch: b, height: height, width: width}
	if err := trainer.execution.Run(&state); err != nil {
		return 0, err
	}
	return state.loss, nil
}

func (trainer *Trainer) forward(state *trainingState) error {
	m := trainer.model
	tile := m.Cfg.PatchSize << uint(m.Cfg.NumLevels-1)
	if state.batch <= 0 || state.height <= 0 || state.width <= 0 || state.height%tile != 0 || state.width%tile != 0 {
		return fmt.Errorf("diffusionimage train: batch=%d size=%dx%d must be positive tile %d multiples", state.batch, state.width, state.height, tile)
	}
	prediction, trace, err := m.forward(state.x, state.batch, state.height, state.width, true)
	if err != nil {
		return err
	}
	state.prediction, state.trace = prediction, trace
	return nil
}

func (trainer *Trainer) backward(state *trainingState) error {
	dPrediction := make([]float32, len(state.prediction))
	loss, err := ScaledMSELossGradInto(dPrediction, state.prediction, state.target, 1)
	if err != nil {
		return err
	}
	state.loss = loss
	state.gradients = Grads{}
	if _, err := trainer.model.backwardFromTrace(state.trace, dPrediction, state.gradients); err != nil {
		return err
	}
	return nil
}

func (trainer *Trainer) optimize(state *trainingState) error {
	if err := trainer.pack.GatherGradients(state.gradients); err != nil {
		return err
	}
	if err := trainer.update.Step(); err != nil {
		return err
	}
	trainer.pack.Scatter()
	trainer.model.refreshScalars()
	return nil
}

func (m *Model) muonGeometry() optimizer.TensorGeometry {
	return func(name string, length int) (int, int, error) {
		if length == 1 {
			return 1, 1, nil
		}
		if strings.HasSuffix(name, ".bias") || strings.Contains(name, ".norm") {
			return 1, length, nil
		}
		if name == "final_proj.weight" {
			return m.Cfg.BaseChannels, length / m.Cfg.BaseChannels, nil
		}
		if base, ok := strings.CutSuffix(name, ".weight"); ok {
			if bias, ok := m.raw[base+".bias"]; ok && len(bias) > 0 && length%len(bias) == 0 {
				return len(bias), length / len(bias), nil
			}
			for _, marker := range []string{".attn.", ".mlp."} {
				if before, _, ok := strings.Cut(name, marker); ok {
					hidden := len(m.raw[before+".norm1.weight"])
					if hidden > 0 && length%hidden == 0 {
						if strings.HasSuffix(name, ".mlp.1.weight") {
							return hidden, length / hidden, nil
						}
						return length / hidden, hidden, nil
					}
				}
			}
		}
		return 0, 0, fmt.Errorf("diffusionimage train: Muon geometry absent for %s[%d]", name, length)
	}
}

// parameters: the checkpoint tensor map — the compiled bindings alias these
// slices, so in-place updates are live (scalar COPIES refresh separately).
func (m *Model) parameters() map[string][]float32 {
	return m.raw
}

// refreshScalars: re-read the compiled scalar copies (residual scales,
// xATGLU alphas) from the updated tensor map after an optimizer step.
func (m *Model) refreshScalars() {
	at := func(name string) float32 { return m.raw[name][0] }
	m.middleResidual = at("learned_middle_residual_scale")
	refreshBlock := func(blk block) {
		switch b := blk.(type) {
		case *resBlock:
			b.residualScale = at(b.name + ".learned_residual_scale")
		case *attnBlock:
			b.attention.AlphaO = float64(at(b.name + ".attn.o_proj.alpha"))
			b.mlpAlpha = at(b.name + ".mlp.0.alpha")
			b.attentionScale = at(b.name + ".learned_residual_scale_attn")
			b.mlpScale = at(b.name + ".learned_residual_scale_mlp")
		}
	}
	for _, levels := range [][]level{m.encoders, m.decoders} {
		for i := range levels {
			for _, blk := range levels[i].blocks {
				refreshBlock(blk)
			}
		}
	}
	for _, blk := range m.middle {
		refreshBlock(blk)
	}
}
