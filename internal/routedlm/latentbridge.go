// Trainable latent-bridge organ for the routed-LM generation terminal: the
// RxBrain-family llm2vae/vae2llm projection pair trains against a bridge
// reconstruction objective — vae2llm(llm2vae(h)) recovering h through the
// rank-limited latent bottleneck — over the shared Muon stepper with fully
// derived hyperparameters. The pair ships as approximate inverses, so the
// objective is the organ's own contract rather than an invented target;
// f32 masters decode from the BF16 serving storage, and full-pipeline
// image-flow training remains the promotion gate beyond the organ.
package routedlm

import (
	"fmt"
	"math"

	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
	"overgo/internal/safetensors"
)

// LatentBridgeBinding: the projection pair's tensor names.
type LatentBridgeBinding struct {
	Down, Up string // hidden -> latent, latent -> hidden
}

// RxBrainLatentBridgeBinding: Hy-Embodied-RxBrain naming.
func RxBrainLatentBridgeBinding() LatentBridgeBinding {
	return LatentBridgeBinding{Down: "llm2vae", Up: "vae2llm"}
}

// LatentBridgeWeights: the loaded pair in serving form.
type LatentBridgeWeights struct {
	Hidden, Latent int
	DownW, UpW     BF16Matrix
	DownB, UpB     []float32
}

// LoadLatentBridgeWeights derives the bridge geometry from the artifact's
// own tensor shapes and materializes the pair.
func LoadLatentBridgeWeights(src *safetensors.Source, b LatentBridgeBinding) (LatentBridgeWeights, error) {
	var w LatentBridgeWeights
	downShape, err := tensorShape(src, b.Down+".weight")
	if err != nil {
		return w, err
	}
	if len(downShape) != 2 || downShape[0] <= 0 || downShape[1] <= 0 {
		return w, fmt.Errorf("routed lm latent bridge: %s.weight shape %v", b.Down, downShape)
	}
	w.Latent, w.Hidden = int(downShape[0]), int(downShape[1])
	if w.DownW, err = materializeBF16(src, b.Down+".weight", w.Hidden, w.Latent); err != nil {
		return w, err
	}
	if w.DownB, err = materializeVectorF32(src, b.Down+".bias", w.Latent); err != nil {
		return w, err
	}
	if w.UpW, err = materializeBF16(src, b.Up+".weight", w.Latent, w.Hidden); err != nil {
		return w, err
	}
	if w.UpB, err = materializeVectorF32(src, b.Up+".bias", w.Hidden); err != nil {
		return w, err
	}
	return w, nil
}

// LatentBridgeTrainer: packed projection pair over the shared Muon stepper.
type LatentBridgeTrainer struct {
	hidden, latent int
	weights        []float32
	gradients      []float32
	downW, downB   span
	upW, upB       span
	stepper        optimizer.Stepper
	config         optimizer.Config
	step           int
}

// NewLatentBridgeTrainer packs the pair as f32 masters and compiles the
// Muon plan with derived hyperparameters.
func NewLatentBridgeTrainer(w LatentBridgeWeights) (*LatentBridgeTrainer, error) {
	if w.Hidden <= 0 || w.Latent <= 0 {
		return nil, fmt.Errorf("routed lm latent bridge: geometry %dx%d", w.Hidden, w.Latent)
	}
	sections := []struct {
		values     []float32
		rows, cols int
	}{
		{bf16Masters(w.DownW.Data), w.Latent, w.Hidden},
		{w.DownB, 1, w.Latent},
		{bf16Masters(w.UpW.Data), w.Hidden, w.Latent},
		{w.UpB, 1, w.Hidden},
	}
	names := []string{"bridge.down.weight", "bridge.down.bias", "bridge.up.weight", "bridge.up.bias"}
	total := 0
	specs := make([]optimizer.GroupSpec, len(sections))
	for i, s := range sections {
		if len(s.values) != s.rows*s.cols {
			return nil, fmt.Errorf("routed lm latent bridge: %s has %d values, want %d", names[i], len(s.values), s.rows*s.cols)
		}
		specs[i] = optimizer.GroupSpec{Name: names[i], Start: total, End: total + len(s.values), Rows: s.rows, Cols: s.cols}
		total += len(s.values)
	}
	plan, err := optimizer.CompilePlan(total, specs)
	if err != nil {
		return nil, err
	}
	trainer := &LatentBridgeTrainer{
		hidden: w.Hidden, latent: w.Latent,
		weights:   make([]float32, total),
		gradients: make([]float32, total),
		config: optimizer.Config{
			BaseLearningRate: optimizer.DeriveBaseLR(total),
			Momentum:         optimizer.DeriveMomentum(),
			Schedule:         optimizer.ScheduleConstant,
		},
	}
	spans := []*span{&trainer.downW, &trainer.downB, &trainer.upW, &trainer.upB}
	for i, s := range sections {
		*spans[i] = span{specs[i].Start, specs[i].End}
		copy(trainer.weights[specs[i].Start:specs[i].End], s.values)
	}
	trainer.stepper, err = optimizer.NewStepper(trainer.weights, trainer.gradients, plan, trainer.config)
	if err != nil {
		return nil, err
	}
	return trainer, nil
}

// ParameterCount reports the packed organ parameter total.
func (t *LatentBridgeTrainer) ParameterCount() int { return len(t.weights) }

// Config exposes the derived hyperparameters for evidence.
func (t *LatentBridgeTrainer) Config() optimizer.Config { return t.config }

// Close releases the stepper backend.
func (t *LatentBridgeTrainer) Close() error { return t.stepper.Close() }

// Reconstruct runs the training-precision roundtrip on the packed weights.
func (t *LatentBridgeTrainer) Reconstruct(hidden []float32) ([]float32, error) {
	rows, err := t.rows(hidden)
	if err != nil {
		return nil, err
	}
	latent := make([]float32, rows*t.latent)
	hostmath.Linear(latent, hidden, t.weights[t.downW.start:t.downW.end], rows, t.hidden, t.latent)
	for r := 0; r < rows; r++ {
		hostmath.AddBias(latent[r*t.latent:(r+1)*t.latent], t.weights[t.downB.start:t.downB.end])
	}
	out := make([]float32, rows*t.hidden)
	hostmath.Linear(out, latent, t.weights[t.upW.start:t.upW.end], rows, t.latent, t.hidden)
	for r := 0; r < rows; r++ {
		hostmath.AddBias(out[r*t.hidden:(r+1)*t.hidden], t.weights[t.upB.start:t.upB.end])
	}
	return out, nil
}

// Loss evaluates the reconstruction MSE without training.
func (t *LatentBridgeTrainer) Loss(hidden []float32) (float64, error) {
	out, err := t.Reconstruct(hidden)
	if err != nil {
		return 0, err
	}
	invN := 1 / float64(len(hidden))
	var loss float64
	for i := range out {
		d := float64(out[i]) - float64(hidden[i])
		loss += d * d * invN
	}
	return loss, nil
}

// Step: one observed training step of the reconstruction objective.
func (t *LatentBridgeTrainer) Step(hidden []float32) (FlowHeadStepResult, error) {
	loss, gradientL2, err := t.lossAndBridgeGradients(hidden)
	if err != nil {
		return FlowHeadStepResult{}, err
	}
	if err := t.stepper.Step(); err != nil {
		return FlowHeadStepResult{}, err
	}
	t.step++
	return FlowHeadStepResult{
		Loss: loss, Step: t.step,
		LearningRate: t.config.LearningRate(t.step), GradientL2: gradientL2,
	}, nil
}

// lossAndBridgeGradients fills the packed gradient buffer for one pair
// without stepping.
func (t *LatentBridgeTrainer) lossAndBridgeGradients(hidden []float32) (float64, float64, error) {
	rows, err := t.rows(hidden)
	if err != nil {
		return 0, 0, err
	}
	latent := make([]float32, rows*t.latent)
	hostmath.Linear(latent, hidden, t.weights[t.downW.start:t.downW.end], rows, t.hidden, t.latent)
	for r := 0; r < rows; r++ {
		hostmath.AddBias(latent[r*t.latent:(r+1)*t.latent], t.weights[t.downB.start:t.downB.end])
	}
	out := make([]float32, rows*t.hidden)
	hostmath.Linear(out, latent, t.weights[t.upW.start:t.upW.end], rows, t.latent, t.hidden)
	for r := 0; r < rows; r++ {
		hostmath.AddBias(out[r*t.hidden:(r+1)*t.hidden], t.weights[t.upB.start:t.upB.end])
	}

	invN := 1 / float64(len(hidden))
	var loss float64
	dOut := make([]float32, len(out))
	for i := range out {
		d := float64(out[i]) - float64(hidden[i])
		loss += d * d * invN
		dOut[i] = float32(2 * d * invN)
	}

	clear(t.gradients)
	dLatent := make([]float32, len(latent))
	hostmath.LinearBackward(dLatent, t.gradients[t.upW.start:t.upW.end], t.gradients[t.upB.start:t.upB.end],
		latent, t.weights[t.upW.start:t.upW.end], dOut, rows, t.latent, t.hidden, false)
	dHidden := make([]float32, len(hidden))
	hostmath.LinearBackward(dHidden, t.gradients[t.downW.start:t.downW.end], t.gradients[t.downB.start:t.downB.end],
		hidden, t.weights[t.downW.start:t.downW.end], dLatent, rows, t.hidden, t.latent, false)

	var gradientSquared float64
	for _, g := range t.gradients {
		gradientSquared += float64(g) * float64(g)
	}
	return loss, math.Sqrt(gradientSquared), nil
}

func (t *LatentBridgeTrainer) rows(hidden []float32) (int, error) {
	if len(hidden) == 0 || len(hidden)%t.hidden != 0 {
		return 0, fmt.Errorf("routed lm latent bridge: hidden len=%d not divisible by %d", len(hidden), t.hidden)
	}
	return len(hidden) / t.hidden, nil
}
