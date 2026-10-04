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

	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
	"overgo/internal/safetensors"
	"overgo/internal/trainingprogram"
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
	pack           *optimizer.TensorPack
	stepper        optimizer.Stepper
	config         optimizer.Config
	step           int
}

// The projection pair's groups in pack order.
const (
	bridgeDownW = iota
	bridgeDownB
	bridgeUpW
	bridgeUpB
)

// NewLatentBridgeTrainer packs the pair as f32 masters and compiles the
// Muon plan with derived hyperparameters.
func NewLatentBridgeTrainer(w LatentBridgeWeights, policy trainingprogram.OptimizerPolicy) (*LatentBridgeTrainer, error) {
	if w.Hidden <= 0 || w.Latent <= 0 {
		return nil, fmt.Errorf("routed lm latent bridge: geometry %dx%d", w.Hidden, w.Latent)
	}
	pack, err := optimizer.NewSectionPack([]optimizer.TensorSection{
		{Name: "bridge.down.weight", Rows: w.Latent, Cols: w.Hidden, BF16: w.DownW.Data},
		{Name: "bridge.down.bias", Rows: 1, Cols: w.Latent, Values: w.DownB},
		{Name: "bridge.up.weight", Rows: w.Hidden, Cols: w.Latent, BF16: w.UpW.Data},
		{Name: "bridge.up.bias", Rows: 1, Cols: w.Hidden, Values: w.UpB},
	})
	if err != nil {
		return nil, fmt.Errorf("routed lm latent bridge: %w", err)
	}
	trainer := &LatentBridgeTrainer{hidden: w.Hidden, latent: w.Latent, pack: pack}
	if trainer.config, err = policy.Config(pack.ParameterCount()); err != nil {
		return nil, err
	}
	if trainer.stepper, err = pack.NewStepper(trainer.config); err != nil {
		return nil, err
	}
	return trainer, nil
}

// ParameterCount reports the packed organ parameter total.
func (t *LatentBridgeTrainer) ParameterCount() int { return t.pack.ParameterCount() }

// Config exposes the derived hyperparameters for evidence.
func (t *LatentBridgeTrainer) Config() optimizer.Config { return t.config }

// OptimizerPlan is the compiled Muon plan over the packed pair: one group
// per projection tensor.
func (t *LatentBridgeTrainer) OptimizerPlan() optimizer.Plan { return t.pack.Plan() }

// Snapshot copies the optimizer state the next step continues from.
func (t *LatentBridgeTrainer) Snapshot() optimizer.State { return t.stepper.Snapshot() }

// SaveWeights writes the trained pair at path, the weights Snapshot pairs
// with.
func (t *LatentBridgeTrainer) SaveWeights(path string) error { return t.pack.Save(path) }

// Close releases the stepper backend.
func (t *LatentBridgeTrainer) Close() error { return t.stepper.Close() }

// Reconstruct runs the training-precision roundtrip on the packed weights.
func (t *LatentBridgeTrainer) Reconstruct(hidden []float32) ([]float32, error) {
	_, out, err := t.forward(hidden)
	return out, err
}

// forward: the roundtrip's latent rows and reconstructed hidden rows.
func (t *LatentBridgeTrainer) forward(hidden []float32) (latent, out []float32, err error) {
	rows, err := t.rows(hidden)
	if err != nil {
		return nil, nil, err
	}
	latent = make([]float32, rows*t.latent)
	hostmath.Linear(latent, hidden, t.pack.Weights(bridgeDownW), rows, t.hidden, t.latent)
	for r := range rows {
		hostmath.AddBias(latent[r*t.latent:(r+1)*t.latent], t.pack.Weights(bridgeDownB))
	}
	out = make([]float32, rows*t.hidden)
	hostmath.Linear(out, latent, t.pack.Weights(bridgeUpW), rows, t.latent, t.hidden)
	for r := range rows {
		hostmath.AddBias(out[r*t.hidden:(r+1)*t.hidden], t.pack.Weights(bridgeUpB))
	}
	return latent, out, nil
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
	return optimizer.Advance(t.stepper, t.config, &t.step, loss, gradientL2)
}

// lossAndBridgeGradients fills the packed gradient buffer for one pair
// without stepping.
func (t *LatentBridgeTrainer) lossAndBridgeGradients(hidden []float32) (float64, float64, error) {
	latent, out, err := t.forward(hidden)
	if err != nil {
		return 0, 0, err
	}
	rows := len(hidden) / t.hidden

	invN := 1 / float64(len(hidden))
	var loss float64
	dOut := make([]float32, len(out))
	for i := range out {
		d := float64(out[i]) - float64(hidden[i])
		loss += d * d * invN
		dOut[i] = float32(2 * d * invN)
	}

	t.pack.ClearGradients()
	dLatent := make([]float32, len(latent))
	hostmath.LinearBackward(dLatent, t.pack.Gradients(bridgeUpW), t.pack.Gradients(bridgeUpB),
		latent, t.pack.Weights(bridgeUpW), dOut, rows, t.latent, t.hidden, false)
	dHidden := make([]float32, len(hidden))
	hostmath.LinearBackward(dHidden, t.pack.Gradients(bridgeDownW), t.pack.Gradients(bridgeDownB),
		hidden, t.pack.Weights(bridgeDownW), dLatent, rows, t.hidden, t.latent, false)
	return loss, t.pack.GradientL2(), nil
}

func (t *LatentBridgeTrainer) rows(hidden []float32) (int, error) {
	if len(hidden) == 0 || len(hidden)%t.hidden != 0 {
		return 0, fmt.Errorf("routed lm latent bridge: hidden len=%d not divisible by %d", len(hidden), t.hidden)
	}
	return len(hidden) / t.hidden, nil
}
