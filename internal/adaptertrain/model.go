// Package adaptertrain trains artifact-declared per-layer residual adapters.
package adaptertrain

import (
	"context"
	"errors"
	"fmt"
	"math"
	"path/filepath"

	"overgo/internal/gguf"
	"overgo/internal/hostmath"
	"overgo/internal/model"
	"overgo/internal/optimizer"
	"overgo/internal/safetensors"
	"overgo/internal/trainingprogram"
)

const (
	gateName       = "per-layer-gate"
	projectionName = "per-layer-projection"
	normName       = "per-layer-post-norm"
)

// Example is one projected-input training record.
type Example struct {
	Input  []float32
	Side   []float32
	Target []float32
	Rows   int
	Kind   string
}

// State is exact Muon progress at an accumulation boundary.
type State struct {
	Optimizer optimizer.State
}

// Model owns one streamed adapter unit and its compiled training program.
type Model struct {
	hidden, width int
	epsilon       float64
	layer         uint32
	weights       []float32
	gradients     []float32
	momentum      []float32
	gate          []float32
	projection    []float32
	norm          []float32
	inputProj     []float32
	inputNorm     []float32
	tokenInfo     gguf.TensorInfo
	targetInfo    gguf.TensorInfo
	plan          optimizer.Plan
	program       trainingprogram.TrainingProgram
	config        optimizer.Config
	step          int
}

// LoadArtifact binds one real artifact adapter without retaining the GGUF file.
func LoadArtifact(ctx context.Context, path string, layer uint32) (*Model, model.ModelPlan, error) {
	file, err := gguf.Open(path)
	if err != nil {
		return nil, model.ModelPlan{}, fmt.Errorf("adapter training: open artifact: %w", err)
	}
	defer file.Close()
	spec, err := model.ReadSpec(file)
	if err != nil {
		return nil, model.ModelPlan{}, err
	}
	weights, err := model.ReadWeights(file, spec)
	if err != nil {
		return nil, model.ModelPlan{}, err
	}
	plan, err := model.CompileModelPlanWithProfile(spec, weights, spec.Profile())
	if err != nil {
		return nil, model.ModelPlan{}, err
	}
	layerPlan, err := plan.Layer(int(layer))
	if err != nil {
		return nil, model.ModelPlan{}, err
	}
	if !layerPlan.PerLayerInput || weights.PerLayerTokenEmbedding == nil ||
		weights.PerLayerModelProjection == nil || weights.PerLayerProjectionNorm == nil {
		return nil, model.ModelPlan{}, errors.New("adapter training: artifact has no per-layer input program")
	}
	if layer >= uint32(len(weights.Layers)) {
		return nil, model.ModelPlan{}, errors.New("adapter training: layer exceeds artifact")
	}
	host, err := model.LoadHostLayer(ctx, file, weights.Layers[layer])
	if err != nil {
		return nil, model.ModelPlan{}, err
	}
	if host.PerLayerInputGate == nil || host.PerLayerProjection == nil || host.PerLayerPostNorm == nil {
		return nil, model.ModelPlan{}, errors.New("adapter training: layer adapter tensors are incomplete")
	}
	hidden, width := int(spec.EmbeddingLength), int(spec.EmbeddingPerLayer)
	if hidden <= 0 || width <= 0 || host.PerLayerInputGate.Shape.Dims[0] != uint64(hidden) ||
		host.PerLayerInputGate.Shape.Dims[1] != uint64(width) ||
		host.PerLayerProjection.Shape.Dims[0] != uint64(width) ||
		host.PerLayerProjection.Shape.Dims[1] != uint64(hidden) {
		return nil, model.ModelPlan{}, errors.New("adapter training: layer adapter geometry differs")
	}
	projectionRows := make([]uint32, width)
	for index := range projectionRows {
		projectionRows[index] = layer*uint32(width) + uint32(index)
	}
	inputProjection, err := model.LoadHostRows(ctx, file, *weights.PerLayerModelProjection, projectionRows)
	if err != nil {
		return nil, model.ModelPlan{}, err
	}
	inputNorm, err := model.LoadHostTensor(ctx, file, *weights.PerLayerProjectionNorm)
	if err != nil {
		return nil, model.ModelPlan{}, err
	}
	if len(inputNorm.Data) != width {
		return nil, model.ModelPlan{}, errors.New("adapter training: projected-input norm geometry differs")
	}

	m := &Model{
		hidden: hidden, width: width, epsilon: float64(spec.RMSNormEpsilon), layer: layer,
		inputProj: inputProjection.Data, inputNorm: inputNorm.Data,
		tokenInfo: *weights.PerLayerTokenEmbedding, targetInfo: weights.TokenEmbedding,
	}
	appendParameter := func(values []float32) []float32 {
		start := len(m.weights)
		m.weights = append(m.weights, values...)
		return m.weights[start : start+len(values)]
	}
	gateValues := append([]float32(nil), host.PerLayerInputGate.Data...)
	projectionValues := append([]float32(nil), host.PerLayerProjection.Data...)
	normValues := append([]float32(nil), host.PerLayerPostNorm.Data...)
	m.weights = make([]float32, 0, len(gateValues)+len(projectionValues)+len(normValues))
	m.gate = appendParameter(gateValues)
	m.projection = appendParameter(projectionValues)
	m.norm = appendParameter(normValues)
	// Appends can move the slab; bind final ranges once.
	gateEnd := len(gateValues)
	projectionEnd := gateEnd + len(projectionValues)
	m.gate = m.weights[:gateEnd]
	m.projection = m.weights[gateEnd:projectionEnd]
	m.norm = m.weights[projectionEnd:]
	m.gradients = make([]float32, len(m.weights))
	m.momentum = make([]float32, len(m.weights))
	m.plan, err = optimizer.CompilePlan(len(m.weights), []optimizer.GroupSpec{
		{Name: gateName, Start: 0, End: gateEnd, Rows: width, Cols: hidden},
		{Name: projectionName, Start: gateEnd, End: projectionEnd, Rows: hidden, Cols: width},
		{Name: normName, Start: projectionEnd, End: len(m.weights), Rows: 1, Cols: hidden},
	})
	if err != nil {
		return nil, model.ModelPlan{}, err
	}
	m.config = optimizer.Config{BaseLearningRate: optimizer.DeriveBaseLR(len(m.weights)), Schedule: optimizer.ScheduleConstant}
	m.program, err = trainingprogram.CompileTrainingProgram(trainingprogram.ProgramSpec{
		Objective: trainingprogram.ObjectiveTokenPrediction,
		Operators: []trainingprogram.OperatorSpec{
			{ID: "adapter-forward", Phase: trainingprogram.PhaseForward},
			{ID: "squared-error", Phase: trainingprogram.PhaseLoss},
			{ID: "adapter-backward", Phase: trainingprogram.PhaseBackward},
			{ID: "muon", Phase: trainingprogram.PhaseOptimize},
		},
		Parameters: []trainingprogram.ParameterSpec{
			{Name: gateName, Rows: width, Cols: hidden, Trainable: true},
			{Name: projectionName, Rows: hidden, Cols: width, Trainable: true},
			{Name: normName, Rows: 1, Cols: hidden, Trainable: true},
		},
		Optimizer: m.plan,
	})
	if err != nil {
		return nil, model.ModelPlan{}, err
	}
	return m, plan, nil
}

// BuildExample derives the selected layer input from artifact rows.
func (m *Model) BuildExample(ctx context.Context, path string, input []float32, tokenIDs, targetIDs []uint32, kind string) (Example, error) {
	if m == nil || len(tokenIDs) == 0 || len(tokenIDs) != len(targetIDs) || kind == "" {
		return Example{}, errors.New("adapter training: example authority is incomplete")
	}
	file, err := gguf.Open(path)
	if err != nil {
		return Example{}, err
	}
	defer file.Close()
	rows := len(tokenIDs)
	if input == nil {
		value, err := model.LoadHostRows(ctx, file, m.targetInfo, tokenIDs)
		if err != nil {
			return Example{}, err
		}
		input = value.Data
	}
	if len(input) != rows*m.hidden {
		return Example{}, fmt.Errorf("adapter training: %s input has %d values, need %d", kind, len(input), rows*m.hidden)
	}
	selected, err := model.LoadHostRows(ctx, file, m.tokenInfo, tokenIDs)
	if err != nil {
		return Example{}, err
	}
	target, err := model.LoadHostRows(ctx, file, m.targetInfo, targetIDs)
	if err != nil {
		return Example{}, err
	}
	projected := make([]float32, rows*m.width)
	hostmath.Linear(projected, input, m.inputProj, rows, m.hidden, m.width)
	inputScale := float32(1 / math.Sqrt(float64(m.hidden)))
	for index := range projected {
		projected[index] *= inputScale
	}
	normalized := make([]float32, len(projected))
	hostmath.RMSNormInto(normalized, projected, m.inputNorm, rows, m.width, m.epsilon)
	combinedWidth := int(m.tokenInfo.Shape[0])
	if combinedWidth < (int(m.layer)+1)*m.width || len(selected.Data) != rows*combinedWidth {
		return Example{}, errors.New("adapter training: per-layer token rows differ")
	}
	side := make([]float32, rows*m.width)
	tokenScale := float32(math.Sqrt(float64(m.width)))
	combineScale := float32(1 / math.Sqrt2)
	for row := range rows {
		source := selected.Data[row*combinedWidth+int(m.layer)*m.width:]
		for column := range m.width {
			index := row*m.width + column
			side[index] = (normalized[index] + source[column]*tokenScale) * combineScale
		}
	}
	return Example{
		Input: append([]float32(nil), input...), Side: side, Target: target.Data,
		Rows: rows, Kind: kind,
	}, nil
}

func (m *Model) Program() trainingprogram.TrainingProgram { return m.program }
func (m *Model) ParameterCount() int                      { return len(m.weights) }
func (m *Model) Layer() uint32                            { return m.layer }

// Output evaluates one typed example without changing trainer state.
func (m *Model) Output(example Example) ([]float32, error) {
	if err := m.validateExample(example); err != nil {
		return nil, err
	}
	output, _, err := hostmath.PerLayerAdapterForward(
		example.Input, example.Side, m.gate, m.projection, m.norm,
		example.Rows, m.hidden, m.width, m.epsilon,
	)
	return output, err
}

func (m *Model) validateExample(example Example) error {
	if m == nil || example.Rows <= 0 || example.Kind == "" ||
		len(example.Input) != example.Rows*m.hidden || len(example.Side) != example.Rows*m.width ||
		len(example.Target) != example.Rows*m.hidden {
		return errors.New("adapter training: invalid step input")
	}
	return nil
}

// Snapshot returns portable optimizer state; weights publish separately.
func (m *Model) Snapshot() State {
	momentum := make([]float64, len(m.momentum))
	for index, value := range m.momentum {
		momentum[index] = float64(value)
	}
	return State{Optimizer: optimizer.State{
		PlanIdentity: m.plan.Identity(), Config: m.config, Step: m.step, Momentum: momentum,
	}}
}

// Restore binds portable Muon state to the current artifact weights.
func (m *Model) Restore(state State) error {
	if err := optimizer.ValidateState(state.Optimizer, m.plan.Identity(), len(m.weights)); err != nil {
		return err
	}
	if state.Optimizer.Config != m.config {
		return errors.New("adapter training: optimizer config differs")
	}
	m.step = state.Optimizer.Step
	for index, value := range state.Optimizer.Momentum {
		m.momentum[index] = float32(value)
	}
	return nil
}

// SaveWeights writes the exact trainable unit as F32 safetensors.
func (m *Model) SaveWeights(path string) error {
	gateEnd := len(m.gate)
	projectionEnd := gateEnd + len(m.projection)
	return safetensors.Save(path, map[string][]float32{
		gateName: m.weights[:gateEnd], projectionName: m.weights[gateEnd:projectionEnd], normName: m.weights[projectionEnd:],
	}, map[string][]int{
		gateName: {m.width, m.hidden}, projectionName: {m.hidden, m.width}, normName: {m.hidden},
	}, map[string]string{"program": m.program.ID().String()})
}

// RestoreWeights loads an exact trainable unit published by SaveWeights.
func (m *Model) RestoreWeights(directory string) error {
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		return err
	}
	defer source.Close()
	for _, binding := range []struct {
		name string
		dst  []float32
	}{
		{gateName, m.gate}, {projectionName, m.projection}, {normName, m.norm},
	} {
		tensor, ok := source.Tensors[binding.name]
		if !ok {
			return fmt.Errorf("adapter training: checkpoint tensor %q absent", binding.name)
		}
		values, err := safetensors.ReadF32(tensor)
		if err != nil || len(values) != len(binding.dst) {
			return fmt.Errorf("adapter training: checkpoint tensor %q differs", binding.name)
		}
		copy(binding.dst, values)
	}
	return nil
}

func CheckpointWeightsPath(stage string) string {
	return filepath.Join(stage, trainingprogram.CheckpointWeights)
}

// WeightSnapshot returns evidence-only copies for exact-resume comparison.
func (m *Model) WeightSnapshot() []float32 { return append([]float32(nil), m.weights...) }
