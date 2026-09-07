package adaptertrain

import (
	"context"
	"errors"
	"math"
	"slices"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
	"overgo/internal/trainingprogram"
)

// LinearCTCSpec binds a square input projection before a frozen linear CTC
// output. OutputWeight and OutputBias are borrowed immutable model storage;
// their owner keeps them alive and includes them in its own memory admission.
// MemoryBytes bounds additional numeric storage, including optimizer scratch.
type LinearCTCSpec struct {
	OutputWeight, OutputBias                        []float32
	Width, Vocabulary, MaxFrames, MaxTargets, Blank int
	MemoryBytes                                     uint64
	Optimizer                                       optimizer.Config
}

// LinearCTCExample supplies one unpadded sequence of frozen hidden features.
// Loss is a sequence sum. Logits and Gradient are borrowed outputs valid until
// the next execution; the optimize phase consumes Gradient. Update is assigned
// only when the compiled optimizer phase executes successfully.
type LinearCTCExample struct {
	Hidden   []float32
	Targets  []int
	Frames   int
	Loss     float64
	Logits   []float32
	Gradient []float32
	Update   optimizer.StepResult
	owner    *LinearCTC
	sequence uint64
}

// LinearCTC owns a trainable input projection and its shared objective binding.
// It has no training loop or encoder tape. Neither it nor bound executions may
// be used concurrently. An execution error leaves scratch outputs invalid.
// Updates are not transactional: discard the component if an optimizer update
// produces non-finite weights; do not treat an error as a recoverable checkpoint.
type LinearCTC struct {
	width, vocabulary, maxFrames, maxTargets, blank int
	outputWeight, outputBias                        []float32
	weights, gradients                              []float32
	projected, dProjected, logits, dLogits          []float32
	scratch                                         []float64
	bytes                                           uint64
	program                                         trainingprogram.TrainingProgram
	optimizer                                       *optimizer.Optimizer
	sequence                                        uint64
	phase                                           trainingprogram.OperatorPhase
	frames                                          int
}

// NewLinearCTC admits all numeric storage before allocation and initializes
// the trainable projection to identity. Only that projection enters Muon;
// the borrowed output projection and bias never receive gradients or updates.
func NewLinearCTC(spec LinearCTCSpec) (*LinearCTC, error) {
	parameters, parametersOK := checked.MulInt(spec.Width, spec.Width)
	outputCount, outputOK := checked.MulInt(spec.Width, spec.Vocabulary)
	featureCount, featureOK := checked.MulInt(spec.MaxFrames, spec.Width)
	logitCount, logitOK := checked.MulInt(spec.MaxFrames, spec.Vocabulary)
	if spec.Width <= 0 || spec.Vocabulary <= 0 || spec.MaxFrames <= 0 || spec.MaxTargets < 0 ||
		spec.MaxTargets > spec.MaxFrames || spec.Blank < 0 || spec.Blank >= spec.Vocabulary ||
		!parametersOK || !outputOK || !featureOK || !logitOK ||
		len(spec.OutputWeight) != outputCount || len(spec.OutputBias) != 0 && len(spec.OutputBias) != spec.Vocabulary {
		return nil, errors.New("linear CTC: invalid projection geometry")
	}
	for _, values := range [][]float32{spec.OutputWeight, spec.OutputBias} {
		if !finiteProjectionValues(values) {
			return nil, errors.New("linear CTC: non-finite frozen projection")
		}
	}
	plan, err := optimizer.CompilePlan(parameters, []optimizer.GroupSpec{{Name: "input-projection", End: parameters, Rows: spec.Width, Cols: spec.Width}})
	if err != nil {
		return nil, err
	}
	optimizerBytes, err := plan.HostStateBytes()
	if err != nil {
		return nil, err
	}
	ctcCount, err := trainingprogram.CTCLossWorkspaceSize(spec.MaxFrames, spec.Vocabulary, spec.MaxTargets)
	if err != nil {
		return nil, err
	}
	floatCount, ok := checked.Add64(uint64(parameters), uint64(parameters), uint64(featureCount), uint64(featureCount), uint64(logitCount), uint64(logitCount))
	if !ok {
		return nil, errors.New("linear CTC: buffer count overflows")
	}
	floatBytes, ok := checked.Bytes(floatCount, binaryschema.Uint32Bytes)
	if !ok {
		return nil, errors.New("linear CTC: float storage overflows")
	}
	ctcBytes, ok := checked.Bytes(uint64(ctcCount), binaryschema.Uint64Bytes)
	if !ok {
		return nil, errors.New("linear CTC: loss storage overflows")
	}
	bytes, ok := checked.Add64(floatBytes, ctcBytes, optimizerBytes)
	if !ok || bytes > spec.MemoryBytes || bytes > uint64(math.MaxInt) {
		return nil, errors.New("linear CTC: numeric storage exceeds memory admission")
	}
	if err := spec.Optimizer.Validate(); err != nil {
		return nil, err
	}
	program, err := trainingprogram.CompileObjectiveProgram(trainingprogram.ObjectiveCTC, nil, plan)
	if err != nil {
		return nil, err
	}
	m := &LinearCTC{
		width: spec.Width, vocabulary: spec.Vocabulary, maxFrames: spec.MaxFrames, maxTargets: spec.MaxTargets, blank: spec.Blank,
		outputWeight: spec.OutputWeight, outputBias: spec.OutputBias, bytes: bytes, program: program,
		weights: make([]float32, parameters), gradients: make([]float32, parameters),
		projected: make([]float32, featureCount), dProjected: make([]float32, featureCount),
		logits: make([]float32, logitCount), dLogits: make([]float32, logitCount), scratch: make([]float64, ctcCount),
	}
	for index := range spec.Width {
		m.weights[index*spec.Width+index] = 1
	}
	m.optimizer, err = optimizer.New(m.weights, m.gradients, plan, spec.Optimizer)
	if err != nil {
		return nil, err
	}
	return m, nil
}

// StorageBytes reports the admitted numeric storage, including lazy optimizer
// scratch but excluding borrowed hidden features and frozen model tensors.
func (m *LinearCTC) StorageBytes() uint64 { return m.bytes }

// WeightSnapshot copies the current input projection for immutable publication
// or evaluation; it does not expose the borrowed frozen output projection.
func (m *LinearCTC) WeightSnapshot() []float32 { return slices.Clone(m.weights) }

// Bind binds the existing forward/backward/Muon program for one context-owned
// session. Callers may select forward or forward/backward phases for evaluation
// and gradient verification. A new binding invalidates any incomplete previous
// phase; inputs remain immutable until their selected phases have completed.
func (m *LinearCTC) Bind(ctx context.Context) (trainingprogram.Execution[LinearCTCExample], error) {
	if ctx == nil || m == nil || m.optimizer == nil {
		return trainingprogram.Execution[LinearCTCExample]{}, errors.New("linear CTC: invalid session binding")
	}
	if err := ctx.Err(); err != nil {
		return trainingprogram.Execution[LinearCTCExample]{}, err
	}
	m.phase = ""
	return trainingprogram.BindObjective(m.program,
		func(example *LinearCTCExample) error {
			m.phase = ""
			example.Loss, example.Logits, example.Gradient, example.Update = 0, nil, nil, optimizer.StepResult{}
			if err := ctx.Err(); err != nil {
				return err
			}
			if err := m.admit(example); err != nil {
				return err
			}
			next, ok := checked.Add64(m.sequence, 1)
			if !ok {
				return errors.New("linear CTC: execution sequence exhausted")
			}
			m.sequence, m.frames = next, example.Frames
			example.owner, example.sequence = m, next
			logits := m.logits[:example.Frames*m.vocabulary]
			hostmath.LinearInputProjection(logits, m.projected[:len(example.Hidden)], example.Hidden, m.weights, m.outputWeight, m.outputBias, example.Frames, m.width, m.vocabulary)
			loss, err := trainingprogram.CTCLossF32(ctx, m.dLogits[:len(logits)], logits, example.Targets, example.Frames, m.vocabulary, m.blank, m.scratch)
			if err != nil {
				return err
			}
			example.Loss, example.Logits = loss, logits
			m.phase = trainingprogram.PhaseForward
			return nil
		},
		func(example *LinearCTCExample) error {
			if err := ctx.Err(); err != nil {
				return err
			}
			if err := m.requirePhase(example, trainingprogram.PhaseForward); err != nil {
				return err
			}
			m.phase = ""
			dProjected := m.dProjected[:len(example.Hidden)]
			hostmath.LinearBackwardInput(dProjected, m.dLogits[:len(example.Logits)], m.outputWeight, example.Frames, m.width, m.vocabulary)
			clear(m.gradients)
			hostmath.LinearWeightGradient(m.gradients, example.Hidden, dProjected, example.Frames, m.width, m.width)
			if !finiteProjectionValues(m.gradients) {
				return errors.New("linear CTC: non-finite input-projection gradient")
			}
			example.Gradient = m.gradients
			if err := ctx.Err(); err != nil {
				return err
			}
			m.phase = trainingprogram.PhaseBackward
			return nil
		},
		func(example *LinearCTCExample) error {
			if err := ctx.Err(); err != nil {
				return err
			}
			if err := m.requirePhase(example, trainingprogram.PhaseBackward); err != nil {
				return err
			}
			m.phase = ""
			if !finiteProjectionValues(m.gradients) {
				return errors.New("linear CTC: non-finite optimizer input")
			}
			result := m.optimizer.Step()
			if !finiteProjectionValues(m.weights) {
				return errors.New("linear CTC: non-finite input-projection update")
			}
			example.Update = result
			return nil
		})
}

func (m *LinearCTC) requirePhase(example *LinearCTCExample, phase trainingprogram.OperatorPhase) error {
	if m.phase != phase || example.owner != m || example.sequence != m.sequence || example.Frames != m.frames ||
		len(example.Hidden) != m.frames*m.width || len(example.Logits) != m.frames*m.vocabulary {
		return errors.New("linear CTC: missing or stale predecessor phase")
	}
	return nil
}

func (m *LinearCTC) admit(example *LinearCTCExample) error {
	if example.Frames <= 0 || example.Frames > m.maxFrames || len(example.Targets) > m.maxTargets ||
		len(example.Hidden) != example.Frames*m.width || !finiteProjectionValues(example.Hidden) {
		return errors.New("linear CTC: example exceeds admitted geometry or contains non-finite features")
	}
	for _, buffer := range [][]float32{m.weights, m.gradients, m.projected, m.dProjected, m.logits, m.dLogits} {
		if checked.SlicesOverlap(example.Hidden, buffer) {
			return errors.New("linear CTC: hidden features alias mutable storage")
		}
	}
	return nil
}

func finiteProjectionValues(values []float32) bool {
	for _, value := range values {
		if !checked.Finite32(value) {
			return false
		}
	}
	return true
}
