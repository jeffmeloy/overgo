package optimizer

import (
	"fmt"
	"slices"
)

// TensorGeometry: named tensor matrix geometry.
type TensorGeometry func(name string, length int) (rows, cols int, err error)

// TensorPack: deterministic tensor binding over flat optimizer storage.
type TensorPack struct {
	weights   []float32
	gradients []float32
	plan      Plan
	bindings  [][]float32
}

// NewTensorPack binds named tensors to one validated optimizer plan.
func NewTensorPack(tensors map[string][]float32, geometry TensorGeometry) (*TensorPack, error) {
	if geometry == nil {
		return nil, fmt.Errorf("optimizer tensor pack: geometry is required")
	}
	names := make([]string, 0, len(tensors))
	total := 0
	maxInt := int(^uint(0) >> 1)
	for name, values := range tensors {
		if len(values) > maxInt-total {
			return nil, fmt.Errorf("optimizer tensor pack: parameter count overflow")
		}
		names = append(names, name)
		total += len(values)
	}
	slices.Sort(names)

	pack := &TensorPack{
		weights:   make([]float32, total),
		gradients: make([]float32, total),
		bindings:  make([][]float32, 0, len(names)),
	}
	specs := make([]GroupSpec, 0, len(names))
	offset := 0
	for _, name := range names {
		values := tensors[name]
		rows, cols, err := geometry(name, len(values))
		if err != nil {
			return nil, fmt.Errorf("optimizer tensor pack: %s: %w", name, err)
		}
		end := offset + len(values)
		copy(pack.weights[offset:end], values)
		pack.bindings = append(pack.bindings, values)
		specs = append(specs, GroupSpec{
			Name: name, Start: offset, End: end, Rows: rows, Cols: cols,
		})
		offset = end
	}
	plan, err := CompilePlan(total, specs)
	if err != nil {
		return nil, err
	}
	pack.plan = plan
	return pack, nil
}

// MatrixGeometry resolves fixed [rows, cols] facts by tensor name.
func MatrixGeometry(shapes map[string][2]int) TensorGeometry {
	return func(name string, _ int) (int, int, error) {
		shape, ok := shapes[name]
		if !ok {
			return 0, 0, fmt.Errorf("shape is absent")
		}
		return shape[0], shape[1], nil
	}
}

func (p *TensorPack) ParameterCount() int { return len(p.weights) }

func (p *TensorPack) Plan() Plan { return p.plan }

// BindMapViews makes packed slabs authoritative for map-backed tensors.
func (p *TensorPack) BindMapViews(tensors map[string][]float32) map[string][]float32 {
	gradients := make(map[string][]float32, len(p.bindings))
	for index := range p.bindings {
		group := p.plan.Groups()[index]
		weights := p.weights[group.Start:group.End:group.End]
		tensors[group.Name] = weights
		p.bindings[index] = weights
		gradients[group.Name] = p.gradients[group.Start:group.End:group.End]
	}
	return gradients
}

// NewOptimizer binds Muon state directly to packed storage.
func (p *TensorPack) NewOptimizer(config Config) (*Optimizer, error) {
	return New(p.weights, p.gradients, p.plan, config)
}

// NewStepper binds the packed slabs to the platform Muon backend.
func (p *TensorPack) NewStepper(config Config) (Stepper, error) {
	return NewStepper(p.weights, p.gradients, p.plan, config)
}

// Scatter publishes packed weights to bound model tensors.
func (p *TensorPack) Scatter() {
	for index, values := range p.bindings {
		group := p.plan.Groups()[index]
		copy(values, p.weights[group.Start:group.End])
	}
}

// GatherGradients replaces the complete packed gradient slab.
func (p *TensorPack) GatherGradients(gradients map[string][]float32) error {
	for index := range p.bindings {
		group := p.plan.Groups()[index]
		destination := p.gradients[group.Start:group.End]
		clear(destination)
		source, ok := gradients[group.Name]
		if !ok {
			continue
		}
		if len(source) != len(destination) {
			return fmt.Errorf(
				"optimizer tensor pack: %s gradient length %d, want %d",
				group.Name, len(source), len(destination),
			)
		}
		copy(destination, source)
	}
	return nil
}
