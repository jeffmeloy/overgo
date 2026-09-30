package optimizer

import (
	"cmp"
	"fmt"
	"maps"
	"math"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/safetensors"
	"overgo/internal/tensor/dtype"
)

// TensorGeometry: named tensor matrix geometry.
type TensorGeometry func(name string, length int) (rows, cols int, err error)

// TensorSection is one group of a pack in layout order: its name, its
// [rows, cols] geometry, and its initial F32 values or, when Values is
// empty, the BF16 storage the pack decodes into F32 master weights.
type TensorSection struct {
	Name       string
	Rows, Cols int
	Values     []float32
	BF16       []uint16
}

// TensorPack: deterministic tensor binding over flat optimizer storage.
type TensorPack struct {
	weights   []float32
	gradients []float32
	plan      Plan
	bindings  [][]float32
}

// NewTensorPack binds named tensors, in name order, to one validated
// optimizer plan; Scatter publishes the packed weights back to them.
func NewTensorPack(tensors map[string][]float32, geometry TensorGeometry) (*TensorPack, error) {
	sections, err := nameOrder(tensors, geometry)
	if err != nil {
		return nil, err
	}
	pack, err := NewSectionPack(sections)
	if err != nil {
		return nil, err
	}
	for _, section := range sections {
		pack.bindings = append(pack.bindings, section.Values)
	}
	return pack, nil
}

// TensorPlan compiles the plan NewTensorPack binds tensors to, without
// storage.
func TensorPlan(tensors map[string][]float32, geometry TensorGeometry) (Plan, error) {
	sections, err := nameOrder(tensors, geometry)
	if err != nil {
		return Plan{}, err
	}
	return sectionPlan(sections)
}

// nameOrder lays named tensors out as sections in name order.
func nameOrder(tensors map[string][]float32, geometry TensorGeometry) ([]TensorSection, error) {
	if geometry == nil {
		return nil, fmt.Errorf("optimizer tensor pack: geometry is required")
	}
	sections := make([]TensorSection, 0, len(tensors))
	for _, name := range slices.Sorted(maps.Keys(tensors)) {
		rows, cols, err := geometry(name, len(tensors[name]))
		if err != nil {
			return nil, fmt.Errorf("optimizer tensor pack: %s: %w", name, err)
		}
		sections = append(sections, TensorSection{Name: name, Rows: rows, Cols: cols, Values: tensors[name]})
	}
	return sections, nil
}

// sectionPlan compiles sections, in order, into one contiguous plan.
func sectionPlan(sections []TensorSection) (Plan, error) {
	specs := make([]GroupSpec, len(sections))
	total := 0
	for index, section := range sections {
		end, valid := checked.AddInt(total, cmp.Or(len(section.Values), len(section.BF16)))
		if !valid {
			return Plan{}, fmt.Errorf("optimizer tensor pack: parameter count overflow")
		}
		specs[index] = GroupSpec{Name: section.Name, Start: total, End: end, Rows: section.Rows, Cols: section.Cols}
		total = end
	}
	return CompilePlan(total, specs)
}

// NewSectionPack packs sections, in declaration order, into one validated
// optimizer plan over storage the pack owns.
func NewSectionPack(sections []TensorSection) (*TensorPack, error) {
	plan, err := sectionPlan(sections)
	if err != nil {
		return nil, err
	}
	pack := &TensorPack{
		weights:   make([]float32, plan.ParameterCount()),
		gradients: make([]float32, plan.ParameterCount()),
		plan:      plan,
	}
	for index, section := range sections {
		weights := pack.Weights(index)
		copy(weights, section.Values)
		for offset, value := range section.BF16 {
			weights[offset] = dtype.BF16ToFloat32(value)
		}
	}
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

// Slabs are the packed weight and gradient storage the plan's groups span.
func (p *TensorPack) Slabs() (weights, gradients []float32) { return p.weights, p.gradients }

// Weights is the packed weight span of the plan's group-th group.
func (p *TensorPack) Weights(group int) []float32 {
	spec := p.plan.Groups()[group]
	return p.weights[spec.Start:spec.End:spec.End]
}

// Gradients is the packed gradient span of the plan's group-th group.
func (p *TensorPack) Gradients(group int) []float32 {
	spec := p.plan.Groups()[group]
	return p.gradients[spec.Start:spec.End:spec.End]
}

// ClearGradients zeroes the whole packed gradient slab.
func (p *TensorPack) ClearGradients() { clear(p.gradients) }

// GradientL2 is the L2 norm of the whole packed gradient slab.
func (p *TensorPack) GradientL2() float64 {
	var squared float64
	for _, gradient := range p.gradients {
		squared += float64(gradient) * float64(gradient)
	}
	return math.Sqrt(squared)
}

// BindMapViews makes packed slabs authoritative for map-backed tensors.
func (p *TensorPack) BindMapViews(tensors map[string][]float32) map[string][]float32 {
	gradients := make(map[string][]float32, len(p.bindings))
	for index := range p.bindings {
		name := p.plan.Groups()[index].Name
		tensors[name] = p.Weights(index)
		p.bindings[index] = tensors[name]
		gradients[name] = p.Gradients(index)
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

// Save writes every packed tensor at path as SavePlanWeights does: the
// weights the Snapshot of the pack's stepper pairs with.
func (p *TensorPack) Save(path string) error { return SavePlanWeights(path, p.plan, p.weights) }

// SavePlanWeights writes each group of plan at path as a named F32
// safetensors tensor over its span of weights, with its [rows, cols]
// geometry: the weights the Snapshot of a stepper over that plan pairs with.
func SavePlanWeights(path string, plan Plan, weights []float32) error {
	tensors := make(map[string][]float32, plan.GroupCount())
	shapes := make(map[string][]int, plan.GroupCount())
	for _, group := range plan.Groups() {
		tensors[group.Name] = weights[group.Start:group.End]
		shapes[group.Name] = []int{group.Rows, group.Cols}
	}
	return safetensors.Save(path, tensors, shapes, nil)
}

// Scatter publishes packed weights to bound model tensors.
func (p *TensorPack) Scatter() {
	for index, values := range p.bindings {
		copy(values, p.Weights(index))
	}
}

// RestoreWeights validates every replacement before changing packed or bound
// weights. The caller retains ownership of the replacement tensors.
func (p *TensorPack) RestoreWeights(tensors map[string][]float32) error {
	if p == nil || p.plan.Identity() == "" || len(tensors) != p.plan.GroupCount() {
		return fmt.Errorf("optimizer tensor pack: restored tensor set differs")
	}
	for _, group := range p.plan.Groups() {
		values, found := tensors[group.Name]
		if !found || len(values) != group.End-group.Start {
			return fmt.Errorf("optimizer tensor pack: %s restored shape differs", group.Name)
		}
		for _, value := range values {
			if !checked.Finite32(value) {
				return fmt.Errorf("optimizer tensor pack: %s restored value is non-finite", group.Name)
			}
		}
	}
	// Stage before publishing: replacement views may alias packed or bound
	// storage, including another group's range.
	restored := make([]float32, len(p.weights))
	for _, group := range p.plan.Groups() {
		copy(restored[group.Start:group.End], tensors[group.Name])
	}
	copy(p.weights, restored)
	clear(p.gradients)
	p.Scatter()
	return nil
}

// GatherGradients replaces the complete packed gradient slab.
func (p *TensorPack) GatherGradients(gradients map[string][]float32) error {
	for index, group := range p.plan.Groups() {
		destination := p.Gradients(index)
		source, ok := gradients[group.Name]
		if !ok {
			clear(destination)
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
