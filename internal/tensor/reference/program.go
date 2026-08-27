package reference

import (
	"errors"
	"fmt"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

const noStorageViewInput = -1

type compiledNode struct {
	node       *tensor.Tensor
	operands   []int
	viewInput  int
	viewOffset uint64
	input      int
	isInput    bool
}

// Program owns indexed reference execution.
type Program struct {
	nodes      []compiledNode
	inputSlots map[*tensor.Tensor]int
	outputs    []*tensor.Tensor
	outputSlot []int
	inputs     int
	operands   int
}

// Inputs owns indexed reference feeds.
type Inputs struct {
	program *Program
	values  []Value
	bound   []bool
}

// Workspace owns reusable execution slices.
type Workspace struct {
	program  *Program
	values   []Value
	operands []Value
}

// Compile validates and indexes one reference graph.
func Compile(outputs ...*tensor.Tensor) (*Program, error) {
	program, err := tensor.CompileProgram(outputs...)
	if err != nil {
		return nil, err
	}
	return CompileProgram(program)
}

// CompileProgram indexes one validated neutral program.
func CompileProgram(program tensor.Program) (*Program, error) {
	if err := program.RequireBackend(tensor.BackendReference); err != nil {
		return nil, err
	}
	order := program.Order()
	indexes := make(map[*tensor.Tensor]int, len(order))
	compiled := &Program{
		nodes: make([]compiledNode, len(order)), inputSlots: make(map[*tensor.Tensor]int),
		outputs: program.Outputs(),
	}
	for index, node := range order {
		if node.Type != dtype.F32 {
			return nil, fmt.Errorf("reference executor does not support %s for tensor %d", node.Type, node.ID)
		}
		indexes[node] = index
		view, aliases, viewErr := tensor.ResolveStorageView(node)
		if viewErr != nil {
			return nil, viewErr
		}
		compiled.nodes[index] = compiledNode{node: node, viewInput: noStorageViewInput}
		if aliases {
			compiled.nodes[index].viewInput = view.Input
			compiled.nodes[index].viewOffset = view.ElementOffset
		}
		if node.Op == tensor.OpInput {
			compiled.nodes[index].input = compiled.inputs
			compiled.nodes[index].isInput = true
			compiled.inputSlots[node] = compiled.inputs
			compiled.inputs++
		}
	}
	for index, node := range order {
		operands := make([]int, len(node.Inputs))
		for input, operand := range node.Inputs {
			operands[input] = indexes[operand]
		}
		compiled.nodes[index].operands = operands
		compiled.operands = max(compiled.operands, len(operands))
	}
	compiled.outputSlot = make([]int, len(compiled.outputs))
	for index, output := range compiled.outputs {
		compiled.outputSlot[index] = indexes[output]
	}
	return compiled, nil
}

// NewInputs allocates one indexed feed set.
func (p *Program) NewInputs() *Inputs {
	if p == nil {
		return nil
	}
	return &Inputs{program: p, values: make([]Value, p.inputs), bound: make([]bool, p.inputs)}
}

// HasInput reports graph membership.
func (p *Program) HasInput(node *tensor.Tensor) bool {
	_, ok := p.inputSlots[node]
	return ok
}

// Set binds one compiled input.
func (i *Inputs) Set(node *tensor.Tensor, value Value) error {
	if i == nil || i.program == nil {
		return errors.New("reference inputs are unavailable")
	}
	if node == nil {
		return errors.New("reference input is nil")
	}
	slot, ok := i.program.inputSlots[node]
	if !ok {
		return fmt.Errorf("reference input %q is not compiled", node.Name)
	}
	i.values[slot], i.bound[slot] = value, true
	return nil
}

// Bind resolves ordered feeds into compiled slots.
func (i *Inputs) Bind(bindings tensor.InputBindings[Value]) error {
	for _, binding := range bindings {
		if err := i.Set(binding.Node, binding.Value); err != nil {
			return err
		}
	}
	return nil
}

// NewWorkspace allocates reusable execution headers.
func (p *Program) NewWorkspace() *Workspace {
	if p == nil {
		return nil
	}
	return &Workspace{program: p, values: make([]Value, len(p.nodes)), operands: make([]Value, p.operands)}
}

// Execute evaluates one indexed feed set.
func (p *Program) Execute(inputs *Inputs, workspace *Workspace) (map[*tensor.Tensor]Value, error) {
	if p == nil || inputs == nil || inputs.program != p || workspace == nil || workspace.program != p {
		return nil, errors.New("reference execution contract differs")
	}
	clear(workspace.values)
	defer clear(workspace.values)
	defer clear(workspace.operands)
	for index, compiled := range p.nodes {
		node := compiled.node
		if compiled.isInput {
			value, ok := inputs.values[compiled.input], inputs.bound[compiled.input]
			if !ok {
				if embedded, embeddedOK := node.Attrs.(tensor.EmbeddedInputAttributes); embeddedOK {
					value, ok = Value{Shape: node.Shape, Data: embedded.Data}, true
				}
			}
			if !ok {
				return nil, fmt.Errorf("missing feed for input %q", node.Name)
			}
			if !value.Shape.Equal(node.Shape) {
				return nil, fmt.Errorf("feed shape for %q does not match graph", node.Name)
			}
			materialized, err := value.materialized()
			if err != nil {
				return nil, fmt.Errorf("feed storage for %q: %w", node.Name, err)
			}
			workspace.values[index] = materialized
			continue
		}
		operands := workspace.operands[:len(compiled.operands)]
		for operand, slot := range compiled.operands {
			operands[operand] = workspace.values[slot]
		}
		if compiled.viewInput != noStorageViewInput {
			value, viewErr := materializeStorageView(
				node.Shape, operands[compiled.viewInput], compiled.viewOffset,
			)
			if viewErr != nil {
				return nil, fmt.Errorf("execute tensor %d (%s): %w", node.ID, node.Op, viewErr)
			}
			workspace.values[index] = value
			continue
		}
		value, err := executeNode(node, operands)
		if err != nil {
			return nil, fmt.Errorf("execute tensor %d (%s): %w", node.ID, node.Op, err)
		}
		workspace.values[index] = value
	}
	results := make(map[*tensor.Tensor]Value, len(p.outputs))
	for index, output := range p.outputs {
		results[output] = workspace.values[p.outputSlot[index]]
	}
	return results, nil
}
