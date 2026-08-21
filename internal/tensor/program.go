package tensor

import (
	"errors"
	"fmt"
	"slices"
)

// Program: one validated neutral tensor graph shared by execution backends.
type Program struct {
	outputs  []*Tensor
	order    []*Tensor
	backends OperationBackend
}

// CompileProgram validates graph structure and derives its common backend set.
func CompileProgram(outputs ...*Tensor) (Program, error) {
	var noBackend OperationBackend
	order, err := topological(outputs...)
	if err != nil {
		return Program{}, err
	}
	backends := allExecutionBackends
	for _, node := range order {
		if node.Op == OpInput {
			continue
		}
		descriptor, ok := DescribeOperation(node.Op)
		if !ok {
			return Program{}, fmt.Errorf("tensor %d has no operation contract", node.ID)
		}
		backends &= descriptor.Backends
	}
	if backends == noBackend {
		return Program{}, errors.New("tensor program has no common execution backend")
	}
	return Program{
		outputs:  slices.Clone(outputs),
		order:    slices.Clone(order),
		backends: backends,
	}, nil
}

// Outputs returns the validated result nodes.
func (p Program) Outputs() []*Tensor { return slices.Clone(p.outputs) }

// Order returns the validated dependency order.
func (p Program) Order() []*Tensor { return slices.Clone(p.order) }

// RequireBackend rejects execution by a backend not shared by every operation.
func (p Program) RequireBackend(backend OperationBackend) error {
	if backend != BackendReference && backend != BackendCUDA {
		return errors.New("tensor program backend request is invalid")
	}
	var noBackend OperationBackend
	if p.backends&backend == noBackend {
		return fmt.Errorf("tensor program does not support backend %d", backend)
	}
	return nil
}
