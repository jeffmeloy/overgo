package tensor

import (
	"errors"
	"fmt"
)

// Topological returns all nodes needed by outputs in dependency order.
func Topological(outputs ...*Tensor) ([]*Tensor, error) {
	if len(outputs) == 0 {
		return nil, errors.New("graph has no outputs")
	}
	const (
		visiting = 1
		visited  = 2
	)
	state := make(map[*Tensor]uint8)
	ids := make(map[uint64]*Tensor)
	nodes := make([]*Tensor, 0)
	var visit func(*Tensor) error
	visit = func(node *Tensor) error {
		if node == nil {
			return errors.New("graph contains a nil tensor")
		}
		switch state[node] {
		case visiting:
			return fmt.Errorf("graph contains a cycle at tensor %d", node.ID)
		case visited:
			return nil
		}
		if previous, exists := ids[node.ID]; exists && previous != node {
			return fmt.Errorf("tensor ID %d is not unique", node.ID)
		}
		ids[node.ID] = node
		state[node] = visiting
		for _, input := range node.Inputs {
			if err := visit(input); err != nil {
				return err
			}
		}
		state[node] = visited
		nodes = append(nodes, node)
		return nil
	}
	for _, output := range outputs {
		if err := visit(output); err != nil {
			return nil, err
		}
	}
	return nodes, nil
}
