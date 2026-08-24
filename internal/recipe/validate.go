package recipe

import (
	"errors"
	"fmt"
	"slices"
)

type nodeContract struct {
	node   Node
	module Module
	inputs map[PortName]Port
	output map[PortName]Port
}

func (d Definition) Validate(catalog *Catalog) error {
	if err := d.ValidateIdentity(); err != nil {
		return err
	}
	if catalog == nil {
		return errors.New("recipe: nil module catalog")
	}
	nodes := make(map[NodeID]nodeContract, len(d.Nodes))
	for _, node := range d.Nodes {
		if !node.Session.Valid() {
			return fmt.Errorf("recipe: node %q has invalid session policy %q", node.ID, node.Session)
		}
		if !node.Residency.Valid() {
			return fmt.Errorf("recipe: node %q has invalid residency policy %q", node.ID, node.Residency)
		}
		module, ok := catalog.Module(node.Module)
		if !ok {
			return fmt.Errorf("recipe: node %q uses unknown module %q", node.ID, node.Module)
		}
		if !slices.Contains(module.Tasks, d.Task) {
			return fmt.Errorf("recipe: module %q does not admit task %q", module.ID, d.Task)
		}
		if !slices.Contains(module.Placements, node.Placement) {
			return fmt.Errorf("recipe: module %q does not admit placement %q", module.ID, node.Placement)
		}
		contract := nodeContract{node: node, module: module, inputs: map[PortName]Port{}, output: map[PortName]Port{}}
		for _, port := range module.Inputs {
			contract.inputs[port.Name] = port
		}
		for _, port := range module.Outputs {
			contract.output[port.Name] = port
		}
		nodes[node.ID] = contract
	}
	bound := map[Endpoint]int{}
	adjacency := map[NodeID][]NodeID{}
	reverse := map[NodeID][]NodeID{}
	for _, edge := range d.Edges {
		from, to, err := edgePorts(nodes, edge)
		if err != nil {
			return err
		}
		if from.Data != to.Data {
			return fmt.Errorf("recipe: edge %s.%s -> %s.%s changes %s to %s", edge.From.Node, edge.From.Port, edge.To.Node, edge.To.Port, from.Data, to.Data)
		}
		bound[edge.To]++
		adjacency[edge.From.Node] = append(adjacency[edge.From.Node], edge.To.Node)
		reverse[edge.To.Node] = append(reverse[edge.To.Node], edge.From.Node)
	}
	for _, input := range d.Inputs {
		target, err := inputPort(nodes, input.Target)
		if err != nil {
			return err
		}
		if input.Data != target.Data {
			return fmt.Errorf("recipe: input %q schema %s does not match %s", input.Name, input.Data, target.Data)
		}
		bound[input.Target]++
	}
	for endpoint, count := range bound {
		port := nodes[endpoint.Node].inputs[endpoint.Port]
		if port.Cardinality != CardinalityMany && count > 1 {
			return fmt.Errorf("recipe: input %s.%s has %d producers", endpoint.Node, endpoint.Port, count)
		}
	}
	for _, contract := range nodes {
		for _, port := range contract.module.Inputs {
			endpoint := Endpoint{Node: contract.node.ID, Port: port.Name}
			if (port.Cardinality == CardinalityOne || port.Cardinality == CardinalityOneOrMany) &&
				bound[endpoint] == 0 {
				return fmt.Errorf("recipe: required input %s.%s is unbound", endpoint.Node, endpoint.Port)
			}
		}
	}
	outputNodes := map[NodeID]struct{}{}
	for _, output := range d.Outputs {
		port, err := outputPort(nodes, output.Source)
		if err != nil {
			return err
		}
		if output.Data != port.Data {
			return fmt.Errorf("recipe: output %q schema %s does not match %s", output.Name, output.Data, port.Data)
		}
		outputNodes[output.Source.Node] = struct{}{}
	}
	if _, _, err := executionOrder(d); err != nil {
		return err
	}
	if err := validateReachability(d, nodes, adjacency, reverse, outputNodes); err != nil {
		return err
	}
	return nil
}

func edgePorts(nodes map[NodeID]nodeContract, edge Edge) (Port, Port, error) {
	from, err := outputPort(nodes, edge.From)
	if err != nil {
		return Port{}, Port{}, err
	}
	to, err := inputPort(nodes, edge.To)
	return from, to, err
}

func inputPort(nodes map[NodeID]nodeContract, endpoint Endpoint) (Port, error) {
	node, ok := nodes[endpoint.Node]
	if !ok {
		return Port{}, fmt.Errorf("recipe: unknown node %q", endpoint.Node)
	}
	port, ok := node.inputs[endpoint.Port]
	if !ok {
		return Port{}, fmt.Errorf("recipe: unknown input %s.%s", endpoint.Node, endpoint.Port)
	}
	return port, nil
}

func outputPort(nodes map[NodeID]nodeContract, endpoint Endpoint) (Port, error) {
	node, ok := nodes[endpoint.Node]
	if !ok {
		return Port{}, fmt.Errorf("recipe: unknown node %q", endpoint.Node)
	}
	port, ok := node.output[endpoint.Port]
	if !ok {
		return Port{}, fmt.Errorf("recipe: unknown output %s.%s", endpoint.Node, endpoint.Port)
	}
	return port, nil
}

func validateReachability(
	d Definition,
	nodes map[NodeID]nodeContract,
	adjacency, reverse map[NodeID][]NodeID,
	outputNodes map[NodeID]struct{},
) error {
	forward := map[NodeID]struct{}{}
	queue := make([]NodeID, 0)
	for _, input := range d.Inputs {
		queue = append(queue, input.Target.Node)
	}
	for id, contract := range nodes {
		if len(contract.module.Inputs) == 0 {
			queue = append(queue, id)
		}
	}
	visitNodes(queue, adjacency, forward)
	backward := map[NodeID]struct{}{}
	queue = queue[:0]
	for id := range outputNodes {
		queue = append(queue, id)
	}
	visitNodes(queue, reverse, backward)
	for _, node := range d.Nodes {
		if _, ok := forward[node.ID]; !ok {
			return fmt.Errorf("recipe: node %q is unreachable from graph input", node.ID)
		}
		if _, ok := backward[node.ID]; !ok {
			return fmt.Errorf("recipe: node %q cannot reach graph output", node.ID)
		}
	}
	return nil
}

func visitNodes(queue []NodeID, adjacency map[NodeID][]NodeID, visited map[NodeID]struct{}) {
	for len(queue) > 0 {
		node := queue[len(queue)-1]
		queue = queue[:len(queue)-1]
		if _, ok := visited[node]; ok {
			continue
		}
		visited[node] = struct{}{}
		queue = append(queue, adjacency[node]...)
	}
}
