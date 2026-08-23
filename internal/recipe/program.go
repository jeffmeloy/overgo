package recipe

import (
	"errors"
	"slices"
)

// Stage defines validated node plus its immutable module contract.
type Stage struct {
	Node   Node   `json:"node"`
	Module Module `json:"module"`
}

// Program defines catalog-resolved executable recipe graph.
type Program struct {
	definition Definition
	stages     []Stage
	readySets  [][]Stage
	visibility InteractionVisibility
	catalog    *Catalog
}

// Definition returns an isolated recipe copy.
func (p Program) Definition() Definition { return cloneProgramDefinition(p.definition) }

// Stages returns isolated compiled-stage contracts.
func (p Program) Stages() []Stage {
	return cloneStages(p.stages)
}

// ReadySets returns deterministic dependency levels eligible for concurrent execution.
func (p Program) ReadySets() [][]Stage {
	sets := make([][]Stage, len(p.readySets))
	for index := range p.readySets {
		sets[index] = cloneStages(p.readySets[index])
	}
	return sets
}

// UsesCatalog reports the immutable module authority used during compilation.
func (p Program) UsesCatalog(catalog *Catalog) bool { return catalog != nil && p.catalog == catalog }

// Catalog returns the immutable module authority used during compilation.
func (p Program) Catalog() *Catalog { return p.catalog }

// InteractionScope binds one node to compiled graph visibility.
func (p Program) InteractionScope(node NodeID) (InteractionScope, error) {
	if !p.visibility.contains(node) {
		return InteractionScope{}, errors.New("recipe: interaction node is absent")
	}
	return InteractionScope{Node: node, Visibility: p.visibility}, nil
}

// CompileProgram resolves module contracts and orders executable stages.
func CompileProgram(definition Definition, catalog *Catalog) (Program, error) {
	if err := definition.Validate(catalog); err != nil {
		return Program{}, err
	}
	ordered, ready, err := executionOrder(definition)
	if err != nil {
		return Program{}, err
	}
	stages := make([]Stage, len(ordered))
	for index, node := range ordered {
		module, ok := catalog.Module(node.Module)
		if !ok {
			return Program{}, errors.New("recipe: compiled node lacks module contract")
		}
		stages[index] = Stage{Node: node, Module: module}
	}
	stageByNode := make(map[NodeID]Stage, len(stages))
	for _, stage := range stages {
		stageByNode[stage.Node.ID] = stage
	}
	readySets := make([][]Stage, len(ready))
	for setIndex, nodes := range ready {
		readySets[setIndex] = make([]Stage, len(nodes))
		for nodeIndex, node := range nodes {
			readySets[setIndex][nodeIndex] = stageByNode[node.ID]
		}
	}
	return Program{
		definition: cloneProgramDefinition(definition), stages: stages, readySets: readySets,
		visibility: compileInteractionVisibility(definition, ordered), catalog: catalog,
	}, nil
}

func cloneProgramDefinition(definition Definition) Definition {
	definition.Dependencies = slices.Clone(definition.Dependencies)
	definition.Nodes = slices.Clone(definition.Nodes)
	definition.Edges = slices.Clone(definition.Edges)
	definition.Inputs = slices.Clone(definition.Inputs)
	definition.Outputs = slices.Clone(definition.Outputs)
	return definition
}

func executionOrder(definition Definition) ([]Node, [][]Node, error) {
	var none int
	nodes := make(map[NodeID]Node, len(definition.Nodes))
	indegree := make(map[NodeID]int, len(definition.Nodes))
	adjacency := make(map[NodeID][]NodeID, len(definition.Nodes))
	for _, node := range definition.Nodes {
		nodes[node.ID] = node
		indegree[node.ID] = none
	}
	for _, edge := range definition.Edges {
		adjacency[edge.From.Node] = append(adjacency[edge.From.Node], edge.To.Node)
		indegree[edge.To.Node]++
	}
	ready := make([]NodeID, none, len(nodes))
	for id, count := range indegree {
		if count == none {
			ready = append(ready, id)
		}
	}
	slices.Sort(ready)
	steps := make([]Node, none, len(nodes))
	sets := make([][]Node, none, len(nodes))
	for len(ready) > none {
		current := slices.Clone(ready)
		ready = ready[:none]
		set := make([]Node, len(current))
		for index, id := range current {
			set[index] = nodes[id]
			steps = append(steps, nodes[id])
			for _, target := range adjacency[id] {
				indegree[target]--
				if indegree[target] == none {
					ready = append(ready, target)
				}
			}
		}
		sets = append(sets, set)
		slices.Sort(ready)
	}
	if len(steps) != len(nodes) {
		return nil, nil, errors.New("recipe: execution graph contains cycle")
	}
	return steps, sets, nil
}

func cloneStages(stages []Stage) []Stage {
	stages = slices.Clone(stages)
	for index := range stages {
		stages[index].Module = cloneModule(stages[index].Module)
	}
	return stages
}
