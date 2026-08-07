package planner

import (
	"errors"
	"fmt"
	"math"
	"sort"

	"overgo/internal/checked"
	"overgo/internal/tensor"
)

// Allocation: arena range assigned to intermediate tensor
type Allocation struct {
	Tensor *tensor.Tensor
	Offset uint64
	Size   uint64
	First  int
	Last   int
}

// Plan: deterministic single-arena allocation plan
type Plan struct {
	ArenaSize   uint64
	Alignment   uint64
	Allocations map[*tensor.Tensor]Allocation
}

type freeBlock struct {
	offset uint64
	size   uint64
}

// Build assigns storage to non-input graph nodes using graph liveness.
func Build(outputs []*tensor.Tensor, alignment uint64) (Plan, error) {
	return BuildWithDependencies(outputs, alignment, nil)
}

// BuildWithDependencies extends graph liveness for rewritten consumers.
func BuildWithDependencies(
	outputs []*tensor.Tensor,
	alignment uint64,
	dependencies map[*tensor.Tensor][]*tensor.Tensor,
) (Plan, error) {
	return BuildWithRewrites(outputs, alignment, dependencies, nil, nil)
}

// BuildWithRewrites extends liveness for fused dependencies, aliases, and eliminated nodes.
func BuildWithRewrites(
	outputs []*tensor.Tensor,
	alignment uint64,
	dependencies map[*tensor.Tensor][]*tensor.Tensor,
	aliases map[*tensor.Tensor]*tensor.Tensor,
	excluded map[*tensor.Tensor]struct{},
) (Plan, error) {
	if alignment == 0 || alignment&(alignment-1) != 0 {
		return Plan{}, fmt.Errorf("planner alignment %d is not a power of two", alignment)
	}
	nodes, err := tensor.Topological(outputs...)
	if err != nil {
		return Plan{}, err
	}
	indexes := make(map[*tensor.Tensor]int, len(nodes))
	roots := make(map[*tensor.Tensor]*tensor.Tensor, len(nodes))
	lastUse := make(map[*tensor.Tensor]int, len(nodes))
	for index, node := range nodes {
		indexes[node] = index
		root := node
		alias := aliases[node]
		if node.Op == tensor.OpReshape || alias != nil {
			if alias == nil && len(node.Inputs) == 1 {
				alias = node.Inputs[0]
			}
			if alias == nil || roots[alias] == nil {
				return Plan{}, fmt.Errorf("alias tensor %d has invalid storage root", node.ID)
			}
			root = roots[alias]
		}
		roots[node] = root
		lastUse[root] = index
	}
	for index, node := range nodes {
		for _, input := range node.Inputs {
			root := roots[input]
			if index > lastUse[root] {
				lastUse[root] = index
			}
		}
	}
	for consumer, inputs := range dependencies {
		index, ok := indexes[consumer]
		if !ok {
			return Plan{}, fmt.Errorf("dependency consumer tensor %d is outside graph", consumer.ID)
		}
		for _, input := range inputs {
			root, ok := roots[input]
			if !ok {
				return Plan{}, fmt.Errorf("dependency input tensor %d is outside graph", input.ID)
			}
			if index > lastUse[root] {
				lastUse[root] = index
			}
		}
	}
	for _, output := range outputs {
		lastUse[roots[output]] = len(nodes)
	}

	plan := Plan{
		Alignment:   alignment,
		Allocations: make(map[*tensor.Tensor]Allocation),
	}
	var active []Allocation
	var free []freeBlock
	for index, node := range nodes {
		kept := active[:0]
		for _, allocation := range active {
			if allocation.Last < index {
				free = append(free, freeBlock{offset: allocation.Offset, size: allocation.Size})
			} else {
				kept = append(kept, allocation)
			}
		}
		active = kept
		free = coalesce(free)

		if node.Op == tensor.OpInput || node.Op == tensor.OpReshape || aliases[node] != nil {
			continue
		}
		if _, skip := excluded[node]; skip {
			continue
		}
		size, err := node.Shape.Bytes(node.Type)
		if err != nil {
			return Plan{}, fmt.Errorf("size tensor %d: %w", node.ID, err)
		}
		size, ok := checked.Align(size, alignment)
		if !ok {
			return Plan{}, fmt.Errorf("aligned size for tensor %d overflows uint64", node.ID)
		}
		offset, remaining, found := takeBestFit(free, size)
		if found {
			free = remaining
		} else {
			offset, ok = checked.Align(plan.ArenaSize, alignment)
			if !ok || offset > math.MaxUint64-size {
				return Plan{}, errors.New("tensor arena size overflows uint64")
			}
			plan.ArenaSize = offset + size
		}
		allocation := Allocation{
			Tensor: node,
			Offset: offset,
			Size:   size,
			First:  indexes[node],
			Last:   lastUse[node],
		}
		plan.Allocations[node] = allocation
		active = append(active, allocation)
	}
	for _, node := range nodes {
		if node.Op != tensor.OpReshape && aliases[node] == nil {
			continue
		}
		if allocation, ok := plan.Allocations[roots[node]]; ok {
			allocation.Tensor = node
			plan.Allocations[node] = allocation
		}
	}
	return plan, nil
}

func takeBestFit(blocks []freeBlock, size uint64) (uint64, []freeBlock, bool) {
	best := -1
	for index, block := range blocks {
		if block.size >= size && (best == -1 || block.size < blocks[best].size) {
			best = index
		}
	}
	if best == -1 {
		return 0, blocks, false
	}
	block := blocks[best]
	if block.size == size {
		blocks = append(blocks[:best], blocks[best+1:]...)
	} else {
		blocks[best].offset += size
		blocks[best].size -= size
	}
	return block.offset, blocks, true
}

func coalesce(blocks []freeBlock) []freeBlock {
	if len(blocks) < 2 {
		return blocks
	}
	sort.Slice(blocks, func(i, j int) bool { return blocks[i].offset < blocks[j].offset })
	result := blocks[:1]
	for _, block := range blocks[1:] {
		last := &result[len(result)-1]
		if last.offset+last.size == block.offset {
			last.size += block.size
		} else {
			result = append(result, block)
		}
	}
	return result
}
