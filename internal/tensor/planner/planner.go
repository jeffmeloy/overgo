package planner

import (
	"errors"
	"fmt"
	"math"
	"sort"

	"llamacpp2go/internal/tensor"
)

// Allocation is an arena range assigned to an intermediate tensor.
type Allocation struct {
	Tensor *tensor.Tensor
	Offset uint64
	Size   uint64
	First  int
	Last   int
}

// Plan is a deterministic single-arena allocation plan.
type Plan struct {
	ArenaSize   uint64
	Alignment   uint64
	Allocations map[*tensor.Tensor]Allocation
}

type freeBlock struct {
	offset uint64
	size   uint64
}

// Build assigns storage to non-input graph nodes using last-use liveness.
func Build(outputs []*tensor.Tensor, alignment uint64) (Plan, error) {
	if alignment == 0 || alignment&(alignment-1) != 0 {
		return Plan{}, fmt.Errorf("planner alignment %d is not a power of two", alignment)
	}
	nodes, err := tensor.Topological(outputs...)
	if err != nil {
		return Plan{}, err
	}
	indexes := make(map[*tensor.Tensor]int, len(nodes))
	lastUse := make(map[*tensor.Tensor]int, len(nodes))
	for index, node := range nodes {
		indexes[node] = index
		lastUse[node] = index
	}
	for index, node := range nodes {
		for _, input := range node.Inputs {
			if index > lastUse[input] {
				lastUse[input] = index
			}
		}
	}
	for _, output := range outputs {
		lastUse[output] = len(nodes)
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

		if node.Op == tensor.OpInput {
			continue
		}
		size, err := node.Shape.Bytes(node.Type)
		if err != nil {
			return Plan{}, fmt.Errorf("size tensor %d: %w", node.ID, err)
		}
		size, overflow := alignUp(size, alignment)
		if overflow {
			return Plan{}, fmt.Errorf("aligned size for tensor %d overflows uint64", node.ID)
		}
		offset, remaining, found := takeBestFit(free, size)
		if found {
			free = remaining
		} else {
			offset, overflow = alignUp(plan.ArenaSize, alignment)
			if overflow || offset > math.MaxUint64-size {
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

func alignUp(value, alignment uint64) (uint64, bool) {
	mask := alignment - 1
	if value > math.MaxUint64-mask {
		return 0, true
	}
	return (value + mask) &^ mask, false
}
