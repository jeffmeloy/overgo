package inference

import (
	"errors"
	"fmt"

	"overgo/internal/cuda/executor"
	"overgo/internal/model"
	"overgo/internal/tensor"
)

type decodeDynamicKind uint8

const (
	decodeDynamicTokenRows decodeDynamicKind = iota
	decodeDynamicPositions
	decodeDynamicAttention
	decodeDynamicCacheAppend
)

type decodeDynamicSlot struct {
	node      *tensor.Tensor
	branch    int
	kind      decodeDynamicKind
	value     tensor.Attributes
	positions [4][]uint32
	attention *tensor.AttentionAttributes
	append    *tensor.CacheAppendAttributes
}

type decodeSessionIdentity struct {
	capacity   uint32
	branches   uint32
	tokenCount uint32
	output     deviceOutputPlan
	lora       [32]byte
}

func (i decodeSessionIdentity) matches(
	capacity, tokenCount uint32,
	output deviceOutputPlan,
	lora [32]byte,
) bool {
	return i.capacity == capacity &&
		i.branches > 0 && i.tokenCount == tokenCount && i.output == output && i.lora == lora
}

type decodeSessionBranchPlan struct {
	cacheInputs []decodeCacheInputPlan
	feedback    decodeInputSlot
}

type decodeInputSlot struct {
	slot    executor.InputSlot
	present bool
}

type decodeStateInputSlot struct {
	name model.CacheStateName
	slot executor.InputSlot
}

type decodeCacheInputPlan struct {
	key, value decodeInputSlot
	states     []decodeStateInputSlot
}

type decodeSessionPlan struct {
	identity   decodeSessionIdentity
	targets    []deviceCacheTargetPlan
	attributes *executor.RuntimeAttributes
	dynamic    []decodeDynamicSlot
	branches   []decodeSessionBranchPlan
}

func compileDecodeSessionPlan(
	compiled *executor.CompiledGraph,
	graphs []deviceBatchGraph,
	targets []deviceCacheTargetPlan,
	identity decodeSessionIdentity,
) (decodeSessionPlan, error) {
	if compiled == nil || identity.capacity == 0 || identity.branches == 0 || identity.tokenCount == 0 ||
		len(graphs) != int(identity.branches) || len(targets) != len(graphs) {
		return decodeSessionPlan{}, errors.New("inference: decode session identity is invalid")
	}
	plan := decodeSessionPlan{
		identity: identity, targets: targets, attributes: compiled.NewRuntimeAttributes(),
		branches: make([]decodeSessionBranchPlan, len(graphs)),
	}
	positionRow := func(graph deviceBatchGraph, node *tensor.Tensor) bool {
		for _, candidate := range graph.positionRows {
			if candidate == node {
				return true
			}
		}
		return false
	}
	appendSlot := func(slot decodeDynamicSlot) error {
		if err := plan.attributes.Set(slot.node, slot.value); err != nil {
			return err
		}
		plan.dynamic = append(plan.dynamic, slot)
		return nil
	}
	for branch, graph := range graphs {
		if graph.feedback == nil && graph.tokenInput == nil {
			return decodeSessionPlan{}, errors.New("inference: decode session has no parameterized token input")
		}
		branchPlan := decodeSessionBranchPlan{cacheInputs: make([]decodeCacheInputPlan, len(graph.cacheInputs))}
		var err error
		branchPlan.feedback, err = compileDecodeInputSlot(compiled, graph.feedback)
		if err != nil {
			return decodeSessionPlan{}, err
		}
		for layer, inputs := range graph.cacheInputs {
			compiledInputs := &branchPlan.cacheInputs[layer]
			compiledInputs.key, err = compileDecodeInputSlot(compiled, inputs.key)
			if err != nil {
				return decodeSessionPlan{}, err
			}
			compiledInputs.value, err = compileDecodeInputSlot(compiled, inputs.value)
			if err != nil {
				return decodeSessionPlan{}, err
			}
			for _, name := range inputs.states.SortedNames() {
				slot, slotErr := compileDecodeInputSlot(compiled, inputs.states[name].Value)
				if slotErr != nil {
					return decodeSessionPlan{}, slotErr
				}
				compiledInputs.states = append(compiledInputs.states, decodeStateInputSlot{name: name, slot: slot.slot})
			}
		}
		plan.branches[branch] = branchPlan
		nodes, err := tensor.Topological(decodeGraphOutputs(graph, identity.output)...)
		if err != nil {
			return decodeSessionPlan{}, err
		}
		for _, node := range nodes {
			slot := decodeDynamicSlot{node: node, branch: branch}
			switch node.Op {
			case tensor.OpGetRows:
				if node != graph.tokenInput && !positionRow(graph, node) {
					continue
				}
				value := node.Attrs.(tensor.GetRowsAttributes)
				value.Rows = append([]uint32(nil), value.Rows...)
				slot.value, slot.positions[0] = &value, value.Rows
				if node == graph.tokenInput {
					slot.kind = decodeDynamicTokenRows
				} else {
					slot.kind = decodeDynamicPositions
				}
			case tensor.OpRoPENormal, tensor.OpRoPENeoX:
				value := node.Attrs.(tensor.RoPEAttributes)
				value.Positions = append([]uint32(nil), value.Positions...)
				slot.kind, slot.value, slot.positions[0] = decodeDynamicPositions, &value, value.Positions
			case tensor.OpRoPEMulti:
				value := node.Attrs.(tensor.RoPEMultiAttributes)
				for axis := range value.Positions {
					value.Positions[axis] = append([]uint32(nil), value.Positions[axis]...)
					slot.positions[axis] = value.Positions[axis]
				}
				slot.kind, slot.value = decodeDynamicPositions, &value
			case tensor.OpAttention:
				value := node.Attrs.(tensor.AttentionAttributes)
				slot.kind, slot.value, slot.attention = decodeDynamicAttention, &value, &value
			case tensor.OpCacheAppend:
				value := node.Attrs.(tensor.CacheAppendAttributes)
				slot.kind, slot.value, slot.append = decodeDynamicCacheAppend, &value, &value
			default:
				continue
			}
			if err := appendSlot(slot); err != nil {
				return decodeSessionPlan{}, fmt.Errorf("inference: compile decode slot %s: %w", node.Op, err)
			}
		}
	}
	return plan, nil
}

func compileDecodeInputSlot(compiled *executor.CompiledGraph, node *tensor.Tensor) (decodeInputSlot, error) {
	if node == nil {
		return decodeInputSlot{}, nil
	}
	slot, ok := compiled.InputSlot(node)
	if !ok {
		return decodeInputSlot{}, fmt.Errorf("inference: decode input %q is not compiled", node.Name)
	}
	return decodeInputSlot{slot: slot, present: true}, nil
}

func decodeGraphOutputs(graph deviceBatchGraph, output deviceOutputPlan) []*tensor.Tensor {
	first := output.graphOutput(graph)
	result := []*tensor.Tensor{first}
	// Shared-KV aliases: collect each output once.
	seen := map[*tensor.Tensor]struct{}{first: {}}
	if graph.hidden != nil {
		result = appendUniqueGraphOutputs(result, seen, graph.hidden)
	}
	for layer := range graph.keys {
		result = appendUniqueGraphOutputs(result, seen, graph.states[layer].AppendValues(
			[]*tensor.Tensor{graph.keys[layer], graph.values[layer]},
		)...)
	}
	return result
}

func appendUniqueGraphOutputs(
	outputs []*tensor.Tensor,
	seen map[*tensor.Tensor]struct{},
	candidates ...*tensor.Tensor,
) []*tensor.Tensor {
	for _, node := range candidates {
		if _, duplicate := seen[node]; duplicate {
			continue
		}
		seen[node] = struct{}{}
		outputs = append(outputs, node)
	}
	return outputs
}

func (p *decodeSessionPlan) updateBranch(
	branch int,
	rows []uint32,
	position, pastTokens uint32,
) error {
	if p == nil || p.attributes == nil {
		return errors.New("inference: decode session plan is unavailable")
	}
	if branch < 0 || branch >= len(p.branches) || int(p.identity.tokenCount) != len(rows) {
		return errors.New("inference: decode session branch is invalid")
	}
	if uint64(pastTokens)+uint64(p.identity.tokenCount) > uint64(p.identity.capacity) {
		return errors.New("inference: decode session exceeds capacity")
	}
	for index := range p.dynamic {
		slot := &p.dynamic[index]
		if slot.branch != branch {
			continue
		}
		switch slot.kind {
		case decodeDynamicTokenRows:
			if len(slot.positions[0]) != len(rows) {
				return errors.New("inference: decode token-row count changed")
			}
			copy(slot.positions[0], rows)
		case decodeDynamicPositions:
			for _, target := range slot.positions {
				if len(target) == 0 {
					continue
				}
				if len(target) != len(rows) {
					return errors.New("inference: decode position count changed")
				}
				for offset := range target {
					target[offset] = position + uint32(offset)
				}
			}
		case decodeDynamicAttention:
			slot.attention.QueryStart = pastTokens
			slot.attention.KeyValueTokens = pastTokens + p.identity.tokenCount
		case decodeDynamicCacheAppend:
			slot.append.Offset = pastTokens
		}
	}
	return nil
}
