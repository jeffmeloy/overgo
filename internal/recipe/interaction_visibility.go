package recipe

import (
	"cmp"
	"slices"

	"overgo/internal/artifact"
)

// Activation identifies one immutable recipe-node event.
type Activation struct {
	Recipe artifact.ID
	Node   NodeID
	Event  artifact.ID
}

// InteractionScope binds context publication to one compiled recipe node.
type InteractionScope struct {
	Node       NodeID
	Visibility InteractionVisibility
}

// Valid reports whether scope identity is complete.
func (s InteractionScope) Valid() bool {
	return s.Visibility.recipe.Kind() == artifact.KindRecipe && s.Visibility.contains(s.Node)
}

// InteractionVisibility owns reflexive recipe ancestry.
type InteractionVisibility struct {
	recipe artifact.ID
	nodes  []interactionVisibilityNode
}

type interactionVisibilityNode struct {
	id        NodeID
	ancestors []NodeID
}

// Allows reports exact recipe and ancestor membership.
func (v InteractionVisibility) Allows(current, candidate Activation) bool {
	if current.Recipe != v.recipe || candidate.Recipe != v.recipe ||
		current.Event.Kind() != artifact.KindEvidence || candidate.Event.Kind() != artifact.KindEvidence {
		return false
	}
	index, found := slices.BinarySearchFunc(v.nodes, current.Node, func(node interactionVisibilityNode, id NodeID) int {
		return compareNodeID(node.id, id)
	})
	return found && slices.Contains(v.nodes[index].ancestors, candidate.Node)
}

func (v InteractionVisibility) contains(node NodeID) bool {
	_, found := slices.BinarySearchFunc(v.nodes, node, func(item interactionVisibilityNode, id NodeID) int {
		return compareNodeID(item.id, id)
	})
	return found
}

func compareNodeID(left, right NodeID) int {
	return cmp.Compare(left, right)
}

func compileInteractionVisibility(definition Definition, ordered []Node) InteractionVisibility {
	reverse := make(map[NodeID][]NodeID, len(ordered))
	for _, edge := range definition.Edges {
		reverse[edge.To.Node] = append(reverse[edge.To.Node], edge.From.Node)
	}
	nodes := make([]interactionVisibilityNode, len(ordered))
	for index, node := range ordered {
		seen := map[NodeID]struct{}{}
		visitNodes([]NodeID{node.ID}, reverse, seen)
		ancestors := make([]NodeID, 0, len(seen))
		for id := range seen {
			ancestors = append(ancestors, id)
		}
		slices.Sort(ancestors)
		nodes[index] = interactionVisibilityNode{id: node.ID, ancestors: ancestors}
	}
	slices.SortFunc(nodes, func(left, right interactionVisibilityNode) int {
		return compareNodeID(left.id, right.id)
	})
	return InteractionVisibility{recipe: definition.ID, nodes: nodes}
}
