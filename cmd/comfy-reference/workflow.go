package main

import (
	"encoding/json"
	"fmt"
)

// uiWorkflow is a ComfyUI workflow as its page saves it: the top graph's nodes and links, and the subgraphs nodes of it
// instantiate by naming one as their type.
type uiWorkflow struct {
	Nodes       []*uiNode `json:"nodes"`
	Links       []uiLink  `json:"links"`
	Definitions struct {
		Subgraphs []uiSubgraph `json:"subgraphs"`
	} `json:"definitions"`
}

// uiSubgraph is a subgraph's definition: its nodes and links, its inputs fed by the slot of an instance's input of the
// same place (subgraphInput) and its outputs read at an instance's output of the same place (subgraphOutput).
type uiSubgraph struct {
	ID    string    `json:"id"`
	Nodes []*uiNode `json:"nodes"`
	Links []uiLink  `json:"links"`
}

// uiNode is one node as the page saves it: its mode (modeBypass passes inputs through, modeMuted drops it), its slots, and
// the values of its widgets in the order its page holds them.
type uiNode struct {
	ID            int             `json:"id"`
	Type          string          `json:"type"`
	Mode          int             `json:"mode"`
	Inputs        []uiSlot        `json:"inputs"`
	Outputs       []uiSlot        `json:"outputs"`
	WidgetsValues json.RawMessage `json:"widgets_values"`
}

// uiSlot is an input or output of a node; an input's link names where it is fed from, and its widget the widget it
// stands in for when a link feeds a value the node would otherwise hold.
type uiSlot struct {
	Name   string `json:"name"`
	Type   string `json:"type"`
	Link   *int   `json:"link"`
	Widget *struct {
		Name string `json:"name"`
	} `json:"widget"`
}

// uiLink joins an output of one node to an input of another.
type uiLink struct {
	ID, OriginID, OriginSlot, TargetID, TargetSlot int
	Type                                           any
}

const (
	// modeBypass and modeMuted are the modes a node does not run in: bypassed, its inputs pass to its outputs; muted, it
	// gives nothing.
	modeMuted, modeBypass = 2, 4
	// subgraphInput and subgraphOutput are the nodes a subgraph's links name for its inputs and its outputs.
	subgraphInput, subgraphOutput = -10, -20
	// linkFields is how many fields the top graph writes a link as: its id, origin node and slot, target node and slot,
	// and type.
	linkFields = 6
)

// UnmarshalJSON reads a link in either form a page writes: an array at the top graph, an object in a subgraph.
func (l *uiLink) UnmarshalJSON(data []byte) error {
	var fields []any
	if json.Unmarshal(data, &fields) == nil {
		// The array holds the numbers in this order, then the type.
		numbers := []*int{&l.ID, &l.OriginID, &l.OriginSlot, &l.TargetID, &l.TargetSlot}
		if len(fields) != linkFields || len(numbers)+1 != linkFields {
			return fmt.Errorf("comfy-reference: a link of %d fields, not %d", len(fields), linkFields)
		}
		for index, number := range numbers {
			read, isNumber := fields[index].(float64)
			if !isNumber {
				return fmt.Errorf("comfy-reference: a link's field %d is %v, not a number", index, fields[index])
			}
			*number = int(read)
		}
		l.Type = fields[len(numbers)]
		return nil
	}
	var object struct {
		ID         int `json:"id"`
		OriginID   int `json:"origin_id"`
		OriginSlot int `json:"origin_slot"`
		TargetID   int `json:"target_id"`
		TargetSlot int `json:"target_slot"`
		Type       any `json:"type"`
	}
	if err := json.Unmarshal(data, &object); err != nil {
		return err
	}
	*l = uiLink{object.ID, object.OriginID, object.OriginSlot, object.TargetID, object.TargetSlot, object.Type}
	return nil
}

// nodeInfo is what ComfyUI's /object_info says of a node class: its inputs by kind, and their order.
type nodeInfo struct {
	Input struct {
		Required map[string][]json.RawMessage `json:"required"`
		Optional map[string][]json.RawMessage `json:"optional"`
	} `json:"input"`
	InputOrder struct {
		Required []string `json:"required"`
		Optional []string `json:"optional"`
	} `json:"input_order"`
}
