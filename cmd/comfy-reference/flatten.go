package main

import (
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// apiNode is a node of ComfyUI's API request: its class and its inputs, each a value or a link [node, slot].
type apiNode struct {
	ClassType string         `json:"class_type"`
	Inputs    map[string]any `json:"inputs"`
}

// scope is one graph the request is read from: the top graph, or a subgraph as one instance of it holds it. Its nodes
// are named in the request by the path of instances to it and their own id ("337:328").
type scope struct {
	path     string
	nodes    map[int]*uiNode
	links    map[int]uiLink
	parent   *scope
	instance *uiNode
	children map[int]*scope
}

// flattener turns a saved workflow into ComfyUI's API request as ComfyUI's page does when a section is switched on.
type flattener struct {
	info map[string]nodeInfo
	top  *scope
	// sets are the KJNodes SetNode nodes by the name a GetNode reads them by, with the scope each lies in.
	sets map[string]setNode
}

type setNode struct {
	node  *uiNode
	scope *scope
}

// newFlattener reads workflow's graphs into scopes, each subgraph once for each instance of it.
func newFlattener(workflow uiWorkflow, info map[string]nodeInfo) (*flattener, error) {
	definitions := map[string]uiSubgraph{}
	for _, definition := range workflow.Definitions.Subgraphs {
		definitions[definition.ID] = definition
	}
	f := &flattener{info: info, sets: map[string]setNode{}}
	var build func(path string, nodes []*uiNode, links []uiLink, parent *scope, instance *uiNode, depth int) (*scope, error)
	build = func(path string, nodes []*uiNode, links []uiLink, parent *scope, instance *uiNode, depth int) (*scope, error) {
		if depth > len(definitions) {
			return nil, fmt.Errorf("comfy-reference: subgraphs instantiate one another without end at %s", path)
		}
		s := &scope{path: path, nodes: map[int]*uiNode{}, links: map[int]uiLink{}, parent: parent, instance: instance, children: map[int]*scope{}}
		for _, link := range links {
			s.links[link.ID] = link
		}
		for _, node := range nodes {
			s.nodes[node.ID] = node
			if node.Type == "SetNode" {
				name, err := firstString(node.WidgetsValues)
				if err != nil {
					return nil, fmt.Errorf("comfy-reference: SetNode %d: %w", node.ID, err)
				}
				f.sets[name] = setNode{node, s}
			}
			if definition, isInstance := definitions[node.Type]; isInstance {
				child, err := build(s.path+strconv.Itoa(node.ID)+":", definition.Nodes, definition.Links, s, node, depth+1)
				if err != nil {
					return nil, err
				}
				s.children[node.ID] = child
			}
		}
		return s, nil
	}
	top, err := build("", workflow.Nodes, workflow.Links, nil, nil, 0)
	f.top = top
	return f, err
}

// switchOn runs the nodes of a section, as ComfyUI's group bypasser does: each member of the top graph, and every node
// inside a member that instantiates a subgraph.
func (f *flattener) switchOn(members []int) error {
	var on func(s *scope, id int)
	on = func(s *scope, id int) {
		s.nodes[id].Mode = 0
		if child := s.children[id]; child != nil {
			for inner := range child.nodes {
				on(child, inner)
			}
		}
	}
	for _, id := range members {
		if f.top.nodes[id] == nil {
			return fmt.Errorf("comfy-reference: the section names node %d, which the workflow lacks", id)
		}
		on(f.top, id)
	}
	return nil
}

// request is the API request of the workflow as it stands: every node ComfyUI runs, in a scope whose instances all run.
func (f *flattener) request() (map[string]apiNode, error) {
	request := map[string]apiNode{}
	var walk func(s *scope) error
	walk = func(s *scope) error {
		ids := make([]int, 0, len(s.nodes))
		for id := range s.nodes {
			ids = append(ids, id)
		}
		slices.Sort(ids)
		for _, id := range ids {
			node := s.nodes[id]
			if node.Mode == modeMuted || node.Mode == modeBypass {
				continue
			}
			if child := s.children[id]; child != nil {
				if err := walk(child); err != nil {
					return err
				}
				continue
			}
			info, known := f.info[node.Type]
			if !known {
				continue
			}
			inputs, err := f.inputs(s, node, info)
			if err != nil {
				return fmt.Errorf("comfy-reference: node %s%d (%s): %w", s.path, id, node.Type, err)
			}
			request[s.path+strconv.Itoa(id)] = apiNode{ClassType: node.Type, Inputs: inputs}
		}
		return nil
	}
	return request, walk(f.top)
}

// link is where the input slot of node in s is fed from in the request, as [node, slot]; false where nothing that runs
// feeds it, which the request leaves out.
func (f *flattener) link(s *scope, slot uiSlot) ([]any, bool, error) {
	if slot.Link == nil {
		return nil, false, nil
	}
	l, held := s.links[*slot.Link]
	if !held {
		return nil, false, fmt.Errorf("the link %d of input %q names no link of its graph", *slot.Link, slot.Name)
	}
	return f.source(s, l.OriginID, l.OriginSlot, 0)
}

// source is the node that runs which feeds output slot of node id in s, followed through subgraph boundaries, bypassed
// nodes, KJNodes Set/Get pairs and reroutes; a link to a node the graph lacks is refused.
func (f *flattener) source(s *scope, id, slot, depth int) ([]any, bool, error) {
	if depth > len(f.sets)+len(s.nodes)+len(s.links)+linkFields {
		return nil, false, fmt.Errorf("the link from node %d of %s goes round without end", id, s.path)
	}
	if id == subgraphInput {
		if s.instance == nil || slot >= len(s.instance.Inputs) {
			return nil, false, fmt.Errorf("a subgraph input %d no instance feeds", slot)
		}
		return f.link(s.parent, s.instance.Inputs[slot])
	}
	node := s.nodes[id]
	if node == nil {
		return nil, false, fmt.Errorf("a link names node %d, which %s lacks", id, scopeName(s.path))
	}
	through := func(input uiSlot) ([]any, bool, error) { return f.link(s, input) }
	switch child := s.children[id]; {
	case node.Mode == modeMuted:
		return nil, false, nil
	case node.Mode == modeBypass:
		if slot >= len(node.Outputs) {
			return nil, false, fmt.Errorf("a link names output %d of bypassed node %d, which has %d", slot, id, len(node.Outputs))
		}
		want := node.Outputs[slot].Type
		if slot < len(node.Inputs) && node.Inputs[slot].Type == want {
			return through(node.Inputs[slot])
		}
		for _, input := range node.Inputs {
			if input.Type == want {
				return through(input)
			}
		}
		return nil, false, nil
	case child != nil:
		for _, l := range child.links {
			if l.TargetID == subgraphOutput && l.TargetSlot == slot {
				return f.source(child, l.OriginID, l.OriginSlot, depth+1)
			}
		}
		return nil, false, nil
	case node.Type == "GetNode":
		name, err := firstString(node.WidgetsValues)
		set, held := f.sets[name]
		if err != nil || !held || len(set.node.Inputs) == 0 {
			return nil, false, fmt.Errorf("GetNode %d reads %q, which no SetNode sets (%v)", id, name, err)
		}
		return f.link(set.scope, set.node.Inputs[0])
	case node.Type == "SetNode" || node.Type == "Reroute":
		if len(node.Inputs) == 0 {
			return nil, false, nil
		}
		return through(node.Inputs[0])
	}
	if _, known := f.info[node.Type]; !known {
		return nil, false, fmt.Errorf("a link names node %d of type %q, which ComfyUI does not run and the recorder does not resolve", id, node.Type)
	}
	return []any{s.path + strconv.Itoa(id), slot}, true, nil
}

// scopeName names a scope for a message: its path, or the top graph.
func scopeName(path string) string {
	if path == "" {
		return "the top graph"
	}
	return "subgraph " + strings.TrimSuffix(path, ":")
}

// firstString is the first of a node's widget values, which a KJNodes Set or Get node names its value by.
func firstString(values json.RawMessage) (string, error) {
	var list []json.RawMessage
	if err := json.Unmarshal(values, &list); err != nil || len(list) == 0 {
		return "", fmt.Errorf("no widget values (%v)", err)
	}
	var name string
	return name, json.Unmarshal(list[0], &name)
}
