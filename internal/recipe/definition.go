package recipe

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sort"

	"llamacpp2go/internal/artifact"
)

type definitionBody struct {
	Version uint16      `json:"version"`
	Task    Task        `json:"task"`
	Model   artifact.ID `json:"model"`
	Nodes   []Node      `json:"nodes"`
	Edges   []Edge      `json:"edges,omitempty"`
	Inputs  []Input     `json:"inputs,omitempty"`
	Outputs []Output    `json:"outputs"`
}

func NewDefinition(task Task, model artifact.ID, nodes []Node, edges []Edge, inputs []Input, outputs []Output) (Definition, error) {
	definition := Definition{
		Version: Version, Task: task, Model: model,
		Nodes: slices.Clone(nodes), Edges: slices.Clone(edges),
		Inputs: slices.Clone(inputs), Outputs: slices.Clone(outputs),
	}
	if err := canonicalize(&definition); err != nil {
		return Definition{}, err
	}
	content, err := definitionContent(definition)
	if err != nil {
		return Definition{}, err
	}
	id, err := artifact.IdentifyBytes(artifact.KindRecipe, content)
	if err != nil {
		return Definition{}, err
	}
	definition.ID = id
	return definition, nil
}

func (d Definition) ValidateIdentity() error {
	if d.Version != Version || d.ID.Kind() != artifact.KindRecipe || d.Model.Kind() != artifact.KindModel {
		return errors.New("recipe: invalid version, recipe identity, or model identity")
	}
	canonical := d
	if err := canonicalize(&canonical); err != nil {
		return err
	}
	if !sameDefinitionShape(d, canonical) {
		return errors.New("recipe: definition is not canonical")
	}
	content, err := definitionContent(canonical)
	if err != nil {
		return err
	}
	want, err := artifact.IdentifyBytes(artifact.KindRecipe, content)
	if err != nil {
		return err
	}
	if d.ID != want {
		return errors.New("recipe: identity mismatch")
	}
	return nil
}

func (d Definition) Content() ([]byte, error) {
	if err := d.ValidateIdentity(); err != nil {
		return nil, err
	}
	return definitionContent(d)
}

func (d Definition) Descriptor() (artifact.Descriptor, error) {
	content, err := d.Content()
	if err != nil {
		return artifact.Descriptor{}, err
	}
	return artifact.Descriptor{ID: d.ID, Size: uint64(len(content)), MediaType: MediaType, Schema: Schema}, nil
}

func canonicalize(d *Definition) error {
	if d == nil || d.Version != Version || d.Model.Kind() != artifact.KindModel {
		return errors.New("recipe: invalid definition envelope")
	}
	if err := validateTask(d.Task); err != nil {
		return err
	}
	if len(d.Nodes) == 0 || len(d.Outputs) == 0 {
		return errors.New("recipe: definition requires nodes and outputs")
	}
	for _, node := range d.Nodes {
		if !validName(string(node.ID)) || !validName(string(node.Module)) {
			return errors.New("recipe: invalid node or module identity")
		}
		if err := validatePlacement(node.Placement); err != nil {
			return err
		}
	}
	for _, input := range d.Inputs {
		if !validName(string(input.Name)) || !validEndpoint(input.Target) {
			return errors.New("recipe: invalid graph input")
		}
		if err := validateDataKind(input.Data); err != nil {
			return err
		}
	}
	for _, output := range d.Outputs {
		if !validName(string(output.Name)) || !validEndpoint(output.Source) {
			return errors.New("recipe: invalid graph output")
		}
		if err := validateDataKind(output.Data); err != nil {
			return err
		}
	}
	for _, edge := range d.Edges {
		if !validEndpoint(edge.From) || !validEndpoint(edge.To) {
			return errors.New("recipe: invalid edge endpoint")
		}
	}
	sort.Slice(d.Nodes, func(i, j int) bool { return d.Nodes[i].ID < d.Nodes[j].ID })
	sort.Slice(d.Edges, func(i, j int) bool { return edgeKey(d.Edges[i]) < edgeKey(d.Edges[j]) })
	sort.Slice(d.Inputs, func(i, j int) bool { return d.Inputs[i].Name < d.Inputs[j].Name })
	sort.Slice(d.Outputs, func(i, j int) bool { return d.Outputs[i].Name < d.Outputs[j].Name })
	if duplicateNodes(d.Nodes) || duplicateEdges(d.Edges) || duplicateInputs(d.Inputs) || duplicateOutputs(d.Outputs) {
		return errors.New("recipe: duplicate graph identity")
	}
	return nil
}

func definitionContent(d Definition) ([]byte, error) {
	body := definitionBody{
		Version: d.Version, Task: d.Task, Model: d.Model,
		Nodes: d.Nodes, Edges: d.Edges, Inputs: d.Inputs, Outputs: d.Outputs,
	}
	content, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("recipe: encode definition: %w", err)
	}
	return content, nil
}

func validEndpoint(endpoint Endpoint) bool {
	return validName(string(endpoint.Node)) && validName(string(endpoint.Port))
}

func edgeKey(edge Edge) string {
	return string(edge.To.Node) + "\x00" + string(edge.To.Port) + "\x00" + string(edge.From.Node) + "\x00" + string(edge.From.Port)
}

func duplicateNodes(nodes []Node) bool {
	for index := 1; index < len(nodes); index++ {
		if nodes[index-1].ID == nodes[index].ID {
			return true
		}
	}
	return false
}

func duplicateEdges(edges []Edge) bool {
	for index := 1; index < len(edges); index++ {
		if edges[index-1] == edges[index] {
			return true
		}
	}
	return false
}

func duplicateInputs(inputs []Input) bool {
	for index := 1; index < len(inputs); index++ {
		if inputs[index-1].Name == inputs[index].Name {
			return true
		}
	}
	return false
}

func duplicateOutputs(outputs []Output) bool {
	for index := 1; index < len(outputs); index++ {
		if outputs[index-1].Name == outputs[index].Name {
			return true
		}
	}
	return false
}

func sameDefinitionShape(left, right Definition) bool {
	return left.Version == right.Version && left.Task == right.Task && left.Model == right.Model &&
		slices.Equal(left.Nodes, right.Nodes) && slices.Equal(left.Edges, right.Edges) &&
		slices.Equal(left.Inputs, right.Inputs) && slices.Equal(left.Outputs, right.Outputs)
}
