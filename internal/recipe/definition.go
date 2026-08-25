package recipe

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

type definitionBody struct {
	Version      uint16       `json:"version"`
	Task         Task         `json:"task"`
	Dependencies []Dependency `json:"dependencies"`
	Nodes        []Node       `json:"nodes"`
	Edges        []Edge       `json:"edges,omitempty"`
	Inputs       []Input      `json:"inputs,omitempty"`
	Outputs      []Output     `json:"outputs"`
}

var definitionCodec = artifact.DocumentCodec[Definition]{
	Name: "recipe definition", Contract: DefinitionDocumentContract(),
	Decode: decodeDefinition,
	Encode: definitionContent, Canonicalize: canonicalize,
	Clone: func(value Definition) Definition {
		value.Dependencies = slices.Clone(value.Dependencies)
		value.Nodes = slices.Clone(value.Nodes)
		value.Edges = slices.Clone(value.Edges)
		value.Inputs = slices.Clone(value.Inputs)
		value.Outputs = slices.Clone(value.Outputs)
		return value
	},
	Identity:    func(value Definition) artifact.ID { return value.ID },
	SetIdentity: func(value *Definition, id artifact.ID) { value.ID = id },
}

func ParseDefinition(content []byte) (Definition, error) {
	return definitionCodec.Parse(content)
}

// RequireDefinition loads one exact recipe definition and verifies its
// document contract and content-derived identity.
func RequireDefinition(ctx context.Context, reader artifact.Reader, id artifact.ID) (Definition, error) {
	return definitionCodec.Require(ctx, reader, id)
}

func decodeDefinition(content []byte, definition *Definition) error {
	var body definitionBody
	if err := strictjson.DecodeBytes(content, &body); err != nil {
		return err
	}
	if body.Version != Version {
		return errors.New("recipe: unsupported definition version")
	}
	*definition = Definition{
		Version: Version, Task: body.Task, Dependencies: body.Dependencies,
		Nodes: body.Nodes, Edges: body.Edges, Inputs: body.Inputs, Outputs: body.Outputs,
	}
	return nil
}

func NewDefinitionWithDependencies(
	task Task,
	dependencies []Dependency,
	nodes []Node,
	edges []Edge,
	inputs []Input,
	outputs []Output,
) (Definition, error) {
	return definitionCodec.New(Definition{
		Version: Version, Task: task, Dependencies: slices.Clone(dependencies),
		Nodes: slices.Clone(nodes), Edges: slices.Clone(edges),
		Inputs: slices.Clone(inputs), Outputs: slices.Clone(outputs),
	})
}

func (d Definition) ValidateIdentity() error {
	return definitionCodec.ValidateIdentity(d)
}

func (d Definition) Content() ([]byte, error) {
	return definitionCodec.ContentBytes(d)
}

func (d Definition) ArtifactContent() (artifact.Content, error) {
	return definitionCodec.Content(d)
}

func DefinitionDocumentContract() artifact.DocumentContract {
	return artifact.DocumentContract{Kind: artifact.KindRecipe, MediaType: MediaType, Schema: Schema}
}

func canonicalize(d *Definition) error {
	if d == nil || d.Version != Version {
		return errors.New("recipe: invalid definition envelope")
	}
	for _, dependency := range d.Dependencies {
		if err := validateDependency(dependency); err != nil {
			return err
		}
	}
	sort.Slice(d.Dependencies, func(i, j int) bool {
		return dependencyKey(d.Dependencies[i]) < dependencyKey(d.Dependencies[j])
	})
	if hasDuplicateKey(d.Dependencies, dependencyKey) {
		return errors.New("recipe: duplicate dependency role and slot")
	}
	model, found := artifact.ID{}, false
	for _, dependency := range d.Dependencies {
		if dependency.Role == DependencyModel && dependency.Slot == primaryDependencySlot {
			model, found = dependency.Artifact, true
		}
	}
	if !found {
		return errors.New("recipe: model dependency is absent")
	}
	d.Model = model
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
		if _, ok := d.Dependency(DependencyModel, node.ModelSlot); !ok {
			return fmt.Errorf("recipe: node %q model slot %d has no model dependency", node.ID, node.ModelSlot)
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
	if hasDuplicateKey(d.Nodes, func(node Node) NodeID { return node.ID }) ||
		hasDuplicateKey(d.Edges, edgeKey) ||
		hasDuplicateKey(d.Inputs, func(input Input) PortName { return input.Name }) ||
		hasDuplicateKey(d.Outputs, func(output Output) PortName { return output.Name }) {
		return errors.New("recipe: duplicate graph identity")
	}
	return nil
}

func definitionContent(d Definition) ([]byte, error) {
	body := definitionBody{
		Version: d.Version, Task: d.Task, Dependencies: d.Dependencies,
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

func dependencyKey(dependency Dependency) string {
	return string(dependency.Role) + fmt.Sprintf("\x00%010d", dependency.Slot)
}

func hasDuplicateKey[T any, K comparable](values []T, key func(T) K) bool {
	seen := make(map[K]struct{}, len(values))
	for _, value := range values {
		identity := key(value)
		if _, found := seen[identity]; found {
			return true
		}
		seen[identity] = struct{}{}
	}
	return false
}

func (d Definition) Dependency(role DependencyRole, slot uint32) (artifact.ID, bool) {
	for _, dependency := range d.Dependencies {
		if dependency.Role == role && dependency.Slot == slot {
			return dependency.Artifact, true
		}
	}
	return artifact.ID{}, false
}

// PrimaryDependency returns slot zero for a required singleton role.
func (d Definition) PrimaryDependency(role DependencyRole) (artifact.ID, bool) {
	return d.Dependency(role, primaryDependencySlot)
}
