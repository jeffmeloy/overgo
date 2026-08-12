package recipe

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

type legacyDefinitionBody struct {
	Version uint16      `json:"version"`
	Task    Task        `json:"task"`
	Model   artifact.ID `json:"model"`
	Nodes   []Node      `json:"nodes"`
	Edges   []Edge      `json:"edges,omitempty"`
	Inputs  []Input     `json:"inputs,omitempty"`
	Outputs []Output    `json:"outputs"`
}

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
	Name: "recipe definition", ContractFor: func(value Definition) artifact.DocumentContract {
		return DefinitionDocumentContract(value.Version)
	},
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

func decodeDefinition(content []byte, definition *Definition) error {
	var envelope struct {
		Version uint16 `json:"version"`
	}
	if err := json.Unmarshal(content, &envelope); err != nil {
		return fmt.Errorf("recipe: decode definition version: %w", err)
	}
	switch envelope.Version {
	case LegacyVersion:
		var body legacyDefinitionBody
		if err := strictjson.DecodeBytes(content, &body); err != nil {
			return err
		}
		*definition = Definition{
			Version: LegacyVersion, Task: body.Task, Model: body.Model,
			Nodes: body.Nodes, Edges: body.Edges, Inputs: body.Inputs, Outputs: body.Outputs,
		}
	case Version:
		var body definitionBody
		if err := strictjson.DecodeBytes(content, &body); err != nil {
			return err
		}
		*definition = Definition{
			Version: Version, Task: body.Task, Dependencies: body.Dependencies,
			Nodes: body.Nodes, Edges: body.Edges, Inputs: body.Inputs, Outputs: body.Outputs,
		}
	default:
		return errors.New("recipe: unsupported definition version")
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
	return newDefinition(Version, task, artifact.ID{}, dependencies, nodes, edges, inputs, outputs)
}

func newDefinition(
	version uint16,
	task Task,
	model artifact.ID,
	dependencies []Dependency,
	nodes []Node,
	edges []Edge,
	inputs []Input,
	outputs []Output,
) (Definition, error) {
	return definitionCodec.New(Definition{
		Version: version, Task: task, Model: model, Dependencies: slices.Clone(dependencies),
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

func (d Definition) Descriptor() (artifact.Descriptor, error) {
	content, err := d.Content()
	if err != nil {
		return artifact.Descriptor{}, err
	}
	return DefinitionDocumentContract(d.Version).Descriptor(d.ID, uint64(len(content)))
}

func (d Definition) ArtifactContent() (artifact.Content, error) {
	return definitionCodec.Content(d)
}

func DefinitionDocumentContract(version uint16) artifact.DocumentContract {
	schema := Schema
	if version == LegacyVersion {
		schema = LegacySchema
	}
	return artifact.DocumentContract{Kind: artifact.KindRecipe, MediaType: MediaType, Schema: schema}
}

func canonicalize(d *Definition) error {
	if d == nil || (d.Version != LegacyVersion && d.Version != Version) {
		return errors.New("recipe: invalid definition envelope")
	}
	if d.Version == LegacyVersion {
		if d.Model.Kind() != artifact.KindModel || len(d.Dependencies) != 0 {
			return errors.New("recipe: invalid legacy definition dependencies")
		}
	} else {
		for _, dependency := range d.Dependencies {
			if err := validateDependency(dependency); err != nil {
				return err
			}
		}
		sort.Slice(d.Dependencies, func(i, j int) bool {
			return dependencyKey(d.Dependencies[i]) < dependencyKey(d.Dependencies[j])
		})
		if duplicateDependencies(d.Dependencies) {
			return errors.New("recipe: duplicate dependency role and slot")
		}
		model, found := artifact.ID{}, false
		for _, dependency := range d.Dependencies {
			if dependency.Role == DependencyModel && dependency.Slot == 0 {
				model, found = dependency.Artifact, true
			}
		}
		if !found {
			return errors.New("recipe: model dependency is absent")
		}
		d.Model = model
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
	var body any
	if d.Version == LegacyVersion {
		body = legacyDefinitionBody{
			Version: d.Version, Task: d.Task, Model: d.Model,
			Nodes: d.Nodes, Edges: d.Edges, Inputs: d.Inputs, Outputs: d.Outputs,
		}
	} else {
		body = definitionBody{
			Version: d.Version, Task: d.Task, Dependencies: d.Dependencies,
			Nodes: d.Nodes, Edges: d.Edges, Inputs: d.Inputs, Outputs: d.Outputs,
		}
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

func duplicateDependencies(dependencies []Dependency) bool {
	for index := 1; index < len(dependencies); index++ {
		if dependencies[index-1].Role == dependencies[index].Role &&
			dependencies[index-1].Slot == dependencies[index].Slot {
			return true
		}
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
