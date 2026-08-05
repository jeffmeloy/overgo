package artifact

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"
)

const (
	ManifestVersion   = uint16(1)
	ManifestMediaType = "application/vnd.llamacpp2go.artifact-manifest+json"
	ManifestSchema    = "llamacpp2go.artifact-manifest.v1"
	maxComponentName  = 1024
	maxComponents     = 1 << 20
)

// ComponentRole: manifest component contract
type ComponentRole uint8

const (
	ComponentInvalid ComponentRole = iota
	ComponentWeights
	ComponentWeightsShard
	ComponentConfig
	ComponentShardIndex
	ComponentTokenizer
	ComponentVocabulary
	ComponentMerges
	ComponentChatTemplate
	ComponentPreprocessor
	ComponentGenerationConfig
	ComponentProjector
	ComponentAdapter
	ComponentCompanion
)

var componentRoleNames = [...]string{
	ComponentInvalid:          "invalid",
	ComponentWeights:          "weights",
	ComponentWeightsShard:     "weights-shard",
	ComponentConfig:           "config",
	ComponentShardIndex:       "shard-index",
	ComponentTokenizer:        "tokenizer",
	ComponentVocabulary:       "vocabulary",
	ComponentMerges:           "merges",
	ComponentChatTemplate:     "chat-template",
	ComponentPreprocessor:     "preprocessor",
	ComponentGenerationConfig: "generation-config",
	ComponentProjector:        "projector",
	ComponentAdapter:          "adapter",
	ComponentCompanion:        "companion",
}

func (r ComponentRole) String() string {
	if int(r) >= len(componentRoleNames) {
		return componentRoleNames[ComponentInvalid]
	}
	return componentRoleNames[r]
}

func ParseComponentRole(value string) (ComponentRole, error) {
	for role, name := range componentRoleNames {
		if role > 0 && value == name {
			return ComponentRole(role), nil
		}
	}
	return ComponentInvalid, fmt.Errorf("artifact: unknown component role %q", value)
}

func (r ComponentRole) MarshalJSON() ([]byte, error) {
	if r == ComponentInvalid || int(r) >= len(componentRoleNames) {
		return nil, errors.New("artifact: invalid component role")
	}
	return json.Marshal(r.String())
}

func (r *ComponentRole) UnmarshalJSON(data []byte) error {
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return fmt.Errorf("artifact: decode component role: %w", err)
	}
	parsed, err := ParseComponentRole(value)
	if err != nil {
		return err
	}
	*r = parsed
	return nil
}

// Component: canonical manifest member
type Component struct {
	Role     ComponentRole `json:"role"`
	Ordinal  uint32        `json:"ordinal"`
	Name     string        `json:"name"`
	Artifact ID            `json:"artifact"`
}

func (c Component) Validate() error {
	if c.Role == ComponentInvalid || int(c.Role) >= len(componentRoleNames) {
		return errors.New("artifact: invalid component role")
	}
	if c.Name == "" || len(c.Name) > maxComponentName || strings.TrimSpace(c.Name) != c.Name || strings.ContainsAny(c.Name, "\r\n\\") {
		return errors.New("artifact: invalid component name")
	}
	if !c.Artifact.Valid() {
		return errors.New("artifact: component has invalid identity")
	}
	return nil
}

// Manifest: canonical logical artifact inventory
type Manifest struct {
	Version    uint16      `json:"version"`
	ID         ID          `json:"id"`
	Components []Component `json:"components"`
}

type manifestBody struct {
	Version    uint16      `json:"version"`
	Components []Component `json:"components"`
}

func NewManifest(kind Kind, components []Component) (Manifest, error) {
	if kind == KindInvalid || int(kind) >= len(kindNames) {
		return Manifest{}, errors.New("artifact: invalid manifest kind")
	}
	canonical, err := canonicalComponents(components)
	if err != nil {
		return Manifest{}, err
	}
	body, err := encodeManifestBody(canonical)
	if err != nil {
		return Manifest{}, err
	}
	id, err := IdentifyBytes(kind, body)
	if err != nil {
		return Manifest{}, err
	}
	return Manifest{Version: ManifestVersion, ID: id, Components: canonical}, nil
}

func (m Manifest) Validate() error {
	if m.Version != ManifestVersion || !m.ID.Valid() {
		return errors.New("artifact: invalid manifest version or identity")
	}
	canonical, err := canonicalComponents(m.Components)
	if err != nil {
		return err
	}
	if !slices.Equal(canonical, m.Components) {
		return errors.New("artifact: non-canonical manifest components")
	}
	body, err := encodeManifestBody(canonical)
	if err != nil {
		return err
	}
	want, err := IdentifyBytes(m.ID.Kind(), body)
	if err != nil {
		return err
	}
	if want != m.ID {
		return errors.New("artifact: manifest identity mismatch")
	}
	return nil
}

func (m Manifest) Descriptor() (Descriptor, error) {
	if err := m.Validate(); err != nil {
		return Descriptor{}, err
	}
	body, err := encodeManifestBody(m.Components)
	if err != nil {
		return Descriptor{}, err
	}
	return Descriptor{ID: m.ID, Size: uint64(len(body)), MediaType: ManifestMediaType, Schema: ManifestSchema}, nil
}

func (m Manifest) Lineage() []Lineage {
	result := make([]Lineage, len(m.Components))
	for index, component := range m.Components {
		result[index] = Lineage{Child: m.ID, Parent: component.Artifact, Relation: RelationContains}
	}
	return result
}

func canonicalComponents(components []Component) ([]Component, error) {
	if len(components) == 0 || len(components) > maxComponents {
		return nil, errors.New("artifact: invalid manifest component count")
	}
	result := slices.Clone(components)
	names := make(map[string]struct{}, len(result))
	for _, component := range result {
		if err := component.Validate(); err != nil {
			return nil, err
		}
		if _, duplicate := names[component.Name]; duplicate {
			return nil, fmt.Errorf("artifact: duplicate component name %q", component.Name)
		}
		names[component.Name] = struct{}{}
	}
	sort.Slice(result, func(i, j int) bool {
		left, right := result[i], result[j]
		if left.Role != right.Role {
			return left.Role < right.Role
		}
		if left.Ordinal != right.Ordinal {
			return left.Ordinal < right.Ordinal
		}
		if left.Name != right.Name {
			return left.Name < right.Name
		}
		return left.Artifact.String() < right.Artifact.String()
	})
	for index := 1; index < len(result); index++ {
		previous, current := result[index-1], result[index]
		if previous.Role == current.Role && previous.Ordinal == current.Ordinal {
			return nil, fmt.Errorf("artifact: duplicate component slot %s/%d", current.Role, current.Ordinal)
		}
	}
	return result, nil
}

func encodeManifestBody(components []Component) ([]byte, error) {
	data, err := json.Marshal(manifestBody{Version: ManifestVersion, Components: components})
	if err != nil {
		return nil, fmt.Errorf("artifact: encode manifest: %w", err)
	}
	return data, nil
}
