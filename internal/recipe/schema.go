package recipe

import (
	"errors"
	"fmt"
	"strings"

	"llamacpp2go/internal/artifact"
)

const (
	LegacyVersion = uint16(1)
	Version       = uint16(2)
	MediaType     = "application/vnd.llamacpp2go.recipe+json"
	LegacySchema  = "llamacpp2go.recipe.v1"
	Schema        = "llamacpp2go.recipe.v2"
	maxName       = 128
)

type Task string

const (
	TaskInference  Task = "inference"
	TaskGeneration Task = "generation"
	TaskEmbedding  Task = "embedding"
	TaskRerank     Task = "rerank"
	TaskProjection Task = "projection"
	TaskTraining   Task = "training"
)

type Placement string

const (
	PlacementHost   Placement = "host"
	PlacementDevice Placement = "device"
	PlacementHybrid Placement = "hybrid"
)

type DataKind string

const (
	DataArtifact   DataKind = "artifact"
	DataText       DataKind = "text"
	DataTokens     DataKind = "tokens"
	DataEmbeddings DataKind = "embeddings"
	DataTensor     DataKind = "tensor"
	DataModelPlan  DataKind = "model-plan"
	DataCache      DataKind = "cache"
	DataLogits     DataKind = "logits"
	DataImage      DataKind = "image"
	DataAudio      DataKind = "audio"
	DataVideo      DataKind = "video"
	DataMetrics    DataKind = "metrics"
	DataCheckpoint DataKind = "checkpoint"
	DataScores     DataKind = "scores"
	DataRanking    DataKind = "ranking"
	DataBatch      DataKind = "batch"
	DataLoss       DataKind = "loss"
	DataGradients  DataKind = "gradients"
)

type Cardinality string

const (
	CardinalityOne       Cardinality = "one"
	CardinalityOptional  Cardinality = "optional"
	CardinalityMany      Cardinality = "many"
	CardinalityOneOrMany Cardinality = "one-or-many"
)

type ModuleID string
type NodeID string
type PortName string

type DependencyRole string

const (
	DependencyModel      DependencyRole = "model"
	DependencyProfile    DependencyRole = "profile"
	DependencyTokenizer  DependencyRole = "tokenizer"
	DependencyProjector  DependencyRole = "projector"
	DependencyAdapter    DependencyRole = "adapter"
	DependencyDataset    DependencyRole = "dataset"
	DependencyCheckpoint DependencyRole = "checkpoint"
	DependencyDefinition DependencyRole = "model-definition"
)

type Dependency struct {
	Role     DependencyRole `json:"role"`
	Slot     uint32         `json:"slot,omitempty"`
	Artifact artifact.ID    `json:"artifact"`
}

type Port struct {
	Name        PortName    `json:"name"`
	Data        DataKind    `json:"data"`
	Cardinality Cardinality `json:"cardinality"`
}

type Module struct {
	ID         ModuleID    `json:"id"`
	Tasks      []Task      `json:"tasks"`
	Placements []Placement `json:"placements"`
	Inputs     []Port      `json:"inputs,omitempty"`
	Outputs    []Port      `json:"outputs,omitempty"`
}

type Node struct {
	ID        NodeID    `json:"id"`
	Module    ModuleID  `json:"module"`
	Placement Placement `json:"placement"`
}

type Endpoint struct {
	Node NodeID   `json:"node"`
	Port PortName `json:"port"`
}

type Edge struct {
	From Endpoint `json:"from"`
	To   Endpoint `json:"to"`
}

type Input struct {
	Name   PortName `json:"name"`
	Data   DataKind `json:"data"`
	Target Endpoint `json:"target"`
}

type Output struct {
	Name   PortName `json:"name"`
	Data   DataKind `json:"data"`
	Source Endpoint `json:"source"`
}

type Definition struct {
	Version      uint16       `json:"version"`
	ID           artifact.ID  `json:"id"`
	Task         Task         `json:"task"`
	Model        artifact.ID  `json:"model"`
	Dependencies []Dependency `json:"dependencies,omitempty"`
	Nodes        []Node       `json:"nodes"`
	Edges        []Edge       `json:"edges,omitempty"`
	Inputs       []Input      `json:"inputs,omitempty"`
	Outputs      []Output     `json:"outputs"`
}

func SupportsSchema(schema string) bool {
	return schema == Schema || schema == LegacySchema
}

func validateDependency(dependency Dependency) error {
	want := artifact.KindInvalid
	switch dependency.Role {
	case DependencyModel:
		want = artifact.KindModel
	case DependencyProfile:
		want = artifact.KindProfile
	case DependencyTokenizer:
		want = artifact.KindTokenizer
	case DependencyProjector:
		want = artifact.KindProjector
	case DependencyAdapter:
		want = artifact.KindAdapter
	case DependencyDataset:
		want = artifact.KindDataset
	case DependencyCheckpoint:
		want = artifact.KindCheckpoint
	case DependencyDefinition:
		want = artifact.KindModelDefinition
	default:
		return fmt.Errorf("recipe: invalid dependency role %q", dependency.Role)
	}
	if dependency.Artifact.Kind() != want {
		return fmt.Errorf("recipe: dependency %q has artifact kind %q, want %q", dependency.Role, dependency.Artifact.Kind(), want)
	}
	return nil
}

func validName(value string) bool {
	if value == "" || len(value) > maxName || strings.TrimSpace(value) != value {
		return false
	}
	for _, char := range value {
		if char >= 'a' && char <= 'z' || char >= '0' && char <= '9' || char == '.' || char == '-' || char == '_' {
			continue
		}
		return false
	}
	return true
}

func validateTask(task Task) error {
	switch task {
	case TaskInference, TaskGeneration, TaskEmbedding, TaskRerank, TaskProjection, TaskTraining:
		return nil
	default:
		return fmt.Errorf("recipe: invalid task %q", task)
	}
}

func validatePlacement(placement Placement) error {
	switch placement {
	case PlacementHost, PlacementDevice, PlacementHybrid:
		return nil
	default:
		return fmt.Errorf("recipe: invalid placement %q", placement)
	}
}

func validateDataKind(kind DataKind) error {
	switch kind {
	case DataArtifact, DataText, DataTokens, DataEmbeddings, DataTensor, DataModelPlan,
		DataCache, DataLogits, DataImage, DataAudio, DataVideo, DataMetrics, DataCheckpoint,
		DataScores, DataRanking, DataBatch, DataLoss, DataGradients:
		return nil
	default:
		return fmt.Errorf("recipe: invalid data kind %q", kind)
	}
}

func (p Port) validate() error {
	if !validName(string(p.Name)) {
		return errors.New("recipe: invalid port name")
	}
	if err := validateDataKind(p.Data); err != nil {
		return err
	}
	switch p.Cardinality {
	case CardinalityOne, CardinalityOptional, CardinalityMany, CardinalityOneOrMany:
		return nil
	default:
		return fmt.Errorf("recipe: invalid cardinality %q", p.Cardinality)
	}
}
