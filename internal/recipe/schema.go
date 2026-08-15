package recipe

import (
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
)

const (
	Version   = uint16(2)
	MediaType = "application/vnd.overgo.recipe+json"
	Schema    = "overgo.recipe.v2"
	maxName   = 128
)

type Task string

const (
	TaskInference  Task = "inference"
	TaskGeneration Task = "generation"
	TaskEmbedding  Task = "embedding"
	TaskRerank     Task = "rerank"
	TaskProjection Task = "projection"
	TaskTraining   Task = "training"
	// TaskForecast: time-series forecasting (ladder rung 1, the first
	// non-token capability; series in, quantile horizon out).
	TaskForecast Task = "forecast"
	// TaskTabular: tabular in-context prediction (ladder rung 2; labeled
	// row prefix in, per-row predictions out).
	TaskTabular Task = "tabular"
	// TaskSeq2Seq: encoder-decoder generation (ladder rung 6; source
	// tokens in, generated target tokens out).
	TaskSeq2Seq Task = "seq2seq"
	// TaskSpeech: speech synthesis (ladder rung 7; text in, audio out).
	TaskSpeech Task = "speech"
	// TaskImageGen: class-conditional image generation (ladder rung 8;
	// condition tensor in, sampled image out). A NEW terminal task: the
	// existing TaskGeneration is token generation (tokenizer-coupled in the
	// workflow catalog) and the oscillator contract has no tokens or logits.
	TaskImageGen Task = "image-gen"
	// TaskVideoGen: text-to-video generation (ladder rung 11; prompt text
	// in, decoded video frames out). Distinct from TaskImageGen: the
	// contract carries a text-conditioning stage and temporal decode.
	TaskVideoGen Task = "video-gen"
	// TaskVQA: vision question-answering (ladder rung 14; image + question
	// text in, answer text out). The multimodal serving contract — a vision
	// tower + modality-routed (MoT) decoder — distinct from token inference
	// (no GGUF, dual-modality prefill) and from image-gen (text out, not image).
	TaskVQA Task = "vqa"
)

type Placement string

const (
	PlacementHost   Placement = "host"
	PlacementDevice Placement = "device"
	PlacementHybrid Placement = "hybrid"
)

type DataKind string

const (
	DataArtifact           DataKind = "artifact"
	DataText               DataKind = "text"
	DataTokens             DataKind = "tokens"
	DataEmbeddings         DataKind = "embeddings"
	DataTensor             DataKind = "tensor"
	DataModelPlan          DataKind = "model-plan"
	DataSessionPlan        DataKind = "session-plan"
	DataCache              DataKind = "cache"
	DataLogits             DataKind = "logits"
	DataImage              DataKind = "image"
	DataImageTensor        DataKind = "image-tensor"
	DataPromptConditioning DataKind = "prompt-conditioning"
	DataClassConditioning  DataKind = "class-conditioning"
	DataAudio              DataKind = "audio"
	DataAudioTensor        DataKind = "audio-tensor"
	DataVideo              DataKind = "video"
	DataVideoTensor        DataKind = "video-tensor"
	DataMetrics            DataKind = "metrics"
	DataCheckpoint         DataKind = "checkpoint"
	DataScores             DataKind = "scores"
	DataRanking            DataKind = "ranking"
	DataBatch              DataKind = "batch"
	DataLoss               DataKind = "loss"
	DataGradients          DataKind = "gradients"
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

// SessionPolicy: decode cache/graph lifetime.
type SessionPolicy string

const (
	SessionRequest  SessionPolicy = "request"
	SessionCapacity SessionPolicy = "capacity"
)

func (p SessionPolicy) Valid() bool {
	return p == "" || p == SessionRequest || p == SessionCapacity
}

// ResidencyPolicy: compiled model-weight storage and execution policy.
type ResidencyPolicy string

const (
	ResidencyStream           ResidencyPolicy = "stream"
	ResidencyHostCache        ResidencyPolicy = "host-cache"
	ResidencyDeviceF32        ResidencyPolicy = "device-f32"
	ResidencyDeviceNative     ResidencyPolicy = "device-native"
	ResidencyDeviceNativeBF16 ResidencyPolicy = "device-native-bf16"
	ResidencyHybridNative     ResidencyPolicy = "hybrid-native"
	ResidencyHostReference    ResidencyPolicy = "host-reference"
)

func (p ResidencyPolicy) Valid() bool {
	switch p {
	case "", ResidencyStream, ResidencyHostCache, ResidencyDeviceF32, ResidencyDeviceNative,
		ResidencyDeviceNativeBF16, ResidencyHybridNative, ResidencyHostReference:
		return true
	default:
		return false
	}
}

type Node struct {
	ID        NodeID          `json:"id"`
	Module    ModuleID        `json:"module"`
	Placement Placement       `json:"placement"`
	Session   SessionPolicy   `json:"session,omitempty"`
	Residency ResidencyPolicy `json:"residency,omitempty"`
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

func validateTask(task Task) error {
	switch task {
	case TaskInference, TaskGeneration, TaskEmbedding, TaskRerank, TaskProjection, TaskTraining, TaskForecast, TaskTabular, TaskSeq2Seq, TaskSpeech, TaskImageGen, TaskVideoGen, TaskVQA:
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
	case DataArtifact, DataText, DataTokens, DataEmbeddings, DataTensor, DataModelPlan, DataSessionPlan,
		DataCache, DataLogits, DataImage, DataImageTensor, DataPromptConditioning, DataClassConditioning,
		DataAudio, DataAudioTensor, DataVideo, DataVideoTensor, DataMetrics, DataCheckpoint,
		DataScores, DataRanking, DataBatch, DataLoss, DataGradients:
		return nil
	default:
		return fmt.Errorf("recipe: invalid data kind %q", kind)
	}
}

func (p Port) validate() error {
	if !textcheck.LowerIdentifier(string(p.Name), maxName) {
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
