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
	// TaskTranscription converts audio into a language-bearing transcript.
	TaskTranscription Task = "transcription"
	// TaskAlignment binds transcript elements to source-audio sample spans.
	TaskAlignment Task = "alignment"
	// TaskDiarization identifies speaker turns in source audio.
	TaskDiarization Task = "diarization"
	// TaskActivityDetection identifies active speech spans in source audio.
	TaskActivityDetection Task = "activity-detection"
	// TaskAudioConversion transforms audio while preserving its semantic content.
	TaskAudioConversion Task = "audio-conversion"
	// TaskAudioGeneration produces audio from a non-audio condition.
	TaskAudioGeneration Task = "audio-generation"
	// TaskImageGen: class-conditional image generation (ladder rung 8;
	// condition tensor in, sampled image out). A NEW terminal task: the
	// existing TaskGeneration is token generation (tokenizer-coupled in the
	// workflow catalog) and the oscillator contract has no tokens or logits.
	TaskImageGen Task = "image-gen"
	// TaskVideoGen: text-to-video generation (ladder rung 11; prompt text
	// in, decoded video frames out). Distinct from TaskImageGen: the
	// contract carries a text-conditioning stage and temporal decode.
	TaskVideoGen Task = "video-gen"
	// TaskVideoEdit performs reference-guided video editing over the
	// reference-edit runtime: source video plus prompt in, edited
	// video out.
	TaskVideoEdit Task = "video-edit"
	// TaskVQA: vision question-answering (ladder rung 14; image + question
	// text in, answer text out). The multimodal serving contract — a vision
	// tower + modality-routed (MoT) decoder — distinct from token inference
	// (no GGUF, dual-modality prefill) and from image-gen (text out, not image).
	TaskVQA Task = "vqa"
)

func (task Task) Valid() bool { return validateTask(task) == nil }

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
	// DataTranscription carries text bound to an exact audio source.
	DataTranscription DataKind = "transcription"
	// DataTimestampedAlignment carries transcript elements with sample spans.
	DataTimestampedAlignment DataKind = "timestamped-alignment"
	// DataSpeechTurns carries ordered, potentially overlapping speaker turns.
	DataSpeechTurns DataKind = "speech-turns"
	// DataActivitySegments carries ordered active-speech sample spans.
	DataActivitySegments DataKind = "activity-segments"
	// DataConvertedAudio carries transformed audio with source lineage.
	DataConvertedAudio DataKind = "converted-audio"
	// DataGeneratedAudio carries condition-derived audio with restart state.
	DataGeneratedAudio  DataKind = "generated-audio"
	DataVideo           DataKind = "video"
	DataVideoTensor     DataKind = "video-tensor"
	DataMetrics         DataKind = "metrics"
	DataCheckpoint      DataKind = "checkpoint"
	DataScores          DataKind = "scores"
	DataRanking         DataKind = "ranking"
	DataBatch           DataKind = "batch"
	DataPreferenceBatch DataKind = "preference-batch"
	DataSequenceScores  DataKind = "sequence-scores"
	DataLoss            DataKind = "loss"
	DataGradients       DataKind = "gradients"
	// DataToolCall carries one strict tool invocation.
	DataToolCall DataKind = "tool-call"
	// DataToolResult carries one typed invocation result.
	DataToolResult DataKind = "tool-result"
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

const primaryDependencySlot uint32 = iota

const (
	DependencyModel   DependencyRole = "model"
	DependencyProfile DependencyRole = "profile"
	// DependencyProcessorProfile binds input transformation policy.
	DependencyProcessorProfile DependencyRole = "processor-profile"
	// DependencyFlowProfile binds flow operator facts.
	DependencyFlowProfile DependencyRole = "flow-profile"
	// DependencyDerivationProfile binds construction policy.
	DependencyDerivationProfile DependencyRole = "derivation-profile"
	DependencyTokenizer         DependencyRole = "tokenizer"
	DependencyProjector         DependencyRole = "projector"
	DependencyAdapter           DependencyRole = "adapter"
	DependencyDataset           DependencyRole = "dataset"
	DependencyCheckpoint        DependencyRole = "checkpoint"
	DependencyDefinition        DependencyRole = "model-definition"
	DependencyObjective         DependencyRole = "training-objective"
	DependencyPrecision         DependencyRole = "training-precision"
	DependencyPlacement         DependencyRole = "training-placement"
	DependencyMemory            DependencyRole = "training-memory"
	DependencyOptimizer         DependencyRole = "training-optimizer"
	DependencyCheckpointPolicy  DependencyRole = "training-checkpoint"
	DependencyEvaluation        DependencyRole = "training-evaluation"
	DependencyEvaluator         DependencyRole = "evaluator"
	DependencyPromotion         DependencyRole = "training-promotion"
	// DependencyCandidateTrial binds a derived graph to the admitted,
	// closed-world candidate compilation it executes or evaluates.
	DependencyCandidateTrial DependencyRole = "candidate-trial"
	// DependencyCandidateEvaluation binds the pre-execution evaluation plan.
	DependencyCandidateEvaluation DependencyRole = "candidate-evaluation"
	// DependencyCandidateComponent binds one compiled domain component plan.
	DependencyCandidateComponent DependencyRole = "candidate-component"
	// DependencyCandidateAblation binds one declared drop-delta arm.
	DependencyCandidateAblation DependencyRole = "candidate-ablation"
	// DependencyCandidateMaterialization binds the exact realized arm closure.
	DependencyCandidateMaterialization DependencyRole = "candidate-materialization"
	// DependencyCodeAuthority binds the compiled code revision used to realize a candidate.
	DependencyCodeAuthority DependencyRole = "code-authority"
	// DependencyEnvironmentAuthority binds the realization environment observation.
	DependencyEnvironmentAuthority DependencyRole = "environment-authority"
	// DependencyFalsifier binds the typed evaluation intent selected before execution.
	DependencyFalsifier DependencyRole = "falsifier"
	// DependencyExecutionRecipe binds an exact graph to evaluations or components.
	DependencyExecutionRecipe DependencyRole = "execution-recipe"
	// DependencyBudget binds a split-scoped resource grant.
	DependencyBudget DependencyRole = "budget"
	// DependencyDatasetShard binds an exact held-out or development split.
	DependencyDatasetShard DependencyRole = "dataset-shard"
	// DependencyTensorInventory binds the exact realized tensor inventory.
	DependencyTensorInventory DependencyRole = "tensor-inventory"
	// DependencyOutput binds the immutable output produced by a realized arm.
	DependencyOutput DependencyRole = "output"
	// DependencyRun binds the supervised realization run.
	DependencyRun DependencyRole = "run"
	// DependencyObservation binds the realization's typed observation contract.
	DependencyObservation DependencyRole = "observation"
	// DependencyCapabilityBundle binds typed instruction and resource data.
	DependencyCapabilityBundle DependencyRole = "capability-bundle"
	// DependencyToolManual binds a compiled workflow to one exact callable manual.
	DependencyToolManual DependencyRole = "tool-manual"
)

type Dependency struct {
	Role     DependencyRole `json:"role"`
	Slot     uint32         `json:"slot,omitzero"`
	Artifact artifact.ID    `json:"artifact"`
}

type Port struct {
	Name        PortName    `json:"name"`
	Data        DataKind    `json:"data"`
	Cardinality Cardinality `json:"cardinality"`
}

// ArtifactRequirement defines a typed stage-state postcondition.
type ArtifactRequirement struct {
	Name     PortName      `json:"name"`
	Kind     artifact.Kind `json:"kind"`
	Preserve bool          `json:"preserve,omitzero"`
}

type Module struct {
	ID             ModuleID              `json:"id"`
	Tasks          []Task                `json:"tasks"`
	Placements     []Placement           `json:"placements"`
	Inputs         []Port                `json:"inputs,omitempty"`
	Outputs        []Port                `json:"outputs,omitempty"`
	Postconditions []ArtifactRequirement `json:"postconditions,omitempty"`
	StageNode      NodeID                `json:"stage_node,omitzero"`
	Next           ModuleID              `json:"next,omitzero"`
}

// SessionPolicy defines decode cache/graph lifetime.
type SessionPolicy string

const (
	SessionRequest  SessionPolicy = "request"
	SessionCapacity SessionPolicy = "capacity"
	// SessionRequestCapacity: one request-owned resident session.
	SessionRequestCapacity = 1
)

func (p SessionPolicy) Valid() bool {
	return p == "" || p == SessionRequest || p == SessionCapacity
}

// ResidencyPolicy defines compiled model-weight storage and execution policy.
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
	Session   SessionPolicy   `json:"session,omitzero"`
	Residency ResidencyPolicy `json:"residency,omitzero"`
	// ModelSlot selects which model dependency the node executes against,
	// keyed by Dependency{Role: DependencyModel, Slot: ModelSlot}. Slot 0 is
	// the definition's primary model; a multi-model chain binds later nodes
	// to higher slots so composition stays typed port wiring, never latent
	// bridging.
	ModelSlot uint32 `json:"model_slot,omitzero"`
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
	case DependencyProfile, DependencyProcessorProfile, DependencyFlowProfile, DependencyDerivationProfile,
		DependencyCapabilityBundle, DependencyObjective, DependencyPrecision, DependencyPlacement,
		DependencyMemory, DependencyOptimizer, DependencyCheckpointPolicy, DependencyEvaluation, DependencyPromotion,
		DependencyCandidateTrial, DependencyCandidateEvaluation, DependencyCandidateComponent,
		DependencyCandidateAblation:
		want = artifact.KindProfile
	case DependencyEvaluator, DependencyCandidateMaterialization, DependencyCodeAuthority,
		DependencyEnvironmentAuthority, DependencyBudget, DependencyObservation:
		want = artifact.KindEvidence
	case DependencyFalsifier, DependencyExecutionRecipe:
		want = artifact.KindRecipe
	case DependencyTokenizer:
		want = artifact.KindTokenizer
	case DependencyProjector:
		want = artifact.KindProjector
	case DependencyAdapter:
		want = artifact.KindAdapter
	case DependencyDataset:
		want = artifact.KindDataset
	case DependencyDatasetShard:
		want = artifact.KindDatasetShard
	case DependencyCheckpoint:
		want = artifact.KindCheckpoint
	case DependencyDefinition:
		want = artifact.KindModelDefinition
	case DependencyTensorInventory:
		want = artifact.KindTensorInventory
	case DependencyOutput:
		want = artifact.KindOutput
	case DependencyRun:
		want = artifact.KindRun
	case DependencyToolManual:
		want = artifact.KindRecipe
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
	case TaskInference, TaskGeneration, TaskEmbedding, TaskRerank, TaskProjection, TaskTraining,
		TaskForecast, TaskTabular, TaskSeq2Seq, TaskSpeech, TaskTranscription, TaskAlignment,
		TaskDiarization, TaskActivityDetection, TaskAudioConversion, TaskAudioGeneration,
		TaskImageGen, TaskVideoGen, TaskVideoEdit, TaskVQA:
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
		DataAudio, DataAudioTensor, DataTranscription, DataTimestampedAlignment, DataSpeechTurns,
		DataActivitySegments, DataConvertedAudio, DataGeneratedAudio,
		DataVideo, DataVideoTensor, DataMetrics, DataCheckpoint,
		DataScores, DataRanking, DataBatch, DataPreferenceBatch, DataSequenceScores, DataLoss, DataGradients,
		DataToolCall, DataToolResult:
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

func validName(value string) bool {
	return textcheck.LowerIdentifier(value, len(value))
}
