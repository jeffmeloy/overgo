package trainingprogram

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/recipecontract"
	"overgo/internal/strictjson"
)

const (
	ObjectiveVersion   uint16 = 2
	ObjectiveMediaType        = "application/vnd.overgo.training-objective+json"
	ObjectiveSchema           = "overgo/training-objective/v2"
)

// ObjectiveKind names loss semantics; recipes bind model-specific execution.
type ObjectiveKind string

const (
	ObjectiveTokenPrediction ObjectiveKind = "token-prediction"
	ObjectiveFNS             ObjectiveKind = "fns"
	ObjectiveLatentL2        ObjectiveKind = "latent-l2"
	ObjectiveLatentSequence  ObjectiveKind = "latent-sequence-l2"
	ObjectiveForecast        ObjectiveKind = "forecast"
	ObjectiveOCR             ObjectiveKind = "ocr-token-prediction"
	ObjectiveFlowMatching    ObjectiveKind = "flow-matching"
	ObjectiveImageLatent     ObjectiveKind = "image-latent"
	ObjectiveDistillation    ObjectiveKind = "logit-distillation"
	ObjectiveTablePrediction ObjectiveKind = "table-prediction"
	ObjectiveDPO             ObjectiveKind = "dpo"
	ObjectiveGRPO            ObjectiveKind = "grpo"
)

var objectiveKinds = []ObjectiveKind{
	ObjectiveTokenPrediction,
	ObjectiveFNS,
	ObjectiveLatentL2,
	ObjectiveLatentSequence,
	ObjectiveForecast,
	ObjectiveOCR,
	ObjectiveFlowMatching,
	ObjectiveImageLatent,
	ObjectiveDistillation,
	ObjectiveTablePrediction,
	ObjectiveDPO,
	ObjectiveGRPO,
}

type ObjectiveAuthority string

const (
	ObjectiveAdaptive ObjectiveAuthority = "adaptive-evidence"
	ObjectiveApproved ObjectiveAuthority = "approved"
)

type EvaluationMetric string

const (
	MetricTokenAccuracy   EvaluationMetric = "token-accuracy"
	MetricImagePSNRSigned EvaluationMetric = "image-psnr-signed"
	MetricImagePSNRUnit   EvaluationMetric = "image-psnr-unit"
	MetricAudioSNR        EvaluationMetric = "audio-snr"
	MetricVideoPSNRSigned EvaluationMetric = "video-psnr-signed"
	MetricVideoPSNRUnit   EvaluationMetric = "video-psnr-unit"
	MetricForecastMAE     EvaluationMetric = "forecast-mae"
	MetricTableAccuracy   EvaluationMetric = "table-accuracy"
)

type ObjectiveSpec struct {
	Name       string
	Kind       ObjectiveKind
	Signature  recipecontract.ModalitySignature
	Dataset    artifact.ID
	Split      artifact.ID
	Processors []artifact.ID
	Projectors []artifact.ID
	Codecs     []artifact.ID
	Loss       artifact.ID
	Evaluation artifact.ID
	Metric     EvaluationMetric
	Evidence   []artifact.ID
	Authority  ObjectiveAuthority
}

// ObjectiveDocument binds one trainable modality pair to RepoDB evidence.
type ObjectiveDocument struct {
	ID      artifact.ID
	Version uint16
	ObjectiveSpec
}

type objectiveBody struct {
	Version    uint16                           `json:"version"`
	Name       string                           `json:"name"`
	Kind       ObjectiveKind                    `json:"kind"`
	Signature  recipecontract.ModalitySignature `json:"signature"`
	Dataset    artifact.ID                      `json:"dataset"`
	Split      artifact.ID                      `json:"split"`
	Processors []artifact.ID                    `json:"processors"`
	Projectors []artifact.ID                    `json:"projectors,omitempty"`
	Codecs     []artifact.ID                    `json:"codecs,omitempty"`
	Loss       artifact.ID                      `json:"loss"`
	Evaluation artifact.ID                      `json:"evaluation"`
	Metric     EvaluationMetric                 `json:"metric"`
	Evidence   []artifact.ID                    `json:"evidence"`
	Authority  ObjectiveAuthority               `json:"authority"`
}

var objectiveCodec = artifact.DocumentCodec[ObjectiveDocument]{
	Name: "training objective",
	Contract: artifact.DocumentContract{
		Kind: artifact.KindProfile, MediaType: ObjectiveMediaType, Schema: ObjectiveSchema,
	},
	Decode: func(data []byte, value *ObjectiveDocument) error {
		var body objectiveBody
		if err := strictjson.DecodeBytes(data, &body); err != nil {
			return err
		}
		*value = ObjectiveDocument{Version: body.Version, ObjectiveSpec: ObjectiveSpec{
			Name: body.Name, Kind: body.Kind, Signature: body.Signature, Dataset: body.Dataset, Split: body.Split,
			Processors: body.Processors, Projectors: body.Projectors, Codecs: body.Codecs,
			Loss: body.Loss, Evaluation: body.Evaluation, Metric: body.Metric,
			Evidence: body.Evidence, Authority: body.Authority,
		}}
		return nil
	},
	Encode: func(value ObjectiveDocument) ([]byte, error) {
		return json.Marshal(objectiveBody{
			Version: value.Version, Name: value.Name, Kind: value.Kind, Signature: value.Signature,
			Dataset: value.Dataset, Split: value.Split, Processors: value.Processors,
			Projectors: value.Projectors, Codecs: value.Codecs, Loss: value.Loss,
			Evaluation: value.Evaluation, Metric: value.Metric, Evidence: value.Evidence, Authority: value.Authority,
		})
	},
	Canonicalize: canonicalizeObjective,
	Clone: func(value ObjectiveDocument) ObjectiveDocument {
		value.Signature = value.Signature.Clone()
		value.Processors = slices.Clone(value.Processors)
		value.Projectors = slices.Clone(value.Projectors)
		value.Codecs = slices.Clone(value.Codecs)
		value.Evidence = slices.Clone(value.Evidence)
		return value
	},
	Identity:    func(value ObjectiveDocument) artifact.ID { return value.ID },
	SetIdentity: func(value *ObjectiveDocument, id artifact.ID) { value.ID = id },
}

func NewObjective(spec ObjectiveSpec) (ObjectiveDocument, error) {
	return objectiveCodec.New(ObjectiveDocument{Version: ObjectiveVersion, ObjectiveSpec: spec})
}

func (d ObjectiveDocument) Content() (artifact.Content, error) { return objectiveCodec.Content(d) }

func LoadObjective(ctx context.Context, reader artifact.Reader, id artifact.ID) (ObjectiveDocument, error) {
	document, ok, err := objectiveCodec.Read(ctx, reader, id)
	if err != nil {
		return ObjectiveDocument{}, err
	}
	if !ok {
		return ObjectiveDocument{}, errors.New("training objective: RepoDB content absent")
	}
	return document, nil
}

func canonicalizeObjective(value *ObjectiveDocument) error {
	if value.Version != ObjectiveVersion || strings.TrimSpace(value.Name) == "" || value.Name != strings.TrimSpace(value.Name) ||
		value.Dataset.Kind() != artifact.KindDataset || value.Split.Kind() != artifact.KindDatasetShard ||
		value.Loss.Kind() != artifact.KindProfile || value.Evaluation.Kind() != artifact.KindProfile ||
		!validObjectiveKind(value.Kind) || (value.Authority != ObjectiveAdaptive && value.Authority != ObjectiveApproved) {
		return errors.New("training objective: invalid authority")
	}
	if err := value.Signature.Validate(); err != nil || len(value.Signature.Inputs) != 1 || len(value.Signature.Outputs) != 1 {
		return errors.New("training objective: one input and one output modality required")
	}
	if !metricValidForModality(value.Signature.Outputs[0], value.Metric) {
		return fmt.Errorf("training objective: metric %q differs from output modality", value.Metric)
	}
	if !objectiveSignatureValid(value.Kind, value.Signature) {
		return fmt.Errorf("training objective: signature differs from %q contract", value.Kind)
	}
	var err error
	if value.Processors, err = canonicalObjectiveIDs(value.Processors, artifact.KindProfile, true); err != nil {
		return fmt.Errorf("training objective: processors: %w", err)
	}
	if value.Projectors, err = canonicalObjectiveIDs(value.Projectors, artifact.KindProjector, false); err != nil {
		return fmt.Errorf("training objective: projectors: %w", err)
	}
	if value.Codecs, err = canonicalObjectiveIDs(value.Codecs, artifact.KindProfile, false); err != nil {
		return fmt.Errorf("training objective: codecs: %w", err)
	}
	if value.Evidence, err = canonicalObjectiveIDs(value.Evidence, artifact.KindEvidence, true); err != nil {
		return fmt.Errorf("training objective: evidence: %w", err)
	}
	return nil
}

func canonicalObjectiveIDs(source []artifact.ID, kind artifact.Kind, required bool) ([]artifact.ID, error) {
	result := slices.Clone(source)
	sort.Slice(result, func(left, right int) bool { return result[left].String() < result[right].String() })
	if required && len(result) == 0 {
		return nil, errors.New("required identities absent")
	}
	for index, id := range result {
		if id.Kind() != kind || index > 0 && result[index-1] == id {
			return nil, errors.New("invalid or duplicate identity")
		}
	}
	return result, nil
}

func validObjectiveKind(kind ObjectiveKind) bool {
	return slices.Contains(objectiveKinds, kind)
}

func objectiveSignatureValid(kind ObjectiveKind, signature recipecontract.ModalitySignature) bool {
	input, output := signature.Inputs[0], signature.Outputs[0]
	switch kind {
	case ObjectiveTokenPrediction, ObjectiveDPO, ObjectiveGRPO:
		return output == recipecontract.ModalityText
	case ObjectiveFNS:
		return input == recipecontract.ModalityText && output == recipecontract.ModalityText
	case ObjectiveLatentL2, ObjectiveLatentSequence:
		return input == recipecontract.ModalityText && output == recipecontract.ModalityAudio
	case ObjectiveForecast:
		return input == recipecontract.ModalityTimeSeries && output == recipecontract.ModalityTimeSeries
	case ObjectiveOCR:
		return input == recipecontract.ModalityImage && output == recipecontract.ModalityText
	case ObjectiveFlowMatching:
		return output == recipecontract.ModalityAudio || output == recipecontract.ModalityImage || output == recipecontract.ModalityVideo
	case ObjectiveImageLatent:
		return input == recipecontract.ModalityImage && output == recipecontract.ModalityImage
	case ObjectiveDistillation:
		return input == recipecontract.ModalityText && output == recipecontract.ModalityText
	default:
		return false
	}
}

type ObjectiveDisposition string

const (
	ObjectiveTrainable ObjectiveDisposition = "trainable"
	ObjectiveRefused   ObjectiveDisposition = "refused"
)

type ObjectiveRow struct {
	Signature   recipecontract.ModalitySignature
	Objective   artifact.ID
	Disposition ObjectiveDisposition
	Reason      string
}

type ObjectiveProgramDisposition string

const (
	ObjectiveProgramCompiled ObjectiveProgramDisposition = "compiled"
	ObjectiveProgramRefused  ObjectiveProgramDisposition = "refused"
)

type TrainingObjectiveRow struct {
	Kind        ObjectiveKind
	Objective   artifact.ID
	Program     artifact.ID
	Disposition ObjectiveProgramDisposition
	Reason      string
}

var objectiveModalities = []recipecontract.Modality{
	recipecontract.ModalityText,
	recipecontract.ModalityImage,
	recipecontract.ModalityAudio,
	recipecontract.ModalityVideo,
	recipecontract.ModalityTimeSeries,
	recipecontract.ModalityTable,
}

// CompileObjectiveMatrix derives trainable rows only from stored objective documents.
func CompileObjectiveMatrix(ctx context.Context, reader artifact.Reader, objectives []artifact.ID) ([]ObjectiveRow, error) {
	if ctx == nil || reader == nil {
		return nil, errors.New("training objective: nil matrix authority")
	}
	approved := make(map[string]ObjectiveDocument, len(objectives))
	for _, id := range objectives {
		document, err := LoadObjective(ctx, reader, id)
		if err != nil {
			return nil, err
		}
		if err := validateObjectiveReferences(ctx, reader, document); err != nil {
			return nil, fmt.Errorf("training objective %q: %w", document.Name, err)
		}
		key := objectivePairKey(document.Signature)
		if _, duplicate := approved[key]; duplicate {
			return nil, fmt.Errorf("training objective: duplicate pair %s", key)
		}
		approved[key] = document
	}
	rows := make([]ObjectiveRow, 0, len(objectiveModalities)*len(objectiveModalities))
	for _, input := range objectiveModalities {
		for _, output := range objectiveModalities {
			signature := recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{input}, Outputs: []recipecontract.Modality{output}}
			row := ObjectiveRow{Signature: signature, Disposition: ObjectiveRefused, Reason: "no RepoDB objective with corpus and evidence"}
			if objective, ok := approved[objectivePairKey(signature)]; ok {
				row.Objective, row.Disposition, row.Reason = objective.ID, ObjectiveTrainable, ""
			}
			rows = append(rows, row)
		}
	}
	return rows, nil
}

// CompileTrainingObjectiveMatrix binds stored objective evidence to shared programs.
func CompileTrainingObjectiveMatrix(ctx context.Context, reader artifact.Reader, objectives []artifact.ID, programs []TrainingProgram) ([]TrainingObjectiveRow, error) {
	if ctx == nil || reader == nil {
		return nil, errors.New("training objective: nil program matrix authority")
	}
	byKind := make(map[ObjectiveKind]TrainingProgram, len(programs))
	for _, program := range programs {
		if program.ID().Kind() != artifact.KindRecipe || !validObjectiveKind(program.Objective()) {
			return nil, errors.New("training objective: invalid compiled program")
		}
		if _, duplicate := byKind[program.Objective()]; duplicate {
			return nil, fmt.Errorf("training objective: duplicate program for %q", program.Objective())
		}
		byKind[program.Objective()] = program
	}
	documents := make([]ObjectiveDocument, len(objectives))
	for index, id := range objectives {
		document, err := LoadObjective(ctx, reader, id)
		if err != nil {
			return nil, err
		}
		if err := validateObjectiveReferences(ctx, reader, document); err != nil {
			return nil, fmt.Errorf("training objective %q: %w", document.Name, err)
		}
		documents[index] = document
	}
	sort.Slice(documents, func(left, right int) bool {
		if documents[left].Kind != documents[right].Kind {
			return documents[left].Kind < documents[right].Kind
		}
		return documents[left].ID.String() < documents[right].ID.String()
	})
	rows := make([]TrainingObjectiveRow, len(documents))
	seen := make(map[string]struct{}, len(documents))
	for index, document := range documents {
		key := string(document.Kind) + "\x00" + objectivePairKey(document.Signature)
		if _, duplicate := seen[key]; duplicate {
			return nil, fmt.Errorf("training objective: duplicate kind/signature %s", key)
		}
		seen[key] = struct{}{}
		row := TrainingObjectiveRow{
			Kind: document.Kind, Objective: document.ID,
			Disposition: ObjectiveProgramRefused, Reason: "shared TrainingProgram absent",
		}
		if program, ok := byKind[document.Kind]; ok {
			row.Program, row.Disposition, row.Reason = program.ID(), ObjectiveProgramCompiled, ""
		}
		rows[index] = row
	}
	return rows, nil
}

// CompileTrainingRunPlanFromRepository binds opaque run IDs to stored objective facts.
func CompileTrainingRunPlanFromRepository(ctx context.Context, reader artifact.Reader, spec RunSpec) (TrainingRunPlan, error) {
	plan, err := CompileTrainingRunPlan(spec)
	if err != nil {
		return TrainingRunPlan{}, err
	}
	for _, id := range []artifact.ID{
		spec.Policies.Precision, spec.Policies.Placement, spec.Policies.Memory,
		spec.Policies.Checkpoint, spec.Policies.Evaluation, spec.Policies.Promotion,
	} {
		if _, ok, err := reader.Artifact(ctx, id); err != nil {
			return TrainingRunPlan{}, err
		} else if !ok {
			return TrainingRunPlan{}, fmt.Errorf("training objective: policy %s absent", id)
		}
	}
	if _, err := RequireOptimizerPolicy(ctx, reader, spec.Policies.Optimizer); err != nil {
		return TrainingRunPlan{}, err
	}
	objective, err := resolveObjective(ctx, reader, spec.Policies.Objective)
	if err != nil {
		return TrainingRunPlan{}, err
	}
	if objective.spec.Kind != plan.Program().Objective() || objective.spec.Dataset != plan.Dataset() || objective.spec.Split != plan.Split() ||
		!slices.Equal(objective.spec.Signature.Inputs, plan.signature.Inputs) ||
		!slices.Equal(objective.spec.Signature.Outputs, plan.signature.Outputs) ||
		!slices.Equal(objective.spec.Processors, plan.processors) ||
		!slices.Equal(objective.spec.Projectors, plan.projectors) ||
		!slices.Equal(objective.spec.Codecs, plan.codecs) || objective.spec.Evaluation != plan.policies.Evaluation {
		return TrainingRunPlan{}, errors.New("training objective: run authority differs from RepoDB objective")
	}
	return plan, nil
}

func validateObjectiveReferences(ctx context.Context, reader artifact.Reader, document ObjectiveDocument) error {
	references := append([]artifact.ID{document.Dataset, document.Split, document.Loss, document.Evaluation}, document.Processors...)
	references = append(references, document.Projectors...)
	references = append(references, document.Codecs...)
	references = append(references, document.Evidence...)
	for _, id := range references {
		if _, ok, err := reader.Artifact(ctx, id); err != nil {
			return err
		} else if !ok {
			return fmt.Errorf("referenced artifact %s absent", id)
		}
	}
	return nil
}

func objectivePairKey(signature recipecontract.ModalitySignature) string {
	return string(signature.Inputs[0]) + "->" + string(signature.Outputs[0])
}

func metricValidForModality(modality recipecontract.Modality, metric EvaluationMetric) bool {
	switch modality {
	case recipecontract.ModalityText:
		return metric == MetricTokenAccuracy
	case recipecontract.ModalityImage:
		return metric == MetricImagePSNRSigned || metric == MetricImagePSNRUnit
	case recipecontract.ModalityAudio:
		return metric == MetricAudioSNR
	case recipecontract.ModalityVideo:
		return metric == MetricVideoPSNRSigned || metric == MetricVideoPSNRUnit
	case recipecontract.ModalityTimeSeries:
		return metric == MetricForecastMAE
	case recipecontract.ModalityTable:
		return metric == MetricTableAccuracy
	default:
		return false
	}
}
