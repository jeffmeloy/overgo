package runrecord

import (
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/trainingprogram"
)

const (
	ImprovementAdmissionVersion   uint16 = 1
	ImprovementAdmissionMediaType        = "application/vnd.overgo.improvement-admission+json"
	ImprovementAdmissionSchema           = "overgo/improvement-admission/v1"
	ImprovementDecisionVersion    uint16 = 1
	ImprovementDecisionMediaType         = "application/vnd.overgo.improvement-decision+json"
	ImprovementDecisionSchema            = "overgo/improvement-decision/v1"
)

type ImprovementAdmission struct {
	Version          uint16      `json:"version"`
	Proposal         artifact.ID `json:"proposal"`
	ParentModel      artifact.ID `json:"parent_model"`
	Candidate        artifact.ID `json:"candidate"`
	Dataset          artifact.ID `json:"dataset"`
	DevelopmentSplit artifact.ID `json:"development_split"`
	PromotionSplit   artifact.ID `json:"promotion_split"`
	Recipe           artifact.ID `json:"recipe"`
	Code             artifact.ID `json:"code"`
	Proposer         artifact.ID `json:"proposer"`
	Evaluator        artifact.ID `json:"evaluator"`
	Authority        artifact.ID `json:"authority"`
	ID               artifact.ID `json:"-"`
}

type ImprovementDecisionState string

const (
	ImprovementPromote ImprovementDecisionState = "promote"
	ImprovementRefuse  ImprovementDecisionState = "refuse"
)

type ImprovementDecision struct {
	Version          uint16                   `json:"version"`
	State            ImprovementDecisionState `json:"state"`
	Admission        artifact.ID              `json:"admission"`
	ParentModel      artifact.ID              `json:"parent_model"`
	ChildModel       artifact.ID              `json:"child_model"`
	Dataset          artifact.ID              `json:"dataset"`
	DevelopmentSplit artifact.ID              `json:"development_split"`
	PromotionSplit   artifact.ID              `json:"promotion_split"`
	Recipe           artifact.ID              `json:"recipe"`
	Code             artifact.ID              `json:"code"`
	Evaluator        artifact.ID              `json:"evaluator"`
	Run              artifact.ID              `json:"run"`
	Evaluation       artifact.ID              `json:"evaluation"`
	Proposer         artifact.ID              `json:"proposer"`
	Authority        artifact.ID              `json:"authority"`
	Decider          artifact.ID              `json:"decider"`
	Rollback         artifact.ID              `json:"rollback"`
	ID               artifact.ID              `json:"-"`
}

var improvementAdmissionCodec = artifact.JSONDocumentCodec(
	"improvement admission", artifact.KindEvidence, ImprovementAdmissionMediaType, ImprovementAdmissionSchema,
	canonicalizeImprovementAdmission,
	func(value ImprovementAdmission) artifact.ID { return value.ID },
	func(value *ImprovementAdmission, id artifact.ID) { value.ID = id }, nil,
)

var improvementDecisionCodec = artifact.JSONDocumentCodec(
	"improvement decision", artifact.KindEvidence, ImprovementDecisionMediaType, ImprovementDecisionSchema,
	canonicalizeImprovementDecision,
	func(value ImprovementDecision) artifact.ID { return value.ID },
	func(value *ImprovementDecision, id artifact.ID) { value.ID = id }, nil,
)

func AdmitImprovement(
	proposal trainingprogram.ImprovementProposal, authority, evaluator, promotionSplit artifact.ID,
) (ImprovementAdmission, error) {
	return improvementAdmissionCodec.New(ImprovementAdmission{
		Version: ImprovementAdmissionVersion, Proposal: proposal.ID(), ParentModel: proposal.ParentModel(),
		Candidate: proposal.Candidate(), Dataset: proposal.Dataset(),
		DevelopmentSplit: proposal.DevelopmentSplit(), PromotionSplit: promotionSplit,
		Recipe: proposal.Recipe(), Code: proposal.Code(), Proposer: proposal.Proposer(),
		Evaluator: evaluator, Authority: authority,
	})
}

func DecideImprovement(
	admission ImprovementAdmission, child artifact.ID, run Run, evaluation Evaluation,
	decider artifact.ID, state ImprovementDecisionState,
) (ImprovementDecision, error) {
	if err := validateImprovementEvidence(admission, child, run, evaluation, decider, state); err != nil {
		return ImprovementDecision{}, err
	}
	return improvementDecisionCodec.New(ImprovementDecision{
		Version: ImprovementDecisionVersion, State: state, Admission: admission.ID,
		ParentModel: admission.ParentModel, ChildModel: child, Dataset: admission.Dataset,
		DevelopmentSplit: admission.DevelopmentSplit, PromotionSplit: admission.PromotionSplit,
		Recipe: admission.Recipe, Code: admission.Code, Evaluator: admission.Evaluator,
		Run: run.ID, Evaluation: evaluation.ID, Proposer: admission.Proposer,
		Authority: admission.Authority, Decider: decider, Rollback: admission.ParentModel,
	})
}

func (value ImprovementAdmission) ValidateIdentity() error {
	return improvementAdmissionCodec.ValidateIdentity(value)
}

func (value ImprovementDecision) ValidateIdentity() error {
	return improvementDecisionCodec.ValidateIdentity(value)
}

func (value ImprovementAdmission) Content() (artifact.Content, error) {
	return improvementAdmissionCodec.Content(value)
}

func (value ImprovementDecision) Content() (artifact.Content, error) {
	return improvementDecisionCodec.Content(value)
}

func (value ImprovementDecision) RollbackTarget() artifact.ID { return value.Rollback }

func (value ImprovementAdmission) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, value.Proposal, value.ParentModel, value.Dataset,
		value.Candidate, value.DevelopmentSplit, value.PromotionSplit, value.Recipe, value.Code,
		value.Proposer, value.Evaluator, value.Authority)
}

func (value ImprovementDecision) Lineage() []artifact.Lineage {
	lineage := artifact.DependencyLineage(value.ID, value.Admission, value.ParentModel, value.ChildModel,
		value.Dataset, value.DevelopmentSplit, value.PromotionSplit, value.Recipe, value.Code,
		value.Evaluator, value.Run, value.Evaluation, value.Proposer, value.Authority, value.Decider)
	lineage = append(lineage, artifact.Lineage{
		Child: value.ChildModel, Parent: value.ParentModel, Relation: artifact.RelationTrainedFrom,
	})
	for _, parent := range []artifact.ID{
		value.Dataset, value.DevelopmentSplit, value.PromotionSplit, value.Recipe, value.Code, value.Evaluator,
	} {
		lineage = append(lineage, artifact.Lineage{
			Child: value.ChildModel, Parent: parent, Relation: artifact.RelationDependsOn,
		})
	}
	return lineage
}

func (value ImprovementAdmission) Batch(key string) (artifact.Batch, error) {
	return improvementAdmissionCodec.Batch(key, value, value.Lineage(), nil)
}

func (value ImprovementDecision) Batch(key string) (artifact.Batch, error) {
	return improvementDecisionCodec.Batch(key, value, value.Lineage(), nil)
}

func canonicalizeImprovementAdmission(value *ImprovementAdmission) error {
	if value == nil || value.Version != ImprovementAdmissionVersion ||
		value.Proposal.Kind() != artifact.KindRecipe || value.ParentModel.Kind() != artifact.KindModel ||
		!value.Candidate.Valid() ||
		value.Dataset.Kind() != artifact.KindDataset || value.DevelopmentSplit.Kind() != artifact.KindDatasetShard ||
		value.PromotionSplit.Kind() != artifact.KindDatasetShard || value.DevelopmentSplit == value.PromotionSplit ||
		value.Recipe.Kind() != artifact.KindRecipe || value.Code.Kind() != artifact.KindEvidence ||
		value.Proposer.Kind() != artifact.KindEvidence || value.Evaluator.Kind() != artifact.KindEvidence ||
		value.Authority.Kind() != artifact.KindEvidence || value.Candidate == value.Evaluator ||
		!distinctIDs(value.Proposer, value.Evaluator, value.Authority, value.Code) {
		return errors.New("run record: invalid improvement admission")
	}
	return nil
}

func canonicalizeImprovementDecision(value *ImprovementDecision) error {
	if value == nil || value.Version != ImprovementDecisionVersion ||
		value.State != ImprovementPromote && value.State != ImprovementRefuse ||
		value.Admission.Kind() != artifact.KindEvidence || value.ParentModel.Kind() != artifact.KindModel ||
		value.ChildModel.Kind() != artifact.KindModel || value.ChildModel == value.ParentModel ||
		value.Dataset.Kind() != artifact.KindDataset || value.DevelopmentSplit.Kind() != artifact.KindDatasetShard ||
		value.PromotionSplit.Kind() != artifact.KindDatasetShard || value.DevelopmentSplit == value.PromotionSplit ||
		value.Recipe.Kind() != artifact.KindRecipe || value.Code.Kind() != artifact.KindEvidence ||
		value.Evaluator.Kind() != artifact.KindEvidence || value.Run.Kind() != artifact.KindRun ||
		value.Evaluation.Kind() != artifact.KindEvaluation || value.Proposer.Kind() != artifact.KindEvidence ||
		value.Authority.Kind() != artifact.KindEvidence || value.Decider.Kind() != artifact.KindEvidence ||
		value.Rollback != value.ParentModel ||
		!distinctIDs(value.Code, value.Proposer, value.Evaluator, value.Authority, value.Decider) {
		return errors.New("run record: invalid improvement decision")
	}
	return nil
}

func validateImprovementEvidence(
	admission ImprovementAdmission, child artifact.ID, run Run, evaluation Evaluation,
	decider artifact.ID, state ImprovementDecisionState,
) error {
	if err := admission.ValidateIdentity(); err != nil {
		return err
	}
	if err := errors.Join(run.ValidateIdentity(), evaluation.ValidateIdentity()); err != nil {
		return err
	}
	if child.Kind() != artifact.KindModel || decider.Kind() != artifact.KindEvidence ||
		!distinctIDs(admission.Proposer, admission.Authority, admission.Evaluator, admission.Code, decider) {
		return errors.New("run record: improvement decision authority is not independent")
	}
	if run.Recipe != admission.Recipe || evaluation.Recipe != admission.Recipe || evaluation.Run != run.ID ||
		evaluation.Dataset != admission.Dataset || !containsAllIDs(run.Inputs, child, admission.Dataset,
		admission.PromotionSplit, admission.Code, admission.ID) {
		return errors.New("run record: held-out improvement evidence graph differs")
	}
	if state == ImprovementPromote && run.Outcome != OutcomeSucceeded {
		return errors.New("run record: failed improvement run cannot promote")
	}
	return nil
}

func distinctIDs(values ...artifact.ID) bool {
	seen := make(map[artifact.ID]struct{}, len(values))
	for _, value := range values {
		if !value.Valid() {
			return false
		}
		if _, exists := seen[value]; exists {
			return false
		}
		seen[value] = struct{}{}
	}
	return true
}

func containsAllIDs(values []artifact.ID, required ...artifact.ID) bool {
	seen := make(map[artifact.ID]struct{}, len(values))
	for _, value := range values {
		seen[value] = struct{}{}
	}
	for _, value := range required {
		if _, exists := seen[value]; !exists {
			return false
		}
	}
	return true
}
