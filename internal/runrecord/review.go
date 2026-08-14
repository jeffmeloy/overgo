package runrecord

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

const ReviewVersion uint16 = 1

const (
	ReviewActorMediaType     = "application/vnd.overgo.review-actor+json"
	ReviewActorSchema        = "overgo/review-actor/v1"
	ReviewWorktreeMediaType  = "application/vnd.overgo.review-worktree+json"
	ReviewWorktreeSchema     = "overgo/review-worktree/v1"
	ReviewEvaluatorMediaType = "application/vnd.overgo.review-evaluator+json"
	ReviewEvaluatorSchema    = "overgo/review-evaluator/v1"
	ReviewCandidateMediaType = "application/vnd.overgo.review-candidate+json"
	ReviewCandidateSchema    = "overgo/review-candidate/v1"
	ReviewFindingMediaType   = "application/vnd.overgo.review-finding+json"
	ReviewFindingSchema      = "overgo/review-finding/v1"
	ReviewVerdictMediaType   = "application/vnd.overgo.review-verdict+json"
	ReviewVerdictSchema      = "overgo/review-verdict/v1"
)

type ReviewRole string

const (
	ReviewDeveloper ReviewRole = "developer"
	ReviewSQA       ReviewRole = "sqa"
)

type ReviewActor struct {
	Version   uint16      `json:"version"`
	Principal string      `json:"principal"`
	Role      ReviewRole  `json:"role"`
	ID        artifact.ID `json:"-"`
}

type ReviewWorktree struct {
	Version uint16      `json:"version"`
	Path    string      `json:"path"`
	Branch  string      `json:"branch"`
	Head    string      `json:"head"`
	Clean   bool        `json:"clean"`
	ID      artifact.ID `json:"-"`
}

type ReviewEvaluator struct {
	Version    uint16      `json:"version"`
	Name       string      `json:"name"`
	Definition artifact.ID `json:"definition"`
	Revision   string      `json:"revision"`
	ID         artifact.ID `json:"-"`
}

type ReviewCandidate struct {
	Version    uint16      `json:"version"`
	BaseCommit string      `json:"base_commit"`
	CodeCommit string      `json:"code_commit"`
	Developer  artifact.ID `json:"developer"`
	Worktree   artifact.ID `json:"worktree"`
	Evaluator  artifact.ID `json:"evaluator"`
	GateResult artifact.ID `json:"gate_result"`
	GateRun    artifact.ID `json:"gate_run"`
	ID         artifact.ID `json:"-"`
}

type ReviewSeverity string
type ReviewFindingStatus string

const (
	ReviewCritical ReviewSeverity = "critical"
	ReviewMajor    ReviewSeverity = "major"
	ReviewMinor    ReviewSeverity = "minor"
	ReviewInfo     ReviewSeverity = "info"

	ReviewFindingOpen     ReviewFindingStatus = "open"
	ReviewFindingResolved ReviewFindingStatus = "resolved"
	ReviewFindingAccepted ReviewFindingStatus = "accepted_risk"
)

type ReviewFinding struct {
	Version   uint16              `json:"version"`
	Candidate artifact.ID         `json:"candidate"`
	Reviewer  artifact.ID         `json:"reviewer"`
	Evaluator artifact.ID         `json:"evaluator"`
	Severity  ReviewSeverity      `json:"severity"`
	Status    ReviewFindingStatus `json:"status"`
	Summary   string              `json:"summary"`
	Check     string              `json:"check"`
	ID        artifact.ID         `json:"-"`
}

type ReviewVerdictOutcome string

const (
	ReviewApproved        ReviewVerdictOutcome = "approved"
	ReviewChangesRequired ReviewVerdictOutcome = "changes_required"
)

type ReviewVerdict struct {
	Version    uint16               `json:"version"`
	Candidate  artifact.ID          `json:"candidate"`
	Reviewer   artifact.ID          `json:"reviewer"`
	Worktree   artifact.ID          `json:"worktree"`
	Evaluator  artifact.ID          `json:"evaluator"`
	TargetHead string               `json:"target_head"`
	Findings   []artifact.ID        `json:"findings"`
	Outcome    ReviewVerdictOutcome `json:"outcome"`
	ID         artifact.ID          `json:"-"`
}

var reviewActorCodec = reviewCodec("review actor", ReviewActorMediaType, ReviewActorSchema,
	func(value *ReviewActor) error {
		if value == nil || value.Version != ReviewVersion || !validReviewText(value.Principal) ||
			value.Role != ReviewDeveloper && value.Role != ReviewSQA {
			return errors.New("run record: invalid review actor")
		}
		return nil
	}, func(value ReviewActor) artifact.ID { return value.ID }, func(value *ReviewActor, id artifact.ID) { value.ID = id })

var reviewWorktreeCodec = reviewCodec("review worktree", ReviewWorktreeMediaType, ReviewWorktreeSchema,
	func(value *ReviewWorktree) error {
		if value == nil || value.Version != ReviewVersion || !validReviewText(value.Path) ||
			strings.Contains(value.Path, "\\") || !validReviewText(value.Branch) || !validCodeCommit(value.Head) {
			return errors.New("run record: invalid review worktree")
		}
		return nil
	}, func(value ReviewWorktree) artifact.ID { return value.ID }, func(value *ReviewWorktree, id artifact.ID) { value.ID = id })

var reviewEvaluatorCodec = reviewCodec("review evaluator", ReviewEvaluatorMediaType, ReviewEvaluatorSchema,
	func(value *ReviewEvaluator) error {
		if value == nil || value.Version != ReviewVersion || !validReviewText(value.Name) ||
			!value.Definition.Valid() || !validCodeCommit(value.Revision) {
			return errors.New("run record: invalid review evaluator")
		}
		return nil
	}, func(value ReviewEvaluator) artifact.ID { return value.ID }, func(value *ReviewEvaluator, id artifact.ID) { value.ID = id })

var reviewCandidateCodec = reviewCodec("review candidate", ReviewCandidateMediaType, ReviewCandidateSchema,
	func(value *ReviewCandidate) error {
		if value == nil || value.Version != ReviewVersion || !validCodeCommit(value.BaseCommit) ||
			!validCodeCommit(value.CodeCommit) || value.BaseCommit == value.CodeCommit ||
			!allEvidenceIDs(value.Developer, value.Worktree, value.Evaluator, value.GateResult, value.GateRun) {
			return errors.New("run record: invalid review candidate")
		}
		return nil
	}, func(value ReviewCandidate) artifact.ID { return value.ID }, func(value *ReviewCandidate, id artifact.ID) { value.ID = id })

var reviewFindingCodec = reviewCodec("review finding", ReviewFindingMediaType, ReviewFindingSchema,
	func(value *ReviewFinding) error {
		if value == nil || value.Version != ReviewVersion || !allEvidenceIDs(value.Candidate, value.Reviewer, value.Evaluator) ||
			!validReviewText(value.Summary) || !validReviewText(value.Check) ||
			value.Severity != ReviewCritical && value.Severity != ReviewMajor && value.Severity != ReviewMinor && value.Severity != ReviewInfo ||
			value.Status != ReviewFindingOpen && value.Status != ReviewFindingResolved && value.Status != ReviewFindingAccepted {
			return errors.New("run record: invalid review finding")
		}
		return nil
	}, func(value ReviewFinding) artifact.ID { return value.ID }, func(value *ReviewFinding, id artifact.ID) { value.ID = id })

var reviewVerdictCodec = reviewCodec("review verdict", ReviewVerdictMediaType, ReviewVerdictSchema,
	func(value *ReviewVerdict) error {
		if value == nil || value.Version != ReviewVersion || !allEvidenceIDs(value.Candidate, value.Reviewer, value.Worktree, value.Evaluator) ||
			!validCodeCommit(value.TargetHead) || value.Findings == nil ||
			value.Outcome != ReviewApproved && value.Outcome != ReviewChangesRequired {
			return errors.New("run record: invalid review verdict")
		}
		seen := map[artifact.ID]bool{}
		for _, id := range value.Findings {
			if id.Kind() != artifact.KindEvidence || seen[id] {
				return errors.New("run record: invalid review verdict findings")
			}
			seen[id] = true
		}
		return nil
	}, func(value ReviewVerdict) artifact.ID { return value.ID }, func(value *ReviewVerdict, id artifact.ID) { value.ID = id })

func reviewCodec[T any](name, mediaType, schema string, canonicalize func(*T) error, identity func(T) artifact.ID, setIdentity func(*T, artifact.ID)) artifact.DocumentCodec[T] {
	contract := artifact.DocumentContract{Kind: artifact.KindEvidence, MediaType: mediaType, Schema: schema}
	return artifact.DocumentCodec[T]{
		Name: name, Contract: contract,
		Decode: func(data []byte, value *T) error { return strictjson.DecodeBytes(data, value) },
		Encode: func(value T) ([]byte, error) { return json.Marshal(value) }, Canonicalize: canonicalize,
		Identity: identity, SetIdentity: setIdentity,
	}
}

func NewReviewActor(principal string, role ReviewRole) (ReviewActor, error) {
	return reviewActorCodec.New(ReviewActor{Version: ReviewVersion, Principal: principal, Role: role})
}

func NewReviewWorktree(path, branch, head string, clean bool) (ReviewWorktree, error) {
	return reviewWorktreeCodec.New(ReviewWorktree{Version: ReviewVersion, Path: path, Branch: branch, Head: head, Clean: clean})
}

func NewReviewEvaluator(name string, definition artifact.ID, revision string) (ReviewEvaluator, error) {
	return reviewEvaluatorCodec.New(ReviewEvaluator{Version: ReviewVersion, Name: name, Definition: definition, Revision: revision})
}

func NewReviewCandidate(candidate ReviewCandidate) (ReviewCandidate, error) {
	candidate.Version = ReviewVersion
	return reviewCandidateCodec.New(candidate)
}

func NewReviewFinding(finding ReviewFinding) (ReviewFinding, error) {
	finding.Version = ReviewVersion
	return reviewFindingCodec.New(finding)
}

func NewReviewVerdict(verdict ReviewVerdict) (ReviewVerdict, error) {
	verdict.Version = ReviewVersion
	verdict.Findings = slices.Clone(verdict.Findings)
	return reviewVerdictCodec.New(verdict)
}

func (value ReviewActor) Content() (artifact.Content, error) { return reviewActorCodec.Content(value) }
func (value ReviewWorktree) Content() (artifact.Content, error) {
	return reviewWorktreeCodec.Content(value)
}
func (value ReviewEvaluator) Content() (artifact.Content, error) {
	return reviewEvaluatorCodec.Content(value)
}
func (value ReviewCandidate) Content() (artifact.Content, error) {
	return reviewCandidateCodec.Content(value)
}
func (value ReviewFinding) Content() (artifact.Content, error) {
	return reviewFindingCodec.Content(value)
}
func (value ReviewVerdict) Content() (artifact.Content, error) {
	return reviewVerdictCodec.Content(value)
}
func (value ReviewActor) ValidateIdentity() error { return reviewActorCodec.ValidateIdentity(value) }
func (value ReviewWorktree) ValidateIdentity() error {
	return reviewWorktreeCodec.ValidateIdentity(value)
}
func (value ReviewEvaluator) ValidateIdentity() error {
	return reviewEvaluatorCodec.ValidateIdentity(value)
}
func (value ReviewCandidate) ValidateIdentity() error {
	return reviewCandidateCodec.ValidateIdentity(value)
}
func (value ReviewFinding) ValidateIdentity() error {
	return reviewFindingCodec.ValidateIdentity(value)
}
func (value ReviewVerdict) ValidateIdentity() error {
	return reviewVerdictCodec.ValidateIdentity(value)
}

func ParseReviewActor(data []byte) (ReviewActor, error)       { return reviewActorCodec.Parse(data) }
func ParseReviewWorktree(data []byte) (ReviewWorktree, error) { return reviewWorktreeCodec.Parse(data) }
func ParseReviewEvaluator(data []byte) (ReviewEvaluator, error) {
	return reviewEvaluatorCodec.Parse(data)
}
func ParseReviewCandidate(data []byte) (ReviewCandidate, error) {
	return reviewCandidateCodec.Parse(data)
}
func ParseReviewFinding(data []byte) (ReviewFinding, error) { return reviewFindingCodec.Parse(data) }
func ParseReviewVerdict(data []byte) (ReviewVerdict, error) { return reviewVerdictCodec.Parse(data) }

func (value ReviewEvaluator) Lineage() []artifact.Lineage {
	return []artifact.Lineage{{Child: value.ID, Parent: value.Definition, Relation: artifact.RelationDependsOn}}
}

func (value ReviewCandidate) Lineage() []artifact.Lineage {
	parents := []artifact.ID{value.Developer, value.Worktree, value.Evaluator, value.GateResult, value.GateRun}
	lineage := make([]artifact.Lineage, 0, len(parents))
	for _, parent := range parents {
		lineage = append(lineage, artifact.Lineage{Child: value.ID, Parent: parent, Relation: artifact.RelationDependsOn})
	}
	return lineage
}

func (value ReviewFinding) Lineage() []artifact.Lineage {
	return []artifact.Lineage{
		{Child: value.ID, Parent: value.Candidate, Relation: artifact.RelationDependsOn},
		{Child: value.ID, Parent: value.Reviewer, Relation: artifact.RelationDependsOn},
		{Child: value.ID, Parent: value.Evaluator, Relation: artifact.RelationDependsOn},
	}
}

func (value ReviewVerdict) Lineage() []artifact.Lineage {
	parents := []artifact.ID{value.Candidate, value.Reviewer, value.Worktree, value.Evaluator}
	parents = append(parents, value.Findings...)
	lineage := make([]artifact.Lineage, 0, len(parents))
	for _, parent := range parents {
		lineage = append(lineage, artifact.Lineage{Child: value.ID, Parent: parent, Relation: artifact.RelationDependsOn})
	}
	return lineage
}

func validReviewText(value string) bool {
	return value != "" && len(value) <= 2048 && strings.TrimSpace(value) == value && !strings.ContainsAny(value, "\x00\r\n")
}

func allEvidenceIDs(ids ...artifact.ID) bool {
	for _, id := range ids {
		if id.Kind() != artifact.KindEvidence {
			return false
		}
	}
	return true
}

func ReviewContents(values ...interface {
	Content() (artifact.Content, error)
}) ([]artifact.Content, error) {
	contents := make([]artifact.Content, 0, len(values))
	for _, value := range values {
		content, err := value.Content()
		if err != nil {
			return nil, fmt.Errorf("run record: review content: %w", err)
		}
		contents = append(contents, content)
	}
	return contents, nil
}
