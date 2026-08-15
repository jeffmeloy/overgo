package runrecord

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
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

type ReviewAdmission struct {
	Developer         ReviewActor
	Reviewer          ReviewActor
	DeveloperWorktree ReviewWorktree
	ReviewWorktree    ReviewWorktree
	Evaluator         ReviewEvaluator
	Candidate         ReviewCandidate
	Findings          []ReviewFinding
	Verdict           ReviewVerdict
}

type ReviewPhase string

const (
	ReviewPhaseImplementation ReviewPhase = "implementation"
	ReviewPhaseSQA            ReviewPhase = "sqa"
	ReviewPhasePriority       ReviewPhase = "priority"
)

// ReviewPriority is the phase implied by immutable review evidence at one Git
// head. It guides dispatch without introducing another task or rank surface.
type ReviewPriority struct {
	Phase     ReviewPhase
	Candidate artifact.ID
	Verdict   artifact.ID
}

var reviewActorCodec = artifact.JSONDocumentCodec("review actor", artifact.KindEvidence, ReviewActorMediaType, ReviewActorSchema,
	func(value *ReviewActor) error {
		if value == nil || value.Version != ReviewVersion || !textcheck.Bounded(value.Principal, 2048, "\x00\r\n") ||
			value.Role != ReviewDeveloper && value.Role != ReviewSQA {
			return errors.New("run record: invalid review actor")
		}
		return nil
	}, func(value ReviewActor) artifact.ID { return value.ID }, func(value *ReviewActor, id artifact.ID) { value.ID = id }, nil)

var reviewWorktreeCodec = artifact.JSONDocumentCodec("review worktree", artifact.KindEvidence, ReviewWorktreeMediaType, ReviewWorktreeSchema,
	func(value *ReviewWorktree) error {
		if value == nil || value.Version != ReviewVersion || !textcheck.Bounded(value.Path, 2048, "\x00\r\n") ||
			strings.Contains(value.Path, "\\") || !textcheck.Bounded(value.Branch, 2048, "\x00\r\n") || !validCodeCommit(value.Head) {
			return errors.New("run record: invalid review worktree")
		}
		return nil
	}, func(value ReviewWorktree) artifact.ID { return value.ID }, func(value *ReviewWorktree, id artifact.ID) { value.ID = id }, nil)

var reviewEvaluatorCodec = artifact.JSONDocumentCodec("review evaluator", artifact.KindEvidence, ReviewEvaluatorMediaType, ReviewEvaluatorSchema,
	func(value *ReviewEvaluator) error {
		if value == nil || value.Version != ReviewVersion || !textcheck.Bounded(value.Name, 2048, "\x00\r\n") ||
			!value.Definition.Valid() || !validCodeCommit(value.Revision) {
			return errors.New("run record: invalid review evaluator")
		}
		return nil
	}, func(value ReviewEvaluator) artifact.ID { return value.ID }, func(value *ReviewEvaluator, id artifact.ID) { value.ID = id }, nil)

var reviewCandidateCodec = artifact.JSONDocumentCodec("review candidate", artifact.KindEvidence, ReviewCandidateMediaType, ReviewCandidateSchema,
	func(value *ReviewCandidate) error {
		if value == nil || value.Version != ReviewVersion || !validCodeCommit(value.BaseCommit) ||
			!validCodeCommit(value.CodeCommit) || value.BaseCommit == value.CodeCommit ||
			!allEvidenceIDs(value.Developer, value.Worktree, value.Evaluator, value.GateResult, value.GateRun) {
			return errors.New("run record: invalid review candidate")
		}
		return nil
	}, func(value ReviewCandidate) artifact.ID { return value.ID }, func(value *ReviewCandidate, id artifact.ID) { value.ID = id }, nil)

var reviewFindingCodec = artifact.JSONDocumentCodec("review finding", artifact.KindEvidence, ReviewFindingMediaType, ReviewFindingSchema,
	func(value *ReviewFinding) error {
		if value == nil || value.Version != ReviewVersion || !allEvidenceIDs(value.Candidate, value.Reviewer, value.Evaluator) ||
			!textcheck.Bounded(value.Summary, 2048, "\x00\r\n") || !textcheck.Bounded(value.Check, 2048, "\x00\r\n") ||
			value.Severity != ReviewCritical && value.Severity != ReviewMajor && value.Severity != ReviewMinor && value.Severity != ReviewInfo ||
			value.Status != ReviewFindingOpen && value.Status != ReviewFindingResolved && value.Status != ReviewFindingAccepted {
			return errors.New("run record: invalid review finding")
		}
		return nil
	}, func(value ReviewFinding) artifact.ID { return value.ID }, func(value *ReviewFinding, id artifact.ID) { value.ID = id }, nil)

var reviewVerdictCodec = artifact.JSONDocumentCodec("review verdict", artifact.KindEvidence, ReviewVerdictMediaType, ReviewVerdictSchema,
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
	}, func(value ReviewVerdict) artifact.ID { return value.ID }, func(value *ReviewVerdict, id artifact.ID) { value.ID = id },
	func(value ReviewVerdict) ReviewVerdict { value.Findings = slices.Clone(value.Findings); return value })

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
	return dependencyLineage(value.ID, value.Definition)
}

func (value ReviewCandidate) Lineage() []artifact.Lineage {
	parents := []artifact.ID{value.Developer, value.Worktree, value.Evaluator, value.GateResult, value.GateRun}
	return dependencyLineage(value.ID, parents...)
}

func (value ReviewFinding) Lineage() []artifact.Lineage {
	return dependencyLineage(value.ID, value.Candidate, value.Reviewer, value.Evaluator)
}

func (value ReviewVerdict) Lineage() []artifact.Lineage {
	parents := []artifact.ID{value.Candidate, value.Reviewer, value.Worktree, value.Evaluator}
	parents = append(parents, value.Findings...)
	return dependencyLineage(value.ID, parents...)
}

// AdmitReview verifies that an approved verdict covers one immutable candidate
// at the exact target head and was produced independently on frozen evidence.
func AdmitReview(targetHead string, admission ReviewAdmission) error {
	if !validCodeCommit(targetHead) {
		return errors.New("run record: review admission target is invalid")
	}
	identities := []error{
		admission.Developer.ValidateIdentity(), admission.Reviewer.ValidateIdentity(),
		admission.DeveloperWorktree.ValidateIdentity(), admission.ReviewWorktree.ValidateIdentity(),
		admission.Evaluator.ValidateIdentity(), admission.Candidate.ValidateIdentity(), admission.Verdict.ValidateIdentity(),
	}
	for _, finding := range admission.Findings {
		identities = append(identities, finding.ValidateIdentity())
	}
	for _, err := range identities {
		if err != nil {
			return fmt.Errorf("run record: review admission identity: %w", err)
		}
	}
	if admission.Developer.Role != ReviewDeveloper || admission.Reviewer.Role != ReviewSQA ||
		admission.Developer.ID == admission.Reviewer.ID || admission.Developer.Principal == admission.Reviewer.Principal {
		return errors.New("run record: review admission requires distinct developer and SQA identities")
	}
	if !admission.DeveloperWorktree.Clean || !admission.ReviewWorktree.Clean ||
		admission.DeveloperWorktree.ID == admission.ReviewWorktree.ID ||
		admission.DeveloperWorktree.Path == admission.ReviewWorktree.Path {
		return errors.New("run record: review admission requires clean separate worktrees")
	}
	candidate, verdict := admission.Candidate, admission.Verdict
	if candidate.Developer != admission.Developer.ID || candidate.Worktree != admission.DeveloperWorktree.ID ||
		candidate.Evaluator != admission.Evaluator.ID || verdict.Candidate != candidate.ID ||
		verdict.Reviewer != admission.Reviewer.ID || verdict.Worktree != admission.ReviewWorktree.ID ||
		verdict.Evaluator != admission.Evaluator.ID {
		return errors.New("run record: review admission identity graph is inconsistent")
	}
	if admission.Evaluator.Revision != candidate.BaseCommit || candidate.CodeCommit != targetHead ||
		admission.DeveloperWorktree.Head != targetHead || admission.ReviewWorktree.Head != targetHead ||
		verdict.TargetHead != targetHead {
		return errors.New("run record: review admission is not frozen at the target head")
	}
	findings := make(map[artifact.ID]ReviewFinding, len(admission.Findings))
	for _, finding := range admission.Findings {
		if _, duplicate := findings[finding.ID]; duplicate || finding.Candidate != candidate.ID ||
			finding.Reviewer != admission.Reviewer.ID || finding.Evaluator != admission.Evaluator.ID {
			return errors.New("run record: review admission finding graph is inconsistent")
		}
		findings[finding.ID] = finding
	}
	if len(findings) != len(verdict.Findings) {
		return errors.New("run record: review admission verdict omits findings")
	}
	for _, id := range verdict.Findings {
		finding, ok := findings[id]
		if !ok {
			return errors.New("run record: review admission verdict names an unknown finding")
		}
		if finding.Status == ReviewFindingOpen {
			return errors.New("run record: review admission has an open finding")
		}
	}
	if verdict.Outcome != ReviewApproved {
		return errors.New("run record: review admission verdict is not approved")
	}
	return nil
}

// LoadReviewAdmission resolves the complete immutable graph named by a
// verdict. Storage mechanics live here so every consumer validates the same
// document contracts before calling AdmitReview.
func LoadReviewAdmission(ctx context.Context, reader artifact.Reader, verdictID artifact.ID) (ReviewAdmission, error) {
	verdict, err := readReviewDocument(ctx, reader, verdictID, reviewVerdictCodec)
	if err != nil {
		return ReviewAdmission{}, err
	}
	return loadReviewAdmission(ctx, reader, verdict)
}

// DeriveReviewPriority maps Git HEAD plus admitted RepoDB review evidence to
// implementation -> SQA -> priority. Invalid or rejected verdicts do not
// advance the phase; incomplete evidence for the current candidate is loud.
func DeriveReviewPriority(ctx context.Context, reader artifact.Reader, targetHead string, descriptors []artifact.Descriptor) (ReviewPriority, error) {
	if !validCodeCommit(targetHead) {
		return ReviewPriority{}, errors.New("run record: review priority target is invalid")
	}
	candidates := make(map[artifact.ID]ReviewCandidate)
	var selected artifact.ID
	for _, descriptor := range descriptors {
		if descriptor.MediaType != ReviewCandidateMediaType || descriptor.Schema != ReviewCandidateSchema {
			continue
		}
		candidate, err := readReviewDocument(ctx, reader, descriptor.ID, reviewCandidateCodec)
		if err != nil {
			return ReviewPriority{}, fmt.Errorf("run record: review priority candidate: %w", err)
		}
		if candidate.CodeCommit != targetHead {
			continue
		}
		candidates[candidate.ID] = candidate
		if !selected.Valid() || candidate.ID.String() < selected.String() {
			selected = candidate.ID
		}
	}
	if len(candidates) == 0 {
		return ReviewPriority{Phase: ReviewPhaseImplementation}, nil
	}
	priority := ReviewPriority{Phase: ReviewPhaseSQA, Candidate: selected}
	for _, descriptor := range descriptors {
		if descriptor.MediaType != ReviewVerdictMediaType || descriptor.Schema != ReviewVerdictSchema {
			continue
		}
		verdict, err := readReviewDocument(ctx, reader, descriptor.ID, reviewVerdictCodec)
		if err != nil {
			return ReviewPriority{}, fmt.Errorf("run record: review priority verdict: %w", err)
		}
		if verdict.TargetHead != targetHead {
			continue
		}
		if _, ok := candidates[verdict.Candidate]; !ok {
			continue
		}
		admission, err := loadReviewAdmission(ctx, reader, verdict)
		if err != nil {
			return ReviewPriority{}, fmt.Errorf("run record: review priority admission: %w", err)
		}
		if AdmitReview(targetHead, admission) != nil {
			continue
		}
		if !priority.Verdict.Valid() || verdict.ID.String() < priority.Verdict.String() {
			priority = ReviewPriority{Phase: ReviewPhasePriority, Candidate: verdict.Candidate, Verdict: verdict.ID}
		}
	}
	return priority, nil
}

func loadReviewAdmission(ctx context.Context, reader artifact.Reader, verdict ReviewVerdict) (ReviewAdmission, error) {
	candidate, err := readReviewDocument(ctx, reader, verdict.Candidate, reviewCandidateCodec)
	if err != nil {
		return ReviewAdmission{}, err
	}
	developer, err := readReviewDocument(ctx, reader, candidate.Developer, reviewActorCodec)
	if err != nil {
		return ReviewAdmission{}, err
	}
	reviewer, err := readReviewDocument(ctx, reader, verdict.Reviewer, reviewActorCodec)
	if err != nil {
		return ReviewAdmission{}, err
	}
	developerWorktree, err := readReviewDocument(ctx, reader, candidate.Worktree, reviewWorktreeCodec)
	if err != nil {
		return ReviewAdmission{}, err
	}
	reviewWorktree, err := readReviewDocument(ctx, reader, verdict.Worktree, reviewWorktreeCodec)
	if err != nil {
		return ReviewAdmission{}, err
	}
	evaluator, err := readReviewDocument(ctx, reader, verdict.Evaluator, reviewEvaluatorCodec)
	if err != nil {
		return ReviewAdmission{}, err
	}
	findings := make([]ReviewFinding, 0, len(verdict.Findings))
	for _, id := range verdict.Findings {
		finding, err := readReviewDocument(ctx, reader, id, reviewFindingCodec)
		if err != nil {
			return ReviewAdmission{}, err
		}
		findings = append(findings, finding)
	}
	return ReviewAdmission{
		Developer: developer, Reviewer: reviewer, DeveloperWorktree: developerWorktree, ReviewWorktree: reviewWorktree,
		Evaluator: evaluator, Candidate: candidate, Findings: findings, Verdict: verdict,
	}, nil
}

func readReviewDocument[T any](ctx context.Context, reader artifact.Reader, id artifact.ID, codec artifact.DocumentCodec[T]) (T, error) {
	var zero T
	content, ok, err := artifact.ReadDocument(ctx, reader, id, codec.Contract)
	if err != nil {
		return zero, fmt.Errorf("load review document %s: %w", id, err)
	}
	if !ok {
		return zero, fmt.Errorf("review document %s is absent", id)
	}
	value, err := codec.Parse(content.Data)
	if err != nil {
		return zero, fmt.Errorf("parse review document %s: %w", id, err)
	}
	return value, nil
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
