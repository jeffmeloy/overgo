package runrecord

import (
	"context"
	"errors"
	"strings"

	"overgo/internal/artifact"
)

// An attempt record is the measurement unit for automation
// effectiveness: one typed document per gate run, success or failure,
// binding the plan step the attempt served, the strategy that produced
// it, the manifest selection the gate observed, and the diff the
// candidate carried, to the gate result already in the store. Nothing
// here is asserted that the gate did not observe; the record makes the
// observation queryable across runs instead of leaving it in logs.
const (
	// AttemptMediaType identifies encoded gate-attempt documents.
	AttemptMediaType = "application/vnd.overgo.gate-attempt+json"
	// AttemptSchema identifies the exact stored gate-attempt schema.
	AttemptSchema = "overgo/gate-attempt/v1"
	// MaximumAttemptPopulation bounds one deterministic in-memory population
	// of attempt records or per-attempt evidence reads.
	MaximumAttemptPopulation = 4096
)

var attemptContract = artifact.DocumentContract{
	Kind: artifact.KindEvidence, MediaType: AttemptMediaType, Schema: AttemptSchema,
}

const maxAttemptStrategyBytes = 128

// AttemptSelection mirrors the gate's manifest selection measurements
// without importing the selector package: counts of check dispositions
// and the planning cost, exactly as measured.
type AttemptSelection struct {
	Defined       int   `json:"defined"`
	Selected      int   `json:"selected"`
	Excluded      int   `json:"excluded"`
	Uncertainty   int   `json:"uncertainty"`
	CacheEligible int   `json:"cache_eligible"`
	CacheHits     int   `json:"cache_hits"`
	PlanningNS    int64 `json:"planning_ns"`
}

// AttemptDiff is the candidate's observed change size.
type AttemptDiff struct {
	Files      int `json:"files"`
	Insertions int `json:"insertions"`
	Deletions  int `json:"deletions"`
}

// AttemptRecord binds one gate run to the plan step it served.
type AttemptRecord struct {
	Version           uint16           `json:"version"`
	PlanItem          string           `json:"plan_item"`
	PlanStep          string           `json:"plan_step"`
	Strategy          string           `json:"strategy,omitempty"`
	Result            artifact.ID      `json:"result"`
	Recipe            artifact.ID      `json:"recipe"`
	CodeCommit        string           `json:"code_commit"`
	Outcome           Outcome          `json:"outcome"`
	Failure           string           `json:"failure,omitempty"`
	WallNS            uint64           `json:"wall_ns"`
	BaseManifest      artifact.ID      `json:"base_manifest,omitzero"`
	CandidateManifest artifact.ID      `json:"candidate_manifest,omitzero"`
	Selection         AttemptSelection `json:"selection"`
	Diff              AttemptDiff      `json:"diff"`
	StrategyID        artifact.ID      `json:"strategy_id,omitzero"`
	TaskContract      artifact.ID      `json:"task_contract,omitzero"`
	WorkLease         artifact.ID      `json:"work_lease,omitzero"`
	Environment       artifact.ID      `json:"environment,omitzero"`
	Trajectory        artifact.ID      `json:"trajectory,omitzero"`
	CostUnits         uint64           `json:"cost_units,omitempty"`
	// Causal explains why the attempt ran; a recovery trigger here
	// replaces the deleted duplicate recovered flag.
	Causal *CausalContext `json:"causal,omitempty"`
	ID     artifact.ID    `json:"-"`
}

var attemptCodec = artifact.JSONDocumentCodec(
	"run record gate attempt", attemptContract.Kind, attemptContract.MediaType, attemptContract.Schema,
	canonicalizeAttempt, func(value AttemptRecord) artifact.ID { return value.ID },
	func(value *AttemptRecord, id artifact.ID) { value.ID = id },
	func(value AttemptRecord) AttemptRecord {
		value.Causal = cloneCausal(value.Causal)
		return value
	},
)

func canonicalizeAttempt(value *AttemptRecord) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion {
		return errors.New("run record: invalid attempt version")
	}
	if strings.TrimSpace(value.PlanItem) == "" || strings.TrimSpace(value.PlanStep) == "" ||
		strings.ContainsAny(value.PlanItem+value.PlanStep, "/\x00\r\n\t ") {
		return errors.New("run record: attempt requires its plan item and step")
	}
	// Strategy is optional -- an interactive session declares none --
	// but a declared identity is one bounded token, never free text.
	if value.Strategy != "" &&
		(len(value.Strategy) > maxAttemptStrategyBytes || strings.ContainsAny(value.Strategy, " /\x00\r\n\t")) {
		return errors.New("run record: attempt strategy must be one bounded token")
	}
	if value.Result.Kind() != artifact.KindEvidence {
		return errors.New("run record: attempt requires the gate result it binds")
	}
	if !value.Recipe.Valid() {
		return errors.New("run record: attempt requires the gate recipe identity")
	}
	if !validCodeCommit(value.CodeCommit) {
		return errors.New("run record: attempt requires the verifying repository commit")
	}
	switch value.Outcome {
	case OutcomeSucceeded:
		if value.Failure != "" {
			return errors.New("run record: a succeeded attempt carries no failure")
		}
	case OutcomeFailed:
		if strings.TrimSpace(value.Failure) == "" {
			return errors.New("run record: a failed attempt names its failing step")
		}
	default:
		return errors.New("run record: invalid attempt outcome")
	}
	if value.WallNS == 0 {
		return errors.New("run record: attempt requires its measured wall")
	}
	if value.Selection.Defined < 0 || value.Selection.Selected < 0 || value.Selection.Excluded < 0 ||
		value.Selection.Uncertainty < 0 || value.Selection.CacheEligible < 0 ||
		value.Selection.CacheHits < 0 || value.Selection.PlanningNS < 0 ||
		value.Selection.CacheHits > value.Selection.CacheEligible {
		return errors.New("run record: attempt selection counts are not observations")
	}
	if value.Diff.Files < 0 || value.Diff.Insertions < 0 || value.Diff.Deletions < 0 {
		return errors.New("run record: attempt diff counts are not observations")
	}
	if err := validCausal(value.Causal); err != nil {
		return err
	}
	for _, authority := range []struct {
		id   artifact.ID
		kind artifact.Kind
	}{
		{value.StrategyID, artifact.KindProfile},
		{value.TaskContract, artifact.KindRecipe},
		{value.WorkLease, artifact.KindEvidence},
		{value.Environment, artifact.KindEvidence},
		{value.Trajectory, artifact.KindEvidence},
	} {
		if authority.id.Valid() && authority.id.Kind() != authority.kind {
			return errors.New("run record: invalid attempt authority")
		}
	}
	return nil
}

// NewAttemptRecord validates and identifies one attempt.
func NewAttemptRecord(record AttemptRecord) (AttemptRecord, error) {
	return attemptCodec.NewInitial(record)
}

// ValidateIdentity proves that the attempt still matches its immutable
// content-addressed identity. Comparison admission uses this to reject a
// strategy ID attached after the gate identified the record.
func (a AttemptRecord) ValidateIdentity() error {
	return attemptCodec.ValidateIdentity(a)
}

// RequireAttemptRecord loads one canonical gate-attempt document.
func RequireAttemptRecord(ctx context.Context, reader artifact.Reader, id artifact.ID) (AttemptRecord, error) {
	return attemptCodec.Require(ctx, reader, id)
}

// Content encodes the attempt for its store batch.
func (a AttemptRecord) Content() (artifact.Content, error) {
	return attemptCodec.Content(a)
}

// Lineage binds the attempt to the gate result and recipe it observed.
func (a AttemptRecord) Lineage() []artifact.Lineage {
	lineage := []artifact.Lineage{
		{Child: a.ID, Parent: a.Result, Relation: artifact.RelationDependsOn},
		{Child: a.ID, Parent: a.Recipe, Relation: artifact.RelationDependsOn},
	}
	for _, manifest := range []artifact.ID{a.BaseManifest, a.CandidateManifest} {
		if manifest.Valid() {
			lineage = append(lineage, artifact.Lineage{
				Child: a.ID, Parent: manifest, Relation: artifact.RelationDependsOn,
			})
		}
	}
	for _, parent := range []artifact.ID{a.StrategyID, a.TaskContract, a.WorkLease, a.Environment, a.Trajectory} {
		if parent.Valid() {
			lineage = append(lineage, artifact.Lineage{Child: a.ID, Parent: parent, Relation: artifact.RelationDependsOn})
		}
	}
	return lineage
}
