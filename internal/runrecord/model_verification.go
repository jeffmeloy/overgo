package runrecord

import (
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
)

const (
	ModelVerificationVersion   uint16 = 1
	ModelVerificationMediaType        = "application/vnd.overgo.model-verification+json"
	ModelVerificationSchema           = "overgo/model-verification/v1"
)

// VerificationTier defines evidenced capability level; no implicit subsumption.
type VerificationTier string

const (
	TierSyntheticDifferential VerificationTier = "synthetic-differential"
	TierRealArtifactSmoke     VerificationTier = "real-artifact-smoke"
	TierExactGolden           VerificationTier = "exact-golden"
	TierCapabilityMeasured    VerificationTier = "capability-measured"
)

// Valid reports whether the tier belongs to the verification evidence contract.
func (t VerificationTier) Valid() bool {
	switch t {
	case TierSyntheticDifferential, TierRealArtifactSmoke, TierExactGolden, TierCapabilityMeasured:
		return true
	default:
		return false
	}
}

// StrongerThan reports strict evidence precedence without assigning numeric scores.
func (t VerificationTier) StrongerThan(other VerificationTier) bool {
	switch t {
	case TierCapabilityMeasured:
		return other.Valid() && other != TierCapabilityMeasured
	case TierExactGolden:
		return other == TierRealArtifactSmoke || other == TierSyntheticDifferential
	case TierRealArtifactSmoke:
		return other == TierSyntheticDifferential
	default:
		return false
	}
}

// CapabilityClaim binds capability proof to code, data, span, and evidence.
type CapabilityClaim struct {
	Capability      string           `json:"capability"`
	Tier            VerificationTier `json:"tier"`
	Commit          string           `json:"commit"`
	Dataset         artifact.ID      `json:"dataset,omitzero"`
	SpanSteps       uint64           `json:"span_steps,omitempty"`
	SpanTokens      uint64           `json:"span_tokens,omitempty"`
	ContextTokens   uint64           `json:"context_tokens,omitempty"`
	WallNS          uint64           `json:"wall_ns,omitempty"`
	PeakDeviceBytes uint64           `json:"peak_device_bytes,omitempty"`
	Evidence        []artifact.ID    `json:"evidence"`
}

// ModelVerification defines immutable model claims and evidence lineage.
type ModelVerification struct {
	Version    uint16            `json:"version"`
	Model      artifact.ID       `json:"model"`
	Name       string            `json:"name"`
	Claims     []CapabilityClaim `json:"claims"`
	Supersedes []artifact.ID     `json:"supersedes,omitempty"`
	ID         artifact.ID       `json:"-"`
}

var modelVerificationCodec = artifact.JSONDocumentCodec(
	"model verification", artifact.KindEvidence, ModelVerificationMediaType, ModelVerificationSchema,
	canonicalizeModelVerification,
	func(value ModelVerification) artifact.ID { return value.ID },
	func(value *ModelVerification, id artifact.ID) { value.ID = id },
	func(value ModelVerification) ModelVerification {
		value.Claims = slices.Clone(value.Claims)
		value.Supersedes = slices.Clone(value.Supersedes)
		for i := range value.Claims {
			value.Claims[i].Evidence = slices.Clone(value.Claims[i].Evidence)
		}
		return value
	},
)

func NewModelVerification(model artifact.ID, name string, claims []CapabilityClaim) (ModelVerification, error) {
	return newModelVerification(model, name, claims, nil)
}

// NewModelVerificationCorrection replaces immutable verification records with
// a corrected claim set. The matrix excludes the named records only when the
// correction and replaced record bind the same model.
func NewModelVerificationCorrection(model artifact.ID, name string, claims []CapabilityClaim, supersedes []artifact.ID) (ModelVerification, error) {
	return newModelVerification(model, name, claims, supersedes)
}

func newModelVerification(model artifact.ID, name string, claims []CapabilityClaim, supersedes []artifact.ID) (ModelVerification, error) {
	cloned := slices.Clone(claims)
	for i := range cloned {
		cloned[i].Evidence = slices.Clone(cloned[i].Evidence)
	}
	return modelVerificationCodec.New(ModelVerification{
		Version: ModelVerificationVersion, Model: model, Name: name, Claims: cloned,
		Supersedes: slices.Clone(supersedes),
	})
}

func ParseModelVerification(content []byte) (ModelVerification, error) {
	return modelVerificationCodec.Parse(content)
}

func (v ModelVerification) Content() (artifact.Content, error) {
	return modelVerificationCodec.Content(v)
}

// Batch binds model, dataset, and evidence parents.
func (v ModelVerification) Batch(key string) (artifact.Batch, error) {
	seen := map[artifact.ID]bool{v.Model: true}
	parents := []artifact.ID{v.Model}
	appendParent := func(id artifact.ID) {
		if id.Valid() && !seen[id] {
			seen[id] = true
			parents = append(parents, id)
		}
	}
	for _, claim := range v.Claims {
		appendParent(claim.Dataset)
		for _, evidence := range claim.Evidence {
			appendParent(evidence)
		}
	}
	for _, superseded := range v.Supersedes {
		appendParent(superseded)
	}
	return modelVerificationCodec.Batch(key, v, artifact.DependencyLineage(v.ID, parents...), nil)
}

// MatrixRow defines strongest evidenced claim per capability.
type MatrixRow struct {
	Model        artifact.ID       `json:"model"`
	Name         string            `json:"name"`
	Capabilities []CapabilityClaim `json:"capabilities"`
}

// VerificationMatrix derives order-independent maximum-evidence rows.
func VerificationMatrix(records []ModelVerification) []MatrixRow {
	records = slices.Clone(records)
	sort.Slice(records, func(i, j int) bool { return records[i].ID.String() < records[j].ID.String() })
	byID := make(map[artifact.ID]ModelVerification, len(records))
	for _, record := range records {
		byID[record.ID] = record
	}
	superseded := make(map[artifact.ID]bool)
	for _, record := range records {
		for _, prior := range record.Supersedes {
			if replaced, ok := byID[prior]; ok && replaced.Model == record.Model {
				superseded[prior] = true
			}
		}
	}
	type slot struct {
		claim CapabilityClaim
		seen  map[artifact.ID]bool
	}
	rows := map[artifact.ID]map[string]*slot{}
	names := map[artifact.ID]string{}
	for _, record := range records {
		if superseded[record.ID] {
			continue
		}
		if name, ok := names[record.Model]; !ok || record.Name < name {
			names[record.Model] = record.Name
		}
		capabilities := rows[record.Model]
		if capabilities == nil {
			capabilities = map[string]*slot{}
			rows[record.Model] = capabilities
		}
		for _, claim := range record.Claims {
			current := capabilities[claim.Capability]
			switch {
			case current == nil || claim.Tier.StrongerThan(current.claim.Tier):
				seen := make(map[artifact.ID]bool, len(claim.Evidence))
				for _, id := range claim.Evidence {
					seen[id] = true
				}
				winning := claim
				winning.Evidence = slices.Clone(claim.Evidence)
				capabilities[claim.Capability] = &slot{claim: winning, seen: seen}
			case claim.Tier == current.claim.Tier:
				// Equal-tier claims union their evidence. A measured claim
				// displaces an unmeasured one's provenance -- performance
				// numbers must surface in the matrix -- otherwise provenance
				// stays with the first record in identity order.
				union := current.seen
				evidence := current.claim.Evidence
				if claim.WallNS > 0 && current.claim.WallNS == 0 {
					adopted := claim
					adopted.Evidence = evidence
					current.claim = adopted
				}
				for _, id := range claim.Evidence {
					if !union[id] {
						union[id] = true
						current.claim.Evidence = append(current.claim.Evidence, id)
					}
				}
			}
		}
	}
	matrix := make([]MatrixRow, 0, len(rows))
	for model, capabilities := range rows {
		row := MatrixRow{Model: model, Name: names[model]}
		for _, entry := range capabilities {
			sort.Slice(entry.claim.Evidence, func(i, j int) bool {
				return entry.claim.Evidence[i].String() < entry.claim.Evidence[j].String()
			})
			row.Capabilities = append(row.Capabilities, entry.claim)
		}
		sort.Slice(row.Capabilities, func(i, j int) bool {
			return row.Capabilities[i].Capability < row.Capabilities[j].Capability
		})
		matrix = append(matrix, row)
	}
	sort.Slice(matrix, func(i, j int) bool {
		if matrix[i].Name != matrix[j].Name {
			return matrix[i].Name < matrix[j].Name
		}
		return matrix[i].Model.String() < matrix[j].Model.String()
	})
	return matrix
}

func canonicalizeModelVerification(value *ModelVerification) error {
	if value == nil || value.Version != ModelVerificationVersion {
		return errors.New("run record: invalid model verification version")
	}
	if value.Model.Kind() != artifact.KindModel {
		return errors.New("run record: model verification requires a model identity")
	}
	if !textcheck.Bounded(value.Name, artifact.MaxContentBytes, "\x00\r\n\t") {
		return errors.New("run record: model verification requires a bounded name")
	}
	if len(value.Claims) == 0 {
		return errors.New("run record: model verification requires capability claims")
	}
	for _, superseded := range value.Supersedes {
		if superseded.Kind() != artifact.KindEvidence {
			return errors.New("run record: model verification supersedes a non-evidence artifact")
		}
	}
	sort.Slice(value.Supersedes, func(i, j int) bool {
		return value.Supersedes[i].String() < value.Supersedes[j].String()
	})
	value.Supersedes = slices.Compact(value.Supersedes)
	for i := range value.Claims {
		claim := &value.Claims[i]
		if !textcheck.Bounded(claim.Capability, artifact.MaxContentBytes, "") ||
			strings.Trim(claim.Capability, "abcdefghijklmnopqrstuvwxyz-") != "" {
			return fmt.Errorf("run record: capability %q must be lowercase kebab-case", claim.Capability)
		}
		if !claim.Tier.Valid() {
			return fmt.Errorf("run record: capability %q carries invalid tier %q", claim.Capability, claim.Tier)
		}
		if !validCodeCommit(claim.Commit) {
			return fmt.Errorf("run record: capability %q requires the verifying repository commit", claim.Capability)
		}
		if claim.Capability == "training" || claim.SpanSteps > 0 || claim.SpanTokens > 0 {
			if !claim.Dataset.Valid() || claim.Dataset == value.Model {
				return fmt.Errorf("run record: capability %q requires its dataset identity", claim.Capability)
			}
			if claim.SpanSteps == 0 && claim.SpanTokens == 0 {
				return fmt.Errorf("run record: capability %q requires its training span (steps or tokens)", claim.Capability)
			}
		}
		if (claim.ContextTokens > 0 || claim.PeakDeviceBytes > 0) && claim.WallNS == 0 {
			return fmt.Errorf("run record: capability %q declares context or peak memory without a measured wall", claim.Capability)
		}
		if len(claim.Evidence) == 0 {
			return fmt.Errorf("run record: capability %q claims tier %s without evidence", claim.Capability, claim.Tier)
		}
		for _, evidence := range claim.Evidence {
			if !evidence.Valid() || evidence == value.Model {
				return fmt.Errorf("run record: capability %q carries invalid evidence", claim.Capability)
			}
		}
		sort.Slice(claim.Evidence, func(a, b int) bool {
			return claim.Evidence[a].String() < claim.Evidence[b].String()
		})
		claim.Evidence = slices.Compact(claim.Evidence)
	}
	sort.Slice(value.Claims, func(a, b int) bool {
		return value.Claims[a].Capability < value.Claims[b].Capability
	})
	claimCount := len(value.Claims)
	value.Claims = slices.CompactFunc(value.Claims, func(left, right CapabilityClaim) bool {
		return left.Capability == right.Capability
	})
	if len(value.Claims) != claimCount {
		return errors.New("run record: capability claims must be unique")
	}
	return nil
}
