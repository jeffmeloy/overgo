package runrecord

import (
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
)

const (
	ModelVerificationVersion   uint16 = 1
	ModelVerificationMediaType        = "application/vnd.overgo.model-verification+json"
	ModelVerificationSchema           = "overgo/model-verification/v1"
)

// VerificationTier is the ordered evidence ladder for one capability claim.
// Higher tiers subsume nothing automatically -- each tier names only the kind
// of evidence that grounds it, and a claim at any tier requires that
// evidence committed beside it.
type VerificationTier string

const (
	TierSyntheticDifferential VerificationTier = "synthetic-differential"
	TierRealArtifactSmoke     VerificationTier = "real-artifact-smoke"
	TierExactGolden           VerificationTier = "exact-golden"
	TierCapabilityMeasured    VerificationTier = "capability-measured"
)

// Rank orders tiers for matrix derivation; zero marks an invalid tier.
func (t VerificationTier) Rank() int {
	switch t {
	case TierSyntheticDifferential:
		return 1
	case TierRealArtifactSmoke:
		return 2
	case TierExactGolden:
		return 3
	case TierCapabilityMeasured:
		return 4
	default:
		return 0
	}
}

// CapabilityClaim is one evidenced verification claim: what was verified
// (inference, training, ...), to what tier, by which repository commit,
// against which dataset and training span, grounded in which evidence.
// A claim without evidence cannot exist; a training claim without its
// dataset and span is a claim about nothing in particular and is refused.
type CapabilityClaim struct {
	Capability string           `json:"capability"`
	Tier       VerificationTier `json:"tier"`
	// Commit is the repository commit whose verifier produced the evidence.
	Commit string `json:"commit"`
	// Dataset names the dataset artifact the verification consumed; required
	// for training claims and whenever a span is declared.
	Dataset artifact.ID `json:"dataset,omitzero"`
	// SpanSteps and SpanTokens bound the training span the claim covers.
	SpanSteps  uint64 `json:"span_steps,omitempty"`
	SpanTokens uint64 `json:"span_tokens,omitempty"`
	// Performance measurement: the context the measured run used, its wall,
	// and peak device memory. Optional as a group, but context or peak
	// without a measured wall is a decoration, not a measurement, and is
	// refused.
	ContextTokens   uint64        `json:"context_tokens,omitempty"`
	WallNS          uint64        `json:"wall_ns,omitempty"`
	PeakDeviceBytes uint64        `json:"peak_device_bytes,omitempty"`
	Evidence        []artifact.ID `json:"evidence"`
}

// ModelVerification is the per-model verification record: typed capability
// claims bound to a model identity, committed to the store with lineage to
// every piece of evidence. The comparison matrix is derived from these
// records at query time -- there is no hand-maintained document to drift.
type ModelVerification struct {
	Version uint16            `json:"version"`
	Model   artifact.ID       `json:"model"`
	Name    string            `json:"name"`
	Claims  []CapabilityClaim `json:"claims"`
	ID      artifact.ID       `json:"-"`
}

var modelVerificationCodec = artifact.JSONDocumentCodec(
	"model verification", artifact.KindEvidence, ModelVerificationMediaType, ModelVerificationSchema,
	canonicalizeModelVerification,
	func(value ModelVerification) artifact.ID { return value.ID },
	func(value *ModelVerification, id artifact.ID) { value.ID = id },
	func(value ModelVerification) ModelVerification {
		value.Claims = slices.Clone(value.Claims)
		for i := range value.Claims {
			value.Claims[i].Evidence = slices.Clone(value.Claims[i].Evidence)
		}
		return value
	},
)

func NewModelVerification(model artifact.ID, name string, claims []CapabilityClaim) (ModelVerification, error) {
	cloned := slices.Clone(claims)
	for i := range cloned {
		cloned[i].Evidence = slices.Clone(cloned[i].Evidence)
	}
	return modelVerificationCodec.New(ModelVerification{
		Version: ModelVerificationVersion, Model: model, Name: name, Claims: cloned,
	})
}

func ParseModelVerification(content []byte) (ModelVerification, error) {
	return modelVerificationCodec.Parse(content)
}

func (v ModelVerification) Content() (artifact.Content, error) {
	return modelVerificationCodec.Content(v)
}

// Batch commits the record with lineage to the model, every dataset it
// verified against, and every evidence artifact grounding its claims.
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
	return modelVerificationCodec.Batch(key, v, artifact.DependencyLineage(v.ID, parents...), nil)
}

// MatrixRow is one derived comparison row: the strongest evidenced tier per
// capability across every committed record for the model.
type MatrixRow struct {
	Model        artifact.ID       `json:"model"`
	Name         string            `json:"name"`
	Capabilities []CapabilityClaim `json:"capabilities"`
}

// VerificationMatrix derives the comparison matrix from committed records:
// rows keyed by model identity, each capability at the maximum tier any
// record evidences, carrying the union of evidence for that winning tier and
// the winning claim's commit, dataset and span provenance. Records are
// sorted by identity first so the derivation is independent of store scan
// order. Verification accumulates -- records are immutable, so the maximum
// is the current state of proof.
func VerificationMatrix(records []ModelVerification) []MatrixRow {
	records = slices.Clone(records)
	sort.Slice(records, func(i, j int) bool { return records[i].ID.String() < records[j].ID.String() })
	type slot struct {
		claim CapabilityClaim
		seen  map[artifact.ID]bool
	}
	rows := map[artifact.ID]map[string]*slot{}
	names := map[artifact.ID]string{}
	for _, record := range records {
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
			case current == nil || claim.Tier.Rank() > current.claim.Tier.Rank():
				seen := make(map[artifact.ID]bool, len(claim.Evidence))
				for _, id := range claim.Evidence {
					seen[id] = true
				}
				winning := claim
				winning.Evidence = slices.Clone(claim.Evidence)
				capabilities[claim.Capability] = &slot{claim: winning, seen: seen}
			case claim.Tier.Rank() == current.claim.Tier.Rank():
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
	name := strings.TrimSpace(value.Name)
	if name == "" || name != value.Name || len(name) > 256 || strings.ContainsAny(name, "\x00\r\n\t") {
		return errors.New("run record: model verification requires a bounded name")
	}
	if len(value.Claims) == 0 {
		return errors.New("run record: model verification requires capability claims")
	}
	for i := range value.Claims {
		claim := &value.Claims[i]
		if claim.Capability == "" || len(claim.Capability) > 64 ||
			strings.Trim(claim.Capability, "abcdefghijklmnopqrstuvwxyz-") != "" {
			return fmt.Errorf("run record: capability %q must be lowercase kebab-case", claim.Capability)
		}
		if claim.Tier.Rank() == 0 {
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
	for i := 1; i < len(value.Claims); i++ {
		if value.Claims[i].Capability == value.Claims[i-1].Capability {
			return errors.New("run record: capability claims must be unique")
		}
	}
	return nil
}
