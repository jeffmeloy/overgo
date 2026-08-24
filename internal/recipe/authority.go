package recipe

import (
	"errors"
	"slices"
	"sort"

	"overgo/internal/artifact"
)

const (
	AuthoritySubjectMediaType = "application/vnd.overgo.authority-subject+json"
	AuthoritySubjectSchema    = "overgo/authority-subject/v1"
)

// AuthoritySubject is the canonical identity of every immutable artifact an
// authorization decision governs. It deliberately says nothing about tensor
// layout, model family, dataset partitioning, or execution topology: those
// facts remain in the referenced artifacts and their OvergoDB lineage.
type AuthoritySubject struct {
	Version uint16        `json:"version"`
	Members []artifact.ID `json:"members"`
	ID      artifact.ID   `json:"-"`
}

var authoritySubjectCodec = artifact.JSONDocumentCodec(
	"authority subject", artifact.KindEvidence, AuthoritySubjectMediaType, AuthoritySubjectSchema,
	canonicalizeAuthoritySubject,
	func(value AuthoritySubject) artifact.ID { return value.ID },
	func(value *AuthoritySubject, id artifact.ID) { value.ID = id },
	func(value AuthoritySubject) AuthoritySubject {
		value.Members = slices.Clone(value.Members)
		return value
	},
)

// NewAuthoritySubject derives one order-independent composite identity. A
// caller must name every governed artifact explicitly; nested subjects remain
// ordinary members and therefore compose without a second storage model.
func NewAuthoritySubject(members ...artifact.ID) (AuthoritySubject, error) {
	return authoritySubjectCodec.New(AuthoritySubject{
		Version: artifact.InitialDocumentVersion,
		Members: slices.Clone(members),
	})
}

func ParseAuthoritySubject(content []byte) (AuthoritySubject, error) {
	return authoritySubjectCodec.Parse(content)
}

func (subject AuthoritySubject) ValidateIdentity() error {
	return authoritySubjectCodec.ValidateIdentity(subject)
}

func (subject AuthoritySubject) Content() (artifact.Content, error) {
	return authoritySubjectCodec.Content(subject)
}

func (subject AuthoritySubject) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(subject.ID, subject.Members...)
}

// AdmitAuthorityReceipt verifies that an existing recipe decision authorizes
// the exact live composite subject at or above the required evidence tier.
// Rebuilding the live subject before calling this function is the re-derivation
// boundary: any member addition, removal, or replacement changes its ID and
// makes the receipt stale. Accepted authority also requires measured evidence;
// a bare approval is not a promotion credential.
func AdmitAuthorityReceipt(live AuthoritySubject, receipt Decision, required EvidenceTier) error {
	if err := live.ValidateIdentity(); err != nil {
		return errors.New("recipe: live authority subject identity differs")
	}
	if err := receipt.ValidateIdentity(); err != nil {
		return errors.New("recipe: authority receipt identity differs")
	}
	if !validEvidenceTier(required) || receipt.Subject != live.ID || receipt.Outcome != DecisionAccepted ||
		!tierAtLeast(receipt.Tier, required) || len(receipt.Evidence) == 0 {
		return errors.New("recipe: authority receipt does not admit the live subject")
	}
	return nil
}

func canonicalizeAuthoritySubject(subject *AuthoritySubject) error {
	if subject == nil || subject.Version != artifact.InitialDocumentVersion || len(subject.Members) == 0 {
		return errors.New("recipe: invalid authority subject")
	}
	sort.Slice(subject.Members, func(left, right int) bool {
		return subject.Members[left].String() < subject.Members[right].String()
	})
	for index, member := range subject.Members {
		if !member.Valid() || index > 0 && subject.Members[index-1] == member {
			return errors.New("recipe: invalid authority subject member")
		}
	}
	return nil
}

func tierAtLeast(actual, required EvidenceTier) bool {
	return actual == required || actual.StrongerThan(required)
}
