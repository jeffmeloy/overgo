package modelrecipe

import (
	"errors"
	"reflect"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/model"
)

type ProfileFactID string

type ProfileFactOrigin string

type ProfileFactSpec struct {
	Fact ProfileFactID `json:"fact"`
	Type string        `json:"type"`
}

const (
	ProfileFactConfig   ProfileFactOrigin = "config-derived"
	ProfileFactTensor   ProfileFactOrigin = "tensor-derived"
	ProfileFactExternal ProfileFactOrigin = "external-evidence"
	ProfileFactRuntime  ProfileFactOrigin = "runtime-measured"
)

type ProfileFactProvenance struct {
	Fact         ProfileFactID     `json:"fact"`
	Origin       ProfileFactOrigin `json:"origin"`
	SourceField  string            `json:"source_field"`
	DerivationID artifact.ID       `json:"derivation_id"`
}

var architectureProfileFactSpecs = discoverProfileFacts()

var architectureProfileFacts = func() []ProfileFactID {
	facts := make([]ProfileFactID, len(architectureProfileFactSpecs))
	for index, spec := range architectureProfileFactSpecs {
		facts[index] = spec.Fact
	}
	return facts
}()

func CatalogProfileProvenance(profile model.ArchitectureProfile) ([]ProfileFactProvenance, error) {
	derivation, err := NewCatalogProfileDerivation(profile)
	if err != nil {
		return nil, err
	}
	provenance := make([]ProfileFactProvenance, len(architectureProfileFacts))
	for index, fact := range architectureProfileFacts {
		provenance[index] = ProfileFactProvenance{
			Fact: fact, Origin: ProfileFactConfig,
			SourceField: "architecture_catalog." + string(fact), DerivationID: derivation.ID,
		}
	}
	return provenance, nil
}

// laterProfileFacts are the facts the v3 profile schema gained after documents under it were published. A document
// derived before one holds it at its zero value, the value its program was compiled with, and records no provenance for
// it; a document holding any other value records its provenance as it does every fact's.
var laterProfileFacts = []ProfileFactID{"Runtime.PromptTensorCores"}

// documentProfileFacts are the facts a stored document's provenance covers, and the fact schema its catalog derivation
// was made under: every fact, but a later one its policy holds at zero and its provenance does not record.
func documentProfileFacts(policy model.ArchitectureProfile, provenance []ProfileFactProvenance) []ProfileFactSpec {
	recorded := make(map[ProfileFactID]bool, len(provenance))
	for _, item := range provenance {
		recorded[item.Fact] = true
	}
	return slices.DeleteFunc(slices.Clone(architectureProfileFactSpecs), func(spec ProfileFactSpec) bool {
		return slices.Contains(laterProfileFacts, spec.Fact) && !recorded[spec.Fact] && profileFactZero(policy, spec.Fact)
	})
}

// profileFactZero reports whether the policy holds the fact, a dotted path of its fields, at its zero value.
func profileFactZero(policy model.ArchitectureProfile, fact ProfileFactID) bool {
	value := reflect.ValueOf(policy)
	for name := range strings.SplitSeq(string(fact), ".") {
		if value = value.FieldByName(name); !value.IsValid() {
			return false
		}
	}
	return value.IsZero()
}

func canonicalizeProfileProvenance(provenance *[]ProfileFactProvenance, facts []ProfileFactSpec) error {
	if provenance == nil || len(*provenance) != len(facts) {
		return errors.New("model recipe: profile provenance coverage differs")
	}
	sort.Slice(*provenance, func(i, j int) bool { return (*provenance)[i].Fact < (*provenance)[j].Fact })
	for index, item := range *provenance {
		if item.Fact != facts[index].Fact || !validProfileFactOrigin(item.Origin) ||
			item.SourceField == "" || strings.TrimSpace(item.SourceField) != item.SourceField ||
			strings.ContainsAny(item.SourceField, "\r\n") || !item.DerivationID.Valid() {
			return errors.New("model recipe: invalid profile fact provenance")
		}
	}
	return nil
}

func validProfileFactOrigin(origin ProfileFactOrigin) bool {
	switch origin {
	case ProfileFactConfig, ProfileFactTensor, ProfileFactExternal, ProfileFactRuntime:
		return true
	default:
		return false
	}
}

func discoverProfileFacts() []ProfileFactSpec {
	var facts []ProfileFactSpec
	collectProfileFacts(reflect.TypeFor[model.ArchitectureProfile](), "", &facts)
	sort.Slice(facts, func(i, j int) bool { return facts[i].Fact < facts[j].Fact })
	return facts
}

func collectProfileFacts(value reflect.Type, prefix string, facts *[]ProfileFactSpec) {
	for field := range value.Fields() {
		if field.PkgPath != "" {
			continue
		}
		name := field.Name
		if prefix != "" {
			name = prefix + "." + name
		}
		if field.Type.Kind() == reflect.Struct {
			collectProfileFacts(field.Type, name, facts)
			continue
		}
		*facts = append(*facts, ProfileFactSpec{Fact: ProfileFactID(name), Type: field.Type.String()})
	}
}
