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

var architectureProfileFacts = discoverProfileFacts()

func ArchitectureProfileFacts() []ProfileFactID {
	return slices.Clone(architectureProfileFacts)
}

func CatalogProfileProvenance(profile model.ArchitectureProfile) []ProfileFactProvenance {
	derivation, _ := artifact.IdentifyBytes(
		artifact.KindEvidence, []byte("overgo/profile-catalog/v2\x00"+profile.Name),
	)
	provenance := make([]ProfileFactProvenance, len(architectureProfileFacts))
	for index, fact := range architectureProfileFacts {
		provenance[index] = ProfileFactProvenance{
			Fact: fact, Origin: ProfileFactConfig,
			SourceField: "architecture_catalog." + string(fact), DerivationID: derivation,
		}
	}
	return provenance
}

func canonicalizeProfileProvenance(provenance *[]ProfileFactProvenance) error {
	if provenance == nil || len(*provenance) != len(architectureProfileFacts) {
		return errors.New("model recipe: profile provenance coverage differs")
	}
	sort.Slice(*provenance, func(i, j int) bool { return (*provenance)[i].Fact < (*provenance)[j].Fact })
	for index, item := range *provenance {
		if item.Fact != architectureProfileFacts[index] || !validProfileFactOrigin(item.Origin) ||
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

func discoverProfileFacts() []ProfileFactID {
	var facts []ProfileFactID
	collectProfileFacts(reflect.TypeOf(model.ArchitectureProfile{}), "", &facts)
	sort.Slice(facts, func(i, j int) bool { return facts[i] < facts[j] })
	return facts
}

func collectProfileFacts(value reflect.Type, prefix string, facts *[]ProfileFactID) {
	for index := 0; index < value.NumField(); index++ {
		field := value.Field(index)
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
		*facts = append(*facts, ProfileFactID(name))
	}
}
