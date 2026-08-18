package trainingprogram

import (
	"errors"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

const (
	ObjectiveMixtureVersion   uint16 = 1
	ObjectiveMixtureMediaType        = "application/vnd.overgo.objective-mixture+json"
	ObjectiveMixtureSchema           = "overgo/objective-mixture/v1"
)

// MixtureComponent weights one objective inside a composed mixture. Weights
// are integers so mixture identity is exact and schedules derive
// deterministically.
type MixtureComponent struct {
	Objective artifact.ID `json:"objective"`
	Weight    uint32      `json:"weight"`
}

// ObjectiveMixtureDocument makes objective composition a typed artifact: a
// weighted set of objective documents the system may propose, judged -- like
// every operator artifact -- solely by the descendants it produces.
type ObjectiveMixtureDocument struct {
	Version    uint16             `json:"version"`
	Components []MixtureComponent `json:"components"`
	ID         artifact.ID        `json:"-"`
}

var objectiveMixtureCodec = artifact.JSONDocumentCodec(
	"objective mixture", artifact.KindRecipe, ObjectiveMixtureMediaType, ObjectiveMixtureSchema,
	canonicalizeObjectiveMixture,
	func(value ObjectiveMixtureDocument) artifact.ID { return value.ID },
	func(value *ObjectiveMixtureDocument, id artifact.ID) { value.ID = id },
	func(value ObjectiveMixtureDocument) ObjectiveMixtureDocument {
		value.Components = slices.Clone(value.Components)
		return value
	},
)

// NewObjectiveMixture identifies one composed objective.
func NewObjectiveMixture(components []MixtureComponent) (ObjectiveMixtureDocument, error) {
	return objectiveMixtureCodec.New(ObjectiveMixtureDocument{
		Version: ObjectiveMixtureVersion, Components: slices.Clone(components),
	})
}

func ParseObjectiveMixture(content []byte) (ObjectiveMixtureDocument, error) {
	return objectiveMixtureCodec.Parse(content)
}

func (d ObjectiveMixtureDocument) Content() (artifact.Content, error) {
	return objectiveMixtureCodec.Content(d)
}

// Schedule derives the deterministic interleaving the mixture prescribes:
// component indices repeated by weight, round-robin. Training realizes the
// mixture by following this schedule over the components' data bindings.
func (d ObjectiveMixtureDocument) Schedule() []int {
	schedule := make([]int, 0)
	remaining := make([]uint32, len(d.Components))
	for index, component := range d.Components {
		remaining[index] = component.Weight
	}
	for {
		emitted := false
		for index := range d.Components {
			if remaining[index] > 0 {
				schedule = append(schedule, index)
				remaining[index]--
				emitted = true
			}
		}
		if !emitted {
			return schedule
		}
	}
}

func canonicalizeObjectiveMixture(value *ObjectiveMixtureDocument) error {
	if value == nil || value.Version != ObjectiveMixtureVersion {
		return errors.New("training program: invalid objective mixture version")
	}
	if len(value.Components) < 2 {
		return errors.New("training program: a mixture composes at least two objectives")
	}
	slices.SortFunc(value.Components, func(left, right MixtureComponent) int {
		return strings.Compare(left.Objective.String(), right.Objective.String())
	})
	seen := map[artifact.ID]bool{}
	for _, component := range value.Components {
		if !component.Objective.Valid() || component.Weight == 0 {
			return errors.New("training program: mixture components require objectives and positive weights")
		}
		if seen[component.Objective] {
			return errors.New("training program: mixture components must be distinct objectives")
		}
		seen[component.Objective] = true
	}
	return nil
}
