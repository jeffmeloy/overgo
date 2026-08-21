package plan

import (
	"context"
	"errors"
	"math"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
)

type LocalityConstraint string

const (
	LocalityPreferred LocalityConstraint = "preferred"
	LocalityRequired  LocalityConstraint = "required"
)

type ArtifactRequirement struct {
	Artifact   artifact.ID        `json:"artifact"`
	Constraint LocalityConstraint `json:"constraint"`
}

// LocalityWorker declares exact RepoDB locations visible to one worker. The
// scheduler cross-checks every declaration against current catalog state.
type LocalityWorker struct {
	Worker    artifact.ID         `json:"worker"`
	Locations []artifact.Location `json:"locations"`
}

type LocalityScheduleRequest struct {
	Requirements []ArtifactRequirement `json:"requirements"`
	Workers      []LocalityWorker      `json:"workers"`
}

type LocalityEligibility string

const (
	LocalityEligible        LocalityEligibility = "eligible"
	LocalityMissingRequired LocalityEligibility = "missing-required-locality"
)

type LocalityAssessment struct {
	Worker          artifact.ID         `json:"worker"`
	Eligibility     LocalityEligibility `json:"eligibility"`
	LocalBytes      uint64              `json:"local_bytes"`
	LocalArtifacts  uint64              `json:"local_artifacts"`
	MissingRequired []artifact.ID       `json:"missing_required,omitempty"`
}

type LocalitySchedule struct {
	Selected    artifact.ID          `json:"selected,omitzero"`
	Assessments []LocalityAssessment `json:"assessments"`
}

// ScheduleByArtifactLocality filters workers on required-local postconditions,
// then prefers the most already-local bytes, the most local artifacts, and
// finally the stable worker identity. It uses no topology or replica model.
func ScheduleByArtifactLocality(
	ctx context.Context,
	repository artifact.Reader,
	request LocalityScheduleRequest,
) (LocalitySchedule, error) {
	if ctx == nil || repository == nil || !populated(request.Requirements) || !populated(request.Workers) {
		return LocalitySchedule{}, errors.New("plan: locality schedule requires repository, artifacts, and workers")
	}
	descriptors := make(map[artifact.ID]artifact.Descriptor, len(request.Requirements))
	current := make(map[artifact.ID]map[artifact.Location]struct{}, len(request.Requirements))
	for _, requirement := range request.Requirements {
		if !requirement.Artifact.Valid() || requirement.Constraint != LocalityPreferred && requirement.Constraint != LocalityRequired {
			return LocalitySchedule{}, errors.New("plan: invalid artifact locality requirement")
		}
		if _, duplicate := descriptors[requirement.Artifact]; duplicate {
			return LocalitySchedule{}, errors.New("plan: duplicate artifact locality requirement")
		}
		descriptor, found, err := repository.Artifact(ctx, requirement.Artifact)
		if err != nil {
			return LocalitySchedule{}, err
		}
		if !found {
			return LocalitySchedule{}, errors.New("plan: required artifact is absent from RepoDB")
		}
		locations, err := repository.Locations(ctx, requirement.Artifact)
		if err != nil {
			return LocalitySchedule{}, err
		}
		descriptors[requirement.Artifact] = descriptor
		current[requirement.Artifact] = locationSet(locations)
	}
	seenWorkers := make(map[artifact.ID]struct{}, len(request.Workers))
	assessments := make([]LocalityAssessment, len(request.Workers))
	for workerIndex, worker := range request.Workers {
		if !worker.Worker.Valid() {
			return LocalitySchedule{}, errors.New("plan: invalid locality worker")
		}
		if _, duplicate := seenWorkers[worker.Worker]; duplicate {
			return LocalitySchedule{}, errors.New("plan: duplicate locality worker")
		}
		seenWorkers[worker.Worker] = struct{}{}
		declared := make(map[artifact.Location]struct{}, len(worker.Locations))
		for _, location := range worker.Locations {
			if err := location.Validate(); err != nil {
				return LocalitySchedule{}, err
			}
			if _, required := descriptors[location.Artifact]; !required {
				return LocalitySchedule{}, errors.New("plan: worker location is unrelated to requirements")
			}
			if _, duplicate := declared[location]; duplicate {
				return LocalitySchedule{}, errors.New("plan: duplicate worker location")
			}
			declared[location] = struct{}{}
		}
		assessment := LocalityAssessment{Worker: worker.Worker, Eligibility: LocalityEligible}
		for _, requirement := range request.Requirements {
			local := intersects(declared, current[requirement.Artifact])
			if local {
				descriptor := descriptors[requirement.Artifact]
				if math.MaxUint64-assessment.LocalBytes < descriptor.Size {
					return LocalitySchedule{}, errors.New("plan: local artifact bytes overflow")
				}
				assessment.LocalBytes += descriptor.Size
				assessment.LocalArtifacts++
			} else if requirement.Constraint == LocalityRequired {
				assessment.MissingRequired = append(assessment.MissingRequired, requirement.Artifact)
			}
		}
		if populated(assessment.MissingRequired) {
			assessment.Eligibility = LocalityMissingRequired
			slices.SortFunc(assessment.MissingRequired, compareArtifactID)
		}
		assessments[workerIndex] = assessment
	}
	sort.Slice(assessments, func(left, right int) bool {
		return assessments[left].Worker.String() < assessments[right].Worker.String()
	})
	result := LocalitySchedule{Assessments: assessments}
	var selected LocalityAssessment
	for _, assessment := range assessments {
		if assessment.Eligibility != LocalityEligible {
			continue
		}
		if !selected.Worker.Valid() || assessment.LocalBytes > selected.LocalBytes ||
			assessment.LocalBytes == selected.LocalBytes && assessment.LocalArtifacts > selected.LocalArtifacts {
			selected = assessment
		}
	}
	result.Selected = selected.Worker
	return result, nil
}

func locationSet(locations []artifact.Location) map[artifact.Location]struct{} {
	result := make(map[artifact.Location]struct{}, len(locations))
	for _, location := range locations {
		result[location] = struct{}{}
	}
	return result
}

func intersects(left, right map[artifact.Location]struct{}) bool {
	for location := range left {
		if _, found := right[location]; found {
			return true
		}
	}
	return false
}

func compareArtifactID(left, right artifact.ID) int {
	return strings.Compare(left.String(), right.String())
}

func populated[T any](values []T) bool {
	for range values {
		return true
	}
	return false
}
