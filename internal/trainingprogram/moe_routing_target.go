package trainingprogram

import (
	"cmp"
	"errors"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/checked"
)

const (
	moeRoutingTargetMediaType = "application/vnd.overgo.moe-routing-target+json"
	moeRoutingTargetSchema    = "overgo/moe-routing-target/v1"
)

type moeRoutingTargetMode string

const (
	moeRoutingTargetUniform          moeRoutingTargetMode = "uniform"
	moeRoutingTargetDeclaredPrior    moeRoutingTargetMode = "declared-prior"
	moeRoutingTargetCapacityWeighted moeRoutingTargetMode = "capacity-weighted"
)

type moeRoutingStratumTarget struct {
	Stratum artifact.ID `json:"stratum"`
	Masses  []uint64    `json:"masses"`
}

type moeRoutingAssumption struct {
	Name          string      `json:"name"`
	Statement     string      `json:"statement"`
	Evidence      artifact.ID `json:"evidence"`
	ReopenTrigger artifact.ID `json:"reopen_trigger"`
}

type moeRoutingTargetSpec struct {
	Mode        moeRoutingTargetMode      `json:"mode"`
	Prior       artifact.ID               `json:"prior"`
	Strata      []moeRoutingStratumTarget `json:"strata"`
	Assumptions []moeRoutingAssumption    `json:"assumptions"`
}

type moeRoutingTarget struct {
	ID        artifact.ID `json:"-"`
	Version   uint16      `json:"version"`
	Candidate artifact.ID `json:"candidate"`
	moeRoutingTargetSpec
}

// moeRoutingTarget is an evidence-bound objective recipe. It describes a target;
// it does not authorize a balancing estimator or change router behavior.
var moeRoutingTargetContract = artifact.JSONDocumentCodec(
	"MoE routing target", artifact.KindRecipe, moeRoutingTargetMediaType, moeRoutingTargetSchema,
	canonicalizeMoERoutingTarget,
	func(target moeRoutingTarget) artifact.ID { return target.ID },
	func(target *moeRoutingTarget, id artifact.ID) { target.ID = id },
	func(target moeRoutingTarget) moeRoutingTarget {
		target.moeRoutingTargetSpec = cloneMoERoutingTargetSpec(target.moeRoutingTargetSpec)
		return target
	},
)

var moeRoutingTargetLineage = func(target moeRoutingTarget) []artifact.Lineage {
	parents := []artifact.ID{target.Candidate, target.Prior}
	for _, stratum := range target.Strata {
		parents = append(parents, stratum.Stratum)
	}
	for _, assumption := range target.Assumptions {
		parents = append(parents, assumption.Evidence, assumption.ReopenTrigger)
	}
	return artifact.DependencyLineage(target.ID, uniqueArtifactIDs(parents)...)
}

func canonicalizeMoERoutingTarget(target *moeRoutingTarget) error {
	if target == nil || target.Version != artifact.InitialDocumentVersion || target.Candidate.Kind() != artifact.KindRecipe ||
		target.Prior.Kind() != artifact.KindEvidence ||
		(target.Mode != moeRoutingTargetUniform && target.Mode != moeRoutingTargetDeclaredPrior && target.Mode != moeRoutingTargetCapacityWeighted) ||
		len(target.Strata) == 0 || len(target.Assumptions) == 0 {
		return errors.New("training program: routing target requires a candidate, declared mode, prior, strata, and assumptions")
	}
	target.Strata = slices.Clone(target.Strata)
	for index := range target.Strata {
		target.Strata[index].Masses = slices.Clone(target.Strata[index].Masses)
	}
	slices.SortFunc(target.Strata, func(left, right moeRoutingStratumTarget) int {
		return cmp.Compare(left.Stratum.String(), right.Stratum.String())
	})
	expertCount := len(target.Strata[0].Masses)
	if expertCount == 0 {
		return errors.New("training program: routing target requires expert masses")
	}
	for index, stratum := range target.Strata {
		if stratum.Stratum.Kind() != artifact.KindDatasetShard || len(stratum.Masses) != expertCount ||
			index > 0 && target.Strata[index-1].Stratum == stratum.Stratum {
			return errors.New("training program: routing target has invalid strata or expert inventory")
		}
		total, ok := checked.Add64(stratum.Masses...)
		if !ok || total == 0 {
			return errors.New("training program: routing target masses are empty or overflow")
		}
		if target.Mode == moeRoutingTargetUniform && !allMassesEqual(stratum.Masses) {
			return errors.New("training program: uniform routing target requires equal declared masses")
		}
	}
	target.Assumptions = slices.Clone(target.Assumptions)
	slices.SortFunc(target.Assumptions, func(left, right moeRoutingAssumption) int {
		return cmp.Compare(left.Name, right.Name)
	})
	for index, assumption := range target.Assumptions {
		if assumption.Name == "" || strings.TrimSpace(assumption.Name) != assumption.Name || strings.ContainsAny(assumption.Name, "\x00\r\n") ||
			assumption.Statement == "" || strings.TrimSpace(assumption.Statement) != assumption.Statement || strings.ContainsAny(assumption.Statement, "\x00\r\n") ||
			assumption.Evidence.Kind() != artifact.KindEvidence || assumption.ReopenTrigger.Kind() != artifact.KindRecipe ||
			index > 0 && target.Assumptions[index-1].Name == assumption.Name {
			return errors.New("training program: routing target assumption lacks evidence or reopen trigger")
		}
	}
	return nil
}

func cloneMoERoutingTargetSpec(spec moeRoutingTargetSpec) moeRoutingTargetSpec {
	copy := spec
	copy.Strata = slices.Clone(spec.Strata)
	for index := range copy.Strata {
		copy.Strata[index].Masses = slices.Clone(spec.Strata[index].Masses)
	}
	copy.Assumptions = slices.Clone(spec.Assumptions)
	return copy
}

func allMassesEqual(masses []uint64) bool {
	for _, mass := range masses[1:] {
		if mass != masses[0] {
			return false
		}
	}
	return true
}

func uniqueArtifactIDs(ids []artifact.ID) []artifact.ID {
	result := make([]artifact.ID, 0, len(ids))
	seen := make(map[artifact.ID]struct{}, len(ids))
	for _, id := range ids {
		if _, found := seen[id]; found {
			continue
		}
		seen[id] = struct{}{}
		result = append(result, id)
	}
	return result
}
