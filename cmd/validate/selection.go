package main

import (
	"cmp"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

// SurfaceID names one code surface whose change can invalidate a class of
// evidence. Every validation cell is pinned to exactly one surface; a commit
// that does not move a cell's surface reuses the cell's accepted evidence.
type SurfaceID string

// ModelValidation is one (model, declared function, validation) obligation:
// the appropriate validation for one declared modality/function of one model,
// with the surface that pins its accepted evidence and the command that
// re-acquires it when the surface moves.
type ModelValidation struct {
	Model          artifact.ID `json:"model"`
	ModelName      string      `json:"model_name,omitzero"`
	Location       string      `json:"location,omitzero"`
	Kind           recipe.Task `json:"kind,omitzero"` // empty: registered without activation
	Modality       string      `json:"modality,omitzero"`
	Validation     string      `json:"validation,omitzero"` // empty: no validation applicable to the kind
	Surface        SurfaceID   `json:"surface,omitzero"`
	Evidence       artifact.ID `json:"evidence,omitzero"`
	Cost           int64       `json:"cost_ns,omitzero"` // measured wall or declared timeout, for smallest-first
	ReacquireCmd   string      `json:"reacquire_cmd,omitzero"`
	Stale          string      `json:"stale,omitzero"` // activation defect, if any
	OperatorOnly   bool        `json:"operator_only,omitzero"`
	OperatorReason string      `json:"operator_reason,omitzero"`
}

// Selected pairs an obligation with the reason it landed in its plan bucket.
type Selected struct {
	ModelValidation
	Reason string `json:"reason"`
}

// AffectRecord records whether a commit's changes moved one surface and why.
type AffectRecord struct {
	Affected bool   `json:"affected"`
	Reason   string `json:"reason"`
}

// Inputs are the resolved facts the selection reduces: the commit range, the
// per-surface affectedness derived from the range's changed paths, and every
// registered model's validation obligations.
type Inputs struct {
	Baseline         string                     `json:"baseline"`
	Head             string                     `json:"head"`
	ChangedPaths     []string                   `json:"changed_paths"`
	AffectedSurfaces map[SurfaceID]AffectRecord `json:"affected_surfaces"`
	Registered       []ModelValidation          `json:"-"`
}

// Plan is the validation decision for one commit: run only what the commit's
// changes affect, reuse what stays pinned, and account for every registered
// model.
type Plan struct {
	Baseline           string      `json:"baseline"`
	Head               string      `json:"head"`
	ChangedPaths       []string    `json:"changed_paths"`
	AffectedSurfaces   []SurfaceID `json:"affected_surfaces"`
	Run                []Selected  `json:"run"`
	Reuse              []Selected  `json:"reuse"`
	OperatorOwned      []Selected  `json:"operator_owned"`
	RegisteredInactive []Selected  `json:"registered_inactive"`
	SkipInappropriate  []Selected  `json:"skip_inappropriate"`
}

// SelectValidation reduces the resolved inputs to a validation plan. It is
// pure: no store, git, filesystem or device access, so the whole decision is
// unit-tested. A cell runs only when this commit's changes move its surface (or
// it has no accepted evidence yet); everything else is reused, operator-owned,
// registered-inactive, or inapplicable. Run is ordered cheapest first so a
// broken run surfaces on a small model in seconds.
func SelectValidation(in Inputs) Plan {
	plan := Plan{
		Baseline: in.Baseline, Head: in.Head, ChangedPaths: in.ChangedPaths,
		Run: []Selected{}, Reuse: []Selected{}, OperatorOwned: []Selected{},
		RegisteredInactive: []Selected{}, SkipInappropriate: []Selected{},
	}
	for id, rec := range in.AffectedSurfaces {
		if rec.Affected {
			plan.AffectedSurfaces = append(plan.AffectedSurfaces, id)
		}
	}
	slices.Sort(plan.AffectedSurfaces)
	for _, mv := range in.Registered {
		sel := Selected{ModelValidation: mv}
		switch {
		case mv.OperatorOnly:
			sel.Reason = "operator-owned: " + mv.OperatorReason
			plan.OperatorOwned = append(plan.OperatorOwned, sel)
		case mv.Kind == "":
			sel.Reason = "registered without task activation"
			plan.RegisteredInactive = append(plan.RegisteredInactive, sel)
		case mv.Validation == "":
			sel.Reason = "no validation applicable to " + string(mv.Kind)
			plan.SkipInappropriate = append(plan.SkipInappropriate, sel)
		default:
			rec := in.AffectedSurfaces[mv.Surface]
			switch {
			case rec.Affected:
				sel.Reason = "run: " + string(mv.Surface) + " surface affected: " + rec.Reason
				plan.Run = append(plan.Run, sel)
			case !mv.Evidence.Valid():
				sel.Reason = "run: no accepted evidence for this cell"
				plan.Run = append(plan.Run, sel)
			default:
				sel.Reason = "reuse: " + string(mv.Surface) + " surface unchanged since baseline; evidence pinned"
				plan.Reuse = append(plan.Reuse, sel)
			}
		}
	}
	slices.SortStableFunc(plan.Run, func(a, b Selected) int {
		if c := cmp.Compare(a.Cost, b.Cost); c != 0 {
			return c
		}
		return cmp.Or(cmp.Compare(a.Model.String(), b.Model.String()), cmp.Compare(a.Validation, b.Validation))
	})
	for _, bucket := range [][]Selected{plan.Reuse, plan.OperatorOwned, plan.RegisteredInactive, plan.SkipInappropriate} {
		slices.SortStableFunc(bucket, func(a, b Selected) int {
			return cmp.Or(cmp.Compare(a.Model.String(), b.Model.String()), cmp.Compare(a.Validation, b.Validation))
		})
	}
	return plan
}
