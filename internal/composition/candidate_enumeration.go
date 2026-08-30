package composition

import (
	"errors"
	"math"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/organ"
)

// SeamBoundary names the clean seam classes a composite may cut at.
type SeamBoundary string

const (
	// SeamLayerBoundary cuts between transformer blocks or at a module's
	// block-internal interface.
	SeamLayerBoundary SeamBoundary = "layer"
	// SeamExpertBoundary cuts at a mixture expert's routing interface.
	SeamExpertBoundary SeamBoundary = "expert"
	// SeamResidualStream cuts where a component writes into the residual
	// stream — attention output and MLP down projections.
	SeamResidualStream SeamBoundary = "residual"
)

// AdapterRung names the adapter class a measured residual admits.
type AdapterRung string

const (
	// AdapterIdentity grafts directly: the residual is numerically zero.
	AdapterIdentity AdapterRung = "identity"
	// AdapterLinear admits a linear interface adapter: the fitted map
	// explains at least as much target energy as it leaves unexplained.
	AdapterLinear AdapterRung = "linear"
	// AdapterBridge requires a trained bridge: the linear map leaves the
	// majority of the target energy unexplained, down-ranking the seam.
	AdapterBridge AdapterRung = "bridge"
)

// DonorSeamMeasurement carries the measured signals for one shortlisted
// donor seam: the alignment residual over bounded seam activations and the
// predicted multidimensional fitness of the composite.
type DonorSeamMeasurement struct {
	Donor     artifact.ID `json:"donor"`
	Component string      `json:"component"`
	Residual  float64     `json:"residual"`
	Fitness   float64     `json:"fitness"`
}

// CompositionCandidate is one enumerated donor-seam-adapter option, ranked
// and ready for the driver's realization queue.
type CompositionCandidate struct {
	Donor     artifact.ID  `json:"donor"`
	Component string       `json:"component"`
	Seam      SeamBoundary `json:"seam"`
	Adapter   AdapterRung  `json:"adapter"`
	Residual  float64      `json:"residual"`
	Fitness   float64      `json:"fitness"`
}

// seamBoundaryFor places the clean seam from the component's typed
// contract: expert components cut at the routing interface, components
// that write into the residual stream cut there, and everything else cuts
// at a layer boundary.
func seamBoundaryFor(descriptor CanonicalComponentDescriptor) SeamBoundary {
	if strings.Contains(descriptor.Name, "exps") || strings.Contains(descriptor.Name, "expert") {
		return SeamExpertBoundary
	}
	switch organ.Role(descriptor.Role) {
	case organ.RoleAttentionOut, organ.RoleMLPDown:
		return SeamResidualStream
	default:
		return SeamLayerBoundary
	}
}

// adapterRungFor chooses the adapter class the residual admits. The rungs
// split at mathematical boundaries, not tuned thresholds: identity when
// the residual is within the float rounding floor of zero, linear while
// the fitted map explains at least half the target energy (residual² at
// most one half), and a trained bridge beyond that.
func adapterRungFor(residual float64) AdapterRung {
	if residual <= math.Sqrt(math.Nextafter(1, 2)-1) {
		return AdapterIdentity
	}
	if residual*residual <= 0.5 {
		return AdapterLinear
	}
	return AdapterBridge
}

// EnumerateCompositionCandidates enumerates the ranked donor, seam, and
// adapter candidates for one target component: donors shortlist from the
// exact catalog index, each shortlisted seam joins its measured residual
// and predicted fitness, the seam class and adapter rung derive from the
// typed contract and the residual, and the queue ranks by residual
// ascending, then predicted fitness descending, then canonical identity.
// Shortlisted seams without measurements are reported, never silently
// dropped, and a target with no measured candidates refuses.
func EnumerateCompositionCandidates(
	index *ExactComponentIndex,
	target CatalogComponent,
	measurements []DonorSeamMeasurement,
	limit int,
) (candidates []CompositionCandidate, unmeasured []CanonicalComponentDescriptor, err error) {
	if index == nil {
		return nil, nil, errors.New("composition: enumeration requires the exact catalog index")
	}
	hits, err := index.Search(target, limit)
	if err != nil {
		return nil, nil, err
	}
	measured := make(map[artifact.ID]map[string]DonorSeamMeasurement, len(measurements))
	for _, measurement := range measurements {
		if math.IsNaN(measurement.Residual) || math.IsInf(measurement.Residual, 0) ||
			measurement.Residual < 0 ||
			math.IsNaN(measurement.Fitness) || math.IsInf(measurement.Fitness, 0) {
			return nil, nil, errors.New("composition: seam measurements must be finite with a nonnegative residual")
		}
		byComponent, seen := measured[measurement.Donor]
		if !seen {
			byComponent = make(map[string]DonorSeamMeasurement)
			measured[measurement.Donor] = byComponent
		}
		byComponent[measurement.Component] = measurement
	}
	for _, hit := range hits {
		if hit.Descriptor.Model == target.Model {
			continue
		}
		measurement, seen := measured[hit.Descriptor.Model][hit.Descriptor.Name]
		if !seen {
			unmeasured = append(unmeasured, hit.Descriptor)
			continue
		}
		candidates = append(candidates, CompositionCandidate{
			Donor: hit.Descriptor.Model, Component: hit.Descriptor.Name,
			Seam:     seamBoundaryFor(hit.Descriptor),
			Adapter:  adapterRungFor(measurement.Residual),
			Residual: measurement.Residual, Fitness: measurement.Fitness,
		})
	}
	if len(candidates) == 0 {
		return nil, unmeasured, errors.New("composition: no shortlisted seam carries a measured residual and fitness")
	}
	slices.SortFunc(candidates, func(a, b CompositionCandidate) int {
		if a.Residual != b.Residual {
			if a.Residual < b.Residual {
				return -1
			}
			return 1
		}
		if a.Fitness != b.Fitness {
			if a.Fitness > b.Fitness {
				return -1
			}
			return 1
		}
		if by := strings.Compare(a.Donor.String(), b.Donor.String()); by != 0 {
			return by
		}
		return strings.Compare(a.Component, b.Component)
	})
	return candidates, unmeasured, nil
}
