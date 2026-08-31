package agentloop

import (
	"overgo/internal/agenttool"
	"overgo/internal/artifact"
)

// SteeringPhase is the residual tool-call intervention one proposal decode
// runs under: the masked inspection-only action space and the alpha the
// propensity gate derived for it. The phase carries no grant, approval,
// dispatch, or receipt authority -- mutation proposals stay on the unsteered
// path behind the coordinator's full decision ladder, and the phase cannot
// reach it.
type SteeringPhase struct {
	Direction artifact.ID
	Alpha     float64
	Manuals   []agenttool.Manual
}

// ProposalSteeringPhase derives the steered proposal phase for one decode:
// mutation manuals and the dynamic capability proxy are masked out of the
// visible space, and alpha collapses to zero the moment the ORIGINAL space
// exposes a mutation or a proxy whose reachable set is not inspection-only.
// The unsteered space is untouched; masking removes options from the steered
// phase, it never adds authority to it.
func ProposalSteeringPhase(
	direction artifact.ID,
	selected float64,
	visible, proxyReachable []agenttool.Manual,
) SteeringPhase {
	return SteeringPhase{
		Direction: direction,
		Alpha:     agenttool.SteeringAlpha(selected, visible, proxyReachable),
		Manuals:   agenttool.InspectionOnlyManuals(visible),
	}
}
