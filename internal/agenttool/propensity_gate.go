package agenttool

import (
	"fmt"
)

// InspectionOnlySteering decides whether a decode's action space admits a
// positive residual steering alpha: every visible manual must be an
// inspection, and if the capability proxy is visible its dynamically
// reachable set must be inspection-only too, because a proxy call dispatches
// whatever the active catalog resolves -- a proxy cannot hide a mutation
// from the propensity gate. A refusal names the first violating manual.
func InspectionOnlySteering(visible, proxyReachable []Manual) error {
	proxyVisible := false
	for _, manual := range visible {
		if manual.Effect != EffectInspection {
			return fmt.Errorf("agent tool: steering phase exposes %s manual %q", manual.Effect, manual.Name)
		}
		if manual.Name == CapabilityProxyName {
			proxyVisible = true
		}
	}
	if !proxyVisible {
		return nil
	}
	for _, manual := range proxyReachable {
		if manual.Effect != EffectInspection {
			return fmt.Errorf(
				"agent tool: capability proxy reaches %s manual %q; the proxy cannot hide it from the propensity gate",
				manual.Effect, manual.Name,
			)
		}
	}
	return nil
}

// SteeringAlpha derives the alpha one decode runs at: the selector-derived
// value inside an inspection-only action space, and exactly zero the moment
// any decode exposes a mutation or opaque effect. Alpha is never a request
// knob -- callers pass the selector's derived value and this gate owns the
// zeroing decision.
func SteeringAlpha(selected float64, visible, proxyReachable []Manual) float64 {
	if selected <= 0 || InspectionOnlySteering(visible, proxyReachable) != nil {
		return 0
	}
	return selected
}

// InspectionOnlyManuals masks a proposal phase's action space: mutation and
// non-inspection manuals leave the steered phase entirely, so a steered
// proposal can look and verify but can never name a mutation. Masking never
// grants anything: the unsteered mutation path keeps its full ladder.
func InspectionOnlyManuals(manuals []Manual) []Manual {
	masked := make([]Manual, 0, len(manuals))
	for _, manual := range manuals {
		if manual.Effect == EffectInspection && manual.Name != CapabilityProxyName {
			masked = append(masked, manual)
		}
	}
	return masked
}
