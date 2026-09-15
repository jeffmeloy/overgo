package worklease

import "overgo/internal/textcheck"

const (
	// AutomationTextMaxBytes bounds every lease, role and plan identifier field.
	AutomationTextMaxBytes = 2048
	// UnassignedRole selects work without an explicit owner; a dispatch claim never carries it as its worker.
	UnassignedRole = "unassigned"
)

// ValidAutomationText accepts one bounded single-line field.
func ValidAutomationText(value string) bool {
	return textcheck.Bounded(value, AutomationTextMaxBytes, "\x00\r\n")
}

// ValidPlanID accepts one bounded plan item or step identifier.
func ValidPlanID(value string) bool {
	return textcheck.BoundedToken(value, AutomationTextMaxBytes, "/\\\x00\r\n")
}
