// Package executionfailure is the closed failure taxonomy for external
// execution: raw failure evidence normalizes deterministically onto one
// canonical cause, and every classification names the classifier
// version that produced it, so improved classifiers derive new records
// instead of rewriting old ones.
package executionfailure

import (
	"slices"
	"strings"
)

// Cause is one canonical normalized failure cause.
type Cause string

const (
	// CauseContextExhaustion reports a prompt or session exceeding the
	// model's context budget.
	CauseContextExhaustion Cause = "context-exhaustion"
	// CauseConfiguration reports invalid or missing declared settings.
	CauseConfiguration Cause = "configuration"
	// CauseAuthorization reports refused credentials or permissions.
	CauseAuthorization Cause = "authorization"
	// CauseQuota reports an exhausted allowance imposed by a provider.
	CauseQuota Cause = "quota"
	// CauseCapacity reports exhausted local or remote resources.
	CauseCapacity Cause = "capacity"
	// CauseNetwork reports transport-level connectivity failure.
	CauseNetwork Cause = "network"
	// CauseModelAvailability reports an absent or unservable model.
	CauseModelAvailability Cause = "model-availability"
	// CauseTimeout reports a bounded wait that expired.
	CauseTimeout Cause = "timeout"
	// CauseMissingExecutable reports a command that could not be found.
	CauseMissingExecutable Cause = "missing-executable"
	// CauseProcessFailure reports a process that exited nonzero for a
	// reason the classifier does not recognize.
	CauseProcessFailure Cause = "process-failure"
	// CauseToolFailure reports an in-process tool reporting failure.
	CauseToolFailure Cause = "tool-failure"
	// CauseUnknown reports evidence no rule recognizes; it stays raw
	// until a newer classifier learns the cause.
	CauseUnknown Cause = "unknown"
)

// CanonicalCauses is the closed cause vocabulary in declaration order.
var CanonicalCauses = []Cause{
	CauseContextExhaustion, CauseConfiguration, CauseAuthorization,
	CauseQuota, CauseCapacity, CauseNetwork, CauseModelAvailability,
	CauseTimeout, CauseMissingExecutable, CauseProcessFailure,
	CauseToolFailure, CauseUnknown,
}

// ValidCause reports membership in the canonical cause vocabulary.
func ValidCause(cause Cause) bool {
	return slices.Contains(CanonicalCauses, cause)
}

// ClassifierVersion is the version of the normalization rules below.
// It grows whenever a rule is added, removed, or reordered, so every
// derived classification names the exact rules that produced it.
const ClassifierVersion uint16 = 1

// Evidence is the raw failure material a classifier reads. It is
// preserved verbatim by whoever records it; normalization never edits
// it.
type Evidence struct {
	// Message is the failure text the failing component reported.
	Message string
	// Detail carries secondary raw evidence, typically a stderr tail.
	Detail string
	// ExitCode is the process exit status; zero when not a process.
	ExitCode int32
}

// Classification is one derived normalization of evidence: the cause,
// the rule that decided it, and the classifier version that ran.
type Classification struct {
	ClassifierVersion uint16
	Cause             Cause
	Rule              string
}

// normalizationRule matches evidence by marker substring; the ordered
// table below is the classifier, and first match wins, so the mapping
// from evidence to cause is a pure function of the table version.
type normalizationRule struct {
	name    string
	cause   Cause
	markers []string
}

var normalizationRules = []normalizationRule{
	{name: "context-budget", cause: CauseContextExhaustion, markers: []string{
		"context length", "maximum context", "context window", "prompt is too long"}},
	{name: "credentials", cause: CauseAuthorization, markers: []string{
		"unauthorized", "permission denied", "invalid api key", "authentication failed", "forbidden"}},
	{name: "allowance", cause: CauseQuota, markers: []string{
		"quota exceeded", "rate limit", "too many requests", "insufficient credit"}},
	{name: "resources", cause: CauseCapacity, markers: []string{
		"out of memory", "overloaded", "insufficient capacity", "no space left on device", "resource exhausted"}},
	{name: "servable", cause: CauseModelAvailability, markers: []string{
		"model not found", "model unavailable", "no servable model", "model is loading"}},
	{name: "command-lookup", cause: CauseMissingExecutable, markers: []string{
		"executable file not found", "not recognized as an internal or external command", "no such executable"}},
	{name: "expired-wait", cause: CauseTimeout, markers: []string{
		"deadline exceeded", "timed out", "timeout"}},
	{name: "transport", cause: CauseNetwork, markers: []string{
		"connection refused", "connection reset", "no such host", "network is unreachable", "tls handshake", "broken pipe"}},
	{name: "declared-settings", cause: CauseConfiguration, markers: []string{
		"unknown flag", "flag provided but not defined", "missing required", "invalid configuration", "unknown option"}},
	{name: "tool-report", cause: CauseToolFailure, markers: []string{
		"tool failed", "tool call failed", "tool error"}},
}

// Rule names for the two positions the ordered table cannot express:
// the nonzero-exit fallback and the terminal unmatched rule.
const (
	ruleNonzeroExit = "nonzero-exit"
	ruleUnmatched   = "unmatched"
)

// Normalize deterministically maps raw evidence onto one canonical
// cause: first marker match over the ordered rule table wins, a
// nonzero exit with unrecognized text is a process failure, and
// anything else stays unknown for a newer classifier to learn.
func Normalize(evidence Evidence) Classification {
	text := strings.ToLower(evidence.Message + "\n" + evidence.Detail)
	for _, rule := range normalizationRules {
		for _, marker := range rule.markers {
			if strings.Contains(text, marker) {
				return Classification{ClassifierVersion: ClassifierVersion, Cause: rule.cause, Rule: rule.name}
			}
		}
	}
	if evidence.ExitCode != 0 {
		return Classification{ClassifierVersion: ClassifierVersion, Cause: CauseProcessFailure, Rule: ruleNonzeroExit}
	}
	return Classification{ClassifierVersion: ClassifierVersion, Cause: CauseUnknown, Rule: ruleUnmatched}
}
