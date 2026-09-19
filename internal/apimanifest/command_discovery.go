package apimanifest

import (
	"errors"
	"strings"
)

// Classification records who calls a shipped command, decided by its actual
// callers and contracts rather than a name suffix: an operator at a terminal, an
// automation lane, or both.
type Classification string

const (
	// OperatorFacing is run by a person at a terminal.
	OperatorFacing Classification = "operator-facing"
	// AutomationInternal is invoked only by the automation lane or another command.
	AutomationInternal Classification = "automation-internal"
	// OperatorAndAutomation is run both ways.
	OperatorAndAutomation Classification = "both"
)

func validClassification(value Classification) bool {
	switch value {
	case OperatorFacing, AutomationInternal, OperatorAndAutomation:
		return true
	default:
		return false
	}
}

// CommandDescriptor is the owner-declared discovery metadata a command projects
// into its manifest Binary: the purpose and audience a FlagSet cannot infer, the
// classification decided by actual callers, and the flags parsed from the
// execution FlagSet. It is data only; projecting it runs no command and triggers
// no init side effect.
type CommandDescriptor struct {
	Name           string
	Package        string
	BuildContexts  []string
	Purpose        string
	Audience       string
	Classification Classification
	Flags          []Parameter
}

// ProjectCommandBinary projects a command descriptor into its manifest Binary,
// so help and the manifest share one owner-declared source. It refuses an
// incomplete command -- missing purpose, audience, or classification -- rather
// than advertising it as discoverable, and refuses duplicate flags.
func ProjectCommandBinary(descriptor CommandDescriptor) (Binary, error) {
	if !validText(descriptor.Name) || !validText(descriptor.Package) || len(descriptor.BuildContexts) == 0 {
		return Binary{}, errors.New("apimanifest: command descriptor needs a name, package, and build context")
	}
	if strings.TrimSpace(descriptor.Purpose) == "" || strings.TrimSpace(descriptor.Audience) == "" {
		return Binary{}, errors.New("apimanifest: command " + descriptor.Name + " declares no purpose or audience")
	}
	if !validClassification(descriptor.Classification) {
		return Binary{}, errors.New("apimanifest: command " + descriptor.Name + " has no caller classification")
	}
	if duplicateParameters(descriptor.Flags) {
		return Binary{}, errors.New("apimanifest: command " + descriptor.Name + " declares a flag twice")
	}
	return Binary{
		Name: descriptor.Name, Package: descriptor.Package, BuildContexts: descriptor.BuildContexts,
		Purpose: descriptor.Purpose, Audience: descriptor.Audience,
		Classification: string(descriptor.Classification), Flags: descriptor.Flags,
	}, nil
}
