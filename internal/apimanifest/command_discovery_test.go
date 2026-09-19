package apimanifest

import (
	"slices"
	"testing"
)

// TestCommandDiscoveryProjection holds the command discovery contract: a
// complete descriptor projects its purpose, audience, caller classification, and
// parsed flags into a manifest Binary that the manifest still accepts; a
// descriptor missing purpose, audience, or classification is refused rather than
// advertised as discoverable; changed flag registration is reflected; and the
// projection runs no command.
func TestCommandDiscoveryProjection(t *testing.T) {
	base := CommandDescriptor{
		Name: "plan", Package: "overgo/cmd/plan", BuildContexts: []string{"windows/amd64"},
		Purpose: "edit and dispatch the validated campaign", Audience: "the master-lead session",
		Classification: OperatorAndAutomation,
		Flags:          []Parameter{{Name: "next", Type: "bool"}, {Name: "prompt", Type: "bool"}},
	}
	binary, err := ProjectCommandBinary(base)
	if err != nil {
		t.Fatal(err)
	}
	if binary.Purpose != base.Purpose || binary.Audience != base.Audience || binary.Classification != string(OperatorAndAutomation) {
		t.Fatalf("projection dropped discovery metadata: %+v", binary)
	}
	if !slices.Equal(binary.Flags, base.Flags) {
		t.Fatalf("projection dropped flags: %+v", binary.Flags)
	}

	// The projected binary remains a valid manifest entry (manifest compatibility).
	if err := validateBinaries([]Binary{binary}, map[string]bool{"windows/amd64": true}, map[string]bool{}); err != nil {
		t.Fatalf("projected binary rejected by the manifest: %v", err)
	}

	// A changed flag registration is reflected, not stale.
	renamed := base
	renamed.Flags = []Parameter{{Name: "frontier", Type: "bool"}, {Name: "prompt", Type: "bool", Required: true}}
	changed, err := ProjectCommandBinary(renamed)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(changed.Flags, renamed.Flags) || slices.Equal(changed.Flags, base.Flags) {
		t.Fatalf("flag registration change not reflected: %+v", changed.Flags)
	}

	// An incomplete command is refused rather than advertised as complete.
	for name, mutate := range map[string]func(*CommandDescriptor){
		"no purpose":        func(d *CommandDescriptor) { d.Purpose = "" },
		"no audience":       func(d *CommandDescriptor) { d.Audience = "" },
		"no classification": func(d *CommandDescriptor) { d.Classification = "" },
		"unknown classification": func(d *CommandDescriptor) {
			d.Classification = Classification("mystery")
		},
		"duplicate flag": func(d *CommandDescriptor) {
			d.Flags = []Parameter{{Name: "next", Type: "bool"}, {Name: "next", Type: "bool"}}
		},
	} {
		t.Run(name, func(t *testing.T) {
			descriptor := base
			mutate(&descriptor)
			if _, err := ProjectCommandBinary(descriptor); err == nil {
				t.Fatalf("%s projected as a complete command", name)
			}
		})
	}
}
