package codeprofile

import (
	"cmp"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
)

// stagedSurfaceRetentionClass is the closed set of reasons an exported
// declaration may remain without a production call site or future plan row.
type stagedSurfaceRetentionClass string

const (
	// stagedSurfaceTestSupport retains a cross-package test-support contract.
	stagedSurfaceTestSupport stagedSurfaceRetentionClass = "test-support"
	// stagedSurfaceProductionInterface retains a dynamically satisfied
	// production interface that a static call-site census cannot observe.
	stagedSurfaceProductionInterface stagedSurfaceRetentionClass = "production-interface"
)

// stagedSurfaceRetention binds a retained classification to evidence.
type stagedSurfaceRetention struct {
	Classification stagedSurfaceRetentionClass `json:"classification"`
	Evidence       artifact.ID                 `json:"evidence"`
}

// StagedSurfaceEntry declares one exported symbol whose production consumer
// is deliberately deferred or whose non-call-site role is evidence-bound.
type StagedSurfaceEntry struct {
	Package string `json:"package"`
	Name    string `json:"name"`
	Reason  string `json:"reason"`
	// RetireWith names an exact canonical open step. A projected lane may keep
	// an unchanged foreign step through the declaration's InheritedFrom proof.
	RetireWith string `json:"retire_with,omitzero"`
	// Retained is mutually exclusive with RetireWith.
	Retained *stagedSurfaceRetention `json:"retained,omitempty"`
}

// StagedSurfaceDeclaration is the versioned staged-surface file.
type StagedSurfaceDeclaration struct {
	Version uint16 `json:"version"`
	Doc     string `json:"doc,omitzero"`
	// InheritedFrom pins unchanged foreign retirement declarations after a lane
	// projects its queue. It grants no execution or completion authority.
	InheritedFrom string               `json:"inherited_from,omitzero"`
	Staged        []StagedSurfaceEntry `json:"staged"`
}

// LoadStagedSurface reads the declaration; an absent file declares
// nothing and stages nothing. Decoding is strict because this file is
// gate authority: a duplicate name that silently last-wins would drop
// a reviewed entry, so it refuses like every other gate document.
func LoadStagedSurface(path string) (StagedSurfaceDeclaration, error) {
	var declaration StagedSurfaceDeclaration
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			return StagedSurfaceDeclaration{Version: artifact.SecondDocumentVersion}, nil
		}
		return StagedSurfaceDeclaration{}, err
	}
	if err := jsonfile.DecodeStrict(path, &declaration); err != nil {
		return StagedSurfaceDeclaration{}, err
	}
	if declaration.Version != artifact.SecondDocumentVersion {
		return StagedSurfaceDeclaration{}, fmt.Errorf("codeprofile: unsupported staged-surface version %d", declaration.Version)
	}
	if declaration.InheritedFrom != "" && !gitauthority.ValidObjectID(declaration.InheritedFrom) {
		return StagedSurfaceDeclaration{}, fmt.Errorf("codeprofile: inherited staged declarations require an exact commit")
	}
	needsPlan := false
	seen := make(map[string]bool, len(declaration.Staged))
	for _, entry := range declaration.Staged {
		if entry.Package == "" || entry.Name == "" || entry.Reason == "" {
			return StagedSurfaceDeclaration{}, fmt.Errorf("codeprofile: staged entry %s.%s needs package, name, and reason", entry.Package, entry.Name)
		}
		key := entry.Package + "\x00" + entry.Name
		if seen[key] {
			return StagedSurfaceDeclaration{}, fmt.Errorf("codeprofile: duplicate staged entry %s.%s", entry.Package, entry.Name)
		}
		seen[key] = true
		hasPlan := entry.RetireWith != ""
		hasRetention := entry.Retained != nil
		if hasPlan == hasRetention {
			return StagedSurfaceDeclaration{}, fmt.Errorf("codeprofile: staged entry %s.%s needs exactly one of retire_with or retained", entry.Package, entry.Name)
		}
		if hasRetention {
			if err := validateStagedSurfaceRetention(*entry.Retained); err != nil {
				return StagedSurfaceDeclaration{}, fmt.Errorf("codeprofile: staged entry %s.%s: %w", entry.Package, entry.Name, err)
			}
		}
		needsPlan = needsPlan || hasPlan
		if hasPlan && !canonicalPlanStep(entry.RetireWith) {
			return StagedSurfaceDeclaration{}, fmt.Errorf("codeprofile: staged entry %s.%s retire_with %q must name one exact item/step", entry.Package, entry.Name, entry.RetireWith)
		}
	}
	if !needsPlan {
		return declaration, nil
	}
	var live stagedSurfacePlan
	// Decode only the item/step/status projection. The plan owner performs
	// strict whole-document validation; importing it here would cycle through
	// its code-manifest census back into codeprofile.
	err := jsonfile.Decode(filepath.Join(filepath.Dir(path), "plan.json"), &live)
	if err != nil {
		return StagedSurfaceDeclaration{}, fmt.Errorf("codeprofile: load canonical plan for staged surface: %w", err)
	}
	open, err := live.openSteps()
	if err != nil {
		return StagedSurfaceDeclaration{}, err
	}
	var inherited *inheritedStagedSurface
	for _, entry := range declaration.Staged {
		_, present := open[entry.RetireWith]
		if entry.RetireWith == "" || present {
			continue
		}
		if live.Scope == "lane" && live.Lane != "" && declaration.InheritedFrom != "" {
			if live.containsItem(entry.RetireWith) {
				return StagedSurfaceDeclaration{}, fmt.Errorf("codeprofile: inherited retirement %s conflicts with the live queue", entry.RetireWith)
			}
			if inherited == nil {
				source, err := loadInheritedStagedSurface(path, declaration.InheritedFrom)
				if err != nil {
					return StagedSurfaceDeclaration{}, err
				}
				inherited = &source
			}
			if err := inherited.verify(entry, live.Lane); err != nil {
				return StagedSurfaceDeclaration{}, err
			}
			continue
		}
		return StagedSurfaceDeclaration{}, fmt.Errorf(
			"codeprofile: staged entry %s.%s retire_with %q is not an exact canonical open plan step (it is missing, blocked, or historical)",
			entry.Package, entry.Name, entry.RetireWith)
	}
	return declaration, nil
}

func (live stagedSurfacePlan) openSteps() (map[string]string, error) {
	open := make(map[string]string)
	seenPlanSteps := make(map[string]bool)
	for _, item := range live.Items {
		for _, step := range item.Steps {
			reference := item.ID + "/" + step.ID
			if !canonicalPlanStep(reference) || seenPlanSteps[reference] {
				return nil, fmt.Errorf("codeprofile: canonical plan has invalid or duplicate step %q", reference)
			}
			seenPlanSteps[reference] = true
			if item.Status != "open" {
				continue
			}
			if step.Status == "open" {
				open[reference] = cmp.Or(item.Owner, live.Lane)
			}
		}
	}
	return open, nil
}

func canonicalPlanStep(reference string) bool {
	item, step, found := strings.Cut(reference, "/")
	return found && item != "" && step != "" && !strings.Contains(step, "/")
}

func validateStagedSurfaceRetention(retention stagedSurfaceRetention) error {
	switch retention.Classification {
	case stagedSurfaceTestSupport, stagedSurfaceProductionInterface:
	default:
		return fmt.Errorf("unknown retained classification %q", retention.Classification)
	}
	if !retention.Evidence.Valid() || retention.Evidence.Kind() != artifact.KindEvidence {
		return fmt.Errorf("retained classification %q needs evidence identity", retention.Classification)
	}
	return nil
}

// PartitionStagedSurface splits new unconsumed declarations into those
// the staged declaration accepts and those that remain blocking.
func PartitionStagedSurface(
	unconsumed []ConsumerDeclaration,
	declaration StagedSurfaceDeclaration,
) (accepted, blocking []ConsumerDeclaration) {
	staged := make(map[string]bool, len(declaration.Staged))
	for _, entry := range declaration.Staged {
		staged[entry.Package+"\x00"+entry.Name] = true
	}
	for _, candidate := range unconsumed {
		if staged[candidate.Package+"\x00"+candidate.Name] {
			accepted = append(accepted, candidate)
		} else {
			blocking = append(blocking, candidate)
		}
	}
	return accepted, blocking
}
