package gate

import (
	"bytes"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/repoanalysis"
)

// closureProjection is the active closure authority the gate's own repairs
// would leave: decisions rebound to where their literals moved, and decisions
// whose code is gone retired.
type closureProjection struct {
	documents []closureledger.Document
	aliases   map[string]artifact.ID
	// moved counts decisions rebound to a new offset.
	moved int
	// removed names unmatched decisions whose value no candidate in their
	// package still carries, nor, for a named constant, its name: their code
	// is gone, so retiring them discards nothing a reviewer would keep. kept
	// names the rest -- a literal moved where no rebind follows it, a
	// constant whose value changed -- which a person must judge. A literal's
	// name is its offset, so only a constant's name identifies it.
	removed, kept []string
}

// projectedMagicBindings rebinds the active closure decisions to the source in
// memory, exactly as the gate's own same-store rebind would publish them, and
// drops the decisions whose code is gone. The preflight writes no store, so
// without this a catalogued literal that only moved offset reads as new
// policy and stops the landing before the gate that rebinds it. A decision
// whose value survives in its file stays as it was, so it is still reported.
func projectedMagicBindings(snapshot repoanalysis.SourceSnapshot, documents []closureledger.Document, aliases map[string]artifact.ID) (closureProjection, error) {
	candidates, err := closurescan.ScanSnapshot(snapshot, nil, closurescan.CandidateAll)
	if err != nil {
		return closureProjection{}, err
	}
	projection, err := closurescan.ProjectActiveClosures(closurescan.CompileRebindIndex(candidates), candidates, documents, aliases, true, false, nil)
	if err != nil {
		return closureProjection{}, err
	}
	result := closureProjection{aliases: make(map[string]artifact.ID, len(aliases)), moved: len(projection.Rebound)}
	gone := map[artifact.ID]bool{}
	for _, pending := range projection.Pending {
		binding := pending.Document.Bindings[0]
		survives := false
		for _, candidate := range candidates {
			named := binding.Kind == closureledger.BindingConstant && candidate.Name == binding.Name
			if candidate.Package == binding.Package && (named || bytes.Equal(candidate.ValueJSON(), pending.Document.Value)) {
				survives = true
				break
			}
		}
		if survives {
			result.kept = append(result.kept, pending.Document.Name)
			continue
		}
		gone[pending.Document.ID] = true
		result.removed = append(result.removed, pending.Document.Name)
	}
	for _, document := range documents {
		if gone[document.ID] {
			continue
		}
		if current, resolved := projection.Resolved[document.ID]; resolved {
			document = current
		}
		result.documents = append(result.documents, document)
		for _, binding := range document.Bindings {
			alias, err := closureledger.ActiveAlias(binding)
			if err != nil {
				return closureProjection{}, err
			}
			result.aliases[alias] = document.ID
		}
	}
	return result, nil
}
