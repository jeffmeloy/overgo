package gate

import (
	"overgo/internal/artifact"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/repoanalysis"
)

// projectedMagicBindings rebinds the active closure decisions to the source in
// memory, exactly as the gate's own same-store rebind would publish them, and
// reports how many moved. The preflight writes no store, so without this a
// catalogued literal that only moved offset reads as new policy and stops the
// landing before the gate that rebinds it. A decision nothing rebinds stays as
// it was, so what remains unmatched is still reported.
func projectedMagicBindings(snapshot repoanalysis.SourceSnapshot, documents []closureledger.Document, aliases map[string]artifact.ID) ([]closureledger.Document, map[string]artifact.ID, int, error) {
	candidates, err := closurescan.ScanSnapshot(snapshot, nil, closurescan.CandidateAll)
	if err != nil {
		return nil, nil, 0, err
	}
	projection, err := closurescan.ProjectActiveClosures(closurescan.CompileRebindIndex(candidates), candidates, documents, aliases, true, false, nil)
	if err != nil {
		return nil, nil, 0, err
	}
	active := make([]closureledger.Document, 0, len(documents))
	bound := make(map[string]artifact.ID, len(aliases))
	for _, document := range documents {
		if current, resolved := projection.Resolved[document.ID]; resolved {
			document = current
		}
		active = append(active, document)
		for _, binding := range document.Bindings {
			alias, err := closureledger.ActiveAlias(binding)
			if err != nil {
				return nil, nil, 0, err
			}
			bound[alias] = document.ID
		}
	}
	return active, bound, len(projection.Rebound), nil
}
