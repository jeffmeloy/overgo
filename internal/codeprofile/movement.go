package codeprofile

import "overgo/internal/repoanalysis"

// ProductionMovement reports gross AST additions and deletions by file. It
// deliberately does not collapse them into a quality score.
type ProductionMovement struct {
	Added, Deleted               int
	GoLinesAdded, GoLinesDeleted int
}

func MeasureProductionMovement(base, candidate repoanalysis.SourceSnapshot) (ProductionMovement, error) {
	baseNodes, baseLines, err := snapshotSurface(base)
	if err != nil {
		return ProductionMovement{}, err
	}
	candidateNodes, candidateLines, err := snapshotSurface(candidate)
	if err != nil {
		return ProductionMovement{}, err
	}
	var movement ProductionMovement
	for path, nodes := range candidateNodes {
		if delta := nodes - baseNodes[path]; delta > 0 {
			movement.Added += delta
		}
	}
	for path, nodes := range baseNodes {
		if delta := nodes - candidateNodes[path]; delta > 0 {
			movement.Deleted += delta
		}
	}
	for path, lines := range candidateLines {
		if delta := lines - baseLines[path]; delta > 0 {
			movement.GoLinesAdded += delta
		}
	}
	for path, lines := range baseLines {
		if delta := lines - candidateLines[path]; delta > 0 {
			movement.GoLinesDeleted += delta
		}
	}
	return movement, nil
}

func snapshotSurface(snapshot repoanalysis.SourceSnapshot) (map[string]int, map[string]int, error) {
	nodes := map[string]int{}
	lines := map[string]int{}
	for _, source := range snapshot.Files {
		generated, err := source.Generated()
		if err != nil {
			return nil, nil, err
		}
		if generated {
			continue
		}
		file, err := source.Syntax()
		if err != nil {
			return nil, nil, err
		}
		lines[source.Path] = source.Line(file.End())
		if !source.Test {
			nodes[source.Path] = NodeCount(file)
		}
	}
	return nodes, lines, nil
}
