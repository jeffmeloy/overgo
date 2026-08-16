package codeprofile

import "overgo/internal/repoanalysis"

// ProductionMovement reports gross AST additions and deletions by file. It
// deliberately does not collapse them into a quality score.
type ProductionMovement struct {
	Added, Deleted int
}

func MeasureProductionMovement(base, candidate repoanalysis.SourceSnapshot) (ProductionMovement, error) {
	baseNodes, err := productionNodes(base)
	if err != nil {
		return ProductionMovement{}, err
	}
	candidateNodes, err := productionNodes(candidate)
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
	return movement, nil
}

func productionNodes(snapshot repoanalysis.SourceSnapshot) (map[string]int, error) {
	nodes := map[string]int{}
	for _, source := range snapshot.Files {
		if source.Test {
			continue
		}
		generated, err := source.Generated()
		if err != nil {
			return nil, err
		}
		if generated {
			continue
		}
		file, err := source.Syntax()
		if err != nil {
			return nil, err
		}
		nodes[source.Path] = NodeCount(file)
	}
	return nodes, nil
}
