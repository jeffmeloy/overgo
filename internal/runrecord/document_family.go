package runrecord

import "overgo/internal/artifact"

func dependencyLineage(child artifact.ID, parents ...artifact.ID) []artifact.Lineage {
	lineage := make([]artifact.Lineage, len(parents))
	for index, parent := range parents {
		lineage[index] = artifact.Lineage{
			Child: child, Parent: parent, Relation: artifact.RelationDependsOn,
		}
	}
	return lineage
}
