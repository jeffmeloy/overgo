package runrecord

import (
	"encoding/json"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

// evidenceDocumentCodec is the single construction path for immutable
// run-record evidence documents with a fixed media type and schema.
func evidenceDocumentCodec[T any](
	name, mediaType, schema string,
	canonicalize func(*T) error,
	identity func(T) artifact.ID,
	setIdentity func(*T, artifact.ID),
	clone func(T) T,
) artifact.DocumentCodec[T] {
	return artifact.DocumentCodec[T]{
		Name: name,
		Contract: artifact.DocumentContract{
			Kind: artifact.KindEvidence, MediaType: mediaType, Schema: schema,
		},
		Decode:       func(data []byte, value *T) error { return strictjson.DecodeBytes(data, value) },
		Encode:       func(value T) ([]byte, error) { return json.Marshal(value) },
		Canonicalize: canonicalize,
		Clone:        clone,
		Identity:     identity,
		SetIdentity:  setIdentity,
	}
}

func dependencyLineage(child artifact.ID, parents ...artifact.ID) []artifact.Lineage {
	lineage := make([]artifact.Lineage, len(parents))
	for index, parent := range parents {
		lineage[index] = artifact.Lineage{
			Child: child, Parent: parent, Relation: artifact.RelationDependsOn,
		}
	}
	return lineage
}
