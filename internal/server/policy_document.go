package server

import (
	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

const (
	policyDocumentSchemaVersion       = int(artifact.InitialDocumentVersion)
	maxPolicyDocumentBytes      int64 = 1 << 20
)

func loadPolicyDocument(path string, destination any) error {
	data, err := readBoundedFile(path, maxPolicyDocumentBytes, false)
	if err != nil {
		return err
	}
	return strictjson.DecodeBytes(data, destination)
}
