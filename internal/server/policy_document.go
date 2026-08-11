package server

import "overgo/internal/strictjson"

const maxPolicyDocumentBytes int64 = 1 << 20

func loadPolicyDocument(path string, destination any) error {
	data, err := readBoundedFile(path, maxPolicyDocumentBytes, false)
	if err != nil {
		return err
	}
	return strictjson.DecodeBytes(data, destination)
}
