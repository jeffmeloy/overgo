package server

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
)

// decodeStrictJSON: one unknown-field-rejecting document, reject trailing content.
func decodeStrictJSON(data []byte, destination any, documentError string) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return errors.New(documentError)
	}
	return nil
}
