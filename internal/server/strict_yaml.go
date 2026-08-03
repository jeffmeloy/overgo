package server

import (
	"errors"
	"io"
	"strings"

	"gopkg.in/yaml.v3"
)

// decodeStrictYAML: one known-field document
func decodeStrictYAML(data []byte, destination any, documentError string) error {
	decoder := yaml.NewDecoder(strings.NewReader(string(data)))
	decoder.KnownFields(true)
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return errors.New(documentError)
	}
	return nil
}
