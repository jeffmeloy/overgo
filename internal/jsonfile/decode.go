// Package jsonfile decodes complete JSON files.
package jsonfile

import (
	"encoding/json"
	"os"
)

func Decode(path string, destination any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, destination)
}
