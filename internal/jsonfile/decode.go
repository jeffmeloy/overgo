//overgo:runtime-inputs caller

// Package jsonfile decodes complete JSON files.
package jsonfile

import (
	"encoding/json"
	"io/fs"
	"os"

	"overgo/internal/atomicfile"
	"overgo/internal/strictjson"
)

func Decode(path string, destination any) error {
	return decode(path, destination, json.Unmarshal)
}

func DecodeStrict(path string, destination any) error {
	return decode(path, destination, strictjson.DecodeBytes)
}

func decode(path string, destination any, decode func([]byte, any) error) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return decode(data, destination)
}

func Write(path string, source any, mode fs.FileMode) error {
	data, err := json.MarshalIndent(source, "", "  ")
	if err != nil {
		return err
	}
	return atomicfile.Write(path, append(data, '\n'), mode)
}
