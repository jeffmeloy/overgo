package clioptions

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
)

// Main: common command error exit.
func Main(run func() error) {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// MainNamed preserves the command prefix while sharing exit mechanics.
func MainNamed(name string, run func() error) {
	Main(func() error {
		if err := run(); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		return nil
	})
}

// WriteJSON: one JSON document.
func WriteJSON(writer io.Writer, value any) error {
	return json.NewEncoder(writer).Encode(value)
}

// WritePrettyJSON: indented JSON document.
func WritePrettyJSON(writer io.Writer, value any) error {
	encoder := json.NewEncoder(writer)
	encoder.SetIndent("", "  ")
	return encoder.Encode(value)
}

// OutputGenerated: check, update, or print generated data.
func OutputGenerated(data []byte, path string, check, update bool, stale string, writer io.Writer) error {
	if check {
		current, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if !bytes.Equal(current, data) {
			return errors.New(stale)
		}
		return nil
	}
	if update {
		return os.WriteFile(path, data, 0o644)
	}
	_, err := writer.Write(data)
	return err
}
