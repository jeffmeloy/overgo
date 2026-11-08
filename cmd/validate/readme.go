package main

import (
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/scorecard"
)

// The README's speed table stands between these two lines; -readme writes what stands between them.
const (
	readmeSpeedBegin             = "<!-- speed table: go run ./cmd/validate -readme writes it from the newest scorecard record -->"
	readmeSpeedEnd               = "<!-- end of the speed table -->"
	readmeFile                   = "README.md"
	readmeMode       fs.FileMode = 0o644
)

// writeReadmeSpeed writes the newest scorecard record in records' speed table into the README beneath root, between
// its two marker lines. With check it writes nothing and refuses a README whose table is not what the record gives, so
// a hand-typed figure, a cold one among them, cannot stand in the table.
func writeReadmeSpeed(root, records string, check bool, output io.Writer) error {
	published, err := scorecard.Records(records)
	if err != nil {
		return err
	}
	if len(published) == 0 {
		return fmt.Errorf("validate: no scorecard record in %s to write the README's speed table from", records)
	}
	newest := published[len(published)-1]
	path := filepath.Join(root, readmeFile)
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	text := string(data)
	before, rest, opened := strings.Cut(text, readmeSpeedBegin)
	_, after, closed := strings.Cut(rest, readmeSpeedEnd)
	if !opened || !closed {
		return fmt.Errorf("validate: %s holds no speed table between the lines %q and %q", path, readmeSpeedBegin, readmeSpeedEnd)
	}
	written := before + readmeSpeedBegin + "\n" + newest.SpeedTable() + newest.GenerationTable() + readmeSpeedEnd + after
	if check {
		if written != text {
			return fmt.Errorf("validate: %s's speed table is not what %s gives; write it with go run ./cmd/validate -readme", path, newest.Name)
		}
		return nil
	}
	if err := os.WriteFile(path, []byte(written), readmeMode); err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "%s: speed table written from %s\n", path, newest.Name)
	return err
}
