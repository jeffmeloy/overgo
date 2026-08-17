package export

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
)

func CopyFiles(source, target string, names []string) error {
	if strings.TrimSpace(source) == "" || strings.TrimSpace(target) == "" || len(names) == 0 {
		return errors.New("export: source, target, and files are required")
	}
	for _, name := range names {
		if name == "" || filepath.Base(name) != name {
			return errors.New("export: file name must be a base name")
		}
		if err := copyFile(filepath.Join(source, name), filepath.Join(target, name)); err != nil {
			return err
		}
	}
	return nil
}

func copyFile(source, target string) (err error) {
	input, err := os.Open(source)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, input.Close()) }()
	output, err := os.OpenFile(target, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	committed := false
	defer func() {
		err = errors.Join(err, output.Close())
		if !committed {
			err = errors.Join(err, os.Remove(target))
		}
	}()
	if _, err = io.Copy(output, input); err != nil {
		return err
	}
	if err = output.Sync(); err != nil {
		return err
	}
	committed = true
	return nil
}
