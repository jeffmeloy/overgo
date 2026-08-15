package clioptions

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
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

// Tail bounds diagnostic output while retaining its most recent bytes.
func Tail(text string, limit int) string {
	if len(text) <= limit {
		return text
	}
	return "..." + text[len(text)-limit:]
}

// CombinedOutput runs one command with an explicit environment.
func CombinedOutput(env []string, name string, args ...string) (string, error) {
	return CombinedOutputIn("", env, name, args...)
}

// CombinedOutputIn runs one command from an explicit working directory.
func CombinedOutputIn(directory string, env []string, name string, args ...string) (string, error) {
	command := exec.Command(name, args...)
	command.Dir = directory
	command.Env = env
	output, err := command.CombinedOutput()
	return string(output), err
}

// POSIXShell resolves the repository command shell on supported hosts.
func POSIXShell() (string, error) {
	if shell, err := exec.LookPath("sh"); err == nil {
		return shell, nil
	}
	if runtime.GOOS == "windows" {
		for _, path := range []string{
			`C:\Program Files\Git\bin\bash.exe`,
			`C:\Program Files\Git\usr\bin\bash.exe`,
		} {
			if info, err := os.Stat(filepath.Clean(path)); err == nil && !info.IsDir() {
				return path, nil
			}
		}
	}
	return "", errors.New("POSIX shell unavailable")
}

// Verdict renders command success without treating skipped work as green.
func Verdict(err error) string {
	if err != nil {
		return "FAIL"
	}
	return "ok"
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
