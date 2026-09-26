package clioptions

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"overgo/internal/processcontrol"
)

const (
	// DiagnosticTailBytes bounds command failure details while retaining enough
	// output to identify the failing package and assertion.
	DiagnosticTailBytes = 2000
	// OutputFileMode is the owner-writable, world-readable mode every
	// generated output file is written with.
	OutputFileMode = os.FileMode(0o644)
	// OutputDirectoryMode is the traversable mode for generated output
	// directories.
	OutputDirectoryMode = os.FileMode(0o755)
	// PrivateFileMode is the owner-only mode for local state that can contain
	// command arguments, failure context, or other non-published evidence.
	PrivateFileMode = os.FileMode(0o600)
	// ErrorFileEnvironment names a file a failing command also writes its
	// final error to, so a supervisor reads the refusal without parsing output.
	ErrorFileEnvironment = "OVERGO_ERROR_FILE"
)

// Main: common command error exit. The error file is this command's alone:
// its subprocesses do not inherit the name.
func Main(run func() error) {
	errorFile := os.Getenv(ErrorFileEnvironment)
	_ = os.Unsetenv(ErrorFileEnvironment)
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		if errorFile != "" {
			_ = os.WriteFile(errorFile, []byte(err.Error()), PrivateFileMode)
		}
		if errors.Is(err, processcontrol.ErrResourceBusy) {
			os.Exit(processcontrol.ResourceBusyExitCode)
		}
		if errors.Is(err, processcontrol.ErrDeviceMemory) {
			os.Exit(processcontrol.DeviceMemoryExitCode)
		}
		os.Exit(1)
	}
}

// Command declares the help identity a FlagSet cannot infer: a command's
// purpose, its intended audience, and the operation constraints that shape valid
// invocations. The owner declares them once so explicit help and the executed
// flags stay the same source.
type Command struct {
	Name        string
	Purpose     string
	Audience    string
	Constraints []string
}

// ParseCommandLine parses the process arguments against flags with the same
// help handling as ParseForHelp, so a command with no separately testable
// argument slice keeps the os.Args boundary in one place.
func (c Command) ParseCommandLine(flags *flag.FlagSet, out io.Writer) (handled bool, err error) {
	return c.ParseForHelp(flags, os.Args[1:], out)
}

// ParseForHelp parses args against flags, which must use flag.ContinueOnError,
// rendering help from the registered flags. On -h or -help it reports
// handled=true so the caller returns successfully without opening a store,
// loading a model, or mutating anything; a genuine parse error is returned with
// the same usage guidance already written to out so the caller exits non-zero.
func (c Command) ParseForHelp(flags *flag.FlagSet, args []string, out io.Writer) (handled bool, err error) {
	flags.SetOutput(out)
	flags.Usage = func() { c.WriteUsage(flags, out) }
	switch err := flags.Parse(args); {
	case err == nil:
		return false, nil
	case errors.Is(err, flag.ErrHelp):
		return true, nil
	default:
		return false, err
	}
}

// WriteUsage renders the command purpose, audience, operation constraints, and
// the registered flags with their types and defaults, so help reflects exactly
// what execution parses.
func (c Command) WriteUsage(flags *flag.FlagSet, out io.Writer) {
	fmt.Fprintf(out, "%s: %s\n", c.Name, c.Purpose)
	if strings.TrimSpace(c.Audience) != "" {
		fmt.Fprintf(out, "audience: %s\n", c.Audience)
	}
	for _, constraint := range c.Constraints {
		fmt.Fprintf(out, "constraint: %s\n", constraint)
	}
	fmt.Fprintln(out, "flags:")
	previous := flags.Output()
	flags.SetOutput(out)
	flags.PrintDefaults()
	flags.SetOutput(previous)
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

// TailWriter is an io.Writer that retains only the last limit bytes written to
// it, so a supervised process whose output is reported only as a tail never
// holds its whole stream in memory. Tail renders the retained bytes exactly as
// Tail renders a full string.
type TailWriter struct {
	limit    int
	buffer   []byte
	overflow bool
}

// NewTailWriter returns a TailWriter bounded to limit retained bytes; callers
// pass a positive diagnostic bound.
func NewTailWriter(limit int) *TailWriter { return &TailWriter{limit: limit} }

// Write appends data, keeping only the last limit bytes.
func (w *TailWriter) Write(data []byte) (int, error) {
	n := len(data)
	if len(data) >= w.limit {
		w.overflow = w.overflow || len(w.buffer) != 0 || len(data) > w.limit
		w.buffer = append(w.buffer[:0], data[len(data)-w.limit:]...)
		return n, nil
	}
	if len(w.buffer)+len(data) > w.limit {
		w.buffer = append(w.buffer[:0], w.buffer[len(w.buffer)+len(data)-w.limit:]...)
		w.overflow = true
	}
	w.buffer = append(w.buffer, data...)
	return n, nil
}

// Reset discards the retained bytes for reuse across process launches.
func (w *TailWriter) Reset() {
	w.buffer, w.overflow = w.buffer[:0], false
}

// Len reports the retained byte count, which never exceeds the limit however
// large the written stream is.
func (w *TailWriter) Len() int { return len(w.buffer) }

// Tail renders the retained bytes, prefixed with an ellipsis once anything was
// dropped, matching Tail over the whole stream.
func (w *TailWriter) Tail() string {
	if w.overflow {
		return "..." + string(w.buffer)
	}
	return string(w.buffer)
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
		return WriteOutputFile(path, data)
	}
	_, err := writer.Write(data)
	return err
}

// WriteOutputFile writes a command-owned public artifact.
func WriteOutputFile(path string, data []byte) error {
	return os.WriteFile(path, data, OutputFileMode)
}

// EnsureOutputDirectory creates a command-owned output path.
func EnsureOutputDirectory(path string) error {
	return os.MkdirAll(path, OutputDirectoryMode)
}
