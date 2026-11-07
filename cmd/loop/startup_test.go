package main

import (
	"bytes"
	"encoding/binary"
	"encoding/xml"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
	"unicode/utf16"

	"overgo/internal/testcheck"
)

// TestTheStartupTaskRunsTheDriverAtBoot holds what starts the driver with no session to start it, as the power loss of
// 2026-11-03 left the work for 16.8 hours. The scheduled task runs go run ./cmd/loop in the checkout at the user's
// logon, in the user's own session at the least privilege, never a second copy beside a running one, with no time
// limit, and again a minute after it fails, three times; a path that needs escaping is escaped. The worker starts with
// no variable of the Claude Code session the driver was started from, the rest of the environment kept. A run that
// never exited leaves its running record, and the next start logs the step it was working and removes the record; a
// start with none logs nothing.
func TestTheStartupTaskRunsTheDriverAtBoot(t *testing.T) {
	t.Parallel()
	repo := `C:\Users\someone\over&go`
	encoded := testcheck.Must(startupTaskXML(repo, `MACHINE\someone`))(t)
	if mark := littleEndianUTF16(""); !bytes.HasPrefix(encoded, mark) || len(encoded)%2 != 0 {
		t.Fatalf("the task document opens %x; want UTF-16 little-endian behind its byte-order mark, as schtasks reads it", encoded[:min(len(encoded), 4)])
	}
	units := make([]uint16, 0, len(encoded)/2)
	for at := len(littleEndianUTF16("")); at < len(encoded); at += 2 {
		units = append(units, binary.LittleEndian.Uint16(encoded[at:]))
	}
	text := string(utf16.Decode(units))
	document := []byte(text)
	// The document declares UTF-16, which its text, decoded here, no longer is.
	decoder := xml.NewDecoder(strings.NewReader(text))
	decoder.CharsetReader = func(_ string, input io.Reader) (io.Reader, error) { return input, nil }
	var task startupTask
	testcheck.NoError(t, decoder.Decode(&task))
	if !strings.HasPrefix(text, startupTaskHeader) {
		t.Fatalf("the task document opens %.60q; want the UTF-16 declaration", text)
	}
	if task.User != `MACHINE\someone` || !task.Enabled || task.Logon != "InteractiveToken" || task.Level != "LeastPrivilege" ||
		task.Command != "go" || task.Arguments != "run ./cmd/loop" || task.Directory != repo || task.Settings.Instances != "IgnoreNew" ||
		task.Settings.Limit != "PT0S" || task.Settings.Interval != startupRestartInterval || task.Settings.Count != startupRestartCount {
		t.Fatalf("the startup task = %+v; want the driver at the user's logon, one copy, no limit, restarted after a failure", task)
	}
	if !bytes.Contains(document, []byte("over&amp;go")) || !bytes.Contains(document, []byte("<LogonTrigger>")) || !bytes.Contains(document, []byte(startupTaskNamespace)) {
		t.Fatalf("the task document does not escape its path or names no logon trigger or schema:\n%s", document)
	}
	kept := workerEnvironment([]string{"CLAUDECODE=1", "CLAUDE_CODE_SESSION_ID=b48d", "Claude_Pid=17680", "OVERGO_DATA_ROOT=D:/data", "PATH=C:/go/bin"})
	if !slices.Equal(kept, []string{"OVERGO_DATA_ROOT=D:/data", "PATH=C:/go/bin"}) {
		t.Fatalf("the worker's environment = %v; want the driver's without the Claude Code session's variables", kept)
	}
	root := t.TempDir()
	var said bytes.Buffer
	testcheck.NoError(t, noteUncleanEnd(root, time.Now(), &said))
	if said.Len() != 0 {
		t.Fatalf("a start with no running record said %q", said.String())
	}
	started := time.Date(2026, time.November, 3, 11, 26, 0, 0, time.UTC)
	testcheck.NoError(t, keepRunning(root, "fault-finding/queued-load", started))
	testcheck.NoError(t, noteUncleanEnd(root, started.Add(17*time.Hour), &said))
	logged := string(testcheck.Must(os.ReadFile(filepath.Join(root, ".overgo-runtime", uncleanLogFile)))(t))
	if !strings.Contains(logged, "fault-finding/queued-load") || !strings.Contains(said.String(), "fault-finding/queued-load") {
		t.Fatalf("the start after an unclean end logged %q and said %q; want the step it was working", logged, said.String())
	}
	if _, err := os.Stat(filepath.Join(root, ".overgo-runtime", runningRecordFile)); err == nil {
		t.Fatal("the running record outlived the start that read it")
	}
}
