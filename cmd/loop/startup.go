package main

import (
	"bytes"
	"encoding/binary"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"
	"unicode/utf16"

	"overgo/internal/jsonfile"
	"overgo/internal/landing"
	"overgo/internal/processlock"
)

// The scheduled task that starts the driver: named once, run at the user's logon in the user's own session, never a
// second copy beside a running one, with no time limit, and started again a minute after it fails, three times.
const (
	startupTaskName        = "overgo-driver"
	startupRestartInterval = "PT1M"
	startupRestartCount    = 3
	startupTaskFile        = "overgo-driver-task.xml"
	startupTaskNamespace   = "http://schemas.microsoft.com/windows/2004/02/mit/task"
	startupFileMode        = 0o600
)

// startupTask is the Task Scheduler document that runs go run ./cmd/loop in the checkout at the logon of user.
type startupTask struct {
	XMLName  xml.Name `xml:"Task"`
	Version  string   `xml:"version,attr"`
	XMLNS    string   `xml:"xmlns,attr"`
	Purpose  string   `xml:"RegistrationInfo>Description"`
	User     string   `xml:"Triggers>LogonTrigger>UserId"`
	Enabled  bool     `xml:"Triggers>LogonTrigger>Enabled"`
	Logon    string   `xml:"Principals>Principal>LogonType"`
	Level    string   `xml:"Principals>Principal>RunLevel"`
	Settings struct {
		Instances    string `xml:"MultipleInstancesPolicy"`
		Battery      bool   `xml:"DisallowStartIfOnBatteries"`
		StopsBattery bool   `xml:"StopIfGoingOnBatteries"`
		Limit        string `xml:"ExecutionTimeLimit"`
		Interval     string `xml:"RestartOnFailure>Interval"`
		Count        int    `xml:"RestartOnFailure>Count"`
	} `xml:"Settings"`
	Command   string `xml:"Actions>Exec>Command"`
	Arguments string `xml:"Actions>Exec>Arguments"`
	Directory string `xml:"Actions>Exec>WorkingDirectory"`
}

// startupTaskXML is the task that starts the driver in repo at the logon of user, as Task Scheduler reads it.
func startupTaskXML(repo, user string) ([]byte, error) {
	task := startupTask{Version: "1.2", XMLNS: startupTaskNamespace, User: user, Enabled: true,
		Purpose: "Starts overgo's unattended driver, which works the plan until a recorded stop: go run ./cmd/loop -h",
		Logon:   "InteractiveToken", Level: "LeastPrivilege", Command: "go", Arguments: "run ./cmd/loop", Directory: repo}
	task.Settings.Instances, task.Settings.Limit = "IgnoreNew", "PT0S"
	task.Settings.Interval, task.Settings.Count = startupRestartInterval, startupRestartCount
	encoded, err := xml.MarshalIndent(task, "", "  ")
	return littleEndianUTF16(startupTaskHeader + string(encoded)), err
}

// startupTaskHeader declares the document UTF-16, the one encoding schtasks reads a task in: a UTF-8 document is
// refused as malformed, "unable to switch the encoding".
const startupTaskHeader = `<?xml version="1.0" encoding="UTF-16"?>` + "\n"

// byteOrderMark opens the document, which written little-endian says its order; schtasks refuses the big-endian one.
const byteOrderMark rune = 0xFEFF

// littleEndianUTF16 is text as schtasks reads a task document: UTF-16 little-endian behind its byte-order mark.
func littleEndianUTF16(text string) []byte {
	units := utf16.Encode(append([]rune{byteOrderMark}, []rune(text)...))
	var document []byte
	for _, unit := range units {
		document = binary.LittleEndian.AppendUint16(document, unit)
	}
	return document
}

// installStartup writes the task into the checkout's process state and registers it with Task Scheduler for the user
// running the command, replacing a task of its name; the driver then starts at each logon and after a failure.
func installStartup(repo string, said io.Writer) error {
	user := strings.TrimSpace(os.Getenv("USERDOMAIN") + `\` + os.Getenv("USERNAME"))
	if os.Getenv("USERNAME") == "" {
		return errors.New("no user is named in the environment to start the driver at the logon of")
	}
	document, err := startupTaskXML(repo, user)
	if err != nil {
		return err
	}
	path, err := statePath(repo, startupTaskFile)
	if err != nil {
		return err
	}
	if err := os.WriteFile(path, document, startupFileMode); err != nil {
		return err
	}
	var output bytes.Buffer
	if err := runToolEnv(os.Environ(), nil, &output, &output, "schtasks", "/Create", "/TN", startupTaskName, "/XML", path, "/F"); err != nil {
		return fmt.Errorf("schtasks: %w: %s", err, strings.TrimSpace(output.String()))
	}
	_, err = fmt.Fprintf(said, "loop: Windows starts the driver at %s's logon and after a failure (task %s); remove it with schtasks /Delete /TN %s /F\n",
		user, startupTaskName, startupTaskName)
	return err
}

// workerEnvironment is the environment a worker starts in: the driver's own, less the variables of a Claude Code
// session the driver was started from, so the worker is a session of its own and not a part of that one.
func workerEnvironment(environment []string) []string {
	return slices.DeleteFunc(slices.Clone(environment), func(entry string) bool {
		name, _, _ := strings.Cut(entry, "=")
		return strings.HasPrefix(strings.ToUpper(name), claudeSessionPrefix)
	})
}

// claudeSessionPrefix opens the name of every variable a Claude Code session sets for its own processes.
const claudeSessionPrefix = "CLAUDE"

// runningRecord is what the driver keeps while a worker works a step, and removes when the worker exits: one standing
// when the driver starts says the last run ended without its exit, as a machine that powers off leaves it.
type runningRecord struct {
	Step    string    `json:"step"`
	Started time.Time `json:"started"`
}

// The running record and the log of runs that ended without their exit, in the checkout's process state.
const (
	runningRecordFile = "driver_running.json"
	uncleanLogFile    = "driver_unclean.log"
)

// keepRunning records that a worker works step from started, the attempt's own start, so a run that never exits is
// closed at the next start as that attempt; it is removed by dropRunning when the worker exits.
func keepRunning(root, step string, started time.Time) error {
	path, err := statePath(root, runningRecordFile)
	if err != nil {
		return err
	}
	return jsonfile.Write(path, runningRecord{Step: step, Started: started}, startupFileMode)
}

// dropRunning removes the running record once its worker has exited.
func dropRunning(root string) error {
	path, err := statePath(root, runningRecordFile)
	if err != nil {
		return err
	}
	if err := os.Remove(path); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	return nil
}

// noteUncleanEnd reads a running record the last run left, appends what it was working to the log of unclean ends and
// says so, and removes the record; a start with none standing notes nothing.
func noteUncleanEnd(root string, now time.Time, said io.Writer) error {
	path, err := statePath(root, runningRecordFile)
	if err != nil {
		return err
	}
	var left runningRecord
	if err := jsonfile.DecodeStrict(path, &left); err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil
		}
		return err
	}
	line := fmt.Sprintf("%s: the driver's last run ended without its exit while a worker worked %s, from %s\n",
		now.UTC().Format(time.RFC3339), left.Step, left.Started.UTC().Format(time.RFC3339))
	log, err := os.OpenFile(filepath.Join(filepath.Dir(path), uncleanLogFile), os.O_APPEND|os.O_CREATE|os.O_WRONLY, startupFileMode)
	if err != nil {
		return err
	}
	_, written := log.WriteString(line)
	if err := errors.Join(written, log.Close()); err != nil {
		return err
	}
	if _, err := io.WriteString(said, "loop: "+line); err != nil {
		return err
	}
	// The attempt it was is closed as interrupted, its wall unknown, so a landing of the step counts it and does not
	// charge it the time the machine was down (SQA-356).
	interrupted := landing.Attempt{Step: left.Step, Started: left.Started, Interrupted: true}
	if err := landing.RecordAttempt(root, interrupted); err != nil {
		return err
	}
	return dropRunning(root)
}

// statePath is the file name holds in the process state of the checkout at root, the directory made where it is missing.
func statePath(root, name string) (string, error) {
	if err := processlock.EnsureStateDirectory(root); err != nil {
		return "", err
	}
	return filepath.Join(root, processlock.StateDirectory, name), nil
}
