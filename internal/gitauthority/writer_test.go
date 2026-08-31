package gitauthority

import (
	"os/exec"
	"reflect"
	"strings"
	"testing"
)

func TestRequireWriterVersion(t *testing.T) {
	tests := []struct {
		name    string
		output  string
		wantErr bool
	}{
		{name: "minimum", output: "git version 2.36.0"},
		{name: "Windows suffix", output: "git version 2.51.0.windows.1\r\n"},
		{name: "Apple decoration", output: "git version 2.39.3 (Apple Git-146)"},
		{name: "new major", output: "git version 3.0.0"},
		{name: "too old", output: "git version 2.35.9", wantErr: true},
		{name: "prerelease", output: "git version 2.36.0.rc1", wantErr: true},
		{name: "missing patch", output: "git version 2.36", wantErr: true},
		{name: "malformed", output: "version 2.51.0", wantErr: true},
		{name: "multiline", output: "git version 2.51.0\nignored", wantErr: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := requireWriterVersion(test.output)
			if (err != nil) != test.wantErr {
				t.Fatalf("require writer version error = %v, wantErr=%t", err, test.wantErr)
			}
		})
	}
}

func TestWriterArgumentsAreStableAndDoNotAliasInput(t *testing.T) {
	input := []string{"update-ref", "refs/heads/main", "new", "old"}
	got := WriterArguments(input...)
	want := []string{
		"--no-replace-objects",
		"-c", "core.fsync=all",
		"-c", "core.fsyncMethod=fsync",
		"update-ref", "refs/heads/main", "new", "old",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("writer arguments = %q, want %q", got, want)
	}
	got[len(got)-1] = "changed"
	if input[len(input)-1] != "old" {
		t.Fatalf("writer arguments aliased caller input: %q", input)
	}
}

func TestWriterArgumentsOverrideWeakRepositoryFsyncConfiguration(t *testing.T) {
	repository := t.TempDir()
	gitTestCommand(t, repository, "init", "-q")
	gitTestCommand(t, repository, "config", "core.fsync", "none")
	gitTestCommand(t, repository, "config", "core.fsyncMethod", "writeout-only")

	for _, test := range []struct {
		key  string
		want string
	}{
		{key: "core.fsync", want: "all"},
		{key: "core.fsyncMethod", want: "fsync"},
	} {
		command := exec.Command("git", WriterArguments("config", "--get", test.key)...)
		command.Dir = repository
		command.Env = RepositoryEnvironment()
		output, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("resolve %s through writer arguments: %v: %s", test.key, err, output)
		}
		if got := strings.TrimSpace(string(output)); got != test.want {
			t.Fatalf("writer %s = %q, want %q", test.key, got, test.want)
		}
	}
}
