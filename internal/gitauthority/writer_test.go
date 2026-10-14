package gitauthority

import (
	"os/exec"
	"reflect"
	"strings"
	"testing"
)

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
