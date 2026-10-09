package inferencesurface

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestRepositoryCountsEveryCapabilityPackage holds the repository's own tree
// to an identity for every package of the module its capability catalog
// imports: none is refused for a read of the environment nothing declares,
// which would leave its capabilities keeping no answer and saying nothing.
// The linker flag names each, and the count is the same when taken again.
func TestRepositoryCountsEveryCapabilityPackage(t *testing.T) {
	ctx := t.Context()
	counted, refused, err := identities(ctx, "../..")
	if err != nil {
		t.Fatal(err)
	}
	for entry, reads := range refused {
		t.Errorf("%s has no identity: declare each read as a result setting or a neutral read (identity.go): %s", entry, strings.Join(reads, "; "))
	}
	flag, err := LinkFlag(ctx, "../..")
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range []string{"overgo/internal/mediacapability", "overgo/internal/latentvideo", "overgo/internal/speechsynth", "overgo/internal/vqaserve"} {
		if identity := counted[entry]; identity == "" || !strings.Contains(flag, entry+"="+identity) {
			t.Errorf("%s is counted %q, and the linker flag does not name it so", entry, identity)
		}
	}
	if !strings.HasPrefix(flag, "-X=overgo/internal/inferencesurface.stamp=") || strings.ContainsAny(flag, " \t\"'") {
		t.Errorf("the linker flag is %q: want one word that sets this package's stamp", flag)
	}
	t.Logf("%d packages counted, a flag of %d bytes", len(counted), len(flag))
}

// TestCapabilityIdentityMovesOnlyWithItsOwnSources counts a tree with two
// capabilities and holds each identity to what its results depend on. A
// source outside an identity's closure leaves it where it was. A source of
// its closure, a file it embeds, a kernel source where it runs kernels, a
// source of the executor, a source of the catalog and a build setting each
// move it; the capability that runs no kernel keeps its identity when a
// kernel source changes. A read of the environment nothing declares leaves
// the package that reaches it with no identity. In a process, a library on
// the search path and the value of a result setting each move the identity,
// and a build that counted nothing holds none, so it gives again nothing an
// older process answered.
func TestCapabilityIdentityMovesOnlyWithItsOwnSources(t *testing.T) {
	root := t.TempDir()
	write := func(name, content string) {
		t.Helper()
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module fixture\n\ngo 1.26\n")
	write("internal/mediacapability/catalog.go", "package mediacapability\n\nimport (\n\t_ \"fixture/internal/capabilityruntime\"\n\t_ \"fixture/internal/picture\"\n\t_ \"fixture/internal/speech\"\n)\n")
	write("internal/capabilityruntime/runtime.go", "package capabilityruntime\n")
	write("internal/picture/picture.go", "package picture\n\nimport (\n\t_ \"embed\"\n\n\t_ \"fixture/internal/cuda/kernel\"\n\t_ \"fixture/internal/shared\"\n)\n\n//go:embed profile.json\nvar profile []byte\n")
	write("internal/picture/profile.json", `{"steps":4}`)
	write("internal/speech/speech.go", "package speech\n\nimport _ \"fixture/internal/shared\"\n")
	write("internal/shared/shared.go", "package shared\n")
	write("internal/cuda/kernel/kernel.go", "package kernel\n")
	write("internal/training/train.go", "package training\n")
	write("kernels/manifest.json", "{}")
	write("kernels/cuda/ops.cu", "")
	ctx := t.Context()
	const picture, speech = "fixture/internal/picture", "fixture/internal/speech"
	count := func() map[string]string {
		t.Helper()
		counted, _, err := identities(ctx, root)
		if err != nil {
			t.Fatal(err)
		}
		return counted
	}
	before := count()
	if before[picture] == "" || before[speech] == "" || before[picture] == before[speech] {
		t.Fatalf("the tree counts %v: want an identity for each capability's package, and not the same one", before)
	}
	for _, change := range []struct {
		what, file, content string
		moves               []string
	}{
		{"a source no capability depends on", "internal/training/train.go", "package training // changed\n", nil},
		{"a source of one capability", "internal/picture/picture.go", "package picture\n\nimport (\n\t_ \"embed\"\n\n\t_ \"fixture/internal/cuda/kernel\"\n\t_ \"fixture/internal/shared\"\n)\n\n//go:embed profile.json\nvar profile []byte // changed\n", []string{picture}},
		{"a file one capability embeds", "internal/picture/profile.json", `{"steps":8}`, []string{picture}},
		{"a kernel source", "kernels/cuda/ops.cu", "__global__ void op() {}\n", []string{picture}},
		{"a source both depend on", "internal/shared/shared.go", "package shared // changed\n", []string{picture, speech}},
		{"a source of the executor", "internal/capabilityruntime/runtime.go", "package capabilityruntime // changed\n", []string{picture, speech}},
		{"a source of the catalog", "internal/mediacapability/catalog.go", "package mediacapability\n\nimport (\n\t_ \"fixture/internal/capabilityruntime\"\n\t_ \"fixture/internal/picture\"\n\t_ \"fixture/internal/speech\"\n) // changed\n", []string{picture, speech}},
	} {
		write(change.file, change.content)
		after := count()
		var moved []string
		for _, entry := range []string{picture, speech} {
			if after[entry] != before[entry] {
				moved = append(moved, entry)
			}
		}
		if !slices.Equal(moved, change.moves) {
			t.Errorf("%s moved the identities of %v; want %v", change.what, moved, change.moves)
		}
		before = after
	}

	t.Setenv("GOFLAGS", "-tags=wide")
	if tagged := count(); tagged[picture] == before[picture] || tagged[speech] == before[speech] {
		t.Error("a build setting left an identity where it was")
	}
	t.Setenv("GOFLAGS", "")

	write("internal/speech/speech.go", "package speech\n\nimport (\n\t\"os\"\n\n\t_ \"fixture/internal/shared\"\n)\n\nvar precision = os.Getenv(\"FIXTURE_PRECISION\")\n")
	counted, refused, err := identities(ctx, root)
	if err != nil {
		t.Fatal(err)
	}
	if _, held := counted[speech]; held || len(refused[speech]) != 1 || !strings.HasSuffix(refused[speech][0], `speech.go reads "FIXTURE_PRECISION"`) || counted[picture] == "" {
		t.Errorf("a read of the environment nothing declares counted %v and refused %v; want the package that reads it refused with the read, and the other counted", counted, refused)
	}
	write("internal/speech/speech.go", "package speech\n\nimport (\n\t\"os\"\n\n\t_ \"fixture/internal/shared\"\n)\n\nvar toolkit, trace = os.Getenv(\"CUDA_PATH\"), os.Getenv(\"OVERGO_LEDGER_TRACE\")\n")
	if declared := count(); declared[speech] == "" {
		t.Error("a package that reads only a result setting and a neutral name has no identity")
	}

	// In a process: the libraries and the settings it computes under.
	directory := t.TempDir()
	library := filepath.Join(directory, libraries[0])
	if err := os.WriteFile(library, []byte("driver 1"), 0o644); err != nil {
		t.Fatal(err)
	}
	settings := map[string]string{"PATH": directory, "CUDA_PATH": `C:\toolkit\v1`}
	lookup := func(name string) string { return settings[name] }
	executable := filepath.Join(t.TempDir(), "server.exe")
	first := environment(lookup, executable)
	if again := environment(lookup, executable); again != first || !strings.Contains(first, library) {
		t.Fatalf("the environment reads %q then %q: want the same twice, naming the library on the search path", first, again)
	}
	if err := os.WriteFile(library, []byte("driver 2 is longer"), 0o644); err != nil {
		t.Fatal(err)
	}
	updated := environment(lookup, executable)
	if updated == first {
		t.Error("a changed library left the environment where it was")
	}
	settings["CUDA_PATH"] = `C:\toolkit\v2`
	if environment(lookup, executable) == updated {
		t.Error("a changed result setting left the environment where it was")
	}
	settings["OVERGO_LEDGER_TRACE"] = "1"
	withTrace := environment(lookup, executable)
	delete(settings, "OVERGO_LEDGER_TRACE")
	if environment(lookup, executable) != withTrace {
		t.Error("a name that changes no result moved the environment")
	}

	held := stamp
	defer func() { stamp = held }()
	stamp = ""
	if identity, held := Execution(picture); held || identity != "" || Linked() != "" {
		t.Errorf("a build that counted nothing holds identity %q and passes on the flag %q", identity, Linked())
	}
	stamp = picture + "=" + before[picture] + "," + speech + "=" + before[speech]
	if linked := Linked(); linked != "-X=overgo/internal/inferencesurface.stamp="+stamp {
		t.Errorf("a counted build passes on the flag %q; want the one that sets this package's stamp to its own count", linked)
	}
	one, heldOne := Execution(picture)
	other, heldOther := Execution(speech)
	if _, heldNone := Execution("fixture/internal/training"); !heldOne || !heldOther || heldNone || one == other || one == before[picture] {
		t.Errorf("a counted build holds %q (%t) and %q (%t), and one for an uncounted package: %t; want one for each counted package, each joined with the process's environment", one, heldOne, other, heldOther, heldNone)
	}
	stamp = picture + "=" + strings.Repeat("0", len(before[picture]))
	if moved, _ := Execution(picture); moved == one {
		t.Error("another count gave the same identity")
	}
}
