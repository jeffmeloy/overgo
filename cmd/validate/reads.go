package main

import (
	"cmp"
	"context"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/testevidence"
)

// readsCommits is how many of the trunk's latest commits -reads counts each input's moves over.
const readsCommits = 100

// answerReads answers -reads: it runs the tests of pkg in the repository at root once, with no kept pass, and says what
// their test binary read (testevidence.ReadsOf) and which of it moves at each commit (observed-inputs/measure). go
// test keeps a pass while the files the binary opened and the variables it read are unchanged, so an input outside the
// package that the trunk's commits move reruns the package at each of them. It names each file the binary opened or
// looked at inside the repository and outside the package, with how many of the last readsCommits commits changed
// it, most first; the files outside the repository; and the variables it read.
func answerReads(ctx context.Context, output io.Writer, root, pkg string) error {
	reads, err := testevidence.ReadsOf(ctx, root, pkg)
	if err != nil {
		return err
	}
	absoluteRoot, err := filepath.Abs(root)
	if err != nil {
		return err
	}
	// A package is named as ./ and its directory under the repository.
	packageDir := filepath.Join(absoluteRoot, filepath.FromSlash(strings.TrimPrefix(pkg, "./")))
	// Each commit's changed files, so a directory read (a stat of a package) counts every commit that moved a file
	// under it.
	logged, err := gitauthority.Query(ctx, root, "log", "--format=tformat:commit", "--name-only", "-n", fmt.Sprint(readsCommits))
	if err != nil {
		return err
	}
	var commits [][]string
	for name := range strings.FieldsSeq(string(logged)) {
		if name == "commit" {
			commits = append(commits, nil)
		} else if len(commits) > 0 {
			commits[len(commits)-1] = append(commits[len(commits)-1], name)
		}
	}
	moved := func(read string) int {
		count := 0
		for _, changed := range commits {
			if slices.ContainsFunc(changed, func(name string) bool { return name == read || strings.HasPrefix(name, read+"/") }) {
				count++
			}
		}
		return count
	}
	inside, outside := map[string]bool{}, map[string]bool{}
	for _, paths := range []map[string]bool{boolKeys(reads.Opened), reads.Looked} {
		for path := range paths {
			relative, err := filepath.Rel(absoluteRoot, path)
			switch {
			case err != nil || strings.HasPrefix(relative, ".."):
				outside[path] = true
			case !strings.HasPrefix(path, packageDir+string(filepath.Separator)):
				inside[filepath.ToSlash(relative)] = true
			}
		}
	}
	moves := map[string]int{}
	for name := range inside {
		moves[name] = moved(name)
	}
	read := slices.SortedFunc(maps.Keys(inside), func(a, b string) int { return cmp.Or(cmp.Compare(moves[b], moves[a]), cmp.Compare(a, b)) })
	fmt.Fprintf(output, "validate -reads %s: its tests opened %d files and looked at %d paths; %d in the repository outside the package, %d outside the repository; it read %d variables\n",
		pkg, len(reads.Opened), len(reads.Looked), len(read), len(outside), len(reads.Environment))
	// The null device a process opens for a child's unset stdio is logged as the relative path NUL, which go test
	// resolves inside the package and finds modified in the future (2185 on Windows), so it keeps no pass of the
	// package whatever changes (cmd/go: an input modified under 2 s ago, errFileTooNew).
	if slices.ContainsFunc(slices.Collect(maps.Keys(reads.Opened)), func(path string) bool { return strings.EqualFold(filepath.Base(path), "NUL") }) {
		fmt.Fprintln(output, "  its tests open the null device, NUL, which go test reads as a file in the package modified in the future, so it keeps no pass of them")
	}
	for _, name := range read {
		fmt.Fprintf(output, "  %3d of the last %d commits moved %s\n", moves[name], readsCommits, name)
	}
	// A path under the temporary directory is a test's own (t.TempDir), new at every run, so go test finds none of them
	// again and keeps no pass of a package that reads one: they are counted, the rest named.
	temporary, other := 0, []string{}
	for _, path := range slices.Sorted(maps.Keys(outside)) {
		if within, err := filepath.Rel(os.TempDir(), path); err == nil && !strings.HasPrefix(within, "..") {
			temporary++
			continue
		}
		other = append(other, path)
	}
	fmt.Fprintf(output, "  outside the repository: %d paths under the temporary directory, new at each run\n", temporary)
	for _, path := range other {
		fmt.Fprintf(output, "  outside the repository: %s\n", path)
	}
	fmt.Fprintf(output, "  variables: %s\n", strings.Join(slices.Sorted(maps.Keys(reads.Environment)), ", "))
	return nil
}

// boolKeys are the keys of a map as a set.
func boolKeys(values map[string]string) map[string]bool {
	keys := map[string]bool{}
	for key := range values {
		keys[key] = true
	}
	return keys
}
