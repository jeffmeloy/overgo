package main

import (
	"bytes"
	"context"
	"slices"
	"strings"
	"unicode"
)

// listedTest says a line go test -list prints names a test, an example or a fuzz target, which -run selects: one word,
// where the line it ends with names the package and its time.
func listedTest(line string) bool {
	name := strings.TrimSpace(line)
	return name != "" && !strings.ContainsFunc(name, unicode.IsSpace) &&
		(strings.HasPrefix(name, "Test") || strings.HasPrefix(name, "Example") || strings.HasPrefix(name, "Fuzz"))
}

// testSteps are the steps a lane run package by package takes (lanes-yield): each package's tests one at a time, the
// package's arguments with -run naming one test, as go test -list lists them with the lane's flags, tags and
// environment, so the card and the test turn pass to a waiting commit or probe between tests; in the run at
// c79cfc121 one package held them 1 h 26 min. A package that cannot be listed, or lists no test, runs whole, its run
// saying what failed.
func testSteps(ctx context.Context, root string, packages [][]string, environment []string) [][]string {
	var steps [][]string
	for _, arguments := range packages {
		names, err := listedTests(ctx, root, arguments, environment)
		if err != nil || len(names) == 0 {
			steps = append(steps, arguments)
			continue
		}
		for _, name := range names {
			steps = append(steps, oneTest(arguments, name))
		}
	}
	return steps
}

// listedTests are the tests, examples and fuzz targets go test lists for a package step, its arguments the go command's
// with the package last.
func listedTests(ctx context.Context, root string, arguments, environment []string) ([]string, error) {
	var listed bytes.Buffer
	pkg := len(arguments) - 1
	listing := slices.Concat(arguments[:pkg], []string{"-list", "."}, arguments[pkg:])
	if err := runCell(ctx, &listed, root, "go", listing, environment...); err != nil {
		return nil, err
	}
	var names []string
	for line := range strings.Lines(listed.String()) {
		if listedTest(line) {
			names = append(names, strings.TrimSpace(line))
		}
	}
	return names, nil
}

// oneTest is a package step's arguments running only the test named.
func oneTest(arguments []string, name string) []string {
	pkg := len(arguments) - 1
	return slices.Concat(arguments[:pkg], []string{"-run", "^" + name + "$"}, arguments[pkg:])
}
