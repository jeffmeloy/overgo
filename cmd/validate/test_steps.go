package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
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

// laneStep is one go test run of a lane: its arguments, and the place among the lane's packages of the package it runs.
type laneStep struct {
	arguments []string
	pkg       int
}

// testSteps are the steps a lane run package by package takes (lanes-yield): each package's tests one at a time, the
// package's arguments with -run naming one test, as go test -list lists them with the lane's flags, tags and
// environment, so the card and the test turn pass to a waiting commit or probe between tests; in the run at
// c79cfc121 one package held them 1 h 26 min. A package that cannot be listed, or lists no test, runs whole, its run
// saying what failed.
func testSteps(ctx context.Context, root string, packages [][]string, environment []string) []laneStep {
	var steps []laneStep
	for pkg, arguments := range packages {
		names, err := listedTests(ctx, root, arguments, environment)
		if err != nil || len(names) == 0 {
			steps = append(steps, laneStep{arguments: arguments, pkg: pkg})
			continue
		}
		for _, name := range names {
			steps = append(steps, laneStep{arguments: oneTest(arguments, name), pkg: pkg})
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

// laneVerdict judges a lane's steps, each step's verdict as acquire gave it, as the lane's packages were judged run
// whole (lanes-verdict): a package passes when a step of it passed and none failed, is inapplicable when declared
// exclusions alone answered, and fails when a step failed; a pass in one package lends another nothing. The lane fails
// when a package failed, its error then wrapping no errInapplicable, so a failure is never reported as inapplicable;
// else it is inapplicable when a package was. Judged test by test, a package whose one test declared itself
// inapplicable had been an inapplicable lane, and one with a failure beside such a test an inapplicable lane too.
func laneVerdict(steps []laneStep, verdicts []error) error {
	var failed, declared []error
	packages := 0
	for _, step := range steps {
		packages = max(packages, step.pkg+1)
	}
	for pkg := range packages {
		passed, broke, excluded := false, []error(nil), []error(nil)
		for index, step := range steps {
			switch verdict := verdicts[index]; {
			case step.pkg != pkg:
			case verdict == nil:
				passed = true
			case errors.Is(verdict, errInapplicable):
				excluded = append(excluded, verdict)
			default:
				broke = append(broke, verdict)
			}
		}
		switch {
		case len(broke) > 0:
			failed = append(failed, broke...)
		case !passed:
			declared = append(declared, excluded...)
		}
	}
	if len(failed) > 0 {
		return errors.Join(failed...)
	}
	return errors.Join(declared...)
}

// errNotRun is the verdict of a lane's step before it runs, which a step that never ran keeps: it lends its package no
// pass.
func errNotRun(step laneStep) error {
	return fmt.Errorf("go %s did not run", strings.Join(step.arguments, " "))
}
