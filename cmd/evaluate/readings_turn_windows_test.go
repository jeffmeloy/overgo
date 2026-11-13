//go:build windows

package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"testing"

	"overgo/internal/processcontrol"
	"overgo/internal/testcheck"
)

// readingWaiterEnvironment makes a process of this test binary the waiter of TestAReadingGivesTheCardUpBetweenPicturesToAWaiter,
// waiting for the turn it names.
const readingWaiterEnvironment = "OVERGO_TEST_READING_WAITER"

// readingTurn is the turn the readings' tests hold as a model's session holds the card's: one name for every run, so
// its files beside the resource admissions are written again rather than left behind.
const readingTurn = "overgo-test-reading-turn"

// readingWaiter starts a waiter for the reading's turn, a process another started, as a landing's tests are, and not
// the reading's child, which would ride its place; it returns the waiter, once it says it waits for the reading, and its
// next line.
func readingWaiter(t *testing.T) (*exec.Cmd, func() string) {
	t.Helper()
	waiter := exec.Command(os.Args[0], "-test.run=^TestAReadingGivesTheCardUpBetweenPicturesToAWaiter$", "-test.timeout=0")
	waiter.Env = append(os.Environ(), processcontrol.TurnEnvironment+"=", readingWaiterEnvironment+"="+readingTurn)
	output := bufio.NewReader(testcheck.Must(waiter.StdoutPipe())(t))
	waiter.Stderr = os.Stderr
	testcheck.NoError(t, waiter.Start())
	t.Cleanup(func() { _ = waiter.Process.Kill(); _ = waiter.Wait() })
	line := func() string { t.Helper(); return strings.TrimSpace(testcheck.Must(output.ReadString('\n'))(t)) }
	if waits := line(); !strings.Contains(waits, "waits for the turn "+readingTurn) || !strings.Contains(waits, "for the reading since") {
		t.Fatalf("the waiter said %q, want it to wait for the reading", waits)
	}
	return waiter, line
}

// waiterTook reads that the waiter took the turn the reading gave up, and ended.
func waiterTook(t *testing.T, waiter *exec.Cmd, line func() string) {
	t.Helper()
	if took := line(); !strings.Contains(took, "took the turn "+readingTurn) {
		t.Fatalf("the waiter said %q, want it to take the turn the reading gave up", took)
	}
	if said := line(); said != "taken" {
		t.Fatalf("the waiter said %q, want taken", said)
	}
	testcheck.NoError(t, waiter.Wait())
}

// TestAReadingGivesTheCardUpBetweenPicturesToAWaiter draws three pictures of a reading whose session holds a turn as a
// model's session holds the card's: with no waiter the session stays loaded from the first picture to the second;
// a waiter appears, and after the second picture the session closes, which gives the turn up, the waiter takes it and
// ends, and the third picture loads the session again in its turn.
func TestAReadingGivesTheCardUpBetweenPicturesToAWaiter(t *testing.T) {
	if name := os.Getenv(readingWaiterEnvironment); name != "" {
		release := testcheck.Must(processcontrol.TakeTurn(name, "the waiter", os.Stdout))(t)
		fmt.Println("taken")
		testcheck.NoError(t, release())
		return
	}
	var session func() error
	loads := 0
	draw := func() {
		t.Helper()
		if session == nil {
			session = testcheck.Must(processcontrol.TakeTurn(readingTurn, "the reading", io.Discard))(t)
			loads++
		}
	}
	closeSession := func(context.Context) (int, error) {
		if session == nil {
			return 0, nil
		}
		err := session()
		session = nil
		return 1, err
	}
	betweenPictures := func(want bool) {
		t.Helper()
		if gave := testcheck.Must(giveTheCardToAWaiter(t.Context(), closeSession))(t); gave != want {
			t.Fatalf("the reading gave the card up: %t, want %t", gave, want)
		}
	}
	draw()
	betweenPictures(false)
	waiter, line := readingWaiter(t)
	draw()
	betweenPictures(true)
	waiterTook(t, waiter, line)
	draw()
	betweenPictures(false)
	testcheck.Must(closeSession(t.Context()))(t)
	if loads != 2 {
		t.Fatalf("the reading loaded its session %d times, want 2: once for the first two pictures, once after the waiter", loads)
	}
}

// TestAReadingThatKeepsTheCardSaysSo holds a reading whose release between pictures leaves its turn held, as Wan's left
// its text encoder on the card (card-handover/encoder-release): with a waiter there, the reading gives no card up and
// refuses, naming the turn it kept, where it had said it gave the card to the waiter; once its session closes the
// waiter takes the turn.
func TestAReadingThatKeepsTheCardSaysSo(t *testing.T) {
	session := testcheck.Must(processcontrol.TakeTurn(readingTurn, "the reading", io.Discard))(t)
	waiter, line := readingWaiter(t)
	keeps := func(context.Context) (int, error) { return 0, nil }
	if gave, err := giveTheCardToAWaiter(t.Context(), keeps); gave || err == nil || !strings.Contains(err.Error(), "left this render holding its turn") {
		t.Fatalf("a release that kept the turn gave the card up: %t, %v; want no card given and a refusal naming the turn kept", gave, err)
	}
	testcheck.NoError(t, session())
	waiterTook(t, waiter, line)
}
