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
	// One name for every run, so its files beside the resource admissions are written again rather than left behind.
	name := "overgo-test-reading-turn"
	var session func() error
	loads := 0
	draw := func() {
		t.Helper()
		if session == nil {
			session = testcheck.Must(processcontrol.TakeTurn(name, "the reading", io.Discard))(t)
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
	waiter := exec.Command(os.Args[0], "-test.run=^TestAReadingGivesTheCardUpBetweenPicturesToAWaiter$", "-test.timeout=0")
	// A process another started, as a landing's tests are, and not the reading's child, which would ride its place.
	waiter.Env = append(os.Environ(), processcontrol.TurnEnvironment+"=", readingWaiterEnvironment+"="+name)
	output := bufio.NewReader(testcheck.Must(waiter.StdoutPipe())(t))
	waiter.Stderr = os.Stderr
	testcheck.NoError(t, waiter.Start())
	t.Cleanup(func() { _ = waiter.Process.Kill(); _ = waiter.Wait() })
	line := func() string { t.Helper(); return strings.TrimSpace(testcheck.Must(output.ReadString('\n'))(t)) }
	if waits := line(); !strings.Contains(waits, "waits for the turn "+name) || !strings.Contains(waits, "for the reading since") {
		t.Fatalf("the waiter said %q, want it to wait for the reading", waits)
	}
	draw()
	betweenPictures(true)
	if took := line(); !strings.Contains(took, "took the turn "+name) {
		t.Fatalf("the waiter said %q, want it to take the turn the reading gave up", took)
	}
	if taken := line(); taken != "taken" {
		t.Fatalf("the waiter said %q, want taken", taken)
	}
	testcheck.NoError(t, waiter.Wait())
	draw()
	betweenPictures(false)
	testcheck.Must(closeSession(t.Context()))(t)
	if loads != 2 {
		t.Fatalf("the reading loaded its session %d times, want 2: once for the first two pictures, once after the waiter", loads)
	}
}
