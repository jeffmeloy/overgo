package testevidence

import (
	"slices"
	"strings"
	"testing"
	"time"
)

// TestPackageProgressFromStream pins what the progress hook receives: each
// package terminal event in stream order with go test's elapsed, no test
// events, and every event still reaches the deadline observer.
func TestPackageProgressFromStream(t *testing.T) {
	t.Parallel()
	stream := strings.Join([]string{
		`{"Action":"run","Package":"a","Test":"TestA"}`,
		`{"Action":"pass","Package":"a","Test":"TestA","Elapsed":0.5}`,
		`{"Action":"pass","Package":"a","Elapsed":1.25}`,
		`{"Action":"skip","Package":"b","Elapsed":0}`,
		`{"Action":"output","Package":"c","Output":"FAIL\n"}`,
		`{"Action":"fail","Package":"c","Elapsed":2}`,
	}, "\n") + "\n"
	var seen []PackageProgress
	events := 0
	observer := packageProgressObserver(func(progress PackageProgress) { seen = append(seen, progress) }, func(goTestEvent) { events++ })
	if _, err := readGoTestJSON(strings.NewReader(stream), false, false, 1024, nil, observer); err != nil {
		t.Fatal(err)
	}
	want := []PackageProgress{
		{Package: "a", Action: "pass", Elapsed: 1250 * time.Millisecond},
		{Package: "b", Action: "skip"},
		{Package: "c", Action: "fail", Elapsed: 2 * time.Second},
	}
	if !slices.Equal(seen, want) {
		t.Fatalf("progress = %+v, want %+v", seen, want)
	}
	if events != 6 {
		t.Fatalf("deadline observer saw %d events, want every one of 6", events)
	}
	next := func(goTestEvent) {}
	if packageProgressObserver(nil, next) == nil {
		t.Fatal("no progress hook drops the deadline observer")
	}
}
