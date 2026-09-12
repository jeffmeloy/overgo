package main

import (
	"slices"
	"strings"
	"testing"
)

func TestBrowserPackageSelection(t *testing.T) {
	const listed = `{"Package":"overgo/internal/server","Output":"ok server [no tests to run]\n"}
{"Package":"overgo/internal/webuilane","Output":"TestWebUIBrowserLayoutAudit\n"}
{"Package":"overgo/internal/server","Output":"TestWebUIBrowserFirstRun\n"}
{"Package":"overgo/internal/webuilane","Output":"TestWebUIBrowserLayoutAudit\n"}
`
	for _, test := range []struct {
		name, pattern, listing string
		want                   []string
	}{
		{"layout only", "^TestWebUIBrowserLayoutAudit$", listed, []string{"overgo/internal/webuilane"}},
		{"model journey", "^TestWebUIBrowserFirstRun$", listed, []string{"overgo/internal/server"}},
		{"all browser tests", "^TestWebUIBrowser", listed, browserTestPackages},
		{"empty filter", "", listed, browserTestPackages},
		{"subtest fallback", "TestWebUIBrowserFirstRun/turn", "", browserTestPackages},
		{"missing target", "^TestWebUIBrowserAbsent$", listed, nil},
		{"empty discovery", "^TestWebUIBrowser", "", nil},
		{"invalid filter", "[", listed, nil},
		{"malformed discovery", "^TestWebUIBrowser", "{", nil},
		{"foreign owner", "^TestWebUIBrowser", `{"Package":"other/package","Output":"TestWebUIBrowserFirstRun"}`, nil},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := browserPackages(test.pattern, strings.NewReader(test.listing))
			if (err == nil) != (test.want != nil) || !slices.Equal(got, test.want) {
				t.Fatalf("packages=%v err=%v want=%v", got, err, test.want)
			}
		})
	}
}
