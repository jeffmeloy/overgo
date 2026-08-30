package gitauthority

import (
	"strings"
	"testing"
)

func TestValidObjectID(t *testing.T) {
	for _, test := range []struct {
		name  string
		value string
		want  bool
	}{
		{name: "sha1", value: strings.Repeat("a", 40), want: true},
		{name: "sha256", value: strings.Repeat("b", 64), want: true},
		{name: "uppercase", value: strings.Repeat("A", 40), want: true},
		{name: "short", value: strings.Repeat("a", 39)},
		{name: "non hexadecimal", value: strings.Repeat("g", 40)},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := ValidObjectID(test.value); got != test.want {
				t.Fatalf("ValidObjectID(%q) = %t, want %t", test.value, got, test.want)
			}
		})
	}
}
