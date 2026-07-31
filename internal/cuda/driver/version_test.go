package driver

import "testing"

func TestVersionString(t *testing.T) {
	tests := []struct {
		value Version
		want  string
	}{
		{0, "0"},
		{11080, "11.8"},
		{12090, "12.9"},
		{13000, "13.0"},
	}

	for _, tt := range tests {
		if got := tt.value.String(); got != tt.want {
			t.Errorf("Version(%d).String() = %q, want %q", tt.value, got, tt.want)
		}
	}
}
