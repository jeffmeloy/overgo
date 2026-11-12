package artifact

import (
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestAContentSizeRefusalSaysWhichBoundItBroke holds inline content's size refusal to its cause
// (guidance-ops/content-size): empty, over the inline limit, or other than its descriptor says, each said apart with
// the sizes. One message for the three left MOSS's failed draw of partiprompt_592 on 2026-11-11 without its cause.
func TestAContentSizeRefusalSaysWhichBoundItBroke(t *testing.T) {
	t.Parallel()
	data := []byte("four")
	id := testcheck.Must(IdentifyBytes(KindFile, data))(t)
	for _, test := range []struct {
		name string
		data []byte
		size uint64
		want string
	}{
		{"empty", nil, 0, "content is empty"},
		{"over the limit", make([]byte, MaxContentBytes+1), MaxContentBytes + 1, "over the inline limit of"},
		{"other than its descriptor", data, 5, "content is 4 bytes, and its descriptor says 5"},
	} {
		err := Content{Descriptor: Descriptor{ID: id, Size: test.size}, Data: test.data}.Validate()
		if err == nil || !strings.Contains(err.Error(), test.want) {
			t.Errorf("content %s is refused with %v; want a refusal saying %q", test.name, err, test.want)
		}
	}
}
