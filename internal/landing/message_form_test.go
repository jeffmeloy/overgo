package landing

import (
	"strings"
	"testing"
)

// TestAMessageCarriesNoCoAuthorTrailer refuses the lead's words when a line of them credits a co-author, in any case.
func TestAMessageCarriesNoCoAuthorTrailer(t *testing.T) {
	t.Parallel()
	for message, refused := range map[string]bool{
		"Words of the lead.\n\nCo-Authored-By: A Model <model@example.invalid>\n": true,
		"Words of the lead.\n\nco-authored-by: someone\n":                         true,
		"Words of the lead, authored by the review.\n":                            false,
	} {
		if err := creditsNoCoAuthor([]byte(message)); (err != nil) != refused || refused && !strings.Contains(err.Error(), "co-author") {
			t.Errorf("message %q ended as %v, want it refused %v naming the co-author", message, err, refused)
		}
	}
}
