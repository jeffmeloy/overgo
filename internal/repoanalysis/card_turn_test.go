package repoanalysis

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// cardAdmissions are the ways a process once said it had the card, each read by some users and not others, which the
// card's one turn replaced (card-turn): a claim on a machine-wide resource, the device reservation of a measurement,
// the card hold a session wrote for the plan's dispatch, and the benchmark's unavailable record for a claimed card.
var cardAdmissions = struct {
	claims       []string
	declarations []string
	words        []string
}{
	claims:       []string{"ShareResource", "ClaimResource", "HoldResource"},
	declarations: []string{"ReserveDevice", "CardHold", "HoldCard", "ReadCardHold"},
	words:        []string{"card_hold.json", "hold-card", "unavailable: "},
}

// resourceClaimers are the production packages that may claim a machine-wide resource: the quiet machine, whose two
// sides keep a timed measurement off a busy machine, and processcontrol itself, whose campaigns pass their resources to
// their children. Neither claims the card.
var resourceClaimers = []string{"internal/quietmachine", "internal/processcontrol"}

// TestTheCardTurnIsTheOnlyCardAdmission reads the module's production code and wants no card admission but the turn
// a context takes as it is created: no claim on a machine-wide resource outside resourceClaimers, no declaration of a
// device reservation or a card hold, and no string that is the card hold's file, its flag or the unavailable record's
// opening.
func TestTheCardTurnIsTheOnlyCardAdmission(t *testing.T) {
	t.Parallel()
	for _, tree := range []string{"cmd", "internal"} {
		testcheck.NoError(t, filepath.WalkDir(filepath.Join(checkoutRoot, tree), func(file string, entry fs.DirEntry, err error) error {
			if err != nil || entry.IsDir() || !strings.HasSuffix(file, ".go") || strings.HasSuffix(file, "_test.go") {
				return err
			}
			relative := filepath.ToSlash(testcheck.Must(filepath.Rel(checkoutRoot, file))(t))
			syntax := testcheck.Must(parser.ParseFile(token.NewFileSet(), file, nil, parser.SkipObjectResolution))(t)
			claimer := slices.Contains(resourceClaimers, filepath.ToSlash(filepath.Dir(relative)))
			ast.Inspect(syntax, func(node ast.Node) bool {
				switch value := node.(type) {
				case *ast.SelectorExpr:
					if slices.Contains(cardAdmissions.claims, value.Sel.Name) && !claimer {
						t.Errorf("%s claims a machine-wide resource with %s; the card is taken by its turn as a context is created", relative, value.Sel.Name)
					}
				case *ast.FuncDecl:
					if slices.Contains(cardAdmissions.declarations, value.Name.Name) {
						t.Errorf("%s declares %s, a card admission the card's turn replaced", relative, value.Name.Name)
					}
				case *ast.TypeSpec:
					if slices.Contains(cardAdmissions.declarations, value.Name.Name) {
						t.Errorf("%s declares %s, a card admission the card's turn replaced", relative, value.Name.Name)
					}
				case *ast.BasicLit:
					if text, _ := strconv.Unquote(value.Value); value.Kind == token.STRING && slices.Contains(cardAdmissions.words, text) {
						t.Errorf("%s names %q, a card admission the card's turn replaced", relative, text)
					}
				}
				return true
			})
			return nil
		}))
	}
}
