package plan

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// standingDoctrine is the conversion of a plan that holds its doctrine as one
// string: every paragraph of the string, the text between two blank lines,
// becomes a paragraph that governs standing, word for word, and the string is
// dropped. A person then decides which paragraphs govern less than that.
func standingDoctrine(document Plan) Plan {
	for text := range strings.SplitSeq(document.DoctrineBefore, "\n\n") {
		document.Doctrine = append(document.Doctrine, Paragraph{Governs: governsStanding, Text: text})
	}
	document.DoctrineBefore = ""
	return document
}

// TestDoctrineParagraphsNameWhatTheyGovern holds the plan's doctrine to
// paragraphs that each say what they govern: the campaign, the rows named, or,
// standing, nothing that ends. The plan's own validation refuses a paragraph
// that names nothing, by its number, and refuses none for a row that has left:
// a row's landing changes no paragraph. Only the close-out refuses, when its
// last step would leave a paragraph that governs the campaign, or rows none of
// which the plan still holds, and it passes once a person has removed the
// paragraph or stored it as standing. The doctrine as one string, which an
// earlier commit and a lane's plan hold, is still read and is never saved; it
// converts to standing paragraphs word for word, and the checkout's own plan
// is already converted.
func TestDoctrineParagraphsNameWhatTheyGovern(t *testing.T) {
	t.Parallel()
	// A plan document that holds the rows named and the paragraphs given, each as raw JSON.
	written := func(rows []string, paragraphs ...string) []byte {
		items := make([]string, len(rows))
		for index, row := range rows {
			items[index] = `{"id":"` + row + `","title":"row","status":"open","steps":[{"id":"do","title":"do","status":"open","verify":"true"}]}`
		}
		return []byte(`{"campaign":"fixture","paragraphs":[` + strings.Join(paragraphs, ",") + `],"items":[` + strings.Join(items, ",") + `]}`)
	}
	const standing = `{"governs":"standing","text":"A rule that outlives every campaign."}`
	item := func(id string, steps ...string) Item {
		row := Item{ID: id, Title: id, Status: StatusOpen}
		for _, step := range steps {
			row.Steps = append(row.Steps, Step{ID: step, Title: step, Status: StatusOpen, Verify: "true"})
		}
		return row
	}
	governing := func(governs string, rows ...string) Paragraph {
		return Paragraph{Governs: governs, Rows: rows, Text: "A rule that governs " + governs + "."}
	}

	t.Run("a paragraph that says what it governs is read", func(t *testing.T) {
		for name, paragraph := range map[string]string{
			"standing":                         standing,
			"the campaign":                     `{"governs":"campaign","text":"A rule of this campaign."}`,
			"a row the plan holds":             `{"governs":"row","rows":["kept"],"text":"A rule of one row."}`,
			"rows, one of them landed":         `{"governs":"row","rows":["kept","landed"],"text":"A rule of two rows."}`,
			"a row that has left before close": `{"governs":"row","rows":["landed"],"text":"A rule the close-out will refuse."}`,
		} {
			if _, err := Parse(written([]string{"kept"}, standing, paragraph)); err != nil {
				t.Errorf("a paragraph that governs %s is refused: %v", name, err)
			}
		}
	})

	t.Run("a paragraph that names nothing is refused by its number", func(t *testing.T) {
		for name, paragraph := range map[string]string{
			"nothing":                    `{"governs":"","text":"A rule."}`,
			"everything":                 `{"governs":"everything","text":"A rule."}`,
			"a row, with none named":     `{"governs":"row","text":"A rule."}`,
			"standing, with a row named": `{"governs":"standing","rows":["kept"],"text":"A rule."}`,
			"campaign, with a row named": `{"governs":"campaign","rows":["kept"],"text":"A rule."}`,
			"a row whose id is no id":    `{"governs":"row","rows":["kept/do"],"text":"A rule."}`,
			"one row named twice":        `{"governs":"row","rows":["kept","kept"],"text":"A rule."}`,
			"standing, with no text":     `{"governs":"standing","text":" "}`,
			"standing, in two parts":     `{"governs":"standing","text":"One rule.\n\nAnother rule."}`,
		} {
			if _, err := Parse(written([]string{"kept"}, standing, paragraph)); err == nil || !strings.Contains(err.Error(), "paragraph 2") {
				t.Errorf("a paragraph that governs %s passed, or its refusal does not name paragraph 2: %v", name, err)
			}
		}
		if _, err := Parse(written(nil, `{"governs":"standing","text":"A rule.","note":"kept"}`)); err == nil {
			t.Error("a paragraph with a key the plan does not know was read")
		}
	})

	t.Run("a row that lands changes no paragraph", func(t *testing.T) {
		document := Plan{Campaign: "fixture", Items: []Item{item("first", "do"), item("second", "do")}, Doctrine: []Paragraph{
			governing(governsStanding), governing(governsCampaign), governing(governsRow, "first"), governing(governsRow, "first", "second"),
		}}
		advanced, err := Advance(document, "first", "do")
		if err != nil {
			t.Fatalf("landing a row's last step while paragraphs govern it and the campaign is refused: %v", err)
		}
		if len(advanced.Items) != 1 || !reflect.DeepEqual(advanced.Doctrine, document.Doctrine) {
			t.Fatalf("landing the first row's last step left %d rows and the doctrine %+v, want one row and every paragraph as it was", len(advanced.Items), advanced.Doctrine)
		}
		if len(document.Items) != 2 || len(document.Doctrine) != 4 {
			t.Fatal("the advance changed the caller's plan")
		}
	})

	t.Run("the close-out refuses a paragraph whose campaign or rows are closed", func(t *testing.T) {
		// The close-out row has two steps, and a row beside it is still open when it closes.
		closed := func(paragraphs ...Paragraph) error {
			document := Plan{Campaign: "fixture", Items: []Item{item(closeoutItem, "check", "close"), item("open", "do")}, Doctrine: paragraphs}
			checked, err := Advance(document, closeoutItem, "check")
			if err != nil {
				t.Fatalf("the close-out row's first step, which closes nothing, is refused: %v", err)
			}
			_, err = Advance(checked, closeoutItem, "close")
			return err
		}
		for name, paragraph := range map[string]Paragraph{
			"standing":                    governing(governsStanding),
			"a row that is still open":    governing(governsRow, "open"),
			"rows, one of them open":      governing(governsRow, "landed", "open"),
			"the campaign, made standing": {Governs: governsStanding, Text: governing(governsCampaign).Text},
		} {
			if err := closed(governing(governsStanding), paragraph); err != nil {
				t.Errorf("the close-out is refused for a paragraph that governs %s: %v", name, err)
			}
		}
		for name, paragraph := range map[string]Paragraph{
			"the campaign":           governing(governsCampaign),
			"a row that has landed":  governing(governsRow, "landed"),
			"rows that have landed":  governing(governsRow, "landed", "earlier"),
			"the close-out row only": governing(governsRow, closeoutItem),
		} {
			if err := closed(governing(governsStanding), paragraph); err == nil || !strings.Contains(err.Error(), "paragraph 2") {
				t.Errorf("the close-out passed with a paragraph that governs %s, or its refusal does not name paragraph 2: %v", name, err)
			}
		}
		if err := closed(governing(governsStanding)); err != nil {
			t.Errorf("the close-out is refused once the paragraph is removed: %v", err)
		}
		// A close-out row with no step leaves whole, and is held to the same.
		whole := Plan{Campaign: "fixture", Items: []Item{item(closeoutItem)}, Doctrine: []Paragraph{governing(governsCampaign)}}
		if _, err := Advance(whole, closeoutItem, "."); err == nil || !strings.Contains(err.Error(), "paragraph 1") {
			t.Errorf("a close-out row with no step left the plan with a campaign paragraph, or its refusal does not name paragraph 1: %v", err)
		}
	})

	const before = `{"campaign":"fixture","doctrine":"One rule, of two sentences. The second.\n\nAnother rule.\n\nA third.","items":[` +
		`{"id":"kept","title":"row","status":"open","steps":[{"id":"do","title":"do","status":"open","verify":"true"}]}]}`

	t.Run("the doctrine as one string is read and never saved", func(t *testing.T) {
		document, err := Parse([]byte(before))
		if err != nil || len(document.Doctrine) != 0 || !strings.HasPrefix(document.DoctrineBefore, "One rule, of two sentences.") {
			t.Fatalf("a plan that holds its doctrine as one string read as %+v (%v), want it read with the string kept apart from the paragraphs", document, err)
		}
		path := filepath.Join(t.TempDir(), "plan.json")
		if err := Save(path, document); err == nil || !strings.Contains(err.Error(), "paragraph") {
			t.Fatalf("saving the one-string doctrine: %v, want it refused with the instruction to store paragraphs", err)
		}
		if _, err := os.Stat(path); !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("the refused save wrote the plan (stat: %v)", err)
		}
		both := strings.Replace(before, `"items"`, `"paragraphs":[`+standing+`],"items"`, 1)
		if _, err := Parse([]byte(both)); err == nil {
			t.Fatal("a plan that holds its doctrine both as one string and as paragraphs was read")
		}
	})

	t.Run("the one string converts to standing paragraphs word for word", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "plan.json")
		if err := os.WriteFile(path, []byte(before), 0o644); err != nil {
			t.Fatal(err)
		}
		held, err := Load(path)
		if err != nil {
			t.Fatal(err)
		}
		converted := standingDoctrine(held)
		if err := Save(path, converted); err != nil {
			t.Fatalf("saving the converted plan: %v", err)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var keys map[string]json.RawMessage
		if err := json.Unmarshal(data, &keys); err != nil {
			t.Fatal(err)
		}
		if _, old := keys["doctrine"]; old || keys["paragraphs"] == nil {
			t.Fatalf("the saved plan holds the keys of %s, want paragraphs and no doctrine string", data)
		}
		loaded, err := Load(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := Validate(loaded); err != nil || !reflect.DeepEqual(loaded, converted) {
			t.Fatalf("the saved plan loads as %+v (%v), want what was saved: %+v", loaded, err, converted)
		}
		var texts []string
		for _, paragraph := range loaded.Doctrine {
			if paragraph.Governs != governsStanding || len(paragraph.Rows) != 0 {
				t.Errorf("the converted paragraph %q governs %s %q, want standing", paragraph.Text, paragraph.Governs, paragraph.Rows)
			}
			texts = append(texts, paragraph.Text)
		}
		if len(texts) != 3 || strings.Join(texts, "\n\n") != held.DoctrineBefore {
			t.Fatalf("the conversion made the paragraphs %q of the string %q, want its three paragraphs word for word", texts, held.DoctrineBefore)
		}
	})

	t.Run("the checkout's plan holds its doctrine as paragraphs", func(t *testing.T) {
		document := loadCampaignPlan(t)
		if document.DoctrineBefore != "" {
			t.Fatalf("%s still holds its doctrine as one string of %d bytes; store each paragraph with what it governs", Path, len(document.DoctrineBefore))
		}
		governed := map[string]int{}
		for _, paragraph := range document.Doctrine {
			governed[paragraph.Governs]++
		}
		t.Logf("%s holds %d doctrine paragraphs: %d standing, %d of the campaign, %d of rows", Path, len(document.Doctrine), governed[governsStanding], governed[governsCampaign], governed[governsRow])
	})
}
