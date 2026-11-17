package evaluation

import (
	"testing"

	"overgo/internal/testcheck"
)

// TestAReadingsRoomIsWhatItsReferenceAnswersWrong holds the room a reading has (reading-power/do): media whose every
// question the reference answers right give a preset nothing to turn right; a reference wrong in four media leaves a
// best tail of one in sixteen, over the pass mark, so no gain can show; one wrong in six can, a preset needing to turn
// an answer right in five of them and break none; and a case of no medium is refused.
func TestAReadingsRoomIsWhatItsReferenceAnswersWrong(t *testing.T) {
	t.Parallel()
	read := func(wrongIn int) ReadingRoom {
		t.Helper()
		outcomes, media := map[string]bool{}, map[string]string{}
		for medium := range 10 {
			for question := range 3 {
				name := string(rune('a'+medium)) + string(rune('0'+question))
				outcomes[name], media[name] = !(medium < wrongIn && question == 0), string(rune('a'+medium))
			}
		}
		return testcheck.Must(ReadingRoomOf(outcomes, media))(t)
	}
	if none := read(0); none.Room != 0 || none.Media != 10 || none.Possible() || none.Needed != 0 {
		t.Errorf("a reference right on every question reads %+v; want no room and no gain possible", none)
	}
	if four := read(4); four.Room != 4 || four.BestTail != 1.0/16 || four.Possible() || four.Needed != 0 {
		t.Errorf("a reference wrong in four media reads %+v; want a best tail of 1/16, over the pass mark", four)
	}
	if six := read(6); six.Room != 6 || !six.Possible() || six.Needed != 5 {
		t.Errorf("a reference wrong in six media reads %+v; want a gain possible, needing five of them", six)
	}
	if _, err := ReadingRoomOf(map[string]bool{"x": true}, map[string]string{}); err == nil {
		t.Error("a case of no medium was read into a room")
	}
}
