package messages

import "testing"

func TestStreamSnapshotMessages(t *testing.T) {
	sc := NewStreamConverter("openai-response")
	sc.Feed([]byte("event: ping\ndata: {}\n\n"))
	if sc.StreamSnapshot().Progress != 0 {
		t.Fatal(sc.StreamSnapshot())
	}
	sc.Feed([]byte("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\"}}\n\n"))
	sc.Feed([]byte("event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n"))
	s := sc.StreamSnapshot()
	if !s.Started || !s.FinishSeen || s.TerminalSeen {
		t.Fatal(s)
	}
	sc.Feed([]byte("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"))
	s = sc.StreamSnapshot()
	if !s.DoneSeen || !s.TerminalSeen {
		t.Fatal(s)
	}
}

func TestMessagesNullStopAndIncompleteToolAreNotTerminal(t *testing.T) {
	sc := NewStreamConverter("openai-response")
	frames := []string{"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\"}}\n\n", "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"c\",\"name\":\"f\"}}\n\n", "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"x\\\":\"}}\n\n", "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":null}}\n\n"}
	for _, f := range frames {
		if _, _, e := sc.Feed([]byte(f)); e != nil {
			t.Fatal(e)
		}
	}
	if sc.StreamSnapshot().FinishSeen {
		t.Fatal(sc.StreamSnapshot())
	}
	_, _, e := sc.Feed([]byte("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"))
	if e == nil || sc.StreamSnapshot().TerminalSeen {
		t.Fatalf("error=%v snapshot=%+v", e, sc.StreamSnapshot())
	}
}
