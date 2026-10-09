package responses

import "testing"

func TestStreamSnapshotNativeResponses(t *testing.T) {
	sc := NewStreamConverter("openai-response")
	sc.Feed([]byte(": ping\n\n"))
	if sc.StreamSnapshot().Progress != 0 {
		t.Fatal(sc.StreamSnapshot())
	}
	sc.Feed([]byte("event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"r\"}}\n\n"))
	sc.Feed([]byte("event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\"}\n\n"))
	if sc.StreamSnapshot().TerminalSeen {
		t.Fatal("item done is not terminal")
	}
	sc.Feed([]byte("event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{}}\n\n"))
	s := sc.StreamSnapshot()
	if !s.Started || !s.DoneSeen || !s.TerminalSeen || s.TerminalKind != "completed" {
		t.Fatal(s)
	}
}

func TestResponsesDuplicateStartsDoNotAdvance(t *testing.T) {
	sc := NewStreamConverter("openai-response")
	created := "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"r\"}}\n\n"
	sc.Feed([]byte(created))
	p := sc.StreamSnapshot().Progress
	sc.Feed([]byte(created))
	if sc.StreamSnapshot().Progress != p {
		t.Fatalf("duplicate created advanced: %+v", sc.StreamSnapshot())
	}
	added := "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"m\"}}\n\n"
	sc.Feed([]byte(added))
	p = sc.StreamSnapshot().Progress
	sc.Feed([]byte(added))
	if sc.StreamSnapshot().Progress != p {
		t.Fatalf("duplicate item advanced")
	}
}

// UsageTrace captures the usage member of the terminal response event for
// the stream-finished log: openai-response passthrough (source route) and
// converted targets must both record it; a terminal without usage reports
// Seen=false.
func TestResponsesUsageTraceTerminal(t *testing.T) {
	newConverter := func(source string) *StreamConverter {
		sc := NewStreamConverter(source)
		return sc
	}
	completed := "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"r1\",\"usage\":{\"input_tokens\":34,\"output_tokens\":557,\"total_tokens\":591}}}\n\n"
	noUsage := "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"r1\"}}\n\n"

	sc := newConverter("openai-response")
	if _, _, err := sc.Feed([]byte(completed)); err != nil {
		t.Fatalf("Feed passthrough: %v", err)
	}
	trace := sc.UsageTrace()
	if !trace.Seen || trace.InputTokens != 34 || trace.OutputTokens != 557 {
		t.Fatalf("passthrough trace = %+v, want seen 34/557", trace)
	}

	sc = newConverter("openai")
	if _, _, err := sc.Feed([]byte(completed)); err != nil {
		t.Fatalf("Feed convert: %v", err)
	}
	trace = sc.UsageTrace()
	if !trace.Seen || trace.InputTokens != 34 || trace.OutputTokens != 557 {
		t.Fatalf("converted trace = %+v, want seen 34/557", trace)
	}

	sc = newConverter("openai-response")
	if _, _, err := sc.Feed([]byte(noUsage)); err != nil {
		t.Fatalf("Feed no-usage: %v", err)
	}
	if trace := sc.UsageTrace(); trace.Seen {
		t.Fatalf("terminal without usage must report Seen=false, got %+v", trace)
	}
}
