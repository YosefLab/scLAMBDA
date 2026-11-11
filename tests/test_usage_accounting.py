"""Usage follows provider invocations, including calls absent from prompt fields."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace

import pytest

from sclambda.llm.api import LLMClient
from sclambda.llm.usage import summarize, usage_context
from sclambda.reporting.pipeline import PipelineReports


def sdk_reply(content='answer', usage=None):
    return SimpleNamespace(content=content, usage_metadata=usage if usage is not None else {
        'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120,
        'input_token_details': {'cache_read': 80, 'cache_creation': 10},
        'output_token_details': {'reasoning': 15}},
        response_metadata={'stop_reason': 'end_turn'}, id='response')


def client_with_replies(monkeypatch, replies, **kwargs):
    client = LLMClient(provider='anthropic', initial_retry_delay=0, **kwargs)
    def invoke(messages):
        value = replies.pop(0)
        if isinstance(value, Exception):
            raise value
        return value
    async def ainvoke(messages):
        return invoke(messages)
    monkeypatch.setattr(client, '_create_llm', lambda **kw: SimpleNamespace(invoke=invoke, ainvoke=ainvoke))
    return client


def test_request_usage_includes_system_images_and_reasoning_without_double_count(monkeypatch, tmp_path):
    client = client_with_replies(monkeypatch, [sdk_reply()], usage_log_path=tmp_path / 'requests.jsonl')
    response = client.query('very short', system='system text', image_base64=['payload1', 'payload2'])
    events = client.usage_ledger.snapshot()
    assert response.usage['output_token_details']['reasoning'] == 15
    assert events[0]['image_count'] == 2 and 'system text' in events[0]['input_prompt']
    assert 'payload1' not in json.dumps(events)
    totals = summarize(events + events)  # repeated copies of one event are not extra requests
    assert (totals['input_tokens'], totals['output_tokens']) == (100, 20)
    assert totals['attempts'] == 1 and totals['complete']
    assert json.loads((tmp_path / 'requests.jsonl').read_text()) == events[0]


def test_transport_retry_and_missing_usage_are_unknown_not_zero(monkeypatch):
    client = client_with_replies(monkeypatch, [TimeoutError('timeout'), sdk_reply(), sdk_reply(usage={})])
    client.query('retry')
    client.query('missing usage')
    events = client.usage_ledger.snapshot()
    assert events[0]['request_id'] == events[1]['request_id'] != events[2]['request_id']
    assert [e['attempt'] for e in events] == [1, 2, 1]
    summary = summarize(events)
    assert summary['attempts'] == 3 and summary['responses'] == 2
    assert summary['input_tokens'] == 100 and summary['attempts_missing_usage'] == 2
    assert not summary['complete']


def test_async_and_parallel_contexts_are_isolated(monkeypatch):
    client = client_with_replies(monkeypatch, [sdk_reply() for _ in range(9)])
    def query(cid):
        with usage_context(cluster_id=cid, role='actor'):
            client.query(cid)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(query, map(str, range(8))))
    asyncio.run(client.aquery('async', system='async system'))
    events = client.usage_ledger.snapshot()
    assert {e['cluster_id'] for e in events[:-1]} == set(map(str, range(8)))
    assert 'cluster_id' not in events[-1] and 'role' not in events[-1]
    assert summarize(events)['input_tokens'] == 900


def test_celltype_loop_counts_every_round_and_critic_correction(monkeypatch, tmp_path):
    from test_simplified_annotation import harness, ACTOR, decision
    invalid = decision(); invalid['support'][0]['measurement_id'] = 'missing'
    replies = [ACTOR, invalid, decision('best_effort', 'Another useful test'), ACTOR, decision()]
    node, state, _ = harness(monkeypatch, [], rounds=2)
    node.llm_client = client_with_replies(monkeypatch, [sdk_reply(json.dumps(v)) for v in replies])
    result = node(state)
    assert result['cell_type'] == 'Chosen identity'
    events = node.llm_client.usage_ledger.snapshot()
    assert [e['role'] for e in events] == ['actor', 'critic', 'critic_repair', 'actor', 'critic']
    assert all(e['node'] == 'CTDossierAgentNode' and e['cluster_id'] == '0' for e in events)
    report = PipelineReports()
    report.name = 'celltype'; report._usage_events = events; report._last_results = [result]
    path = report._write_prompt_history(tmp_path / 'prompts.md')
    text = path.read_text()
    assert 'Total input tokens: `500`' in text and 'Total output tokens: `100`' in text
    assert 'critic_repair' in text
    usage_path = report._write_usage_report(None, {'excel_path': tmp_path / 'labels.xlsx', 'prompts_path': False})
    assert json.loads(usage_path.read_text())['summary']['attempts'] == 5


def test_spatial_final_recovery_calls_are_counted(monkeypatch):
    from test_spatial_decisions import harness, ACTOR, decision
    final = decision(); final.pop('resolution')
    node, state, _, _ = harness(monkeypatch, [], rounds=1)
    # Main critic fails, first final critic fails, second final critic succeeds.
    replies = [ACTOR, {}, {}, final]
    node.llm_client = client_with_replies(monkeypatch, [sdk_reply(json.dumps(v)) for v in replies])
    result = node(state)
    assert result['niche_dossier']['critic_decision']['output_origin'] == 'final_critic'
    events = node.llm_client.usage_ledger.snapshot()
    assert [e['role'] for e in events] == ['actor', 'critic', 'final_critic', 'final_critic']
    assert summarize(events)['output_tokens'] == 80


def test_annotation_run_totals_include_setup_and_do_not_leak_between_runs(monkeypatch, tmp_path):
    import sclambda.main as api
    from sclambda.framework.base_node import BaseNode
    class Node(BaseNode):
        def execute(self, state):
            self.llm_client.query('parse correction', power='weak')
            return dict(state)
    class Pipeline(PipelineReports):
        name = 'fake'
        max_depth_override = None
        validate_params = staticmethod(lambda **kw: None)
        resolve_initial_clusters = staticmethod(lambda *a: ['0', '1'])
        def __init__(self, accessor, llm_client, **kwargs):
            self.llm_client = llm_client
        def run(self, initial_clusters, **kw):
            self.llm_client.query('shared catalog')
            node = Node(None, None, self.llm_client)
            with ThreadPoolExecutor(max_workers=2) as pool:
                rows = list(pool.map(node, [{'cluster_id': cid} for cid in initial_clusters]))
            self._last_results = rows
            return rows
        def format_result(self, results, **kw):
            return {'results': results}
    monkeypatch.setattr(api, 'get_pipeline_class', lambda name: Pipeline)
    lam = object.__new__(api.Lambda)
    lam.accessor = None; lam._n_top_genes = 20; lam._subcluster_n_neighbors = 30
    lam._column_cache = False
    lam.llm_client = client_with_replies(monkeypatch, [sdk_reply() for _ in range(6)])
    first = lam.annotate(obs_columns='cluster')
    first_pipe = lam._last_pipeline
    second = lam.annotate(obs_columns='cluster')
    assert first['llm_usage']['input_tokens'] == second['llm_usage']['input_tokens'] == 300
    assert first['llm_usage']['run_id'] != second['llm_usage']['run_id']
    assert len(first_pipe._get_usage_events()) == 3
    assert len(lam.llm_client.usage_ledger.snapshot()) == 6
    events = second['llm_usage_events']
    assert 'cluster_id' not in events[0] and {e['cluster_id'] for e in events[1:]} == {'0', '1'}


def test_legacy_prompt_estimates_are_not_claimed_as_measured_usage(tmp_path):
    report = PipelineReports()
    report.accessor = SimpleNamespace()
    report.name = 'celltype'; report._cluster_name_map = {}; report._prompt_history_fields = ('generation_prompt',)
    report._last_results = [{'cluster_id': '0', 'generation_prompt': 'old prompt', 'generation_response': 'old response'}]
    text = report._write_prompt_history(tmp_path / 'legacy.md').read_text()
    assert 'Incomplete' in text and 'Do not use as total run usage' in text
    assert '- Total input tokens:' not in text
