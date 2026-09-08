#!/usr/bin/env python3
"""Bounded real-server packed-prefill regression; never starts/stops a service.

Run against an otherwise idle loopback candidate with APC and packed prefill
active. Requires its local tokenizer. Equal busy/idle budgets make the reported
lifetime packed-row maximum an unambiguous selected-budget check; unequal
budgets prove only the maximum-budget bound, not tail-inclusive iteration cost.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import ipaddress
import json
import os
from pathlib import Path
import secrets
import socket
import sys
import threading
import time
import urllib.parse

from serve_smoke_xpu import no_markup, validate_tool_calls


class CheckError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise CheckError(message)


def delta(before, after):
    result = {}
    for key, value in after.items():
        previous = before.get(key)
        if type(value) in (int, float) and type(previous) in (int, float):
            result[key] = value - previous
        elif isinstance(value, dict) and isinstance(previous, dict):
            difference = delta(previous, value)
            for added, count in value.items():
                if added not in previous and type(count) in (int, float):
                    difference[added] = count
            if difference:
                result[key] = difference
    return result


class Client:
    def __init__(self, args):
        parsed = urllib.parse.urlsplit(args.base_url.rstrip('/'))
        require(parsed.scheme == 'http' and not parsed.username and
                not parsed.query and not parsed.fragment,
                '--base-url must be a loopback HTTP URL')
        host = parsed.hostname
        require(host is not None, 'missing base URL hostname')
        if host != 'localhost':
            require(ipaddress.ip_address(host).is_loopback,
                    'external network targets are prohibited')
        require(parsed.path in ('', '/v1'), 'base URL path must be /v1 or empty')
        self.host, self.port = host, parsed.port or 80
        self.timeout = args.timeout
        self.deadline = time.monotonic() + args.deadline
        self.model = None
        self.samples = []
        self.sample_errors = []
        self.stop_monitor = threading.Event()
        self.monitor = None

    def remaining(self):
        left = min(self.timeout, self.deadline - time.monotonic())
        require(left > 0, 'overall client deadline exceeded')
        return left

    def connect(self, path, body=None):
        connection = http.client.HTTPConnection(
            self.host, self.port, timeout=self.remaining())
        try:
            data = None if body is None else json.dumps(body).encode()
            connection.request('GET' if body is None else 'POST', path, data,
                               {'Content-Type': 'application/json',
                                'Accept': 'text/event-stream' if body and
                                body.get('stream') else 'application/json'})
            response = connection.getresponse()
            if response.status != 200:
                raise CheckError(
                    f'{path}: HTTP {response.status}: '
                    f'{response.read(2048).decode(errors="replace")}')
            return connection, response
        except Exception:
            connection.close()
            raise

    def request(self, path, body=None):
        started = time.monotonic()
        connection, response = self.connect(path, body)
        try:
            payload = json.loads(response.read())
            require('error' not in payload, f'{path}: {payload.get("error")}')
            return payload, time.monotonic() - started
        finally:
            response.close()
            connection.close()

    def health(self):
        health, _ = self.request('/health')
        require(health.get('status') == 'ok' and not health.get('engine_error'),
                f'engine unhealthy: {health}')
        require(not health.get('quarantined_slots'), 'quarantined engine slots')
        for name in ('busy', 'idle'):
            budget = health[f'prefill_budget_{name}']
            require(type(budget) is int and budget >= 8 and budget % 8 == 0,
                    f'invalid reported {name} budget: {budget}')
        bound = max(health['prefill_budget_busy'], health['prefill_budget_idle'])
        require(health['prefill_max_rows'] <= bound,
                f'lifetime packed wave rows exceed maximum budget {bound}')
        require(health['prefill_last_rows'] <= bound,
                'last packed wave exceeds maximum budget')
        if health.get('paged'):
            require(health['kv_pool_blocks_used'] + health['kv_pool_blocks_free']
                    == health['kv_pool_blocks_total'], 'KV pool conservation failed')
            require(0 <= health['kv_remaining_reserved_blocks'] <=
                    health['kv_pool_blocks_free'], 'KV reservations exceed free blocks')
        self.samples.append({'elapsed_seconds': time.monotonic() - self.zero,
                             'health': health})
        return health

    def start_monitor(self):
        self.zero = time.monotonic()

        def poll():
            while not self.stop_monitor.wait(0.1):
                try:
                    self.health()
                except Exception as error:
                    self.sample_errors.append(str(error))
                    break
        self.monitor = threading.Thread(target=poll, daemon=True)
        self.monitor.start()

    def settle(self, cancelled_at_least=None):
        deadline = time.monotonic() + self.remaining()
        while time.monotonic() < deadline:
            health = self.health()
            idle = health['active'] == health['queued'] == 0
            idle = idle and health.get('capacity_waiting', 0) == 0
            if health.get('paged'):
                idle = idle and health['kv_active_blocks'] == 0
                idle = idle and health['kv_remaining_reserved_blocks'] == 0
            if cancelled_at_least is not None:
                idle = idle and health['cancelled_requests'] >= cancelled_at_least
            if idle:
                return health
            time.sleep(0.05)
        raise CheckError('scheduler/resources did not settle before timeout')

    def completion(self, prompt, count=16):
        body = {'model': self.model, 'prompt': prompt['text'],
                'max_tokens': count, 'ignore_eos': True, 'temperature': 0}
        payload, elapsed = self.request('/v1/completions', body)
        choice = payload['choices'][0]
        usage, metadata = payload['usage'], payload['x_knivesysl']
        require(usage['prompt_tokens'] == prompt['tokens'],
                f'prompt token count mismatch: {usage}')
        require(usage['completion_tokens'] == metadata['generated_tokens'] == count,
                f'generated token count mismatch: {usage}, {metadata}')
        require(choice['finish_reason'] == 'length', f'unexpected finish: {choice}')
        require(metadata['reused_tokens'] + metadata['prefilled_tokens'] ==
                prompt['tokens'], 'reused + prefilled prompt invariant failed')
        return {'prompt_tokens': prompt['tokens'], 'prompt_sha256': prompt['sha256'],
                'text': choice['text'], 'usage': usage, 'metadata': metadata,
                'seconds': elapsed}

    def wave(self, prompts):
        before = self.settle()
        started = time.monotonic()
        barrier = threading.Barrier(len(prompts))

        def worker(prompt):
            barrier.wait(timeout=self.remaining())
            try:
                return self.completion(prompt)
            except Exception as error:
                return {'prompt_tokens': prompt['tokens'], 'error': str(error)}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            rows = list(pool.map(worker, prompts))
        after = self.settle()
        return {'requests': rows, 'seconds': time.monotonic() - started,
                'before': before, 'after': after, 'health_delta': delta(before, after)}

    def stream(self, path, body):
        started = time.monotonic()
        connection, response = self.connect(path, {**body, 'stream': True,
                                                  'stream_options': {'include_usage': True}})
        result = {'fragments': [], 'reasoning': [], 'tool_deltas': [],
                  'finish_reason': None, 'usage': None, 'metadata': None,
                  'done': False}
        try:
            require('text/event-stream' in response.getheader('Content-Type', ''),
                    'stream response has incorrect content type')
            for line in response:
                require(time.monotonic() - started < self.timeout,
                        'stream exceeded client timeout')
                self.remaining()
                if not line.strip() or line.startswith(b':'):
                    continue
                require(line.startswith(b'data:'), f'invalid SSE line: {line!r}')
                data = line[5:].strip()
                if data == b'[DONE]':
                    result['done'] = True
                    break
                event = json.loads(data)
                require('error' not in event, f'stream error: {event}')
                if event.get('usage') is not None:
                    result['usage'] = event['usage']
                    result['metadata'] = event.get('x_knivesysl')
                for choice in event.get('choices', []):
                    result['finish_reason'] = choice.get('finish_reason') or result['finish_reason']
                    change = choice.get('delta') or {}
                    content = change.get('content') or choice.get('text')
                    if content:
                        result['fragments'].append(content)
                    if change.get('reasoning_content'):
                        result['reasoning'].append(change['reasoning_content'])
                    result['tool_deltas'].extend(change.get('tool_calls') or [])
            require(result['done'] and result['usage'] is not None and
                    result['metadata'] is not None, 'stream missing DONE/usage/metadata')
            result['text'] = ''.join(result['fragments'])
            result['seconds'] = time.monotonic() - started
            return result
        finally:
            response.close()
            connection.close()


class Prompts:
    def __init__(self, model_dir):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.expanduser(model_dir), local_files_only=True, trust_remote_code=True)
        self.variant = secrets.randbits(24)

    def make(self, tokens, index):
        words = (' alpha', ' beta', ' gamma')
        prefix = [words[index % 3]]
        bits = self.variant + index
        prefix.extend(words[(bits >> bit) & 1] for bit in range(min(24, tokens - 1)))
        text = ''.join(prefix) + words[index % 3] * (tokens - len(prefix))
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        require(len(ids) == tokens, f'real tokenizer count {len(ids)} != {tokens}')
        return {'text': text, 'tokens': tokens,
                'sha256': hashlib.sha256(text.encode()).hexdigest()}


def check_wave(wave):
    errors = [row['error'] for row in wave['requests'] if 'error' in row]
    require(not errors, '; '.join(errors))
    for name in ('request_errors', 'prefill_errors'):
        require(wave['health_delta'].get(name) == 0, f'{name} increased')


def stream_count(client, prompts, detail):
    prompt = prompts.make(17, 79)
    stream = client.stream('/v1/completions', {
        'model': client.model, 'prompt': prompt['text'],
        'max_tokens': 16, 'ignore_eos': True, 'temperature': 0})
    detail.update(stream)
    require(stream['usage']['prompt_tokens'] == prompt['tokens'],
            'raw streaming prompt count mismatch')
    require(stream['usage']['completion_tokens'] ==
            stream['metadata']['generated_tokens'] == 16,
            'raw streaming exact output count mismatch')
    require(stream['finish_reason'] == 'length', 'raw stream did not reach token limit')
    # Raw completions intentionally do not separate reasoning/tool markup.


def stop_split(client, prompts, detail):
    # Respect EOS here: forced post-EOS tokens can synthesize additional turns.
    # Exact ignore_eos counts are tested independently by stream_count/tails.
    body = {'model': client.model, 'max_tokens': 32, 'ignore_eos': False,
            'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False},
            'messages': [{'role': 'user', 'content':
                          'Reply with this exact phrase: cedar amber cobalt silver.'}]}
    encoded = prompts.tokenizer.apply_chat_template(
        body['messages'], tokenize=True, add_generation_prompt=True, enable_thinking=False)
    require(len(encoded) < 2048, 'stop prompt too large')
    baseline = client.stream('/v1/chat/completions', body)
    detail['baseline'] = baseline
    no_markup(baseline['text'], ''.join(baseline['reasoning']))
    require(0 < baseline['usage']['completion_tokens'] <= 32,
            'invalid baseline stream count')
    require(baseline['usage']['completion_tokens'] ==
            baseline['metadata']['generated_tokens'], 'baseline stream count mismatch')
    fragments = baseline['fragments']
    require(len(fragments) >= 2, 'need two real content events to exercise split stop')
    stop = None
    for end in range(2, len(fragments) + 1):
        candidate = ''.join(fragments[:end])
        ids = prompts.tokenizer(candidate, add_special_tokens=False).input_ids
        if len(ids) > 1 and candidate.isascii():
            stop = candidate
            detail['stop_spans_baseline_events'] = end
            detail['stop_token_count'] = len(ids)
            break
    require(stop is not None, 'no ASCII multi-token, multi-event stop in baseline')
    detail['stop'] = stop
    replay = client.stream('/v1/chat/completions', {**body, 'stop': [stop]})
    detail['stopped'] = replay
    require(replay['finish_reason'] == 'stop', 'split stop did not terminate generation')
    require(replay['text'] == '', 'stop or partial stop text leaked before match')
    require(0 < replay['usage']['completion_tokens'] <= 32, 'invalid stopped token count')
    require(replay['usage']['completion_tokens'] == replay['metadata']['generated_tokens'],
            'stopped stream usage/generated count disagree')
    no_markup(replay['text'], ''.join(replay['reasoning']))
    detail['settled'] = client.settle()
    detail['short_after'] = client.completion(prompts.make(9, 80), 8)


def tool_path(client, prompts, detail):
    tools = [{'type': 'function', 'function': {
        'name': 'lookup_temperature', 'description': 'Look up temperature for a city.',
        'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}},
                       'required': ['city']}}}]
    messages = [{'role': 'user', 'content': 'Call lookup_temperature for Oslo.'}]
    ids = prompts.tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    require(len(ids) < 2048, 'tool prompt too large')
    payload, elapsed = client.request('/v1/chat/completions', {
        'model': client.model, 'messages': messages, 'tools': tools,
        'max_tokens': 96, 'ignore_eos': False,
        'chat_template_kwargs': {'enable_thinking': False}})
    detail.update(response=payload, seconds=elapsed, prompt_tokens=len(ids))
    choice = payload['choices'][0]
    message = choice['message']
    no_markup(message.get('content'), message.get('reasoning_content'))
    require(0 < payload['usage']['completion_tokens'] <= 96,
            'invalid tool-path output count')
    require(payload['usage']['completion_tokens'] ==
            payload['x_knivesysl']['generated_tokens'], 'tool-path output count mismatch')
    calls = message.get('tool_calls') or []
    if choice['finish_reason'] == 'tool_calls':
        validate_tool_calls(calls)
    else:
        require(not calls, 'parsed tool calls lack tool_calls finish reason')
    detail['model_emitted_tool_call'] = choice['finish_reason'] == 'tool_calls'


def disconnect(client, prompts, stage, detail):
    before = client.settle()
    detail['before'] = before
    prompt = prompts.make(1024 if stage == 'admitted' else 17,
                          90 if stage == 'admitted' else 91)
    body = {'model': client.model, 'prompt': prompt['text'], 'stream': True,
            'ignore_eos': True, 'max_tokens': 256}
    started = time.monotonic()
    connection, response = client.connect('/v1/completions', body)
    try:
        if stage == 'admitted':
            deadline = time.monotonic() + client.remaining()
            while time.monotonic() < deadline:
                health = client.health()
                if health['active'] > 0:
                    detail['active_observed'] = health
                    break
                time.sleep(0.02)
            require('active_observed' in detail, 'disconnect request was never active')
        else:
            for line in response:
                require(time.monotonic() - started < client.timeout,
                        'disconnect stream exceeded timeout')
                client.remaining()
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                require(data != b'[DONE]', 'stream finished before disconnect')
                event = json.loads(data)
                if any(choice.get('text') for choice in event.get('choices', [])):
                    detail['first_content_event'] = event
                    break
            require('first_content_event' in detail, 'no decode content before disconnect')
    finally:
        # HTTP/1.0 responses can detach the socket from HTTPConnection.
        sock = connection.sock
        if sock is None and response.fp is not None:
            sock = getattr(getattr(response.fp, 'raw', None), '_sock', None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        response.close()
        connection.close()
    detail['disconnect_seconds'] = time.monotonic() - started
    after = client.settle(before['cancelled_requests'] + 1)
    detail.update(after=after, health_delta=delta(before, after))
    detail['short_after'] = client.completion(prompts.make(9, 92), 8)
    detail['recovery'] = client.settle()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout', type=float, default=60,
                        help='maximum seconds per request/recovery wait')
    parser.add_argument('--deadline', type=float, default=360,
                        help='overall HTTP run deadline in seconds')
    args = parser.parse_args()
    if args.timeout <= 0 or args.deadline <= 0:
        parser.error('timeouts must be positive')
    report = {'base_url': args.base_url, 'cases': [], 'failures': [],
              'budget_scope': 'Lifetime packed-row maximum <= max(reported busy,idle); '
                              'not tail-inclusive. Equal budgets disambiguate selection.'}
    client = None
    started = time.monotonic()

    def case(name, function):
        detail = {'name': name}
        report['cases'].append(detail)
        zero = time.monotonic()
        try:
            function(detail)
            detail['ok'] = True
        except Exception as error:
            detail['ok'] = False
            detail['error'] = str(error)
            report['failures'].append(f'{name}: {error}')
        detail['seconds'] = time.monotonic() - zero
        print(f'{name}: {"PASS" if detail["ok"] else "FAIL"}', flush=True)

    try:
        prompts = Prompts(args.model_dir)
        client = Client(args)
        client.start_monitor()
        initial = client.settle()
        report['initial_health'] = initial
        require(initial.get('packed_prefill'), 'candidate must enable packed prefill')
        require(initial['slots'] >= 2, 'width>=2 proof requires at least two slots')
        models, _ = client.request('/v1/models')
        client.model = models['data'][0]['id']
        report['model'] = client.model

        def tails(detail):
            wave = client.wave([prompts.make(n, i) for i, n in enumerate((1, 8, 9, 16, 17))])
            detail.update(wave)
            check_wave(wave)
        case('simultaneous-1-8-9-16-17', tails)

        def apc(detail):
            require(initial['apc_enabled'], 'APC reuse coverage requires APC enabled')
            medium = [prompts.make(n, i + 10) for i, n in enumerate((512, 768, 1024))]
            waves = detail['waves'] = []
            for _ in range(3):
                wave = client.wave(medium)
                waves.append(wave)
                check_wave(wave)
            cold = waves[0]['requests']
            require(all(row['metadata']['reused_tokens'] == 0 for row in cold),
                    'cold wave reused prior state; cold/warm comparison is not attested')
            for wave in waves[1:]:
                for reference, row in zip(cold, wave['requests']):
                    require(row['text'] == reference['text'],
                            f'cold/warm exact output mismatch for {row["prompt_tokens"]} tokens')
            require(all(row['metadata']['reused_tokens'] > 0
                        for row in waves[-1]['requests']), 'last wave lacks actual APC reuse')
            require(waves[-1]['health_delta']['apc_hits'] > 0, 'last wave has no APC hits')
            widths = waves[0]['health_delta'].get('prefill_width_hist', {})
            require(any(int(width) >= 2 and count > 0 for width, count in widths.items()),
                    f'no observed multi-request packed wave: {widths}')
        case('medium-packed-three-wave-apc', apc)
        case('raw-stream-exact-count', lambda detail: stream_count(client, prompts, detail))
        case('streaming-split-stop', lambda detail: stop_split(client, prompts, detail))
        case('structured-tool-marker-and-count', lambda detail: tool_path(client, prompts, detail))
        for stage in ('admitted', 'decode'):
            case(f'disconnect-{stage}',
                 lambda detail, stage=stage: disconnect(client, prompts, stage, detail))
        final = client.settle()
        report['final_health'] = final
        report['health_delta'] = delta(initial, final)
        for key in ('request_errors', 'prefill_errors'):
            require(final[key] == initial[key], f'{key} increased over regression run')
    except Exception as error:
        report['failures'].append(f'run: {error}')
    finally:
        if client is not None:
            client.stop_monitor.set()
            if client.monitor is not None:
                client.monitor.join(timeout=max(1, args.timeout + 1))
            report['health_samples'] = client.samples
            report['failures'].extend(f'health: {error}' for error in client.sample_errors)
        report['seconds'] = time.monotonic() - started
        report['ok'] = not report['failures']
        path = Path(args.output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'{"PASS" if report["ok"] else "FAIL"}: {args.output}', flush=True)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
