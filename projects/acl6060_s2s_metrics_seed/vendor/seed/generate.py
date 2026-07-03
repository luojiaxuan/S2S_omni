# -*- coding: utf-8 -*-
"""
Seed (ByteDance AST) S2S: wav → tgt wav (with silence gaps) + timeline.json + transcript.txt

用法:
  # 单个文件
  python seed/generate.py input/full_wavs/2022.acl-long.268.wav \
      --out-dir output/seed/en_ja --src-lang en --tgt-lang ja

  # 批量（整个目录）
  python seed/generate.py --input-dir input/full_wavs/ \
      --out-dir output/seed/en_ja --src-lang en --tgt-lang ja

依赖:
  pip install soundfile numpy websockets "protobuf>=6.31"
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import websockets
from websockets import Headers
from google.protobuf.json_format import MessageToDict

# -- protobuf path --
_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_cur, "python_protogen"))
from products.understanding.ast.ast_service_pb2 import TranslateRequest, TranslateResponse
from common.events_pb2 import Type

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
INPUT_RATE       = 16000
CHUNK_SAMPLES    = 1600        # 100ms @ 16kHz mono 16-bit → 3200 bytes
CHUNK_DURATION   = 0.1         # seconds，与发送间隔对应
TARGET_RATE      = 24000
TARGET_FORMAT    = "pcm_s16le"
BYTES_PER_SAMPLE = 2           # 16-bit


@dataclass
class Config:
    ws_url:      str = "wss://openspeech.bytedance.com/api/v4/ast/v2/translate"
    api_key:     str = ""    # 新版控制台单一 Key（优先使用）
    app_key:     str = ""    # 旧版两段式鉴权
    access_key:  str = ""
    resource_id: str = "volc.service_type.10053"


# ---------------------------------------------------------------------------
# 音频工具
# ---------------------------------------------------------------------------
def detect_audio_signature(data: bytes) -> str:
    if not data:
        return "empty"
    if data[:4] == b"OggS":
        return "ogg"
    if data[:4] == b"RIFF":
        return "wav_container"
    return "pcm"


def save_pcm_as_wav(pcm_bytes: bytes, wav_path: Path,
                    sample_rate: int = TARGET_RATE,
                    sample_width: int = BYTES_PER_SAMPLE,
                    channels: int = 1):
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def render_pcm_with_gaps(timeline: List[dict], pcm_chunks: List[bytes],
                         rate: int = TARGET_RATE,
                         t0: Optional[float] = None) -> bytes:
    """
    按 heard_start 时间轴将各 PCM chunk 拼接，gap 处补零静音。
    t0 = 时间轴基准（通常为 first_send_timestamp）。
    """
    if not timeline or not pcm_chunks:
        return b""
    if t0 is None:
        t0 = timeline[0]["heard_start"]

    rendered = bytearray()
    write_cursor = 0  # 已写入的 samples 数
    for ent, chunk in zip(timeline, pcm_chunks):
        target_pos = max(0, int(round((ent["heard_start"] - t0) * rate)))
        if target_pos > write_cursor:
            silence_samples = target_pos - write_cursor
            rendered.extend(b"\x00\x00" * silence_samples)
            write_cursor += silence_samples
        rendered.extend(chunk)
        write_cursor += len(chunk) // BYTES_PER_SAMPLE
    return bytes(rendered)


def read_wav_as_chunks(audio_path: str) -> List[bytes]:
    """读取 wav，重采样到 16kHz 单声道，返回 raw PCM bytes 列表（无 WAV 头）。"""
    data, sr = sf.read(audio_path, dtype="int16", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True).astype(np.int16)
    data = data[:, 0]
    if sr != INPUT_RATE:
        n = len(data)
        new_n = int(n * INPUT_RATE / sr)
        data = np.interp(
            np.linspace(0, n - 1, new_n),
            np.arange(n),
            data.astype(np.float64),
        ).astype(np.int16)
    chunks = []
    for i in range(0, len(data), CHUNK_SAMPLES):
        chunk = data[i: i + CHUNK_SAMPLES]
        if chunk.size > 0:
            chunks.append(chunk.tobytes())
    return chunks


# ---------------------------------------------------------------------------
# Protobuf 消息构建
# ---------------------------------------------------------------------------
def _base_req(session_id: str, event, src_lang: str, tgt_lang: str) -> TranslateRequest:
    req = TranslateRequest()
    req.request_meta.SessionID = session_id
    req.event = event
    req.user.uid = "seed_generate"
    req.user.did = "seed_generate"
    req.source_audio.format = "wav"
    req.source_audio.rate = INPUT_RATE
    req.source_audio.bits = 16
    req.source_audio.channel = 1
    req.target_audio.format = TARGET_FORMAT
    req.target_audio.rate = TARGET_RATE
    req.request.mode = "s2s"
    req.request.source_language = src_lang
    req.request.target_language = tgt_lang
    return req


def make_start(session_id, src, tgt):
    return _base_req(session_id, Type.StartSession, src, tgt).SerializeToString()

def make_chunk(session_id, chunk: bytes, src, tgt):
    req = _base_req(session_id, Type.TaskRequest, src, tgt)
    req.source_audio.binary_data = chunk
    return req.SerializeToString()

def make_finish(session_id, src, tgt):
    return _base_req(session_id, Type.FinishSession, src, tgt).SerializeToString()


def build_headers(conf: Config, conn_id: str) -> Headers:
    if conf.api_key:
        return Headers({
            "X-Api-Key":         conf.api_key,
            "X-Api-Resource-Id": conf.resource_id,
            "X-Api-Connect-Id":  conn_id,
        })
    return Headers({
        "X-Api-App-Key":     conf.app_key,
        "X-Api-Access-Key":  conf.access_key,
        "X-Api-Resource-Id": conf.resource_id,
        "X-Api-Connect-Id":  conn_id,
    })


# ---------------------------------------------------------------------------
# 核心翻译函数
# ---------------------------------------------------------------------------
async def translate_one(conf: Config, audio_path: str, out_dir: str,
                        src_lang: str = "en", tgt_lang: str = "ja"):
    base         = Path(audio_path).stem
    out_wav      = Path(out_dir) / f"{base}.wav"
    out_timeline = Path(out_dir) / f"{base}_timeline.json"
    out_txt      = Path(out_dir) / f"{base}.txt"
    os.makedirs(out_dir, exist_ok=True)

    # 1. 读取并重采样源音频
    try:
        audio_chunks = read_wav_as_chunks(audio_path)
    except Exception as e:
        logging.error("读取音频失败 %s: %s", audio_path, e)
        return
    total_dur = sum(len(c) for c in audio_chunks) / (INPUT_RATE * BYTES_PER_SAMPLE)
    logging.info("源音频: %.1fs, %d chunks", total_dur, len(audio_chunks))

    # 2. 建立 WebSocket 连接
    conn_id = str(uuid.uuid4())
    try:
        conn = await websockets.connect(
            conf.ws_url,
            additional_headers=build_headers(conf, conn_id),
            max_size=1_000_000_000,
            ping_interval=None,
        )
        logging.info("已连接, X-Tt-Logid=%s", conn.response.headers.get("X-Tt-Logid"))
    except Exception as e:
        logging.error("连接失败: %s", e)
        if "401" in str(e):
            logging.error("HTTP 401: 鉴权失败，检查 AST_APP_KEY / AST_ACCESS_KEY")
        return

    # 3. StartSession
    session_id = str(uuid.uuid4())
    await conn.send(make_start(session_id, src_lang, tgt_lang))
    init_raw = await conn.recv()
    init_resp = TranslateResponse(); init_resp.ParseFromString(init_raw)
    if init_resp.event != Type.SessionStarted:
        logging.error("建联失败 event=%d msg=%s", init_resp.event,
                      init_resp.response_meta.Message)
        await conn.close()
        return
    logging.info("Session 已启动 (ID=%s)", session_id)

    # 4. 并发发送 + 接收
    first_send_timestamp: Optional[float] = None
    recv_audio   = bytearray()
    recv_text:   List[str]   = []
    pcm_chunks:  List[bytes] = []
    timeline:    List[dict]  = []
    chunk_id     = 0
    prev_heard_end = 0.0
    first_logged = False

    async def send_audio():
        # 绝对时基：第 i 块的发送时刻锁定在 first_send_timestamp + i*CHUNK_DURATION。
        # - 永不超前（等价于真麦克风：模型在该时刻前拿不到未来音频）
        # - 落后时不睡、立即补发追平，误差不累积（避免持续低于 1x 触发 AudioSendSlow）
        nonlocal first_send_timestamp
        first_send_timestamp = time.time()
        for i, chunk in enumerate(audio_chunks):
            target = first_send_timestamp + i * CHUNK_DURATION
            now = time.time()
            if now < target:
                await asyncio.sleep(target - now)
            await conn.send(make_chunk(session_id, chunk, src_lang, tgt_lang))
        await conn.send(make_finish(session_id, src_lang, tgt_lang))
        logging.info("已发送 FinishSession")

    sender = asyncio.create_task(send_audio())

    try:
        while True:
            raw = await conn.recv()
            resp = TranslateResponse(); resp.ParseFromString(raw)

            if resp.event in (Type.SessionFailed, Type.SessionCanceled):
                logging.error("Session 异常 event=%d msg=%s",
                              resp.event, resp.response_meta.Message)
                break
            if resp.event == Type.SessionFinished:
                break
            if resp.event == Type.UsageResponse:
                continue

            if resp.data:
                if not first_logged:
                    first_logged = True
                    sig0 = detect_audio_signature(resp.data)
                    logging.info("首包 signature=%s bytes=%d", sig0, len(resp.data))
                    if sig0 != "pcm":
                        logging.warning("请求 PCM 但收到 %s，将尝试回退处理", sig0)

                duration_sec    = len(resp.data) / (TARGET_RATE * BYTES_PER_SAMPLE)
                receive_ts      = time.time()
                prev_recv_ts    = timeline[-1]["receive_timestamp"] if timeline else None
                receive_gap_sec = (receive_ts - prev_recv_ts) if prev_recv_ts else 0.0

                if not timeline:
                    heard_start    = receive_ts
                    gap_before_sec = 0.0
                else:
                    heard_start    = max(receive_ts, prev_heard_end)
                    gap_before_sec = max(0.0, heard_start - prev_heard_end)

                heard_end      = heard_start + duration_sec
                prev_heard_end = heard_end

                timeline.append({
                    "chunk_id":        chunk_id,
                    "receive_timestamp": float(receive_ts),
                    "duration_sec":    float(duration_sec),
                    "receive_gap_sec": float(receive_gap_sec),
                    "gap_before_sec":  float(gap_before_sec),
                    "heard_start":     float(heard_start),
                    "heard_end":       float(heard_end),
                    "receive_time_sec": float(receive_ts - first_send_timestamp)
                                        if first_send_timestamp else None,
                    "output_time_sec": float(heard_start - first_send_timestamp)
                                       if first_send_timestamp else None,
                })
                pcm_chunks.append(bytes(resp.data))
                recv_audio.extend(resp.data)
                chunk_id += 1

            if resp.text and resp.event == Type.TranslationSubtitleEnd:
                recv_text.append(resp.text)

    except Exception as e:
        logging.error("接收异常: %s", e)
    finally:
        # session 已失败/结束时，sender 并不知情，会继续把剩余音频空发进死连接
        # （TCP send 不会立刻报错），白白空转到发完为止。这里主动 cancel 掉。
        sender.cancel()
        try:
            await sender
        except asyncio.CancelledError:
            pass
        await conn.close()

    if not recv_audio:
        logging.error("未收到任何音频，跳过保存")
        return

    # 5. 保存 WAV
    sig = detect_audio_signature(bytes(recv_audio))
    logging.info("最终音频 signature=%s  total_bytes=%d", sig, len(recv_audio))

    if sig == "ogg":
        try:
            import av, io as _io
            buf = _io.BytesIO(bytes(recv_audio))
            container = av.open(buf, format="ogg")
            stream = container.streams.audio[0]
            rate = stream.sample_rate
            parts = []
            for frame in container.decode(stream):
                arr = frame.to_ndarray()
                if arr.dtype.kind == "f":
                    arr = (arr * 32767).clip(-32768, 32767).astype(np.int16)
                else:
                    arr = arr.astype(np.int16)
                arr = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0).astype(np.int16)
                parts.append(arr)
            container.close()
            pcm_decoded = np.concatenate(parts).tobytes() if parts else b""
            save_pcm_as_wav(pcm_decoded, out_wav, sample_rate=rate)
            logging.info("OGG→PCM 已保存: %s", out_wav)
        except Exception as e:
            logging.error("OGG 解码失败（pip install av）: %s", e)

    elif sig == "wav_container":
        out_wav.write_bytes(bytes(recv_audio))
        logging.info("WAV container 直接保存: %s", out_wav)

    else:  # pcm
        t0 = first_send_timestamp or (timeline[0]["heard_start"] if timeline else 0.0)
        rendered = render_pcm_with_gaps(timeline, pcm_chunks, rate=TARGET_RATE, t0=t0)

        content_sec = len(recv_audio) / (TARGET_RATE * BYTES_PER_SAMPLE)
        heard_sec   = len(rendered)   / (TARGET_RATE * BYTES_PER_SAMPLE)
        total_gap   = sum(x.get("gap_before_sec", 0.0) for x in timeline)
        logging.info("[LEN] content=%.3fs  heard=%.3fs  silence=%.3fs  gap_sum=%.3fs",
                     content_sec, heard_sec, heard_sec - content_sec, total_gap)

        save_pcm_as_wav(rendered, out_wav, sample_rate=TARGET_RATE)
        logging.info("Gap-WAV 已保存: %s", out_wav)

    # 6. 保存 timeline
    payload = {"first_send_timestamp": first_send_timestamp, "timeline": timeline}
    with open(out_timeline, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logging.info("timeline 已保存: %s", out_timeline)

    # 7. 保存 transcript
    transcript = "\n".join(t for t in recv_text if t)
    out_txt.write_text(transcript, encoding="utf-8")
    logging.info("transcript 已保存: %s", out_txt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(
        description="Seed AST S2S: wav → tgt wav (gap-rendered) + timeline.json + transcript.txt"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("audio_path", nargs="?", help="单个输入 wav 文件")
    group.add_argument("--input-dir", help="批量：输入目录（处理所有 *.wav）")
    parser.add_argument("--out-dir",    default="output/seed",  help="输出目录")
    parser.add_argument("--src-lang",   default="en",           help="源语言（默认 en）")
    parser.add_argument("--tgt-lang",   default="ja",           help="目标语言（默认 ja）")
    parser.add_argument("--api-key",    default="",
                        help="新版控制台单一 API Key（优先于 app-key/access-key）")
    parser.add_argument("--app-key",    default="")
    parser.add_argument("--access-key", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    conf = Config(api_key=args.api_key, app_key=args.app_key, access_key=args.access_key)
    if not conf.api_key and not (conf.app_key and conf.access_key):
        logging.error("请设置 --api-key，或同时设置 --app-key/--access-key")
        return
    logging.info("鉴权方式: %s", "X-Api-Key（新版）" if conf.api_key else "App-Key + Access-Key（旧版）")

    if args.input_dir:
        wavs = sorted(Path(args.input_dir).glob("*.wav"))
        logging.info("批量处理 %d 个文件", len(wavs))
        for wav in wavs:
            logging.info("===== %s =====", wav.name)
            await translate_one(conf, str(wav), args.out_dir, args.src_lang, args.tgt_lang)
    else:
        await translate_one(conf, args.audio_path, args.out_dir, args.src_lang, args.tgt_lang)


if __name__ == "__main__":
    asyncio.run(main())
